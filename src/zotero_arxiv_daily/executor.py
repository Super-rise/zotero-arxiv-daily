from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
import re
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm

_ARXIV_ID_RE = re.compile(r'^(\d{4}\.\d{4,5})(?:v\d+)?\.(?:pdf|dvi)$', re.IGNORECASE)


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        if corpus:
            logger.info(f"Fetched {len(corpus)} zotero papers")
            return [CorpusPaper(
                title=c['data']['title'],
                abstract=c['data']['abstractNote'],
                added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
                paths=c['paths']
            ) for c in corpus]
        logger.info("No zotero papers with abstracts found. Falling back to arXiv attachment IDs...")
        return self._fetch_attachment_corpus(zot)

    def _fetch_attachment_corpus(self, zot) -> list[CorpusPaper]:
        """Build corpus from orphan PDF attachments named by arXiv ID (e.g. 2605.08764.pdf).

        Libraries created by bulk-adding arXiv PDFs may only contain attachments
        without parent metadata. Resolve each ID's title/abstract via the arXiv API.
        """
        items = zot.everything(zot.items(limit=200))
        id_to_date = {}
        for it in items:
            if it['data']['itemType'] != 'attachment':
                continue
            m = _ARXIV_ID_RE.match(it['data'].get('title', '') or '')
            if not m:
                continue
            id_to_date.setdefault(m.group(1), it['data'].get('dateAdded', ''))
        if not id_to_date:
            logger.warning("No arXiv-named attachments found. Corpus stays empty.")
            return []
        logger.info(f"Found {len(id_to_date)} arXiv IDs in attachments. Resolving metadata...")
        metas = self._resolve_arxiv_metadata(list(id_to_date.keys()))
        corpus = []
        for aid, date in id_to_date.items():
            meta = metas.get(aid)
            if meta is None:
                continue
            try:
                added = datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
            except (ValueError, TypeError):
                continue
            corpus.append(CorpusPaper(
                title=meta['title'],
                abstract=meta['abstract'],
                added_date=added,
                paths=[]
            ))
        logger.info(f"Built attachment corpus with {len(corpus)} papers")
        return corpus

    def _resolve_arxiv_metadata(self, arxiv_ids: list[str]) -> dict[str, dict]:
        """Resolve title/abstract for arXiv IDs.

        Primary: OpenAlex by arXiv DOI (10.48550/...). Secondary: Semantic
        Scholar batch API. The arXiv export API often returns HTTP 429 from
        shared GitHub Actions runner IPs, so it is avoided entirely.
        """
        import requests as _requests
        metas: dict[str, dict] = {}

        def _reconstruct_abstract(inv: dict | None) -> str:
            if not inv:
                return ""
            positions: dict[int, str] = {}
            for word, idxs in inv.items():
                for i in idxs:
                    positions[i] = word
            return " ".join(positions[i] for i in sorted(positions))

        session = _requests.Session()
        session.headers.update({"User-Agent": "zotero-arxiv-daily/1.0 (mailto:lijinxi2481@163.com)"})
        missing: list[str] = []
        for aid in arxiv_ids:
            try:
                resp = session.get(
                    f"https://api.openalex.org/works/doi:10.48550/arXiv.{aid}",
                    timeout=30,
                )
                if resp.status_code == 200:
                    m = resp.json()
                    title = (m.get("display_name") or "").strip()
                    abstract = _reconstruct_abstract(m.get("abstract_inverted_index"))
                    if title or abstract:
                        metas[aid] = {"title": title, "abstract": abstract}
                        continue
            except Exception as e:
                logger.debug(f"OpenAlex lookup failed for {aid}: {e}")
            missing.append(aid)
        if missing:
            try:
                resp = _requests.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": "title,abstract,externalIds"},
                    json={"ids": [f"ARXIV:{aid}" for aid in missing]},
                    timeout=90,
                )
                resp.raise_for_status()
                for item in resp.json():
                    if not item:
                        continue
                    ext = item.get("externalIds") or {}
                    aid = ext.get("ArXiv")
                    if not aid:
                        continue
                    title = item.get("title") or ""
                    abstract = (item.get("abstract") or "").replace("\n", " ")
                    if title or abstract:
                        metas[aid] = {"title": title, "abstract": abstract}
            except Exception as e:
                logger.warning(f"Semantic Scholar fallback failed: {e}")
        return metas
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    
    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
