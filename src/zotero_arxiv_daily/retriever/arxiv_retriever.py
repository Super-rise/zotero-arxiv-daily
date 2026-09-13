import re

from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar, resolve_arxiv_metadata
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


class _Author:
    def __init__(self, name: str):
        self.name = name


class _FeedPaper:
    """Minimal arxiv.Result stand-in built from an RSS entry.

    Avoids the export API (which 429s from GitHub Actions runner IPs).
    Implements the attribute surface that Paper conversion and full-text
    extraction rely on.
    """

    def __init__(self, *, title: str, summary: str, entry_id: str,
                 authors: list[str], pdf_url: str | None, arxiv_id: str):
        self.title = title
        self.summary = summary
        self.entry_id = entry_id
        self.authors = [_Author(a) for a in authors if a]
        self.pdf_url = pdf_url
        self._arxiv_id = arxiv_id

    def source_url(self) -> str | None:
        if not self._arxiv_id:
            return None
        return f"https://arxiv.org/e-print/{self._arxiv_id}"


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest papers straight from the arxiv RSS feed. The entries
        # carry title/summary/authors/links, so the export API (which 429s
        # from GitHub Actions runner IPs) is not needed at all.
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        raw_papers = [
            self._entry_to_paper(i)
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            raw_papers = raw_papers[:5]
            if not raw_papers:
                logger.info("Debug: RSS feed empty (weekend/holiday). Falling back to listing page + OpenAlex...")
                raw_papers = self._debug_fallback_papers()

        return raw_papers

    def _debug_fallback_papers(self) -> list[ArxivResult]:
        import requests

        cat = self.config.source.arxiv.category[0]
        resp = requests.get(f"https://arxiv.org/list/{cat}/recent", timeout=60)
        resp.raise_for_status()
        ids: list[str] = []
        for aid in re.findall(r"/abs/(\d{4}\.\d{4,5})", resp.text):
            if aid not in ids:
                ids.append(aid)
            if len(ids) >= 5:
                break
        if not ids:
            logger.warning("Debug fallback found no arXiv IDs on listing page.")
            return []
        metas = resolve_arxiv_metadata(ids)
        papers = []
        for aid in ids:
            meta = metas.get(aid)
            if meta is None:
                continue
            papers.append(_FeedPaper(
                title=meta.get("title", ""),
                summary=meta.get("abstract", ""),
                entry_id=f"https://arxiv.org/abs/{aid}",
                authors=meta.get("authors", []),
                pdf_url=f"https://arxiv.org/pdf/{aid}",
                arxiv_id=aid,
            ))
        logger.info(f"Debug fallback produced {len(papers)} papers from listing + OpenAlex.")
        return papers

    @staticmethod
    def _entry_to_paper(entry) -> "_FeedPaper":
        aid = (entry.get("id") or "").removeprefix("oai:arXiv.org:")
        pdf_url = None
        for link in entry.get("links") or []:
            if link.get("rel") == "related" and link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break
        if pdf_url is None and aid:
            pdf_url = f"https://arxiv.org/pdf/{aid}"
        return _FeedPaper(
            title=(entry.get("title") or "").strip(),
            summary=re.sub(r"\s+", " ", entry.get("summary") or "").strip(),
            entry_id=entry.get("link") or (f"https://arxiv.org/abs/{aid}" if aid else ""),
            authors=[a.get("name", "") for a in (entry.get("authors") or [])],
            pdf_url=pdf_url,
            arxiv_id=aid,
        )

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
