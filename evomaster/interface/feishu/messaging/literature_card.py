"""Daily literature digest card builder for Feishu interactive messages."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_CARD_MAX_BYTES = 90 * 1024
_MAX_TITLE_LEN = 180
_MAX_SUMMARY_LEN = 1200
_HARD_PAPER_LIMIT = 40
_SOFT_PAPERS_PER_CARD = 8

_URL_RE = re.compile(r"https?://[^\s)]+")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*]\((https?://[^)]+)\)")
_NUMBERED_PAPER_RE = re.compile(r"(?m)^\s*(\d+)\.\s+\*\*(.+?)\*\*\s*$")
_RESULT_COUNT_RE = re.compile(r"(\d+)\s*(?:篇|papers?)\s*(?:新增|new)?", re.IGNORECASE)

_DATE_PATTERNS = [
    r"(?im)^\s*(?:[-*]\s*)?日期\s*[：:]\s*(.+?)\s*$",
    r"(?im)^\s*(?:[-*]\s*)?date\s*[：:]\s*(.+?)\s*$",
]
_TRACK_KEY_PATTERNS = [
    r"(?im)^\s*(?:[-*]\s*)?追踪键\s*[：:]\s*(.+?)\s*$",
    r"(?im)^\s*track_key\s*:\s*(.+?)\s*$",
]
_SCOPE_PATTERNS = [
    r"(?im)^\s*(?:[-*]\s*)?监控范围\s*[：:]\s*(.+?)\s*$",
]
_RESULT_PATTERNS = [
    r"(?im)^\s*(?:[-*]\s*)?结果\s*[：:]\s*(.+?)\s*$",
]

_FAIL_PATTERNS = [
    r"failed to scan arxiv",
    r"抓取失败",
]
_NO_NEW_PATTERNS = [
    r"today_no_new_papers",
    r"今日无新增",
    r"no new papers",
]


@dataclass
class DigestPaper:
    title: str
    paper_id: str = ""
    published: str = ""
    url: str = ""
    abstract_en: str = ""
    abstract_zh: str = ""
    relevance: str = ""

    @property
    def display_abstract(self) -> str:
        return self.abstract_zh or self.abstract_en


@dataclass
class DailyDigest:
    topic: str
    report_date: str
    track_key: str
    scope: str
    result_text: str
    papers: list[DigestPaper]
    no_new: bool = False
    fetch_failed: bool = False
    error_message: str = ""


def build_daily_literature_card_payloads(result_text: str, prefer_template: bool = True) -> list[str]:
    """Build one or more Feishu card payloads from daily digest text."""
    digest = _parse_daily_digest(result_text)
    if digest is None:
        return []

    papers = digest.papers[:_HARD_PAPER_LIMIT]
    if digest.no_new or digest.fetch_failed or not papers:
        card_obj = _build_card(
            digest=digest,
            papers=[],
            part_index=1,
            total_parts=1,
            global_start_index=1,
            prefer_template=prefer_template,
        )
        return [json.dumps(card_obj, ensure_ascii=False)]

    chunks = _split_papers_for_card_size(digest, papers, prefer_template=prefer_template)
    payloads: list[str] = []
    total_parts = len(chunks)
    start_index = 1

    for idx, chunk in enumerate(chunks, 1):
        card_obj = _build_card(
            digest=digest,
            papers=chunk,
            part_index=idx,
            total_parts=total_parts,
            global_start_index=start_index,
            prefer_template=prefer_template,
        )
        payloads.append(json.dumps(card_obj, ensure_ascii=False))
        start_index += len(chunk)

    logger.info(
        "Daily digest cards built: topic=%s papers=%d chunks=%d",
        digest.topic,
        len(papers),
        len(payloads),
    )
    return payloads


def _parse_daily_digest(raw_text: str) -> Optional[DailyDigest]:
    text = (raw_text or "").strip()
    if not text or not _looks_like_daily_digest(text):
        return None

    # Tolerate markdown key labels like:
    # "**追踪 Profile**：alice" -> "追踪 Profile：alice"
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)

    topic = _extract_topic(text) or "ArXiv Today"
    report_date = _extract_value(text, _DATE_PATTERNS) or datetime.now().strftime("%Y-%m-%d")
    track_key = _extract_value(text, _TRACK_KEY_PATTERNS)
    scope = _extract_scope(text)

    papers = _parse_markdown_papers(text)
    if not papers:
        papers = _parse_tool_observation_papers(text)
    if not papers:
        papers = _parse_loose_papers(text)
    papers = _dedupe_papers([_normalize_paper(p) for p in papers if _paper_has_value(p)])

    fetch_failed = any(re.search(pat, text, flags=re.IGNORECASE) for pat in _FAIL_PATTERNS)
    no_new = any(re.search(pat, text, flags=re.IGNORECASE) for pat in _NO_NEW_PATTERNS)

    result_line = _extract_value(text, _RESULT_PATTERNS)
    if result_line:
        result_text = _clean_line(result_line)
    elif fetch_failed:
        result_text = "抓取失败"
    elif no_new:
        result_text = "今日无新增"
    else:
        count = _parse_new_count(text)
        if count is None:
            count = len(papers)
        result_text = f"{count}篇新增"

    error_message = _extract_value(text, [r"(?im)^\s*Failed to scan arXiv:\s*(.+?)\s*$"])
    return DailyDigest(
        topic=_clean_line(topic),
        report_date=_clean_line(report_date),
        track_key=_clean_line(track_key),
        scope=_clean_line(scope),
        result_text=_clean_line(result_text),
        papers=papers,
        no_new=no_new,
        fetch_failed=fetch_failed,
        error_message=_clean_line(error_message),
    )


def _looks_like_daily_digest(text: str) -> bool:
    # 1) canonical tool output marker
    if "daily_arxiv_digest_ready" in text:
        return True

    # 2) markdown report header
    if re.search(r"(?im)^#{1,3}\s*(?:今日论文早报|AI 论文早报|ArXiv Today)\b", text):
        return True

    # 3) structured key-value block from tool output
    has_track = bool(re.search(r"(?im)^\s*track_key\s*:\s*.+$", text))
    has_result = bool(
        re.search(
            r"(?im)^\s*(?:new_total|returned_total|today_no_new_papers|papers:)\b",
            text,
        )
    )
    if has_track and has_result:
        return True

    # 4) Chinese report structure with date + result + body section
    has_cn_date = bool(re.search(r"(?im)^\s*(?:[-*]\s*)?日期\s*[：:]\s*.+$", text))
    has_cn_result = bool(re.search(r"(?im)^\s*(?:[-*]\s*)?结果\s*[：:]\s*.+$", text))
    has_body = ("新增论文" in text) or ("今日无新增" in text) or ("抓取失败" in text)
    return has_cn_date and has_cn_result and has_body


def _extract_topic(text: str) -> str:
    header_match = re.search(r"(?im)^#{1,3}\s*(?:今日论文早报|AI 论文早报|ArXiv Today)\s*[|｜]\s*(.+?)\s*$", text)
    if header_match:
        return header_match.group(1).strip()
    return _extract_value(
        text,
        [
            r"(?im)^\s*(?:[-*]\s*)?方向\s*[：:]\s*(.+?)\s*$",
            r"(?im)^\s*(?:[-*]\s*)?topic\s*[：:]\s*(.+?)\s*$",
            r"(?im)^\s*ArXiv Today\s*$",
        ],
    )


def _extract_scope(text: str) -> str:
    direct = _extract_value(text, _SCOPE_PATTERNS)
    if direct:
        return direct
    categories = _extract_value(text, [r"(?im)^\s*categories\s*:\s*(.+?)\s*$"])
    keyword = _extract_value(text, [r"(?im)^\s*keyword_expr\s*:\s*(.+?)\s*$"])
    if categories and keyword:
        return f"{categories} / {keyword}"
    return categories or keyword


def _extract_value(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def _parse_markdown_papers(text: str) -> list[DigestPaper]:
    matches = list(_NUMBERED_PAPER_RE.finditer(text))
    if not matches:
        return []

    papers: list[DigestPaper] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        title = _clean_line(match.group(2))
        paper_id = _extract_value(
            block,
            [
                r"(?im)^\s*[-*]\s*arXiv\s*[：:]\s*(.+?)\s*$",
                r"(?im)^\s*arXiv ID\s*[：:]\s*(.+?)\s*$",
            ],
        )
        published = _extract_value(block, [r"(?im)^\s*[-*]\s*(?:日期|date)\s*[：:]\s*(.+?)\s*$"])
        relevance = _extract_value(block, [r"(?im)^\s*[-*]\s*(?:相关性|relevance)\s*[：:]\s*(.+?)\s*$"])
        abstract_en = _extract_value(
            block,
            [
                r"(?im)^\s*[-*]\s*(?:摘要\(EN\)|abstract\(EN\)|abstract_en)\s*[：:]\s*(.+?)\s*$",
                r"(?im)^\s*abstract_en\s*[=:]\s*(.+?)\s*$",
            ],
        )
        abstract_zh = _extract_value(
            block,
            [
                r"(?im)^\s*[-*]\s*(?:摘要\(ZH\)|摘要\(中\)|zh_abstract|abstract_zh)\s*[：:]\s*(.+?)\s*$",
                r"(?im)^\s*zh_abstract\s*[=:]\s*(.+?)\s*$",
                r"(?im)^\s*[-*]\s*摘要\s*[：:]\s*(.+?)\s*$",
            ],
        )
        url = _extract_link(block)
        papers.append(
            DigestPaper(
                title=title,
                paper_id=paper_id,
                published=published,
                url=url,
                abstract_en=abstract_en,
                abstract_zh=abstract_zh,
                relevance=relevance,
            )
        )
    return papers


def _parse_tool_observation_papers(text: str) -> list[DigestPaper]:
    lines = text.splitlines()
    papers: list[DigestPaper] = []
    current: Optional[DigestPaper] = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = re.match(r"^\d+\.\s+id=([^\s|]+)\s+\|\s+date=([^|]+)\|\s+title=(.+)$", line)
        if m:
            if current is not None and _paper_has_value(current):
                papers.append(current)
            current = DigestPaper(
                title=_clean_line(m.group(3)),
                paper_id=_clean_line(m.group(1)),
                published=_clean_line(m.group(2)),
            )
            continue

        if current is None:
            continue
        if line.startswith("url="):
            current.url = _clean_line(line.split("=", 1)[1])
        elif line.startswith("abstract_short="):
            # backward compatibility: old field is usually Chinese one-line summary
            current.abstract_zh = _clean_line(line.split("=", 1)[1])
        elif line.startswith("abstract_en="):
            current.abstract_en = _clean_line(line.split("=", 1)[1])
        elif line.startswith("abstract_zh="):
            current.abstract_zh = _clean_line(line.split("=", 1)[1])

    if current is not None and _paper_has_value(current):
        papers.append(current)
    return papers


def _parse_loose_papers(text: str) -> list[DigestPaper]:
    lines = [_clean_line(line) for line in text.splitlines()]
    papers: list[DigestPaper] = []
    current: Optional[DigestPaper] = None
    in_paper_region = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if re.search(r"新增论文|papers?|top\s*\d+", line, flags=re.IGNORECASE):
            in_paper_region = True
            continue

        if re.search(r"今日结论|说明|summary", line, flags=re.IGNORECASE):
            if current is not None and _paper_has_value(current):
                papers.append(current)
            break

        if _is_field_line(line):
            if current is None:
                current = DigestPaper(title="")
            _fill_field(current, line)
            continue

        if in_paper_region and _looks_like_title_line(line):
            if current is not None and _paper_has_value(current):
                papers.append(current)
            current = DigestPaper(title=_clean_title(line))

    if current is not None and _paper_has_value(current):
        papers.append(current)
    return papers


def _is_field_line(line: str) -> bool:
    lowered = line.lower()
    if lowered.startswith(("arxiv", "url=", "abstract_short=", "abstract_en=", "abstract_zh=", "zh_abstract=")):
        return True
    if re.match(
        r"^(?:[-*]\s*)?(?:日期|链接|相关性|摘要(?:\(EN\)|\(ZH\))?|date|link|relevance|abstract(?:_en|_zh)?|zh_abstract)\s*[：:=]",
        line,
        flags=re.IGNORECASE,
    ):
        return True
    if "http://" in lowered or "https://" in lowered:
        return True
    return False


def _fill_field(paper: DigestPaper, line: str) -> None:
    lower = line.lower()
    if lower.startswith("url="):
        paper.url = _extract_link(line) or _clean_line(line.split("=", 1)[1])
        return
    if lower.startswith("abstract_short="):
        paper.abstract_zh = _clean_line(line.split("=", 1)[1])
        return
    if lower.startswith("abstract_en="):
        paper.abstract_en = _clean_line(line.split("=", 1)[1])
        return
    if lower.startswith(("abstract_zh=", "zh_abstract=")):
        paper.abstract_zh = _clean_line(line.split("=", 1)[1])
        return

    arxiv_match = re.match(r"^(?:[-*]\s*)?arxiv(?:\s+id)?\s*[：:]\s*(.+)$", line, flags=re.IGNORECASE)
    if arxiv_match:
        value = _clean_line(arxiv_match.group(1))
        if "|" in value:
            left, right = [part.strip() for part in value.split("|", 1)]
            paper.paper_id = left
            if right and not paper.published:
                paper.published = right
        else:
            paper.paper_id = value
        return

    date_match = re.match(r"^(?:[-*]\s*)?(?:日期|date)\s*[：:]\s*(.+)$", line, flags=re.IGNORECASE)
    if date_match:
        paper.published = _clean_line(date_match.group(1))
        return

    link_match = re.match(r"^(?:[-*]\s*)?(?:链接|link)\s*[：:]\s*(.+)$", line, flags=re.IGNORECASE)
    if link_match:
        paper.url = _extract_link(link_match.group(1)) or _clean_line(link_match.group(1))
        return

    rel_match = re.match(r"^(?:[-*]\s*)?(?:相关性|relevance)\s*[：:]\s*(.+)$", line, flags=re.IGNORECASE)
    if rel_match:
        paper.relevance = _clean_line(rel_match.group(1))
        return

    abs_en_match = re.match(
        r"^(?:[-*]\s*)?(?:摘要\(EN\)|abstract(?:_en|\(en\)))\s*[：:]\s*(.+)$",
        line,
        flags=re.IGNORECASE,
    )
    if abs_en_match:
        paper.abstract_en = _clean_line(abs_en_match.group(1))
        return

    abs_zh_match = re.match(
        r"^(?:[-*]\s*)?(?:摘要\(ZH\)|摘要|abstract(?:_zh|\(zh\))|zh_abstract)\s*[：:]\s*(.+)$",
        line,
        flags=re.IGNORECASE,
    )
    if abs_zh_match:
        paper.abstract_zh = _clean_line(abs_zh_match.group(1))
        return

    url = _extract_link(line)
    if url:
        paper.url = url


def _looks_like_title_line(line: str) -> bool:
    if not line:
        return False
    if line.startswith("#"):
        return False
    if re.match(
        r"^(?:[-*]\s*)?(?:日期|链接|相关性|摘要|arxiv|result|结果|track_key|监控范围|categories|keyword_expr)\b",
        line,
        flags=re.IGNORECASE,
    ):
        return False
    if re.match(r"^\d+\s*篇新增$", line):
        return False
    if re.search(r"https?://", line):
        return False
    return len(_clean_title(line)) >= 8


def _clean_title(line: str) -> str:
    text = re.sub(r"^\d+[.)]\s*", "", line)
    return _clean_line(text.strip().strip("*"))


def _paper_has_value(paper: DigestPaper) -> bool:
    return bool(paper.title and (paper.paper_id or paper.url or paper.display_abstract or paper.published))


def _normalize_paper(paper: DigestPaper) -> DigestPaper:
    title = _truncate(_clean_line(paper.title), _MAX_TITLE_LEN)
    paper_id = _clean_line(paper.paper_id)
    published = _clean_line(paper.published)
    url = _clean_line(paper.url)
    abstract_en = _truncate(_clean_line(paper.abstract_en), _MAX_SUMMARY_LEN)
    abstract_zh = _truncate(_clean_line(paper.abstract_zh), _MAX_SUMMARY_LEN)
    relevance = _clean_line(paper.relevance)

    if not url:
        from_id = _extract_link(paper_id)
        if from_id:
            url = from_id
    if not paper_id and url:
        paper_id = _guess_paper_id_from_url(url)
    if paper_id.lower().startswith("arxiv:"):
        paper_id = _clean_line(paper_id.split(":", 1)[1])
    if "|" in paper_id:
        paper_id = _clean_line(paper_id.split("|", 1)[0])
    if not url and paper_id:
        url = f"https://arxiv.org/abs/{paper_id}"

    return DigestPaper(
        title=title,
        paper_id=paper_id,
        published=published,
        url=url,
        abstract_en=abstract_en,
        abstract_zh=abstract_zh,
        relevance=relevance,
    )


def _dedupe_papers(papers: list[DigestPaper]) -> list[DigestPaper]:
    seen: set[str] = set()
    deduped: list[DigestPaper] = []
    for paper in papers:
        key = (paper.paper_id or "").lower() or (paper.title or "").lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(paper)
    return deduped


def _parse_new_count(text: str) -> Optional[int]:
    result_line = _extract_value(text, _RESULT_PATTERNS)
    if result_line:
        match = _RESULT_COUNT_RE.search(result_line)
        if match:
            return int(match.group(1))
    m2 = re.search(r"(?im)^\s*new_total\s*:\s*(\d+)\s*$", text)
    if m2:
        return int(m2.group(1))
    return None


def _split_papers_for_card_size(
    digest: DailyDigest,
    papers: list[DigestPaper],
    prefer_template: bool = True,
) -> list[list[DigestPaper]]:
    chunks: list[list[DigestPaper]] = []
    current: list[DigestPaper] = []
    for paper in papers:
        trial = current + [paper]
        trial_card = _build_card(
            digest=digest,
            papers=trial,
            part_index=1,
            total_parts=1,
            global_start_index=1,
            prefer_template=prefer_template,
        )
        trial_bytes = len(json.dumps(trial_card, ensure_ascii=False).encode("utf-8"))
        if current and (len(current) >= _SOFT_PAPERS_PER_CARD or trial_bytes > _CARD_MAX_BYTES):
            chunks.append(current)
            current = [paper]
        else:
            current = trial
    if current:
        chunks.append(current)
    return chunks


def _build_card(
    digest: DailyDigest,
    papers: list[DigestPaper],
    part_index: int,
    total_parts: int,
    global_start_index: int,
    prefer_template: bool = True,
) -> dict:
    if prefer_template:
        template_card = _build_template_card_if_configured(
            digest=digest,
            papers=papers,
            part_index=part_index,
            total_parts=total_parts,
            global_start_index=global_start_index,
        )
        if template_card is not None:
            return template_card

    title = f"{digest.topic} | {digest.report_date}"
    if total_parts > 1:
        title = f"{title} ({part_index}/{total_parts})"

    header_template = "red"
    if digest.no_new:
        header_template = "grey"
    elif not digest.fetch_failed:
        header_template = "blue"

    summary_lines = [f"**日期**：{digest.report_date}"]
    if digest.track_key:
        summary_lines.append(f"**追踪键**：`{digest.track_key}`")
    if digest.scope:
        summary_lines.append(f"**监控范围**：{digest.scope}")
    if digest.result_text:
        summary_lines.append(f"**结果**：{digest.result_text}")

    elements: list[dict] = [{"tag": "markdown", "content": "\n".join(summary_lines)}]

    if digest.fetch_failed:
        elements.append({"tag": "hr"})
        msg = digest.error_message or "抓取失败，请稍后重试。"
        elements.append({"tag": "markdown", "content": f"**错误信息**：{_truncate(msg, 500)}"})
    elif digest.no_new:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "今天没有新增论文。"})
    elif papers:
        elements.append({"tag": "hr"})
        for offset, paper in enumerate(papers):
            idx = global_start_index + offset
            block_lines = [f"**{idx}. {_truncate(_clean_line(paper.title), _MAX_TITLE_LEN)}**"]
            meta = []
            if paper.paper_id:
                meta.append(f"arXiv: `{paper.paper_id}`")
            if paper.published:
                meta.append(f"日期: {paper.published}")
            if paper.relevance:
                meta.append(f"相关性: {paper.relevance}")
            if meta:
                block_lines.append(" | ".join(meta))
            if paper.abstract_en:
                block_lines.append(f"摘要(EN)：{paper.abstract_en}")
            if paper.abstract_zh:
                block_lines.append(f"摘要(ZH)：{paper.abstract_zh}")
            elif paper.display_abstract:
                block_lines.append(f"摘要：{paper.display_abstract}")
            if paper.url:
                block_lines.append(f"[查看论文]({paper.url})")
            elements.append({"tag": "markdown", "content": "\n".join(block_lines)})
            if offset != len(papers) - 1:
                elements.append({"tag": "hr"})
    else:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "未解析到论文列表，请检查输出格式。"})

    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "title": {"tag": "plain_text", "content": _truncate(title, 80)},
            "template": header_template,
        },
        "elements": elements,
    }


def _build_template_card_if_configured(
    digest: DailyDigest,
    papers: list[DigestPaper],
    part_index: int,
    total_parts: int,
    global_start_index: int,
) -> Optional[dict]:
    if digest.fetch_failed or digest.no_new or not papers:
        return None

    template_id = os.getenv("FEISHU_LITERATURE_TEMPLATE_ID", "").strip()
    template_version = os.getenv("FEISHU_LITERATURE_TEMPLATE_VERSION", "").strip()
    if not template_id or not template_version:
        return None

    total_paper_all = min(len(digest.papers), _HARD_PAPER_LIMIT)
    table_rows = []
    paper_list = []
    for offset, paper in enumerate(papers):
        idx = global_start_index + offset
        table_rows.append(
            {
                "index": idx,
                "title": _truncate(_clean_line(paper.title), _MAX_TITLE_LEN),
                "id": _clean_line(paper.paper_id),
                "published": _clean_line(paper.published),
                "url": _clean_line(paper.url),
            }
        )
        paper_list.append(
            {
                "counter": idx,
                "title": _truncate(_clean_line(paper.title), _MAX_TITLE_LEN),
                "id": _clean_line(paper.paper_id),
                "abstract": _truncate(_clean_line(paper.abstract_en), _MAX_SUMMARY_LEN),
                "zh_abstract": _truncate(_clean_line(paper.abstract_zh), _MAX_SUMMARY_LEN),
                "url": _clean_line(paper.url),
                "published": _clean_line(paper.published),
                "relevance": _clean_line(paper.relevance),
            }
        )

    card_date = digest.report_date
    if total_parts > 1:
        card_date = f"{digest.report_date} (Part {part_index}/{total_parts})"

    return {
        "type": "template",
        "data": {
            "template_id": template_id,
            "template_version_name": template_version,
            "template_variable": {
                "today_date": card_date,
                "tag": digest.topic,
                "total_paper": total_paper_all,
                "table_rows": table_rows,
                "paper_list": paper_list,
                "track_key": digest.track_key,
                "scope": digest.scope,
                "result_text": digest.result_text,
            },
        },
    }


def _extract_link(text: str) -> str:
    md = _MARKDOWN_LINK_RE.search(text)
    if md:
        return md.group(1).strip()
    raw = _URL_RE.search(text)
    if raw:
        return raw.group(0).strip()
    return ""


def _guess_paper_id_from_url(url: str) -> str:
    match = re.search(r"/abs/([^/?#]+)", url)
    return match.group(1).strip() if match else ""


def _clean_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\r", " ").replace("\n", " ")).strip()


def _truncate(text: str, max_len: int) -> str:
    cleaned = _clean_line(text)
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."
