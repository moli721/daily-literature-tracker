from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import requests
from pydantic import Field
from requests import exceptions as requests_exceptions
from urllib3.exceptions import InsecureRequestWarning

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_CATEGORY_SPLIT_RE = re.compile(r"[,\s]+")
_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_OR_SPLIT_RE = re.compile(r"\bOR\b|\|", flags=re.IGNORECASE)
_AND_SPLIT_RE = re.compile(r"\bAND\b|&&", flags=re.IGNORECASE)
_TOKEN_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'|([^,\s]+)")
_WORD_RE = re.compile(r"[a-z0-9_+\-\.]+")
_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_DEFAULT_FILTER_PROMPT_FILE = (
    Path(__file__).resolve().parents[1] / "paper_to_hunt.md"
)
_DEFAULT_HUNTS_DIR = Path(__file__).resolve().parents[1] / "hunts"


class ArxivRawCheckParams(BaseToolParams):
    """Incrementally scan arXiv and return new papers in a time window."""

    name: ClassVar[str] = "arxiv_raw_check"

    category: str = Field(
        ...,
        description="arXiv categories, supports one or multiple: cs.AI or cs.AI,cs.LG",
    )
    keyword: str = Field(
        ...,
        description=(
            "Keyword expression on title+abstract. Supports OR/AND. "
            'Examples: agent; "large language"; LLM OR "large language"; safety AND agent'
        ),
    )
    keyword_mode: str = Field(
        default="token_set",
        description='Keyword filtering mode: "expression" (OR/AND) or "token_set" (ArXivToday-like).',
    )
    keyword_list: str = Field(
        default="",
        description='Optional comma-separated keyword list, e.g. "agent,safety,jailbreak".',
    )
    token_set_scope: str = Field(
        default="abstract",
        description='Scope for token_set mode: "abstract" (ArXivToday-like) or "title_abstract".',
    )
    days: int = Field(
        ...,
        ge=1,
        le=30,
        description="Only keep papers from latest N days (UTC).",
    )
    max_results: int = Field(
        ...,
        ge=1,
        le=30,
        description="Maximum number of papers returned in the observation/info payload.",
    )
    abstract_max_chars: int = Field(
        default=400,
        ge=120,
        le=2000,
        description="Maximum abstract length per paper in output payload.",
    )
    scan_limit: int = Field(
        ...,
        ge=20,
        le=2000,
        description="How many latest arXiv entries to fetch before filtering.",
    )
    track_key: str = Field(
        default="default",
        description="Stable tracker key for one topic stream, e.g. daily_ai_agent.",
    )
    profile: str = Field(
        default="",
        description=(
            "Optional user/profile key for multi-user isolation. "
            "When set and track_key is default, tool auto-builds a profile-specific track_key. "
            "Also tries hunts/{profile}.md as llm_filter_prompt_file."
        ),
    )
    include_seen: bool = Field(
        ...,
        description="When true, include already-seen papers in results.",
    )
    update_tracker: bool = Field(
        ...,
        description="Whether to persist seen paper ids.",
    )
    use_llm_for_filtering: bool = Field(
        default=True,
        description="Apply LLM yes/no filtering after keyword filtering.",
    )
    llm_filter_prompt: str = Field(
        default="",
        description="Prompt describing what papers to keep; when empty, load from llm_filter_prompt_file.",
    )
    llm_filter_prompt_file: str = Field(
        default=str(_DEFAULT_FILTER_PROMPT_FILE),
        description="Local file path for filter prompt (paper_to_hunt style).",
    )
    llm_filter_fail_open: bool = Field(
        default=True,
        description="When LLM filtering fails, keep paper by default (ArXivToday behavior).",
    )
    strict_llm_filter: bool = Field(
        default=False,
        description=(
            "When true, honor llm_filter_fail_open exactly. "
            "When false (default), tool forces fail-open to avoid false negatives from LLM outages."
        ),
    )
    use_llm_for_translation: bool = Field(
        default=True,
        description="Translate abstracts to Chinese for card template field zh_abstract.",
    )


class ArxivRawCheckTool(BaseTool):
    """Track arXiv papers incrementally with local state."""

    name: ClassVar[str] = "arxiv_raw_check"
    params_class: ClassVar[type[BaseToolParams]] = ArxivRawCheckParams

    def __init__(self) -> None:
        super().__init__()
        project_root = Path(__file__).resolve().parents[3]
        self._state_file = project_root / "data" / "daily_literature_tracker" / "tracker_state.json"

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        del session
        try:
            params = self.parse_params(args_json)
            assert isinstance(params, ArxivRawCheckParams)
        except Exception as e:
            return f"Invalid parameters: {e}", {"error": str(e)}

        try:
            categories = self._normalize_categories(params.category)
            if not categories:
                return "Invalid category: at least one category is required.", {"error": "invalid_category"}

            profile = self._normalize_profile(params.profile)
            track_key = self._normalize_track_key(params.track_key, categories, params.keyword, profile)
            resolved_prompt_file = self._resolve_llm_filter_prompt_file(params, profile)
            entries, fetch_stats = self._fetch_entries(categories, params.scan_limit)

            now_utc = datetime.now(timezone.utc)
            cutoff_dt = now_utc - timedelta(days=params.days)

            in_window_entries = [
                e for e in entries
                if e["published_at"] is not None and e["published_at"] >= cutoff_dt
            ]

            keyword_entries, keyword_meta = self._apply_keyword_filter(in_window_entries, params)
            llm_entries, llm_meta = self._apply_llm_filter(
                keyword_entries,
                params,
                prompt_file_override=resolved_prompt_file,
            )

            state = self._load_state()
            stream = state.get(track_key, {})
            seen_ids = stream.get("seen_ids", [])
            seen = set(seen_ids if isinstance(seen_ids, list) else [])

            new_entries = [e for e in llm_entries if e["paper_id"] not in seen]
            candidate_entries = llm_entries if params.include_seen else new_entries
            output_entries = candidate_entries[: params.max_results]
            translation_meta = self._apply_translation(output_entries, params)

            if params.update_tracker:
                latest_ids = [e["paper_id"] for e in llm_entries]
                merged_ids = list(dict.fromkeys(latest_ids + list(seen)))
                state[track_key] = {
                    "seen_ids": merged_ids[:8000],
                    "last_checked_at": datetime.now(timezone.utc).isoformat(),
                    "last_categories": categories,
                    "last_keyword_expr": params.keyword,
                    "last_scan_limit": params.scan_limit,
                }
                self._save_state(state)

            stats = {
                "source": fetch_stats.get("source", "api"),
                "api_total": fetch_stats["api_total"],
                "api_page_error_count": fetch_stats.get("api_page_error_count", 0),
                "fetched_total": len(entries),
                "window_total": len(in_window_entries),
                "keyword_matched_total": len(keyword_entries),
                "llm_matched_total": len(llm_entries),
                "llm_high_total": llm_meta.get("relevance_counter", {}).get("high", 0),
                "new_total": len(new_entries),
                "returned_total": len(output_entries),
                "translation_success_total": translation_meta["translated_count"],
            }

            observation = self._format_observation(
                params=params,
                categories=categories,
                track_key=track_key,
                stats=stats,
                output_entries=output_entries,
            )
            info = {
                "track_key": track_key,
                "categories": categories,
                "keyword_expr": params.keyword,
                "keyword_mode": params.keyword_mode,
                "token_set_scope": params.token_set_scope,
                "keyword_list": keyword_meta.get("keywords", []),
                "profile": profile or None,
                "days": params.days,
                "include_seen": params.include_seen,
                "scan_limit": params.scan_limit,
                "max_results": params.max_results,
                "use_llm_for_filtering": params.use_llm_for_filtering,
                "use_llm_for_translation": params.use_llm_for_translation,
                "stats": stats,
                "paper_ids": [e["paper_id"] for e in output_entries],
                "papers": [self._paper_payload(e, params.abstract_max_chars) for e in output_entries],
                "meta": {
                    "fetch": {
                        "source": fetch_stats.get("source", "api"),
                        "api_page_error_count": fetch_stats.get("api_page_error_count", 0),
                        "api_page_errors": fetch_stats.get("api_page_errors", []),
                    },
                    "keyword_filter": keyword_meta,
                    "llm_filter": llm_meta,
                    "translation": translation_meta,
                },
            }
            return observation, info
        except Exception as e:
            self.logger.exception("arxiv_raw_check failed")
            return f"Failed to scan arXiv: {e}", {"error": str(e)}

    def _fetch_entries(self, categories: list[str], scan_limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fetch entries from arXiv API, fallback to RSS when the first API page fails."""
        try:
            return self._fetch_entries_via_api(categories, scan_limit)
        except Exception as api_error:
            self.logger.warning("arXiv API fetch failed, fallback to RSS: %s", api_error)
            rss_entries = self._fetch_entries_via_rss(categories, scan_limit)
            if rss_entries:
                return rss_entries, {
                    "api_total": 0,
                    "source": "rss_fallback",
                    "api_page_error_count": 1,
                }
            raise api_error

    def _fetch_entries_via_api(self, categories: list[str], scan_limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fetch entries from arXiv API with conservative pagination.

        ArXivToday-style behavior: try to keep already-fetched pages when later pages fail.
        """
        page_size = min(100, scan_limit)
        query = self._build_query(categories)
        start = 0
        api_total = 0
        all_entries: list[dict[str, Any]] = []
        page_errors: list[str] = []

        while start < scan_limit:
            current_size = min(page_size, scan_limit - start)
            url = (
                "https://export.arxiv.org/api/query?"
                f"search_query={urllib.parse.quote_plus(query)}"
                f"&start={start}"
                f"&max_results={current_size}"
                "&sortBy=submittedDate"
                "&sortOrder=descending"
            )
            try:
                content = self._http_get(url)
            except Exception as page_error:
                err_msg = f"start={start}, size={current_size}, err={page_error}"
                page_errors.append(err_msg)
                if all_entries:
                    self.logger.warning(
                        "arXiv API page failed, keep partial pages and stop paging: %s",
                        err_msg,
                    )
                    break
                raise

            page_entries, page_total = self._parse_atom_feed(content)
            api_total = max(api_total, page_total)
            if not page_entries:
                break

            all_entries.extend(page_entries)
            if len(page_entries) < current_size:
                break
            start += current_size

        deduped = self._dedupe_entries(all_entries)
        deduped.sort(
            key=lambda e: (
                e["published_date"] is not None,
                e["published_date"] or date.min,
                e["paper_id"],
            ),
            reverse=True,
        )
        source = "api_partial" if page_errors else "api"
        return deduped, {
            "api_total": api_total,
            "source": source,
            "api_page_error_count": len(page_errors),
            "api_page_errors": page_errors[:5],
        }

    def _fetch_entries_via_rss(self, categories: list[str], scan_limit: int) -> list[dict[str, Any]]:
        """Fallback source: arXiv RSS feed by category."""
        if not categories:
            return []
        per_cat_limit = max(20, scan_limit // len(categories))
        all_entries: list[dict[str, Any]] = []
        for category in categories:
            url = f"https://rss.arxiv.org/rss/{category}"
            content = self._http_get(url)
            entries = self._parse_rss_feed(content)
            all_entries.extend(entries[:per_cat_limit])

        deduped = self._dedupe_entries(all_entries)
        deduped.sort(
            key=lambda e: (
                e["published_date"] is not None,
                e["published_date"] or date.min,
                e["paper_id"],
            ),
            reverse=True,
        )
        return deduped[:scan_limit]

    def _http_get(self, url: str) -> str:
        """HTTP GET with lightweight fallback (https -> http for export.arxiv.org)."""
        candidates = [url]
        if url.startswith("https://export.arxiv.org/"):
            candidates.append("http://export.arxiv.org/" + url.removeprefix("https://export.arxiv.org/"))

        last_error: Exception | None = None
        for candidate in candidates:
            for attempt in range(3):
                try:
                    response = requests.get(
                        candidate,
                        headers={"User-Agent": "MagiClawDailyTracker/1.0"},
                        timeout=30,
                    )
                    response.raise_for_status()
                    return response.text
                except requests_exceptions.SSLError:
                    # Public arXiv endpoint fallback for environments with broken local TLS.
                    try:
                        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
                        response = requests.get(
                            candidate,
                            headers={"User-Agent": "MagiClawDailyTracker/1.0"},
                            timeout=30,
                            verify=False,
                        )
                        response.raise_for_status()
                        return response.text
                    except Exception as e2:
                        last_error = e2
                except requests_exceptions.HTTPError as e:
                    last_error = e
                    status = getattr(e.response, "status_code", 0) if getattr(e, "response", None) else 0
                    # Retry transient gateway/server errors.
                    if status in {429, 500, 502, 503, 504} and attempt < 2:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                except Exception as e:
                    last_error = e
                    if attempt < 2:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                break

        assert last_error is not None
        raise last_error

    def _parse_atom_feed(self, xml_text: str) -> tuple[list[dict[str, Any]], int]:
        feed = ET.fromstring(xml_text)
        total_text = feed.findtext("opensearch:totalResults", default="0", namespaces=_ATOM_NS)
        try:
            total_results = int((total_text or "0").strip())
        except ValueError:
            total_results = 0

        entries: list[dict[str, Any]] = []
        for node in feed.findall("atom:entry", _ATOM_NS):
            raw_id = (node.findtext("atom:id", default="", namespaces=_ATOM_NS) or "").strip()
            raw_paper_id = self._extract_paper_id(raw_id)
            if not raw_paper_id:
                continue

            paper_id = self._strip_version(raw_paper_id)
            title = self._clean_text(node.findtext("atom:title", default="", namespaces=_ATOM_NS))
            summary = self._clean_text(node.findtext("atom:summary", default="", namespaces=_ATOM_NS))
            published_at = self._parse_iso_datetime(node.findtext("atom:published", default="", namespaces=_ATOM_NS))
            updated_at = self._parse_iso_datetime(node.findtext("atom:updated", default="", namespaces=_ATOM_NS))
            categories = [
                (cat.get("term") or "").strip()
                for cat in node.findall("atom:category", _ATOM_NS)
                if (cat.get("term") or "").strip()
            ]

            entries.append({
                "paper_id": paper_id,
                "paper_id_version": raw_paper_id,
                "title": title,
                "summary": summary,
                "published_at": published_at,
                "published_date": published_at.date() if published_at else None,
                "updated_at": updated_at,
                "categories": categories,
                "url": f"https://arxiv.org/abs/{paper_id}",
            })
        return entries, total_results

    def _parse_rss_feed(self, xml_text: str) -> list[dict[str, Any]]:
        root = ET.fromstring(xml_text)
        channel = root.find("channel")
        if channel is None:
            return []

        entries: list[dict[str, Any]] = []
        for item in channel.findall("item"):
            raw_title = item.findtext("title", default="") or ""
            raw_link = item.findtext("link", default="") or ""
            raw_desc = item.findtext("description", default="") or ""
            raw_pub = item.findtext("pubDate", default="") or ""
            categories = [
                self._clean_text(cat.text or "")
                for cat in item.findall("category")
                if self._clean_text(cat.text or "")
            ]

            raw_id = self._extract_paper_id(raw_link)
            if not raw_id:
                continue
            paper_id = self._strip_version(raw_id)

            desc_plain = self._clean_text(_TAG_RE.sub(" ", html.unescape(raw_desc)))
            summary = desc_plain
            marker = re.search(r"Abstract:\s*(.*)", desc_plain, flags=re.IGNORECASE)
            if marker:
                summary = marker.group(1).strip()

            published_at = self._parse_rss_datetime(raw_pub)

            entries.append({
                "paper_id": paper_id,
                "paper_id_version": raw_id,
                "title": self._clean_text(html.unescape(raw_title)),
                "summary": summary,
                "published_at": published_at,
                "published_date": published_at.date() if published_at else None,
                "updated_at": published_at,
                "categories": categories,
                "url": f"https://arxiv.org/abs/{paper_id}",
            })
        return entries

    @staticmethod
    def _dedupe_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for entry in entries:
            paper_id = entry["paper_id"]
            current = by_id.get(paper_id)
            if current is None:
                by_id[paper_id] = entry
                continue

            current_time = current.get("updated_at") or current.get("published_at")
            new_time = entry.get("updated_at") or entry.get("published_at")
            if new_time and (current_time is None or new_time > current_time):
                by_id[paper_id] = entry
        return list(by_id.values())

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        cleaned = _WS_RE.sub(" ", (text or "").strip())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."

    @staticmethod
    def _paper_payload(entry: dict[str, Any], abstract_max_chars: int) -> dict[str, Any]:
        published_date = entry.get("published_date")
        summary = entry.get("summary") or ""
        summary_short = ArxivRawCheckTool._clip_text(summary, abstract_max_chars)
        abstract_en = ArxivRawCheckTool._clip_text(
            entry.get("abstract_en") or summary,
            abstract_max_chars,
        )
        abstract_zh_raw = entry.get("abstract_zh") or ""
        abstract_zh = ArxivRawCheckTool._clip_text(abstract_zh_raw, abstract_max_chars)
        return {
            "paper_id": entry.get("paper_id"),
            "paper_id_version": entry.get("paper_id_version"),
            "title": entry.get("title"),
            "abstract_short": abstract_zh or summary_short,
            "abstract_en": abstract_en,
            "abstract_zh": abstract_zh or None,
            "relevance": entry.get("llm_relevance", "medium"),
            "relevance_score": entry.get("llm_relevance_score", 50),
            "published_date": published_date.isoformat() if isinstance(published_date, date) else None,
            "url": entry.get("url"),
            "categories": entry.get("categories", []),
        }

    @staticmethod
    def _build_query(categories: list[str]) -> str:
        if len(categories) == 1:
            return f"cat:{categories[0]}"
        return "(" + " OR ".join(f"cat:{c}" for c in categories) + ")"

    @staticmethod
    def _extract_paper_id(raw_id: str) -> str:
        text = (raw_id or "").strip()
        if not text:
            return ""

        parsed = urllib.parse.urlparse(text)
        if parsed.netloc and parsed.path:
            path = parsed.path.rstrip("/")
            if path.startswith("/abs/"):
                return path.split("/abs/", 1)[1].strip()
            return path.rsplit("/", 1)[-1].strip()

        # Fallback for non-URL ids.
        if "/" in text:
            return text.rsplit("/", 1)[-1].strip()
        return text

    @staticmethod
    def _strip_version(paper_id: str) -> str:
        return _VERSION_RE.sub("", paper_id.strip())

    @staticmethod
    def _parse_iso_datetime(value: str) -> datetime | None:
        text = (value or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _parse_rss_datetime(value: str) -> datetime | None:
        text = (value or "").strip()
        if not text:
            return None
        try:
            dt = parsedate_to_datetime(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _clean_text(text: str) -> str:
        return _WS_RE.sub(" ", (text or "").replace("\n", " ").replace("\r", " ")).strip()

    @staticmethod
    def _normalize_categories(category: str) -> list[str]:
        tokens = _CATEGORY_SPLIT_RE.split((category or "").strip())
        cleaned = [token.strip() for token in tokens if token.strip()]
        # keep order + dedupe
        return list(dict.fromkeys(cleaned))

    @staticmethod
    def _normalize_profile(profile: str) -> str:
        text = (profile or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"[^a-z0-9_-]+", "_", text).strip("_")
        return text[:64]

    def _resolve_llm_filter_prompt_file(self, params: ArxivRawCheckParams, profile: str) -> str:
        """Resolve final prompt file path for LLM filtering.

        Priority:
        1) Explicit non-default llm_filter_prompt_file from caller.
        2) profile-based file: hunts/{profile}.md
        3) default paper_to_hunt.md
        """
        raw = (params.llm_filter_prompt_file or "").strip()
        raw_path = Path(raw) if raw else _DEFAULT_FILTER_PROMPT_FILE
        if not raw_path.is_absolute():
            raw_path = Path(__file__).resolve().parents[1] / raw_path

        try:
            is_default_file = raw_path.resolve() == _DEFAULT_FILTER_PROMPT_FILE.resolve()
        except Exception:
            is_default_file = False

        if not is_default_file and raw:
            return raw

        if profile:
            candidate = _DEFAULT_HUNTS_DIR / f"{profile}.md"
            if candidate.exists():
                return str(candidate)

        return str(_DEFAULT_FILTER_PROMPT_FILE)

    @classmethod
    def _parse_keyword_groups(cls, expr: str) -> list[list[str]]:
        expr = (expr or "").strip()
        if not expr or expr == "*":
            return []

        expr = _WS_RE.sub(" ", expr)
        has_boolean = bool(re.search(r"\b(OR|AND)\b|\||&&", expr, flags=re.IGNORECASE))

        if not has_boolean:
            single = cls._normalize_term(expr)
            return [[single]] if single else []

        groups: list[list[str]] = []
        for or_chunk in _OR_SPLIT_RE.split(expr):
            chunk = or_chunk.strip()
            if not chunk:
                continue

            and_terms: list[str] = []
            for and_chunk in _AND_SPLIT_RE.split(chunk):
                and_terms.extend(cls._extract_terms(and_chunk))

            dedup_terms = list(dict.fromkeys(term for term in and_terms if term))
            if dedup_terms:
                groups.append(dedup_terms)
        return groups

    @classmethod
    def _extract_terms(cls, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        terms: list[str] = []
        for g1, g2, g3 in _TOKEN_RE.findall(text):
            token = g1 or g2 or g3
            normalized = cls._normalize_term(token)
            if normalized:
                terms.append(normalized)
        return terms

    @staticmethod
    def _normalize_term(term: str) -> str:
        cleaned = (term or "").strip().strip('"').strip("'")
        return _WS_RE.sub(" ", cleaned.lower()).strip()

    @staticmethod
    def _matches_keyword(entry: dict[str, Any], keyword_groups: list[list[str]]) -> bool:
        if not keyword_groups:
            return True
        haystack = f"{entry.get('title', '')} {entry.get('summary', '')}"
        haystack = _WS_RE.sub(" ", haystack.lower())
        return any(all(term in haystack for term in group) for group in keyword_groups)

    @classmethod
    def _parse_keyword_list(cls, keyword_list: str) -> list[str]:
        if not (keyword_list or "").strip():
            return []
        terms: list[str] = []
        for chunk in re.split(r"[\n,]+", keyword_list):
            terms.extend(cls._extract_terms(chunk))
        return list(dict.fromkeys(term for term in terms if term))

    @staticmethod
    def _tokenize_words(text: str) -> set[str]:
        return set(_WORD_RE.findall((text or "").lower()))

    @staticmethod
    def _normalize_token_set_scope(scope: str) -> str:
        value = (scope or "").strip().lower()
        if value in {"title_abstract", "title+abstract", "all"}:
            return "title_abstract"
        return "abstract"

    @staticmethod
    def _effective_llm_fail_open(params: ArxivRawCheckParams) -> bool:
        # ArXivToday behavior: fail-open when LLM is unavailable or malformed.
        return True

    def _apply_keyword_filter(
        self,
        entries: list[dict[str, Any]],
        params: ArxivRawCheckParams,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        mode = (params.keyword_mode or "token_set").strip().lower()
        keyword_expr = (params.keyword or "").strip()
        keyword_list = self._parse_keyword_list(params.keyword_list)

        if mode == "token_set":
            source_terms = keyword_list or self._extract_terms(keyword_expr)
            scope = self._normalize_token_set_scope(params.token_set_scope)
            if not source_terms:
                return entries, {
                    "mode": "token_set",
                    "scope": scope,
                    "keywords": [],
                    "matched_total": len(entries),
                    "input_total": len(entries),
                    "skipped_reason": "empty_keyword",
                }

            # ArXivToday-style keyword screening:
            # keyword_set & set(abstract.lower().split())
            keyword_tokens = list(dict.fromkeys((term or "").strip().lower() for term in source_terms if term))
            keyword_set = set(keyword_tokens)

            matched: list[dict[str, Any]] = []
            for entry in entries:
                if scope == "title_abstract":
                    text_to_scan = f"{entry.get('title', '')} {entry.get('summary', '')}"
                else:
                    text_to_scan = entry.get("summary", "")
                words = set((text_to_scan or "").lower().split())
                if keyword_set & words:
                    matched.append(entry)
            return matched, {
                "mode": "token_set",
                "scope": scope,
                "keywords": keyword_tokens,
                "matched_total": len(matched),
                "input_total": len(entries),
            }

        # default: expression mode (legacy-compatible)
        groups = self._parse_keyword_groups(keyword_expr)
        if keyword_list:
            groups.extend([[term] for term in keyword_list])
            groups = list(dict.fromkeys(tuple(g) for g in groups))
            groups = [list(g) for g in groups]

        if not groups:
            return entries, {
                "mode": "expression",
                "groups": [],
                "matched_total": len(entries),
                "input_total": len(entries),
                "skipped_reason": "empty_keyword",
            }

        matched = [entry for entry in entries if self._matches_keyword(entry, groups)]
        return matched, {
            "mode": "expression",
            "groups": groups,
            "keywords": keyword_list,
            "matched_total": len(matched),
            "input_total": len(entries),
        }

    def _apply_llm_filter(
        self,
        entries: list[dict[str, Any]],
        params: ArxivRawCheckParams,
        prompt_file_override: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        effective_fail_open = self._effective_llm_fail_open(params)
        meta: dict[str, Any] = {
            "enabled": params.use_llm_for_filtering,
            "input_total": len(entries),
            "matched_total": len(entries),
            "fail_count": 0,
            "fail_open": effective_fail_open,
        }
        if not params.use_llm_for_filtering or not entries:
            meta["skipped_reason"] = "disabled_or_empty"
            return entries, meta

        filter_prompt = (params.llm_filter_prompt or "").strip()
        selected_prompt_file = (prompt_file_override or params.llm_filter_prompt_file or "").strip()
        if not filter_prompt:
            prompt_path_text = selected_prompt_file
            prompt_path = Path(prompt_path_text) if prompt_path_text else _DEFAULT_FILTER_PROMPT_FILE
            if not prompt_path.is_absolute():
                prompt_path = Path(__file__).resolve().parents[1] / prompt_path
            if prompt_path.exists():
                filter_prompt = prompt_path.read_text(encoding="utf-8").strip()
                selected_prompt_file = str(prompt_path)

        if not filter_prompt:
            meta["skipped_reason"] = "empty_filter_prompt"
            meta["prompt_file"] = selected_prompt_file or None
            return entries, meta

        llm_cfg = self._resolve_llm_config(params)
        if llm_cfg is None:
            meta["skipped_reason"] = "missing_llm_credentials"
            if effective_fail_open:
                return entries, meta
            return [], meta

        matched: list[dict[str, Any]] = []
        fail_count = 0
        decisions: list[dict[str, Any]] = []
        relevance_counter = {"high": 0, "medium": 0, "low": 0}

        for entry in entries:
            paper_id = entry.get("paper_id") or ""
            try:
                is_match, relevance, score, raw_reply = self._llm_is_paper_match(entry, filter_prompt, llm_cfg)
                entry["llm_relevance"] = relevance
                entry["llm_relevance_score"] = score
                if is_match:
                    matched.append(entry)
                    relevance_counter[relevance] = relevance_counter.get(relevance, 0) + 1
                decisions.append({
                    "paper_id": paper_id,
                    "is_match": is_match,
                    "relevance": relevance,
                    "score": score,
                    "reply": self._clip_text(raw_reply, 120),
                })
            except Exception as e:
                fail_count += 1
                if effective_fail_open:
                    entry["llm_relevance"] = "medium"
                    entry["llm_relevance_score"] = 55
                    matched.append(entry)
                    relevance_counter["medium"] = relevance_counter.get("medium", 0) + 1
                decisions.append({
                    "paper_id": paper_id,
                    "is_match": bool(effective_fail_open),
                    "relevance": "medium" if effective_fail_open else "unknown",
                    "score": 55 if effective_fail_open else 0,
                    "reply": f"error: {self._clip_text(str(e), 120)}",
                })

        matched.sort(
            key=lambda e: (
                int(e.get("llm_relevance_score") or 0),
                e.get("published_date") or date.min,
                e.get("paper_id") or "",
            ),
            reverse=True,
        )

        # Safety net: if all LLM calls failed, avoid false "no new papers" caused by fail-close.
        if entries and fail_count == len(entries) and not matched and effective_fail_open:
            fallback_matched: list[dict[str, Any]] = []
            for entry in entries:
                if not entry.get("llm_relevance"):
                    entry["llm_relevance"] = "medium"
                if not entry.get("llm_relevance_score"):
                    entry["llm_relevance_score"] = 55
                fallback_matched.append(entry)
            matched = fallback_matched
            meta["all_llm_failed_fallback"] = True
            relevance_counter["medium"] = len(matched)
            self.logger.warning(
                "All LLM filter calls failed; fallback to keyword-filtered papers (%d items).",
                len(matched),
            )

        meta["matched_total"] = len(matched)
        meta["fail_count"] = fail_count
        meta["model"] = llm_cfg.get("model")
        meta["prompt_file"] = selected_prompt_file or None
        meta["relevance_counter"] = relevance_counter
        meta["decisions_sample"] = decisions[:30]
        return matched, meta

    def _apply_translation(
        self,
        entries: list[dict[str, Any]],
        params: ArxivRawCheckParams,
    ) -> dict[str, Any]:
        for entry in entries:
            summary = self._clean_text(entry.get("summary") or "")
            entry["abstract_en"] = summary
            entry["abstract_zh"] = self._clean_text(entry.get("abstract_zh") or "")

        meta: dict[str, Any] = {
            "enabled": params.use_llm_for_translation,
            "input_total": len(entries),
            "attempted_count": 0,
            "translated_count": 0,
            "fail_count": 0,
        }
        if not params.use_llm_for_translation or not entries:
            meta["skipped_reason"] = "disabled_or_empty"
            return meta

        llm_cfg = self._resolve_llm_config(params)
        if llm_cfg is None:
            meta["skipped_reason"] = "missing_llm_credentials"
            return meta

        translated_count = 0
        fail_count = 0
        attempted_count = 0
        for entry in entries:
            abstract_en = (entry.get("abstract_en") or "").strip()
            if not abstract_en:
                continue
            attempted_count += 1
            try:
                translated = self._llm_translate_abstract(abstract_en, llm_cfg)
                if translated:
                    entry["abstract_zh"] = translated
                    translated_count += 1
            except Exception:
                fail_count += 1

        meta["attempted_count"] = attempted_count
        meta["translated_count"] = translated_count
        meta["fail_count"] = fail_count
        meta["model"] = llm_cfg.get("model")
        return meta

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    @staticmethod
    def _normalize_relevance(label: str) -> str:
        value = (label or "").strip().lower()
        if value in {"high", "h", "高度相关", "高"}:
            return "high"
        if value in {"low", "l", "低相关", "低"}:
            return "low"
        return "medium"

    @staticmethod
    def _score_from_relevance(label: str) -> int:
        mapping = {"high": 85, "medium": 60, "low": 30}
        return mapping.get(ArxivRawCheckTool._normalize_relevance(label), 60)

    @staticmethod
    def _extract_first_json_object(text: str) -> dict[str, Any] | None:
        raw = (text or "").strip()
        if not raw:
            return None
        # strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    @staticmethod
    def _looks_like_placeholder(value: str) -> bool:
        text = (value or "").strip().lower()
        if not text:
            return False
        markers = (
            "dummy",
            "test",
            "placeholder",
            "example",
            "your_",
            "xxxx",
            "changeme",
            "none",
            "null",
            "nil",
        )
        return any(m in text for m in markers)

    @staticmethod
    def _extract_chat_content(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        if not isinstance(message, dict):
            return ""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return ""

    def _resolve_llm_config(self, params: ArxivRawCheckParams) -> dict[str, Any] | None:
        del params

        # 1) dedicated envs for this tool
        daily_key = os.getenv("DAILY_ARXIV_LLM_API_KEY", "").strip()
        daily_base = os.getenv("DAILY_ARXIV_LLM_BASE_URL", "").strip()
        daily_model = os.getenv("DAILY_ARXIV_LLM_MODEL", "").strip()
        if daily_key:
            return {
                "model": daily_model or "grok-4.20-0309-reasoning",
                "base_url": (daily_base or "https://api.x.ai/v1").rstrip("/"),
                "api_key": daily_key,
                "timeout": 60,
            }

        # 2) Grok pair
        grok_key = os.getenv("GROK_API_KEY", "").strip()
        grok_base = os.getenv("GROK_BASE_URL", "").strip()
        grok_model = os.getenv("GROK_MODEL", "").strip()
        if grok_key:
            return {
                "model": daily_model or grok_model or "grok-4.20-0309-reasoning",
                "base_url": (grok_base or "https://api.x.ai/v1").rstrip("/"),
                "api_key": grok_key,
                "timeout": 60,
            }

        # 3) OpenAI pair
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        gpt_base = os.getenv("GPT_BASE_URL", "").strip()
        gpt_model = os.getenv("GPT_CHAT_MODEL", "").strip()
        openai_model = os.getenv("OPENAI_MODEL", "").strip()
        if openai_key:
            return {
                "model": gpt_model or openai_model or "gpt-4o-mini",
                "base_url": (gpt_base or "https://api.openai.com/v1").rstrip("/"),
                "api_key": openai_key,
                "timeout": 60,
            }

        return None

    def _llm_chat(
        self,
        messages: list[dict[str, str]],
        llm_cfg: dict[str, Any],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        url = f"{llm_cfg['base_url']}/chat/completions"
        payload = {
            "model": llm_cfg["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {llm_cfg['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=llm_cfg.get("timeout", 60),
        )
        response.raise_for_status()
        data = response.json()
        content = self._extract_chat_content(data)
        if not content:
            raise RuntimeError("empty llm content")
        return self._strip_think_blocks(content)

    def _llm_is_paper_match(
        self,
        entry: dict[str, Any],
        filter_prompt: str,
        llm_cfg: dict[str, Any],
    ) -> tuple[bool, str, int, str]:
        title = self._clean_text(entry.get("title") or "")
        abstract = self._clean_text(entry.get("summary") or "")
        prompt = f"""You are an academic paper screening assistant.
Decide whether this paper matches the target scope.
Reply with only one word: Yes or No.

Paper title:
{title}

Paper abstract:
{abstract}

Target scope:
{filter_prompt}"""
        answer = self._llm_chat(
            messages=[{"role": "user", "content": prompt}],
            llm_cfg=llm_cfg,
            max_tokens=64,
            temperature=0.0,
        )
        # ArXivToday behavior: any response containing "yes" is treated as a match.
        # Errors are handled upstream by fail-open.
        is_match = "yes" in answer.lower()
        return is_match, "medium", 55, answer

    def _llm_translate_abstract(self, abstract_en: str, llm_cfg: dict[str, Any]) -> str:
        prompt = f"""Translate the following academic abstract into Simplified Chinese.
Keep technical terms accurate and concise.
Output translation only, no explanations.

{abstract_en}"""
        translated = self._llm_chat(
            messages=[{"role": "user", "content": prompt}],
            llm_cfg=llm_cfg,
            max_tokens=800,
            temperature=0.1,
        )
        return self._clean_text(translated)

    def _load_state(self) -> dict[str, Any]:
        if not self._state_file.exists():
            return {}
        try:
            raw = self._state_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            self.logger.warning("Tracker state is invalid, recreating: %s", self._state_file)
            return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self._state_file.with_suffix(".tmp")
        temp_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_file.replace(self._state_file)

    @staticmethod
    def _format_observation(
        params: ArxivRawCheckParams,
        categories: list[str],
        track_key: str,
        stats: dict[str, Any],
        output_entries: list[dict[str, Any]],
    ) -> str:
        lines = [
            "daily_arxiv_digest_ready",
            f"track_key: {track_key}",
            f"profile: {params.profile or '-'}",
            f"categories: {', '.join(categories)}",
            f"keyword_expr: {params.keyword or '*'}",
            f"keyword_mode: {params.keyword_mode}",
            f"token_set_scope: {params.token_set_scope}",
            f"time_window_days: {params.days}",
            f"scan_limit: {params.scan_limit}",
            f"max_results: {params.max_results}",
            f"include_seen: {params.include_seen}",
            f"source: {stats['source']}",
            f"new_total: {stats['new_total']}",
            f"returned_total: {stats['returned_total']}",
            f"api_total: {stats['api_total']}",
            f"api_page_error_count: {stats['api_page_error_count']}",
            f"keyword_matched_total: {stats['keyword_matched_total']}",
            f"llm_matched_total: {stats['llm_matched_total']}",
            f"llm_high_total: {stats['llm_high_total']}",
            f"translation_success_total: {stats['translation_success_total']}",
        ]

        if not output_entries:
            lines.append("")
            lines.append("today_no_new_papers")
            return "\n".join(lines)

        lines.append("")
        lines.append("papers:")
        for idx, entry in enumerate(output_entries, 1):
            published_date = entry.get("published_date")
            published_text = published_date.isoformat() if isinstance(published_date, date) else "unknown"
            abstract_en = ArxivRawCheckTool._clip_text(entry.get("abstract_en") or entry.get("summary") or "", 220)
            abstract_zh = ArxivRawCheckTool._clip_text(entry.get("abstract_zh") or "", 220)
            summary = ArxivRawCheckTool._clip_text(entry.get("summary") or "", 140)
            relevance = str(entry.get("llm_relevance") or "medium")
            score = int(entry.get("llm_relevance_score") or 55)
            lines.append(
                f"{idx}. id={entry['paper_id']} | date={published_text} | title={entry['title']}"
            )
            lines.append(f"   url={entry['url']}")
            lines.append(f"   relevance={relevance} | score={score}")
            lines.append(f"   abstract_short={summary or '-'}")
            lines.append(f"   abstract_en={abstract_en or '-'}")
            lines.append(f"   abstract_zh={abstract_zh or '-'}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_track_key(track_key: str, categories: list[str], keyword: str, profile: str = "") -> str:
        cleaned = (track_key or "").strip()
        if cleaned and cleaned.lower() != "default":
            return cleaned[:120]

        category_part = re.sub(r"[^a-z0-9]+", "_", "_".join(c.lower() for c in categories)).strip("_")
        keyword_part = re.sub(r"[^a-z0-9]+", "_", keyword.lower()).strip("_")
        profile_part = re.sub(r"[^a-z0-9_-]+", "_", (profile or "").lower()).strip("_")
        if profile_part:
            if keyword_part:
                return f"daily_{profile_part}_{category_part}_{keyword_part}"[:120]
            return f"daily_{profile_part}_{category_part}"[:120]
        if keyword_part:
            return f"daily_{category_part}_{keyword_part}"[:120]
        return f"daily_{category_part}"[:120] or "daily_default"
