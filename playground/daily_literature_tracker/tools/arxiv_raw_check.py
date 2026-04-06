from __future__ import annotations

import html
import json
import re
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
_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")


class ArxivRawCheckParams(BaseToolParams):
    """Incrementally scan arXiv and return new papers in a time window."""

    name: ClassVar[str] = "arxiv_raw_check"

    category: str = Field(
        default="cs.AI",
        description="arXiv categories, supports one or multiple: cs.AI or cs.AI,cs.LG",
    )
    keyword: str = Field(
        default="",
        description=(
            "Keyword expression on title+abstract. Supports OR/AND. "
            'Examples: agent; "large language"; LLM OR "large language"; safety AND agent'
        ),
    )
    days: int = Field(
        default=3,
        ge=1,
        le=30,
        description="Only keep papers from latest N days (UTC).",
    )
    max_results: int = Field(
        default=8,
        ge=1,
        le=30,
        description="Maximum number of papers returned in the observation/info payload.",
    )
    abstract_max_chars: int = Field(
        default=180,
        ge=60,
        le=1000,
        description="Maximum abstract length per paper in output payload.",
    )
    scan_limit: int = Field(
        default=300,
        ge=20,
        le=2000,
        description="How many latest arXiv entries to fetch before filtering.",
    )
    track_key: str = Field(
        default="default",
        description="Stable tracker key for one topic stream, e.g. daily_ai_agent.",
    )
    include_seen: bool = Field(
        default=False,
        description="When true, include already-seen papers in results.",
    )
    update_tracker: bool = Field(
        default=True,
        description="Whether to persist seen paper ids.",
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

            track_key = self._normalize_track_key(params.track_key, categories, params.keyword)
            entries, fetch_stats = self._fetch_entries(categories, params.scan_limit)

            now_utc = datetime.now(timezone.utc)
            cutoff_dt = now_utc - timedelta(days=params.days)

            in_window_entries = [
                e for e in entries
                if e["published_at"] is not None and e["published_at"] >= cutoff_dt
            ]

            keyword_groups = self._parse_keyword_groups(params.keyword)
            matched_entries = [
                e for e in in_window_entries
                if self._matches_keyword(e, keyword_groups)
            ]

            state = self._load_state()
            stream = state.get(track_key, {})
            seen_ids = stream.get("seen_ids", [])
            seen = set(seen_ids if isinstance(seen_ids, list) else [])

            new_entries = [e for e in matched_entries if e["paper_id"] not in seen]
            candidate_entries = matched_entries if params.include_seen else new_entries
            output_entries = candidate_entries[: params.max_results]

            if params.update_tracker:
                latest_ids = [e["paper_id"] for e in matched_entries]
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
                "fetched_total": len(entries),
                "window_total": len(in_window_entries),
                "keyword_matched_total": len(matched_entries),
                "new_total": len(new_entries),
                "returned_total": len(output_entries),
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
                "days": params.days,
                "include_seen": params.include_seen,
                "scan_limit": params.scan_limit,
                "max_results": params.max_results,
                "stats": stats,
                "paper_ids": [e["paper_id"] for e in output_entries],
                "papers": [self._paper_payload(e, params.abstract_max_chars) for e in output_entries],
            }
            return observation, info
        except Exception as e:
            self.logger.exception("arxiv_raw_check failed")
            return f"Failed to scan arXiv: {e}", {"error": str(e)}

    def _fetch_entries(self, categories: list[str], scan_limit: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Fetch entries from arXiv API, fallback to RSS when API is unstable."""
        try:
            return self._fetch_entries_via_api(categories, scan_limit)
        except Exception as api_error:
            self.logger.warning("arXiv API fetch failed, fallback to RSS: %s", api_error)
            rss_entries = self._fetch_entries_via_rss(categories, scan_limit)
            if rss_entries:
                return rss_entries, {"api_total": 0, "source": "rss_fallback"}
            raise api_error

    def _fetch_entries_via_api(self, categories: list[str], scan_limit: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Fetch entries from the official arXiv API with pagination."""
        page_size = min(200, scan_limit)
        query = self._build_query(categories)
        start = 0
        api_total = 0
        all_entries: list[dict[str, Any]] = []

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
            content = self._http_get(url)

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
        return deduped, {"api_total": api_total, "source": "api"}

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
            except Exception as e:
                last_error = e

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
    def _paper_payload(entry: dict[str, Any], abstract_max_chars: int) -> dict[str, Any]:
        published_date = entry.get("published_date")
        summary = entry.get("summary") or ""
        summary_short = summary[:abstract_max_chars].strip()
        if len(summary) > abstract_max_chars:
            summary_short += "..."
        return {
            "paper_id": entry.get("paper_id"),
            "paper_id_version": entry.get("paper_id_version"),
            "title": entry.get("title"),
            "abstract_short": summary_short,
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
            f"categories: {', '.join(categories)}",
            f"keyword_expr: {params.keyword or '*'}",
            f"time_window_days: {params.days}",
            f"source: {stats['source']}",
            f"new_total: {stats['new_total']}",
            f"returned_total: {stats['returned_total']}",
            f"api_total: {stats['api_total']}",
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
            summary = (entry.get("summary") or "").strip()
            summary = summary[:140].strip() + ("..." if len(summary) > 140 else "")
            lines.append(
                f"{idx}. id={entry['paper_id']} | date={published_text} | title={entry['title']}"
            )
            lines.append(f"   url={entry['url']}")
            lines.append(f"   abstract_short={summary or '-'}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_track_key(track_key: str, categories: list[str], keyword: str) -> str:
        cleaned = (track_key or "").strip()
        if cleaned and cleaned.lower() != "default":
            return cleaned[:120]

        category_part = re.sub(r"[^a-z0-9]+", "_", "_".join(c.lower() for c in categories)).strip("_")
        keyword_part = re.sub(r"[^a-z0-9]+", "_", keyword.lower()).strip("_")
        if keyword_part:
            return f"daily_{category_part}_{keyword_part}"[:120]
        return f"daily_{category_part}"[:120] or "daily_default"
