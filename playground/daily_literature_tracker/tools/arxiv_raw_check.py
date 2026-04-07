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
}
_CATEGORY_SPLIT_RE = re.compile(r"[,\s]+")
_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'|([^,\s]+)")
_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_ABSTRACT_MAX_CHARS = 400
_OBS_ABSTRACT_MAX_CHARS = 1800
_BOOLEAN_KEYWORDS = {"and", "or", "not"}

_DEFAULT_FILTER_PROMPT_FILE = Path(__file__).resolve().parents[1] / "paper_to_hunt.md"
_DEFAULT_HUNTS_DIR = Path(__file__).resolve().parents[1] / "hunts"


class ArxivRawCheckParams(BaseToolParams):
    """Incremental arXiv scanner (ArXivToday-style fetch + Magiclaw output contract)."""

    name: ClassVar[str] = "arxiv_raw_check"

    category: str = Field(
        ...,
        description="arXiv categories, e.g. cs.AI or cs.AI,cs.LG",
    )
    keyword: str = Field(..., description="Keyword query, e.g. agent")
    days: int = Field(
        ...,
        ge=1,
        le=30,
        description="Only keep papers from the latest N days (UTC).",
    )
    max_results: int = Field(
        ...,
        ge=1,
        le=30,
        description="Maximum papers returned in observation/info payload.",
    )
    scan_limit: int = Field(
        ...,
        ge=20,
        le=2000,
        description=(
            "How many latest arXiv entries to fetch per category before filtering "
            "(ArXivToday-style category-by-category fetch)."
        ),
    )
    profile: str = Field(
        default="",
        description="Optional profile for tracker isolation and hunts/{profile}.md lookup.",
    )
    include_seen: bool = Field(
        ...,
        description="When true, include already-seen papers in results.",
    )
    update_tracker: bool = Field(
        ...,
        description="Whether to persist seen paper ids.",
    )


class ArxivRawCheckTool(BaseTool):
    """Track arXiv papers incrementally with an ArXivToday-like minimal flow."""

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
            keyword_tokens = self._parse_keyword_tokens(params.keyword)
            track_key = self._build_track_key(categories, params.keyword, profile)
            filter_prompt, filter_prompt_file = self._load_filter_prompt(profile)

            entries, fetch_stats = self._fetch_entries(categories, params.scan_limit)
            cutoff_dt = datetime.now(timezone.utc) - timedelta(days=params.days)
            in_window_entries = [
                e for e in entries if e.get("published_at") is not None and e["published_at"] >= cutoff_dt
            ]

            keyword_entries, keyword_meta = self._apply_keyword_filter(in_window_entries, params.keyword)
            llm_entries, llm_meta = self._apply_llm_filter(
                keyword_entries,
                enabled=True,
                filter_prompt=filter_prompt,
                filter_prompt_file=filter_prompt_file,
            )

            state = self._load_state()
            seen_ids = set(self._read_seen_ids(state, track_key))
            new_entries = [e for e in llm_entries if e["paper_id"] not in seen_ids]
            candidate_entries = llm_entries if params.include_seen else new_entries
            output_entries = candidate_entries[: params.max_results]

            translation_meta = self._apply_translation(
                output_entries,
                enabled=True,
            )

            if params.update_tracker:
                self._write_seen_ids(
                    state=state,
                    track_key=track_key,
                    seen_ids=[e["paper_id"] for e in llm_entries],
                    categories=categories,
                    keyword=params.keyword,
                    scan_limit=params.scan_limit,
                )
                self._save_state(state)

            stats = {
                "source": fetch_stats.get("source", "api"),
                "mode": fetch_stats.get("mode", "arxivtoday_per_category"),
                "api_total": int(fetch_stats.get("api_total", 0)),
                "api_page_error_count": int(fetch_stats.get("api_page_error_count", 0)),
                "fetched_total": len(entries),
                "window_total": len(in_window_entries),
                "keyword_matched_total": len(keyword_entries),
                "llm_matched_total": len(llm_entries),
                "new_total": len(new_entries),
                "returned_total": len(output_entries),
                "translation_success_total": int(translation_meta.get("translated_count", 0)),
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
                "keyword_tokens": keyword_tokens,
                "profile": profile or None,
                "days": params.days,
                "include_seen": params.include_seen,
                "scan_limit": params.scan_limit,
                "max_results": params.max_results,
                "use_llm_for_filtering": True,
                "use_llm_for_translation": True,
                "stats": stats,
                "paper_ids": [e["paper_id"] for e in output_entries],
                "papers": [self._paper_payload(e) for e in output_entries],
                "meta": {
                    "fetch": {
                        "source": fetch_stats.get("source", "api"),
                        "mode": fetch_stats.get("mode", "arxivtoday_per_category"),
                        "api_page_error_count": int(fetch_stats.get("api_page_error_count", 0)),
                        "api_page_errors": fetch_stats.get("api_page_errors", []),
                        "categories": fetch_stats.get("categories", {}),
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
                    "api_page_errors": [self._clip_text(str(api_error), 200)],
                    "mode": "arxivtoday_rss_fallback",
                    "categories": {},
                }
            raise

    def _fetch_entries_via_api(self, categories: list[str], scan_limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        all_entries: list[dict[str, Any]] = []
        api_total = 0
        category_stats: dict[str, dict[str, Any]] = {}
        page_errors: list[str] = []

        for category in categories:
            try:
                category_entries, category_meta = self._fetch_one_category_via_api(
                    category=category,
                    max_results=scan_limit,
                )
            except Exception as category_error:
                page_errors.append(f"{category}: fatal={self._clip_text(str(category_error), 220)}")
                category_stats[category] = {
                    "api_total": 0,
                    "fetched_total": 0,
                    "page_error_count": 1,
                }
                continue

            all_entries.extend(category_entries)
            api_total += int(category_meta.get("api_total", 0))
            category_stats[category] = {
                "api_total": int(category_meta.get("api_total", 0)),
                "fetched_total": int(category_meta.get("fetched_total", 0)),
                "page_error_count": int(category_meta.get("page_error_count", 0)),
            }
            for err in category_meta.get("page_errors", []):
                page_errors.append(f"{category}: {err}")

        if not all_entries:
            if page_errors:
                raise RuntimeError("; ".join(page_errors[:3]))
            return [], {
                "api_total": 0,
                "source": "api_per_category",
                "api_page_error_count": 0,
                "api_page_errors": [],
                "mode": "arxivtoday_per_category",
                "categories": category_stats,
            }

        deduped = self._dedupe_entries_keep_first(all_entries)
        source = "api_per_category_partial" if page_errors else "api_per_category"
        return deduped, {
            "api_total": api_total,
            "source": source,
            "api_page_error_count": len(page_errors),
            "api_page_errors": page_errors[:8],
            "mode": "arxivtoday_per_category",
            "categories": category_stats,
        }

    def _fetch_one_category_via_api(self, category: str, max_results: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        page_size = min(100, max_results)
        start = 0
        api_total = 0
        entries: list[dict[str, Any]] = []
        page_errors: list[str] = []

        while start < max_results:
            current_size = min(page_size, max_results - start)
            query = f"cat:{category}"
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
                page_errors.append(f"start={start}, size={current_size}, err={self._clip_text(str(page_error), 180)}")
                if entries:
                    break
                raise

            page_entries, page_total = self._parse_atom_feed(content)
            api_total = max(api_total, int(page_total))
            if not page_entries:
                break

            entries.extend(page_entries)
            if len(page_entries) < current_size:
                break
            start += current_size

        return entries, {
            "api_total": api_total,
            "fetched_total": len(entries),
            "page_error_count": len(page_errors),
            "page_errors": page_errors[:4],
        }

    def _fetch_entries_via_rss(self, categories: list[str], scan_limit: int) -> list[dict[str, Any]]:
        if not categories:
            return []
        per_cat_limit = min(200, max(20, scan_limit))

        all_entries: list[dict[str, Any]] = []
        for category in categories:
            url = f"https://rss.arxiv.org/rss/{category}"
            content = self._http_get(url)
            category_entries = self._parse_rss_feed(content)
            all_entries.extend(category_entries[:per_cat_limit])

        return self._dedupe_entries_keep_first(all_entries)

    def _http_get(self, url: str) -> str:
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
            entries.append(
                {
                    "paper_id": paper_id,
                    "paper_id_version": raw_paper_id,
                    "title": title,
                    "summary": summary,
                    "published_at": published_at,
                    "published_date": published_at.date() if published_at else None,
                    "updated_at": updated_at,
                    "categories": categories,
                    "url": f"https://arxiv.org/abs/{raw_paper_id}",
                }
            )
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
            entries.append(
                {
                    "paper_id": paper_id,
                    "paper_id_version": raw_id,
                    "title": self._clean_text(html.unescape(raw_title)),
                    "summary": summary,
                    "published_at": published_at,
                    "published_date": published_at.date() if published_at else None,
                    "updated_at": published_at,
                    "categories": categories,
                    "url": raw_link.strip() if raw_link.strip() else f"https://arxiv.org/abs/{raw_id}",
                }
            )
        return entries

    @staticmethod
    def _dedupe_entries_keep_first(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for entry in entries:
            paper_id = entry["paper_id"]
            if paper_id in seen:
                continue
            seen.add(paper_id)
            out.append(entry)
        return out

    def _apply_keyword_filter(
        self,
        entries: list[dict[str, Any]],
        keyword_expr: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        keywords = self._parse_keyword_tokens(keyword_expr)
        if not keywords:
            return entries, {
                "mode": "token_set",
                "scope": "abstract",
                "keywords": [],
                "matched_total": len(entries),
                "input_total": len(entries),
                "skipped_reason": "empty_keyword",
            }

        keyword_set = set(keywords)
        matched: list[dict[str, Any]] = []
        for entry in entries:
            words = set((entry.get("summary") or "").lower().split())
            if keyword_set & words:
                matched.append(entry)

        return matched, {
            "mode": "token_set",
            "scope": "abstract",
            "keywords": keywords,
            "matched_total": len(matched),
            "input_total": len(entries),
        }

    def _apply_llm_filter(
        self,
        entries: list[dict[str, Any]],
        enabled: bool,
        filter_prompt: str,
        filter_prompt_file: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        fail_open = True
        meta: dict[str, Any] = {
            "enabled": enabled,
            "input_total": len(entries),
            "matched_total": len(entries),
            "fail_count": 0,
            "fail_open": True,
            "prompt_file": filter_prompt_file,
        }
        if not enabled or not entries:
            meta["skipped_reason"] = "disabled_or_empty"
            return entries, meta
        if not filter_prompt:
            meta["skipped_reason"] = "empty_filter_prompt"
            return entries, meta

        llm_cfg = self._resolve_llm_config()
        if llm_cfg is None:
            meta["skipped_reason"] = "missing_llm_credentials"
            return entries, meta

        matched: list[dict[str, Any]] = []
        fail_count = 0
        decisions: list[dict[str, Any]] = []
        for entry in entries:
            paper_id = entry.get("paper_id") or ""
            try:
                is_match, raw_reply = self._llm_is_paper_match(entry, filter_prompt, llm_cfg)
                if is_match:
                    entry["llm_relevance"] = "high"
                    entry["llm_relevance_score"] = 80
                    matched.append(entry)
                decisions.append(
                    {
                        "paper_id": paper_id,
                        "is_match": is_match,
                        "reply": self._clip_text(raw_reply, 120),
                    }
                )
            except Exception as e:
                fail_count += 1
                decisions.append(
                    {
                        "paper_id": paper_id,
                        "is_match": fail_open,
                        "reply": f"error: {self._clip_text(str(e), 120)}",
                    }
                )
                if fail_open:
                    entry["llm_relevance"] = "medium"
                    entry["llm_relevance_score"] = 55
                    matched.append(entry)

        if entries and fail_count == len(entries) and fail_open and not matched:
            for entry in entries:
                entry["llm_relevance"] = "medium"
                entry["llm_relevance_score"] = 55
            matched = list(entries)
            meta["all_llm_failed_fallback"] = True

        matched.sort(
            key=lambda e: (e.get("published_date") or date.min, e.get("paper_id") or ""),
            reverse=True,
        )

        meta["matched_total"] = len(matched)
        meta["fail_count"] = fail_count
        meta["model"] = llm_cfg.get("model")
        meta["decisions_sample"] = decisions[:30]
        return matched, meta

    def _apply_translation(
        self,
        entries: list[dict[str, Any]],
        enabled: bool,
    ) -> dict[str, Any]:
        for entry in entries:
            summary = self._clean_text(entry.get("summary") or "")
            entry["abstract_en"] = summary
            entry["abstract_zh"] = self._clean_text(entry.get("abstract_zh") or "")

        meta: dict[str, Any] = {
            "enabled": enabled,
            "input_total": len(entries),
            "attempted_count": 0,
            "translated_count": 0,
            "fail_count": 0,
        }
        if not enabled or not entries:
            meta["skipped_reason"] = "disabled_or_empty"
            return meta

        llm_cfg = self._resolve_llm_config()
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
    def _resolve_llm_config() -> dict[str, Any] | None:
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
    ) -> tuple[bool, str]:
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
        return "yes" in answer.lower(), answer

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

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

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
        temp_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_file.replace(self._state_file)

    @staticmethod
    def _read_seen_ids(state: dict[str, Any], track_key: str) -> list[str]:
        record = state.get(track_key)
        if isinstance(record, dict):
            seen_ids = record.get("seen_ids")
            if isinstance(seen_ids, list):
                return [str(x).strip() for x in seen_ids if str(x).strip()]
        if isinstance(record, list):
            return [str(x).strip() for x in record if str(x).strip()]
        return []

    @staticmethod
    def _write_seen_ids(
        state: dict[str, Any],
        track_key: str,
        seen_ids: list[str],
        categories: list[str],
        keyword: str,
        scan_limit: int,
    ) -> None:
        existing = ArxivRawCheckTool._read_seen_ids(state, track_key)
        merged = list(dict.fromkeys([*existing, *[str(x).strip() for x in seen_ids if str(x).strip()]]))
        state[track_key] = {
            "seen_ids": merged[-10000:],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "categories": categories,
            "keyword": keyword,
            "scan_limit": scan_limit,
        }

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
            f"time_window_days: {params.days}",
            f"scan_limit: {params.scan_limit}",
            f"max_results: {params.max_results}",
            f"include_seen: {params.include_seen}",
            f"source: {stats['source']}",
            f"mode: {stats.get('mode', '-')}",
            f"new_total: {stats['new_total']}",
            f"returned_total: {stats['returned_total']}",
            f"api_total: {stats['api_total']}",
            f"api_page_error_count: {stats['api_page_error_count']}",
            f"keyword_matched_total: {stats['keyword_matched_total']}",
            f"llm_matched_total: {stats['llm_matched_total']}",
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
            abstract_en = ArxivRawCheckTool._clip_text(
                entry.get("abstract_en") or entry.get("summary") or "",
                _OBS_ABSTRACT_MAX_CHARS,
            )
            abstract_zh = ArxivRawCheckTool._clip_text(entry.get("abstract_zh") or "", _OBS_ABSTRACT_MAX_CHARS)
            summary = ArxivRawCheckTool._clip_text(entry.get("summary") or "", 320)
            relevance = str(entry.get("llm_relevance") or "medium")
            score = int(entry.get("llm_relevance_score") or 55)
            lines.append(f"{idx}. id={entry['paper_id']} | date={published_text} | title={entry['title']}")
            lines.append(f"   url={entry['url']}")
            lines.append(f"   relevance={relevance} | score={score}")
            lines.append(f"   abstract_short={summary or '-'}")
            lines.append(f"   abstract_en={abstract_en or '-'}")
            lines.append(f"   abstract_zh={abstract_zh}")
        return "\n".join(lines)

    @staticmethod
    def _paper_payload(entry: dict[str, Any]) -> dict[str, Any]:
        published_date = entry.get("published_date")
        summary = entry.get("summary") or ""
        summary_short = ArxivRawCheckTool._clip_text(summary, _ABSTRACT_MAX_CHARS)
        abstract_en = ArxivRawCheckTool._clip_text(entry.get("abstract_en") or summary, _ABSTRACT_MAX_CHARS)
        abstract_zh_raw = entry.get("abstract_zh") or ""
        abstract_zh = ArxivRawCheckTool._clip_text(abstract_zh_raw, _ABSTRACT_MAX_CHARS)
        return {
            "paper_id": entry.get("paper_id"),
            "paper_id_version": entry.get("paper_id_version"),
            "title": entry.get("title"),
            "abstract_short": abstract_zh or summary_short,
            "abstract_en": abstract_en,
            "abstract_zh": abstract_zh or None,
            "relevance": entry.get("llm_relevance", "medium"),
            "relevance_score": entry.get("llm_relevance_score", 55),
            "published_date": published_date.isoformat() if isinstance(published_date, date) else None,
            "url": entry.get("url"),
            "categories": entry.get("categories", []),
        }

    @staticmethod
    def _normalize_categories(category: str) -> list[str]:
        tokens = _CATEGORY_SPLIT_RE.split((category or "").strip())
        cleaned = [token.strip() for token in tokens if token.strip()]
        return list(dict.fromkeys(cleaned))

    @staticmethod
    def _normalize_profile(profile: str) -> str:
        text = (profile or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"[^a-z0-9_-]+", "_", text).strip("_")
        return text[:64]

    def _load_filter_prompt(self, profile: str) -> tuple[str, str | None]:
        prompt_path = _DEFAULT_FILTER_PROMPT_FILE
        if profile:
            profile_file = _DEFAULT_HUNTS_DIR / f"{profile}.md"
            if profile_file.exists():
                prompt_path = profile_file
        if not prompt_path.exists():
            return "", None
        try:
            content = prompt_path.read_text(encoding="utf-8").strip()
            return content, str(prompt_path)
        except Exception:
            return "", str(prompt_path)

    @staticmethod
    def _parse_keyword_tokens(keyword_expr: str) -> list[str]:
        text = (keyword_expr or "").strip()
        if not text or text == "*":
            return []
        tokens: list[str] = []
        for g1, g2, g3 in _TOKEN_RE.findall(text):
            token = (g1 or g2 or g3 or "").strip().strip('"').strip("'")
            token = _WS_RE.sub(" ", token.lower()).strip()
            if not token or token in _BOOLEAN_KEYWORDS:
                continue
            tokens.append(token)
        return list(dict.fromkeys(tokens))

    @staticmethod
    def _build_track_key(categories: list[str], keyword: str, profile: str = "") -> str:
        category_part = re.sub(r"[^a-z0-9]+", "_", "_".join(c.lower() for c in categories)).strip("_")
        keyword_part = re.sub(r"[^a-z0-9]+", "_", (keyword or "").lower()).strip("_")
        profile_part = re.sub(r"[^a-z0-9_-]+", "_", (profile or "").lower()).strip("_")
        if profile_part:
            if keyword_part:
                return f"daily_{profile_part}_{category_part}_{keyword_part}"[:120]
            return f"daily_{profile_part}_{category_part}"[:120]
        if keyword_part:
            return f"daily_{category_part}_{keyword_part}"[:120]
        return (f"daily_{category_part}"[:120] or "daily_default")

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

        if "/" in text:
            return text.rsplit("/", 1)[-1].strip()
        return text

    @staticmethod
    def _strip_version(paper_id: str) -> str:
        return _VERSION_RE.sub("", (paper_id or "").strip())

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
    def _clip_text(text: str, max_chars: int) -> str:
        cleaned = _WS_RE.sub(" ", (text or "").strip())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."
