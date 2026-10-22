"""
Supports GoogleTranslator (unofficial Google web endpoint) and locally
installed Argos Translate models, with Argos as the default provider.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from deep_translator import GoogleTranslator
from deep_translator.exceptions import (
    NotValidLength,
    RequestError,
    TooManyRequests,
    TranslationNotFound,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class _DropStanzaMwtNotice(logging.Filter):
    """Stanza notes that it added the mwt processor Argos did not request; harmless."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "expects mwt, which has been added" not in record.getMessage()


logging.getLogger("stanza").addFilter(_DropStanzaMwtNotice())


# Google's web endpoint rejects payloads of 5,000 characters or more.
GOOGLE_MAX_CHARS = 5000
DEFAULT_CHUNK_CHARS = 4500

# Product and organization names that should remain unchanged
DEFAULT_PROTECTED_TERMS = (
    "Hugging Face",
    "GitHub"
)

RETRYABLE_ERRORS = (TooManyRequests, RequestError, ConnectionError, TimeoutError)


class TranslationManager:
    def __init__(
        self,
        enable_translation: bool = True,
        *,
        provider: str = "argos",
        source: str = "en",
        target: str = "de",
        requests_per_second: float = 4.0,
        max_retries: int = 4,
        backoff_base: float = 2.0,
        max_consecutive_rate_limits: int = 6,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        cache_path: str | Path | None = None,
        protected_terms: Optional[List[str]] = None
    ):
        self.enabled = enable_translation
        self.provider = provider.lower().strip()
        if self.provider not in {"google", "argos"}:
            raise ValueError("provider must be 'google' or 'argos'")
        self.source = source
        self.target = target
        self.translator = None
        if enable_translation:
            if self.provider == "google":
                self.translator = GoogleTranslator(source=source, target=target)
            else:
                try:
                    import argostranslate.translate as argos_translate
                except ImportError as exc:
                    raise ImportError(
                        "Argos Translate is not installed. Run: pip install argostranslate"
                    ) from exc
                # Suppress informational and debug messages from Argos Translate
                logging.getLogger("argostranslate.utils").setLevel(logging.WARNING) 
                installed = argos_translate.get_installed_languages()
                src = next((lang for lang in installed if lang.code == source), None)
                dst = next((lang for lang in installed if lang.code == target), None)
                translation = src.get_translation(dst) if src and dst else None
                if translation is None:
                    raise RuntimeError(
                        f"No installed Argos translation model for {source}->{target}. "
                        "Install the corresponding Argos language package first."
                    )
                self.translator = translation

        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_consecutive_rate_limits = max_consecutive_rate_limits
        self.chunk_chars = min(chunk_chars, GOOGLE_MAX_CHARS - 1)

        self._last_request = 0.0
        self._pause_until = 0.0
        self._lock = threading.Lock()
        self._argos_lock = threading.Lock()
        self._consecutive_rate_limits = 0
        self.circuit_open = False

        self.stats = {"requests": 0, "cache_hits": 0, "failures": 0, "retries": 0}

        self.cache: Dict[str, str] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache_dirty = False
        if self.cache_path:
            self._load_cache()
            atexit.register(self.save_cache)

        self.MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
        self.RAW_URL_RE = re.compile(r"https?://[^\s)\]]+", re.IGNORECASE)
        # Google may insert spaces inside the placeholder ("🔒 0 🔒").
        self.TOKEN_RE = re.compile(r"🔒\s*(\d+)\s*🔒")
        # Names that must never be translated ("Hugging Face" -> "Umarmendes Gesicht").
        terms = DEFAULT_PROTECTED_TERMS if protected_terms is None else protected_terms
        terms = sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True)
        term_alt = "|".join(rf"(?<!\w){re.escape(t)}(?!\w)" for t in terms)
        self.TERM_RE = re.compile(f"(?:{term_alt})" if terms else r"(?!x)x", re.IGNORECASE)

        # One capturing group: re.split() keeps the protected spans at odd indices.
        self.URL_SPLIT_RE = re.compile(
            r"(\[[^\]]+\]\(https?://[^\s)]+\)|https?://[^\s)\]]+"
            + (f"|{term_alt}" if terms else "") + ")",
            re.IGNORECASE,
        )
        md_pattern = type(self).MD_INLINE_SPLIT_RE.pattern
        self.MD_INLINE_SPLIT_RE = re.compile(
            md_pattern[:-1] + (f"|{term_alt}" if terms else "") + ")", re.IGNORECASE
        )
        self.HAS_LETTER_RE = re.compile(r"[^\W\d_]")
        # Punctuation at the start of a segment (": > With T5 ...") is often
        # dropped by Argos; keep it outside the text that is translated.
        self.LEADING_PUNCT_RE = re.compile(r"^[\s:;,.>|*\-\u2013\u2014]*")
        # A single lowercase word as link label ("write", "here", "paper") is part
        # of the surrounding sentence; translated alone it comes out wrong.
        self.INLINE_WORD_RE = re.compile(r"[a-z][a-z'\-]*")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate_text(self, text: str) -> str:
        """Translate ``text``; return it unchanged if no translation exists."""
        if not self.enabled or not text or not isinstance(text, str):
            return text

        translated = self.translate_or_none(text)
        return translated if translated is not None else text

    def translate_or_none(self, text: str) -> Optional[str]:
        """Translate ``text``; return ``None`` if translation is unavailable."""
        if not self.enabled or not text or not isinstance(text, str):
            return None

        masked_text, url_map = self._extract_urls(text)

        if not masked_text.strip(" 🔒0123456789\n\t"):
            # Nothing but URLs; a lone Markdown link still has a label to translate.
            if not self.MD_LINK_RE.search(text):
                return None
            return self._translate_link_labels(text)

        # Argos (SentencePiece models) does not know the 🔒 placeholder and
        # drops or mangles it, so never send URLs to it at all.
        if url_map and self.provider == "argos":
            result = self._translate_around_urls(text)
        else:
            translated = self._translate_plain_text(masked_text)
            if translated is None:
                return None
            result = self._restore_urls(translated, url_map)
            if result is None:
                logger.debug(
                    "Translation dropped a URL placeholder for '%s...'; "
                    "retrying segment by segment.",
                    text[:50],
                )
                result = self._translate_around_urls(text)

        if result is None:
            return None
        # Markdown links come back verbatim from both paths; translate their labels.
        return self._translate_link_labels(result)

    def save_cache(self) -> None:
        if not self.cache_path or not self._cache_dirty:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(self.cache, file, ensure_ascii=False, indent=0, sort_keys=True)
            tmp_path.replace(self.cache_path)
            self._cache_dirty = False
        except OSError as exc:
            logger.warning("Could not save translation cache %s: %s", self.cache_path, exc)

    def log_summary(self) -> None:
        logger.info(
            "Translation summary: %(requests)d requests, %(cache_hits)d cache hits, "
            "%(retries)d retries, %(failures)d failures",
            self.stats,
        )
        if self.circuit_open:
            logger.warning(
                "Translation was disabled during this run after repeated rate "
                "limiting; affected literals were written in English only."
            )

    # ------------------------------------------------------------------
    # URL masking
    # ------------------------------------------------------------------

    def _extract_urls(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replace URLs and Markdown links with numbered placeholders."""
        url_map: Dict[str, str] = {}
        counter = 0

        def replace(original: str) -> str:
            nonlocal counter
            key = str(counter)
            url_map[key] = original
            counter += 1
            return f"🔒{key}🔒"

        masked = self.MD_LINK_RE.sub(lambda m: replace(m.group(0)), text)
        masked = self.RAW_URL_RE.sub(lambda m: replace(m.group(0)), masked)
        masked = self.TERM_RE.sub(lambda m: replace(m.group(0)), masked)
        return masked, url_map

    def _translate_segment(self, part: str) -> Optional[str]:
        """Translate one unprotected piece, keeping its surrounding whitespace
        and leading punctuation exactly as in the source."""
        lead = self.LEADING_PUNCT_RE.match(part).group(0)
        body = part[len(lead):]
        trail = body[len(body.rstrip()):]
        core = body.strip()
        if not core:
            return part
        translated = self._translate_plain_text(core)
        if translated is None:
            return None
        return f"{lead}{translated.strip()}{trail}"

    def _translate_around_urls(self, text: str) -> Optional[str]:
        """Translate only the text between URLs; URLs never reach the model."""
        parts = self.URL_SPLIT_RE.split(text)
        out: List[str] = []
        for i, part in enumerate(parts):
            if i % 2 == 1 or not self.HAS_LETTER_RE.search(part):
                out.append(part)  # protected, or only spaces/punctuation/digits
                continue
            translated = self._translate_segment(part)
            if translated is None:
                return None  # never publish a partially translated text
            out.append(translated)
        return "".join(out)

    def _translate_link_labels(self, text: str) -> Optional[str]:
        """Translate the label of each Markdown link, keeping its URL untouched."""
        failed = False

        def translate_label(match: re.Match) -> str:
            nonlocal failed
            label, url = match.group(1), match.group(2)
            core = label.strip()
            # Keep labels without words ("[1]"), URLs, and single inline words ("write").
            if (not self.HAS_LETTER_RE.search(core) or self.RAW_URL_RE.fullmatch(core)
                    or self.INLINE_WORD_RE.fullmatch(core)):
                return match.group(0)
            translated = self._translate_around_urls(core)  # protects names like "Hugging Face"
            if translated is None:
                failed = True
                return match.group(0)
            # Brackets in the translation would break the link syntax.
            translated = translated.strip().replace("[", "\\[").replace("]", "\\]")
            return f"[{translated}]({url})"

        result = self.MD_LINK_RE.sub(translate_label, text)
        return None if failed else result  # never publish a partially translated text

    def _restore_urls(self, text: str, url_map: Dict[str, str]) -> Optional[str]:
        """Restore placeholders; ``None`` if any placeholder was lost."""
        restored_keys: set[str] = set()

        def restore(match: re.Match) -> str:
            key = match.group(1)
            if key in url_map:
                restored_keys.add(key)
                return url_map[key]
            return match.group(0)

        restored = self.TOKEN_RE.sub(restore, text)

        if restored_keys != set(url_map):
            return None
        return restored

    # ------------------------------------------------------------------
    # Markdown documents (e.g. Hugging Face model cards)
    # ------------------------------------------------------------------

    # Inline spans that must never reach the translator. One capturing group,
    # so re.split() puts them at the odd indices.
    MD_INLINE_SPLIT_RE = re.compile(
        r"("
        r"`[^`\n]+`"                                   # inline code
        r"|\[!\[[^\]]*\]\([^)\s]*\)\]\([^)\s]*\)"     # badge: image inside a link
        r"|!\[[^\]]*\]\([^)\s]*(?:\s+\"[^\"]*\")?\)"  # image
        r"|\[[^\[\]]+\]\([^)\s]+(?:\s+\"[^\"]*\")?\)" # link (absolute or relative)
        r"|<https?://[^>\s]+>"                         # autolink
        r"|<!--.*?-->"                                 # inline HTML comment
        r"|</?[A-Za-z][^<>]*>"                         # inline HTML tag
        r"|https?://[^\s)\]<>`]+"                      # bare URL
        r"|\*\*|(?<!\w)__|__(?!\w)"                     # bold markers (kept in place)
        r")",
        re.IGNORECASE,
    )
    MD_LINK_PARTS_RE = re.compile(
        r"^\[([^\[\]]+)\]\(([^)\s]+(?:\s+\"[^\"]*\")?)\)$"
    )
    MD_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
    MD_HR_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
    MD_HTML_BLOCK_RE = re.compile(r"^\s{0,3}</?[A-Za-z!][^>]*>?")
    MD_HEADING_RE = re.compile(r"^(\s{0,3}#{1,6}\s+)(.*?)(\s+#+)?\s*$")
    MD_ITEM_RE = re.compile(
        r"^(\s*(?:>\s*)*(?:(?:[-*+]|\d{1,9}[.)]|[a-zA-Z][.)])\s+(?:\[[ xX]\]\s+)?)?)"
        r"(\S.*)$"
    )
    MD_LIST_MARKER_RE = re.compile(r"^\s*(?:>\s*)*(?:[-*+]|\d{1,9}[.)]|[a-zA-Z][.)])\s+")
    MD_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")
    MD_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")
    # Model names, file names, versions: "all-MiniLM-L6-v2", "train_script.py".
    MD_IDENTIFIER_RE = re.compile(r"(?=\S*[\d_./])[\w.\-/@:+]+")

    def translate_markdown(self, markdown: str, *, allow_partial: bool = False) -> Optional[str]:
        """Translate a Markdown document such as a Hugging Face model card.

        Only prose is translated. YAML front matter, fenced and indented code,
        HTML blocks and comments, math blocks, inline code, URLs, images and
        link targets are kept verbatim. Headings, list markers, blockquotes,
        tables and blank lines keep their structure; link labels are
        translated.

        Returns ``None`` if translation is unavailable. With
        ``allow_partial=True`` blocks that fail stay in the source language
        instead of discarding the whole document.
        """
        if not self.enabled or not markdown or not isinstance(markdown, str):
            return None

        lines = markdown.replace("\r\n", "\n").split("\n")
        out: List[str] = []
        failures = 0
        para_prefix: Optional[str] = None
        para_lines: List[str] = []
        prev_blank = True
        in_list = False

        def emit_text(prefix: str, text: str, suffix: str = "") -> None:
            nonlocal failures
            translated = self._translate_md_inline(text)
            if translated is None:
                failures += 1
                translated = text
            out.append(f"{prefix}{translated}{suffix}")

        def flush(suffix: str = "") -> None:
            nonlocal para_prefix, para_lines
            if para_prefix is not None:
                emit_text(para_prefix, " ".join(para_lines), suffix)
            para_prefix, para_lines = None, []

        i, n = 0, len(lines)

        # YAML front matter: metadata, not prose.
        if lines and lines[0].strip() == "---":
            for j in range(1, n):
                if lines[j].strip() in ("---", "..."):
                    out.extend(lines[: j + 1])
                    i = j + 1
                    break

        while i < n:
            line = lines[i]
            stripped = line.strip()

            # Fenced code block.
            fence = self.MD_FENCE_RE.match(line)
            if fence:
                flush()
                marker = fence.group(1)
                out.append(line)
                i += 1
                while i < n:
                    out.append(lines[i])
                    closing = lines[i].strip()
                    i += 1
                    if closing.startswith(marker[0] * len(marker)) and not closing.strip(marker[0]):
                        break
                prev_blank = False
                continue

            # Multi-line HTML comment or $$ math block.
            block = next(
                (pair for pair in (("<!--", "-->"), ("$$", "$$")) if stripped.startswith(pair[0])),
                None,
            )
            if block and para_prefix is None:
                opener, closer = block
                out.append(line)
                i += 1
                if closer not in stripped[len(opener):]:
                    while i < n:
                        out.append(lines[i])
                        i += 1
                        if closer in lines[i - 1]:
                            break
                prev_blank = False
                continue

            if not stripped:
                flush()
                out.append(line)
                prev_blank = True
                i += 1
                continue

            # Indented code block (not a nested list continuation).
            if (line.startswith(("    ", "\t")) and prev_blank and not in_list
                    and para_prefix is None):
                out.append(line)
                i += 1
                continue

            if self.MD_HR_RE.match(line) or self.MD_HTML_BLOCK_RE.match(line):
                flush()
                out.append(line)
                prev_blank = False
                in_list = False
                i += 1
                continue

            # Table: a row followed by a separator row starts it.
            if "|" in line and i + 1 < n and self.MD_TABLE_SEP_RE.match(lines[i + 1]):
                flush()
                while i < n and "|" in lines[i] and lines[i].strip():
                    row = lines[i]
                    if self.MD_TABLE_SEP_RE.match(row):
                        out.append(row)
                    else:
                        cells = self.MD_CELL_SPLIT_RE.split(row)
                        new_cells = []
                        for cell in cells:
                            core = cell.strip()
                            if not self._md_is_translatable(core) and "](" not in core:
                                new_cells.append(cell)
                                continue
                            translated = self._translate_md_inline(core)
                            if translated is None:
                                failures += 1
                                translated = core
                            lead = cell[: len(cell) - len(cell.lstrip())]
                            trail = cell[len(cell.rstrip()):]
                            translated = translated.replace("|", "\\|")
                            new_cells.append(f"{lead}{translated}{trail}")
                        out.append("|".join(new_cells))
                    i += 1
                prev_blank = False
                in_list = False
                continue

            heading = self.MD_HEADING_RE.match(line)
            if heading:
                flush()
                emit_text(heading.group(1), heading.group(2), heading.group(3) or "")
                prev_blank = False
                in_list = False
                i += 1
                continue

            # A hard line break ("  " or "\" at the end) ends the paragraph text.
            hard_break = line.endswith("  ") or stripped.endswith("\\")
            suffix = "  " if line.endswith("  ") else ("\\" if stripped.endswith("\\") else "")
            content = stripped[:-1] if suffix == "\\" else stripped

            is_item = bool(self.MD_LIST_MARKER_RE.match(line)) or stripped.startswith(">")
            if is_item or para_prefix is None:
                flush()
                item = self.MD_ITEM_RE.match(line[: len(line) - len(line.lstrip())] + content)
                para_prefix = item.group(1) if item else ""
                para_lines = [item.group(2) if item else content]
                in_list = in_list or bool(self.MD_LIST_MARKER_RE.match(line))
            else:
                para_lines.append(content)  # soft-wrapped continuation line
            if hard_break:
                flush(suffix)
            prev_blank = False
            i += 1

        flush()

        if failures:
            logger.warning(
                "Markdown translation: %d block(s) could not be translated%s.",
                failures,
                "; kept in the source language" if allow_partial else "; discarding the document",
            )
            if not allow_partial:
                return None
        return "\n".join(out)

    def _md_is_translatable(self, core: str) -> bool:
        """True if ``core`` contains words worth translating."""
        if not core or not self.HAS_LETTER_RE.search(core):
            return False
        if self.RAW_URL_RE.fullmatch(core) or self.MD_IDENTIFIER_RE.fullmatch(core):
            return False
        return True

    def _translate_md_inline(self, text: str) -> Optional[str]:
        """Translate one Markdown text unit, protecting code, URLs, HTML and images."""
        parts = self.MD_INLINE_SPLIT_RE.split(text)
        prose_parts = parts[0::2]

        if len(parts) == 1:
            if not self._md_is_translatable(text.strip()):
                return text
            result = self._translate_plain_text(text)
        elif not any(self._md_is_translatable(p.strip()) for p in prose_parts):
            result = text  # only protected spans; link labels are handled below
        elif self.provider == "argos":
            result = self._translate_md_segments(parts)
        else:
            url_map: Dict[str, str] = {}
            masked: List[str] = []
            for idx, part in enumerate(parts):
                if idx % 2:
                    key = str(len(url_map))
                    url_map[key] = part
                    masked.append(f"🔒{key}🔒")
                else:
                    masked.append(part)
            translated = self._translate_plain_text("".join(masked))
            result = self._restore_urls(translated, url_map) if translated is not None else None
            if result is None and translated is not None:
                result = self._translate_md_segments(parts)

        if result is None:
            return None
        return self._translate_md_link_labels(result)

    def _translate_md_segments(self, parts: List[str]) -> Optional[str]:
        """Translate only the prose between protected spans."""
        out: List[str] = []
        for idx, part in enumerate(parts):
            if idx % 2 or not self._md_is_translatable(part.strip()):
                out.append(part)
                continue
            translated = self._translate_segment(part)
            if translated is None:
                return None
            out.append(translated)
        return "".join(out)

    def _translate_md_link_labels(self, text: str) -> Optional[str]:
        """Translate labels of Markdown links (not images or badges)."""
        parts = self.MD_INLINE_SPLIT_RE.split(text)
        for idx in range(1, len(parts), 2):
            link = self.MD_LINK_PARTS_RE.match(parts[idx])
            if not link:
                continue
            label, target = link.group(1), link.group(2)
            if self.INLINE_WORD_RE.fullmatch(label.strip()):
                continue  # single inline word ("write", "here"): keep as in the sentence
            # The label itself may hold inline code or HTML; protect those too.
            label_parts = self.MD_INLINE_SPLIT_RE.split(label)
            if not any(self._md_is_translatable(p.strip()) for p in label_parts[0::2]):
                continue
            translated = self._translate_md_segments(label_parts)
            if translated is None:
                return None
            translated = translated.replace("[", "\\[").replace("]", "\\]")
            parts[idx] = f"[{translated}]({target})"
        return "".join(parts)

    # ------------------------------------------------------------------
    # Translation with cache, chunking, throttling and retries
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Preserve existing Google cache entries; separate Argos results by backend.
        prefix = f"{self.source}:{self.target}"
        if self.provider != "google":
            prefix = f"{self.provider}:{prefix}"
        return f"{prefix}:{digest}"

    def _translate_plain_text(self, text: str) -> Optional[str]:
        key = self._cache_key(text)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]

        if len(text) <= self.chunk_chars:
            translated = self._request(text)
        else:
            parts: List[str] = []
            for chunk in self._split(text):
                part = self._translate_plain_text(chunk) if chunk.strip() else chunk
                if part is None:
                    return None  # never publish a partially translated text
                parts.append(part)
            translated = "".join(parts)

        if translated is not None:
            self.cache[key] = translated
            self._cache_dirty = True
        return translated

    def _split(self, text: str) -> List[str]:
        """Split at paragraph, then sentence, then word boundaries.

        Separators stay attached to the preceding chunk, so joining the
        translated chunks keeps the original layout.
        """
        chunks: List[str] = []
        for paragraph in re.split(r"(?<=\n\n)", text):
            if len(paragraph) <= self.chunk_chars:
                chunks.append(paragraph)
                continue
            # Split after sentence ends and line breaks, keeping the
            # original whitespace attached to the preceding piece.
            pieces = re.split(r"(?<=[.!?\n])(\s+)", paragraph)
            sentences = [
                pieces[i] + (pieces[i + 1] if i + 1 < len(pieces) else "")
                for i in range(0, len(pieces), 2)
            ]
            for sentence in sentences:
                while len(sentence) > self.chunk_chars:
                    cut = sentence.rfind(" ", 0, self.chunk_chars)
                    cut = cut + 1 if cut > 0 else self.chunk_chars
                    chunks.append(sentence[:cut])
                    sentence = sentence[cut:]
                chunks.append(sentence)

        # Merge small neighbours to keep the request count low.
        merged: List[str] = []
        for chunk in chunks:
            if merged and len(merged[-1]) + len(chunk) <= self.chunk_chars:
                merged[-1] += chunk
            else:
                merged.append(chunk)
        return merged

    def _throttle(self) -> None:
        """Space request starts across all threads and honour shared pauses.

        hf2dcat converts resources on several worker threads that share one
        manager, so the limit must be global rather than per thread.
        """
        with self._lock:
            now = time.monotonic()
            start = max(self._last_request + self.min_interval, self._pause_until)
            if start > now:
                time.sleep(start - now)
            self._last_request = time.monotonic()

    def _pause_all(self, delay: float) -> None:
        """Make every thread wait after a rate-limit response."""
        with self._lock:
            self._pause_until = max(self._pause_until, time.monotonic() + delay)

    def _request(self, text: str) -> Optional[str]:
        if self.provider == "argos":
            # A single shared translation instance is used by all converter threads.
            with self._argos_lock:
                self.stats["requests"] += 1
                try:
                    translated = self.translator.translate(text)
                    if isinstance(translated, str) and translated.strip():
                        return translated
                    logger.warning("Argos returned an empty translation for '%s...'", text[:50])
                except Exception as exc:
                    logger.warning("Argos translation failed for '%s...': %s", text[:50], exc)
                self.stats["failures"] += 1
                return None

        if self.circuit_open:
            self.stats["failures"] += 1
            return None

        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.stats["requests"] += 1
            try:
                translated = self.translator.translate(text)
            except TooManyRequests as exc:
                self._consecutive_rate_limits += 1
                if self._consecutive_rate_limits >= self.max_consecutive_rate_limits:
                    self.circuit_open = True
                    logger.error(
                        "Rate limited %d times in a row; disabling translation for "
                        "the rest of this run. Cached translations are still used. "
                        "Last error: %s",
                        self._consecutive_rate_limits,
                        exc,
                    )
                    break
                error = exc
            except RETRYABLE_ERRORS as exc:
                error = exc
            except (NotValidLength, TranslationNotFound) as exc:
                logger.warning("Translation not possible for '%s...': %s", text[:50], exc)
                break
            except Exception as exc:  # unexpected: do not retry blindly
                logger.warning("Translation failed for '%s...': %s", text[:50], exc)
                break
            else:
                self._consecutive_rate_limits = 0
                if isinstance(translated, str) and translated.strip():
                    return translated
                break

            if attempt < self.max_retries:
                delay = self.backoff_base ** (attempt + 1)
                self.stats["retries"] += 1
                logger.info(
                    "Translation request failed (%s); retrying in %.0f s.",
                    error.__class__.__name__,
                    delay,
                )
                if isinstance(error, TooManyRequests):
                    self._pause_all(delay)  # _throttle makes this thread wait too
                else:
                    time.sleep(delay)
            else:
                logger.warning(
                    "Translation failed for '%s...' after %d retries: %s",
                    text[:50],
                    self.max_retries,
                    error,
                )

        self.stats["failures"] += 1
        return None

    def _load_cache(self) -> None:
        try:
            with self.cache_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                self.cache.update(
                    {k: v for k, v in data.items() if isinstance(v, str)}
                )
                logger.info(
                    "Loaded %d cached translations from %s",
                    len(self.cache),
                    self.cache_path,
                )
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable translation cache %s: %s", self.cache_path, exc)
