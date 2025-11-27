import hashlib
import logging
import re
from deep_translator import GoogleTranslator
from typing import Tuple, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranslationManager:
    def __init__(self, enable_translation: bool = True):
        self.enabled = enable_translation
        self.translator = GoogleTranslator(source='en', target='de') if enable_translation else None
        self.cache = {}

        self.MD_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\s)]+)\)')
        self.RAW_URL_RE = re.compile(r'https?://[^\s)\]]+', re.IGNORECASE)

    def _extract_urls(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replace URLs with emoji-safe tokens"""
        url_map = {}
        counter = 0

        def md_replacer(match):
            nonlocal counter
            label, url = match.group(1), match.group(2)
            token = f"🔒{counter}🔒"
            url_map[token] = f"[{label}]({url})"
            counter += 1
            return token

        masked_text = self.MD_LINK_RE.sub(md_replacer, text)

        def raw_replacer(match):
            nonlocal counter
            url = match.group(0)
            token = f"🔒{counter}🔒"
            url_map[token] = url
            counter += 1
            return token

        masked_text = self.RAW_URL_RE.sub(raw_replacer, masked_text)
        return masked_text, url_map

    def _restore_urls(self, text: str, url_map: Dict[str, str]) -> str:
        """Restore URLs after translation"""
        for token, original in url_map.items():
            text = text.replace(token, original)
        return text

    def translate_text(self, text: str) -> str:
        if not self.enabled or not text or not isinstance(text, str):
            return text

        masked_text, url_map = self._extract_urls(text)
        translated = self._translate_plain_text(masked_text)
        return self._restore_urls(translated, url_map)

    def _translate_plain_text(self, text: str) -> str:
        if not text:
            return text

        cache_key = f"de_{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            translated = self.translator.translate(text)
            self.cache[cache_key] = translated
            return translated
        except Exception as e:
            logger.warning(f"Translation failed for '{text[:50]}...': {e}")
            return text


