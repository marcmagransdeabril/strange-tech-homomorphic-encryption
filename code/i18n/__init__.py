"""Minimal i18n helper — loads locale strings from JSON files."""

import json
import os
from pathlib import Path

_LOCALE_DIR = Path(__file__).parent
_DEFAULT_LANG = "es"
_cache = {}


def _load(lang: str) -> dict:
    if lang not in _cache:
        path = _LOCALE_DIR / f"{lang}.json"
        if path.exists():
            _cache[lang] = json.loads(path.read_text(encoding="utf-8"))
        else:
            _cache[lang] = {}
    return _cache[lang]


def t(chapter: str, key: str, lang: str | None = None) -> str:
    """Return the localized string for *chapter*.*key*.

    Language priority: BOOK_LANG env var > *lang* argument > 'es'.
    Falls back to the key itself if not found.
    """
    lang = os.environ.get("BOOK_LANG", lang or _DEFAULT_LANG)
    strings = _load(lang)
    return strings.get(chapter, {}).get(key, key)
