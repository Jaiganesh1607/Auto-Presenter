"""
language_router.py
Maps frontend language labels → OmniVoice language codes.
Also exposes the full grouped language catalogue used by the frontend.
"""

from typing import Dict, List


# ---------------------------------------------------------------------------
# Full language catalogue — 100+ entries
# ---------------------------------------------------------------------------

LANGUAGE_GROUPS: Dict[str, List[Dict[str, str]]] = {
    "English Accents": [
        {"value": "en-US", "label": "American English"},
        {"value": "en-GB", "label": "British English"},
        {"value": "en-AU", "label": "Australian English"},
        {"value": "en-IN", "label": "Indian English"},
        {"value": "en-CA", "label": "Canadian English"},
    ],
    "Indian Languages": [
        {"value": "ta",   "label": "Tamil"},
        {"value": "te",   "label": "Telugu"},
        {"value": "hi",   "label": "Hindi"},
        {"value": "ml",   "label": "Malayalam"},
        {"value": "kn",   "label": "Kannada"},
        {"value": "bn",   "label": "Bengali"},
        {"value": "mr",   "label": "Marathi"},
        {"value": "gu",   "label": "Gujarati"},
        {"value": "pa",   "label": "Punjabi"},
        {"value": "or",   "label": "Odia"},
        {"value": "as",   "label": "Assamese"},
        {"value": "sa",   "label": "Sanskrit"},
        {"value": "ne",   "label": "Nepali"},
        {"value": "ur",   "label": "Urdu"},
        {"value": "ks",   "label": "Kashmiri"},
        {"value": "sd",   "label": "Sindhi"},
        {"value": "mai",  "label": "Maithili"},
        {"value": "doi",  "label": "Dogri"},
        {"value": "kok",  "label": "Konkani"},
        {"value": "mni",  "label": "Meitei (Manipuri)"},
        {"value": "sat",  "label": "Santali"},
        {"value": "bho",  "label": "Bhojpuri"},
    ],
    "Foreign Languages": [
        # European
        {"value": "es",    "label": "Spanish"},
        {"value": "es-MX", "label": "Mexican Spanish"},
        {"value": "fr",    "label": "French"},
        {"value": "fr-CA", "label": "Canadian French"},
        {"value": "de",    "label": "German"},
        {"value": "it",    "label": "Italian"},
        {"value": "pt",    "label": "Portuguese"},
        {"value": "pt-BR", "label": "Brazilian Portuguese"},
        {"value": "nl",    "label": "Dutch"},
        {"value": "pl",    "label": "Polish"},
        {"value": "ru",    "label": "Russian"},
        {"value": "uk",    "label": "Ukrainian"},
        {"value": "cs",    "label": "Czech"},
        {"value": "sk",    "label": "Slovak"},
        {"value": "sv",    "label": "Swedish"},
        {"value": "da",    "label": "Danish"},
        {"value": "no",    "label": "Norwegian"},
        {"value": "fi",    "label": "Finnish"},
        {"value": "hu",    "label": "Hungarian"},
        {"value": "ro",    "label": "Romanian"},
        {"value": "el",    "label": "Greek"},
        {"value": "tr",    "label": "Turkish"},
        {"value": "bg",    "label": "Bulgarian"},
        {"value": "hr",    "label": "Croatian"},
        {"value": "sr",    "label": "Serbian"},
        {"value": "sl",    "label": "Slovenian"},
        {"value": "et",    "label": "Estonian"},
        {"value": "lv",    "label": "Latvian"},
        {"value": "lt",    "label": "Lithuanian"},
        {"value": "ca",    "label": "Catalan"},
        {"value": "gl",    "label": "Galician"},
        {"value": "eu",    "label": "Basque"},
        {"value": "is",    "label": "Icelandic"},
        {"value": "lb",    "label": "Luxembourgish"},
        {"value": "mt",    "label": "Maltese"},
        {"value": "cy",    "label": "Welsh"},
        {"value": "ga",    "label": "Irish"},
        {"value": "sq",    "label": "Albanian"},
        {"value": "mk",    "label": "Macedonian"},
        {"value": "bs",    "label": "Bosnian"},
        # East Asian
        {"value": "zh-CN", "label": "Mandarin Chinese (Simplified)"},
        {"value": "zh-TW", "label": "Mandarin Chinese (Traditional)"},
        {"value": "yue",   "label": "Cantonese"},
        {"value": "ja",    "label": "Japanese"},
        {"value": "ko",    "label": "Korean"},
        {"value": "vi",    "label": "Vietnamese"},
        {"value": "th",    "label": "Thai"},
        {"value": "my",    "label": "Burmese"},
        {"value": "km",    "label": "Khmer"},
        {"value": "lo",    "label": "Lao"},
        {"value": "mn",    "label": "Mongolian"},
        {"value": "bo",    "label": "Tibetan"},
        # Southeast Asian
        {"value": "id",    "label": "Indonesian"},
        {"value": "ms",    "label": "Malay"},
        {"value": "tl",    "label": "Filipino (Tagalog)"},
        {"value": "jv",    "label": "Javanese"},
        {"value": "su",    "label": "Sundanese"},
        {"value": "ceb",   "label": "Cebuano"},
        # Middle Eastern & Central Asian
        {"value": "ar",    "label": "Arabic (Modern Standard)"},
        {"value": "ar-EG", "label": "Arabic (Egyptian)"},
        {"value": "ar-SA", "label": "Arabic (Gulf)"},
        {"value": "he",    "label": "Hebrew"},
        {"value": "fa",    "label": "Persian (Farsi)"},
        {"value": "ps",    "label": "Pashto"},
        {"value": "ku",    "label": "Kurdish"},
        {"value": "az",    "label": "Azerbaijani"},
        {"value": "kk",    "label": "Kazakh"},
        {"value": "ky",    "label": "Kyrgyz"},
        {"value": "uz",    "label": "Uzbek"},
        {"value": "tk",    "label": "Turkmen"},
        {"value": "tg",    "label": "Tajik"},
        {"value": "hy",    "label": "Armenian"},
        {"value": "ka",    "label": "Georgian"},
        # African
        {"value": "sw",    "label": "Swahili"},
        {"value": "am",    "label": "Amharic"},
        {"value": "ha",    "label": "Hausa"},
        {"value": "yo",    "label": "Yoruba"},
        {"value": "ig",    "label": "Igbo"},
        {"value": "zu",    "label": "Zulu"},
        {"value": "xh",    "label": "Xhosa"},
        {"value": "st",    "label": "Sotho"},
        {"value": "sn",    "label": "Shona"},
        {"value": "so",    "label": "Somali"},
        {"value": "rw",    "label": "Kinyarwanda"},
        {"value": "mg",    "label": "Malagasy"},
        # South Asian (non-Indian)
        {"value": "si",    "label": "Sinhala"},
        {"value": "dz",    "label": "Dzongkha"},
        # Pacific
        {"value": "mi",    "label": "Māori"},
        {"value": "haw",   "label": "Hawaiian"},
        {"value": "sm",    "label": "Samoan"},
    ],
}


# ---------------------------------------------------------------------------
# Flat code → OmniVoice language-code map
# ---------------------------------------------------------------------------
_FLAT_MAP: Dict[str, str] = {
    entry["value"]: entry["value"]
    for group in LANGUAGE_GROUPS.values()
    for entry in group
}

# ---------------------------------------------------------------------------
# Accent map for Voice Design instruct parameter
# OmniVoice Voice Design is primarily trained on English and Chinese.
# For English variants, pass an accent hint via instruct.
# For all other languages, leave accent blank and rely on the model.
# ---------------------------------------------------------------------------
ACCENT_MAP: Dict[str, str] = {
    "en-US": "american accent",
    "en-GB": "british accent",
    "en-AU": "australian accent",
    "en-IN": "indian accent",
    "en-CA": "canadian accent",
    # Chinese dialects — OmniVoice supports these in Voice Design
    "zh-CN": "",           # default Mandarin (no special accent label needed)
    "zh-TW": "",
    "yue":   "",
}


class LanguageRouter:
    """
    Translates frontend language selectors → OmniVoice language codes
    and provides Voice Design accent hints.
    """

    @staticmethod
    def resolve(lang_value: str) -> str:
        """
        Return the OmniVoice language code for *lang_value*.
        Falls back to 'en-US' if unknown.
        """
        return _FLAT_MAP.get(lang_value, "en-US")

    @staticmethod
    def get_accent(lang_value: str) -> str:
        """
        Return the accent descriptor string for use in OmniVoice instruct.

        Returns "" for non-English / non-Chinese languages — the model
        handles language via the text input + language_id naturally.
        """
        return ACCENT_MAP.get(lang_value, "")

    @staticmethod
    def get_catalogue() -> Dict[str, List[Dict[str, str]]]:
        """Return the full grouped language catalogue for the API."""
        return LANGUAGE_GROUPS

    @staticmethod
    def all_codes() -> List[str]:
        """Return all known language codes."""
        return list(_FLAT_MAP.keys())
