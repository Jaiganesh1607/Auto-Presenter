"""
expression_engine.py
Maps user-facing Expression dropdown values → OmniVoice expression tags.
Tags are appended to the text before inference; users never see them.
"""

import logging
from typing import Dict, List

logger = logging.getLogger("omnivoice_studio")


# ---------------------------------------------------------------------------
# Expression → OmniVoice tag mapping
# ---------------------------------------------------------------------------
EXPRESSION_TAG_MAP: Dict[str, str] = {
    "none":            "",
    "giggle":          "[laughter]",
    "laughter":        "[laughter]",
    "laugh":           "[laughter]",
    "sigh":            "[sigh]",
    "question":        "[question-en]",
    "question_en":     "[question-en]",
    "question_ah":     "[question-ah]",
    "question_oh":     "[question-oh]",
    "question_ei":     "[question-ei]",
    "question_yi":     "[question-yi]",
    "surprise":        "[surprise-ah]",
    "surprise_ah":     "[surprise-ah]",
    "surprise_oh":     "[surprise-oh]",
    "surprise_wa":     "[surprise-wa]",
    "surprise_yo":     "[surprise-yo]",
    "dissatisfaction": "[dissatisfaction-hnn]",
    "confirmation":    "[confirmation-en]",
}

# Display labels shown to the user (Version 2 dropdown)
EXPRESSION_DISPLAY_OPTIONS: List[Dict[str, str]] = [
    {"value": "none",            "label": "None"},
    {"value": "giggle",          "label": "Giggle"},
    {"value": "sigh",            "label": "Sigh"},
    {"value": "question",        "label": "Question"},
    {"value": "surprise",        "label": "Surprise"},
    {"value": "dissatisfaction", "label": "Dissatisfaction"},
    {"value": "confirmation",    "label": "Confirmation"},
]


class ExpressionEngine:
    """
    Resolves an expression label → OmniVoice tag string.
    The tag is inserted using a strict rule-based mapping.
    """

    @staticmethod
    def get_tag(expression: str) -> str:
        """
        Return the OmniVoice tag for the given expression label.
        """
        key = expression.lower().strip().replace(" ", "_").replace("-", "_")
        return EXPRESSION_TAG_MAP.get(key, "")

    @classmethod
    def inject_tag(cls, text: str, expression: str) -> str:
        """
        Insert the expression tag into *text* if applicable.
        Inserts the tag before the final punctuation mark if present.
        """
        tag = cls.get_tag(expression)
        if not tag:
            return text
            
        text = text.strip()
        # Per user request, tags work best across all languages when placed AFTER the punctuation.
        return f"{text} {tag} "


    @staticmethod
    def available_expressions() -> List[Dict[str, str]]:
        """Return the ordered list of display options for the frontend."""
        return EXPRESSION_DISPLAY_OPTIONS
