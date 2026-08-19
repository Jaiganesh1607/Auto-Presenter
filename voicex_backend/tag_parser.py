"""
tag_parser.py
Parses Version-3 advanced emotion tag markup from user text.

Supported tags:
    <happy> … </happy>
    <sad>   … </sad>
    <angry> … </angry>
    <excited> … </excited>
    <calm>  … </calm>
    <whisper> … </whisper>

Output: list of TextSegment objects, each carrying its own emotion.
"""

import re
from dataclasses import dataclass
from typing import List

from expression_engine import EXPRESSION_TAG_MAP


SUPPORTED_EMOTIONS = {
    "happy", "sad", "angry", "excited", "calm", "whisper",
    "neutral", "fearful", "surprised", "disgusted",
}

# Regex that captures any <emotion> … </emotion> block
_TAG_RE = re.compile(
    r"<(?P<emotion>" + "|".join(SUPPORTED_EMOTIONS) + r")>"
    r"(?P<content>.*?)"
    r"</(?P=emotion)>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class TextSegment:
    """A single segment of text with its associated emotion."""
    text: str
    emotion: str = "neutral"
    is_expression: bool = False


class TagParser:
    """
    Parses emotion-tagged text and returns an ordered list of TextSegments.

    Usage
    -----
    segments = TagParser.parse("<happy>Great!</happy> But <sad>it ended.</sad>")
    # → [TextSegment("Great!", "happy"), TextSegment("But", "neutral"),
    #     TextSegment("it ended.", "sad")]
    """

    @staticmethod
    def _replace_inline_expressions(text: str) -> str:
        """Replace inline <expression> tags with their OmniVoice [tag] equivalents."""
        for expr, tag in EXPRESSION_TAG_MAP.items():
            if not tag:
                continue
            
            # Allow users to type either <question_ah> or <question-ah>
            expr_pattern = expr.replace("_", "[-_]")
            
            # Convert <giggle> -> [laughter] with STRICT space isolation.
            # Spaces guarantee OmniVoice tokenizes it as a control token instead of a gibberish word.
            text = re.sub(rf"<{expr_pattern}>", f" {tag} ", text, flags=re.IGNORECASE)
            text = re.sub(rf"\[{expr_pattern}\]", f" {tag} ", text, flags=re.IGNORECASE)
            
        # Clean up any double spaces created by the isolation
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _split_expression_tags(segments: List[TextSegment]) -> List[TextSegment]:
        new_segments = []
        tags_to_find = [tag for tag in EXPRESSION_TAG_MAP.values() if tag]
        
        if not tags_to_find:
            return segments
            
        # Regex to split on exact tag strings
        pattern = r"(" + "|".join([re.escape(t) for t in tags_to_find]) + r")"
        
        for seg in segments:
            parts = re.split(pattern, seg.text)
            for part in parts:
                part = part.strip()
                if not part: continue
                
                if part in tags_to_find:
                    new_segments.append(TextSegment(text=part, emotion=seg.emotion, is_expression=True))
                else:
                    new_segments.append(TextSegment(text=part, emotion=seg.emotion, is_expression=False))
        return new_segments

    @staticmethod
    def parse(text: str) -> List[TextSegment]:
        """
        Parse text containing optional emotion tags and inline expression tags.

        If no emotion tags are found, the entire text is returned as a single
        neutral segment. Expression tags are converted to OmniVoice native tags.

        Parameters
        ----------
        text : str
            Raw user input (may or may not contain emotion tags).

        Returns
        -------
        List[TextSegment]
            Ordered list of segments with emotions assigned.
        """
        segments: List[TextSegment] = []
        cursor = 0

        # Pre-process inline expression tags before splitting emotions
        text = TagParser._replace_inline_expressions(text)

        for match in _TAG_RE.finditer(text):
            # Text before the tag → neutral
            before = text[cursor:match.start()].strip()
            if before:
                segments.append(TextSegment(text=before, emotion="neutral"))

            emotion  = match.group("emotion").lower()
            content  = match.group("content").strip()
            if content:
                segments.append(TextSegment(text=content, emotion=emotion))

            cursor = match.end()

        # Remaining text after the last tag → neutral
        tail = text[cursor:].strip()
        if tail:
            segments.append(TextSegment(text=tail, emotion="neutral"))

        # Fallback: no tags found
        if not segments:
            segments.append(TextSegment(text=text.strip(), emotion="neutral"))

        return TagParser._split_expression_tags(segments)

    @staticmethod
    def has_tags(text: str) -> bool:
        """Return True if *text* contains at least one emotion tag or expression tag."""
        if bool(_TAG_RE.search(text)):
            return True
        for expr, tag in EXPRESSION_TAG_MAP.items():
            if not tag: continue
            expr_pattern = expr.replace("_", "[-_]")
            if re.search(rf"<{expr_pattern}>", text, flags=re.IGNORECASE) or \
               re.search(rf"\[{expr_pattern}\]", text, flags=re.IGNORECASE):
                return True
        return False

    @staticmethod
    def strip_tags(text: str) -> str:
        """Remove all emotion tags from *text*, returning plain content."""
        result = _TAG_RE.sub(lambda m: m.group("content").strip(), text)
        return result.strip()
