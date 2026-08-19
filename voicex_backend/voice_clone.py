"""
voice_clone.py
Orchestrates the Cross-Lingual Voice Cloning pipeline using the real OmniVoice API.

Architecture (correct for OmniVoice):
    reference audio + target text
        ↓
    Expression Engine → inject [expression-tag] into text
        ↓
    Emotion Engine → speed (only speed applies in cloning mode)
        ↓
    Tag Parser (V3: strip <emotion> tags from text)
        ↓
    OmniVoice model.generate(
        text=...,
        ref_audio="path.wav",
        ref_text=...,   # optional — Whisper auto-transcribes if omitted
        speed=...,
        num_step=...,
    )
        ↓
    Audio WAV bytes

Version behaviour:
  V1 — emotion speed only, no expression
  V2 — emotion speed + expression tag injected silently in text
  V3 — any <emotion> tags stripped from text before cloning

Notes
-----
- Cross-lingual cloning does NOT use the instruct parameter.
  The reference audio carries the voice identity; the target language is
  inferred from the text content.
- ref_text is optional: if provided it improves cloning accuracy.
  If omitted, OmniVoice uses Whisper ASR to auto-transcribe the reference.
"""

import io
import random
import numpy as np
import soundfile as sf
import torch

from typing import Optional

from emotion_engine import EmotionEngine
from expression_engine import ExpressionEngine
from tag_parser import TagParser
from omnivoice_engine import OmniVoiceEngine, SAMPLE_RATE

DEFAULT_NUM_STEP = 50


class VoiceCloneService:
    """
    Entry-point for the Cross-Lingual Voice Cloning feature.
    """

    def __init__(self, engine: Optional[OmniVoiceEngine] = None):
        self.engine = engine or OmniVoiceEngine.get_instance()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        text: str,
        target_language: str,
        reference_audio_bytes: bytes,
        gender: str = "neutral",
        age: int = 30,
        emotion: str = "neutral",
        expression: str = "none",
        version: int = 2,
        ref_text: str = "",
        num_step: int = DEFAULT_NUM_STEP,
    ) -> bytes:
        """
        Generate cross-lingual cloned speech.

        Parameters
        ----------
        text                  : Text in the target language.
        target_language       : Frontend language code (e.g. 'ta', 'fr').
        reference_audio_bytes : Raw bytes of reference WAV / MP3 / OGG / FLAC.
        gender                : Gender hint (informational only — not passed to OmniVoice
                                for cloning; cloning preserves the reference speaker).
        age                   : Age hint (same note as gender).
        emotion               : Emotion label → mapped to speed adjustment.
        expression            : Expression label. Always 'none' for V1.
        version               : 1 | 2 | 3.
        ref_text              : Transcription of the reference audio.
                                Optional — OmniVoice auto-transcribes via Whisper if "".
        num_step              : OmniVoice diffusion steps.

        Returns
        -------
        bytes : WAV bytes at 24 kHz mono PCM-16.

        Cross-Lingual Cloning Notes
        ---------------------------
        OmniVoice tip: for standard pronunciation, use reference audio in the
        same language as target speech. For cross-lingual cloning (different
        reference language), the output will carry an accent from the reference
        language — which is expected behaviour.
        """
        if version >= 3 and TagParser.has_tags(text):
            return self._generate_multi_segment(
                text=text,
                reference_audio_bytes=reference_audio_bytes,
                gender=gender,
                age=age,
                ref_text=ref_text,
                num_step=num_step,
            )

        # Versions 1 & 2
        # Resolve emotion → speed only (instruct not used in clone mode)
        params = EmotionEngine.resolve(emotion, gender, age)

        # Inject expression tag (V1: expression is already 'none' from caller)
        final_text = ExpressionEngine.inject_tag(text, expression)

        return self.engine.synthesize_voice_clone(
            text=final_text,
            reference_audio_bytes=reference_audio_bytes,
            ref_text=ref_text,
            speed=params.speed,
            num_step=num_step,
        )

    def _generate_multi_segment(
        self,
        text: str,
        reference_audio_bytes: bytes,
        gender: str,
        age: int,
        ref_text: str,
        num_step: int,
    ) -> bytes:
        """
        V3: parse <emotion> blocks → synthesise each with its emotion speed mapped
        → concatenate with 200 ms silence between segments.
        Expression tags are preserved strictly inline.
        """
        segments = TagParser.parse(text)
        audio_parts: list[np.ndarray] = []

        # Fix seed to ensure voice stability across segments
        request_seed = random.randint(1, 999999)

        for idx, seg in enumerate(segments):
            is_last = (idx == len(segments) - 1)

            # Resolve emotion per segment (clone uses speed adjustments)
            params = EmotionEngine.resolve(seg.emotion, gender, age)

            random.seed(request_seed)
            np.random.seed(request_seed)
            torch.manual_seed(request_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request_seed)

            wav_bytes = self.engine.synthesize_voice_clone(
                text=seg.text,
                reference_audio_bytes=reference_audio_bytes,
                ref_text=ref_text,  # ref_text is the transcription of the reference_audio_bytes (not seg.text), reused for all segments
                speed=params.speed,
                num_step=num_step,
            )

            arr, _ = sf.read(io.BytesIO(wav_bytes))
            arr = arr.astype(np.float32)

            # Apply 10ms fade-in/out to prevent audio clicking/popping at boundaries
            fade_len = int(SAMPLE_RATE * 0.01)
            if len(arr) > fade_len * 2:
                fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                arr[:fade_len] *= fade_in
                arr[-fade_len:] *= fade_out

            audio_parts.append(arr)

            # 200 ms silence gap
            if not is_last:
                silence = np.zeros(int(SAMPLE_RATE * 0.20), dtype=np.float32)
                audio_parts.append(silence)

        if not audio_parts:
            raise ValueError("No valid text segments found.")

        combined = np.concatenate(audio_parts)
        buf = io.BytesIO()
        sf.write(buf, combined, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
