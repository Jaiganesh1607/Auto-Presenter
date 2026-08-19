"""
voice_design.py
Orchestrates the Voice Design pipeline using the real OmniVoice API.

Architecture (correct for OmniVoice):
    user input
        ↓
    Tag Parser (V3: split text into emotion segments)
        ↓
    Emotion Engine → EmotionParams (pitch_hint, style_hint, speed)
        ↓
    EmotionEngine.build_instruct() → "female, high pitch, British accent"
        ↓
    Expression Engine → inject [expression-tag] into text
        ↓
    OmniVoice model.generate(text=..., instruct=..., speed=...)
        ↓
    Audio WAV bytes

Version behaviour:
  V1 — single emotion, no expression
  V2 — single emotion + expression tag injected silently in text
  V3 — per-sentence <emotion> tags → multi-segment synthesis → concatenate

OmniVoice API used:
    model.generate(text=..., instruct=..., speed=..., num_step=...)
    # audio is List[np.ndarray] at 24 kHz
"""

import io
import random
import numpy as np
import soundfile as sf
import librosa
import torch
from typing import Optional

from emotion_engine import EmotionEngine
from expression_engine import ExpressionEngine
from language_router import LanguageRouter
from tag_parser import TagParser
from omnivoice_engine import OmniVoiceEngine, SAMPLE_RATE

# Diffusion steps: 50 = highest quality, 16 = faster inference
DEFAULT_NUM_STEP = 50


class VoiceDesignService:
    """
    Entry-point for the Voice Design feature.

    Handles:
      - Single-emotion synthesis (V1 / V2)
      - Multi-emotion tagged synthesis (V3)
    """

    def __init__(self, engine: Optional[OmniVoiceEngine] = None):
        self.engine = engine or OmniVoiceEngine.get_instance()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        text: str,
        language: str,
        gender: str = "neutral",
        age: int = 30,
        emotion: str = "neutral",
        expression: str = "none",
        version: int = 2,
        num_step: int = DEFAULT_NUM_STEP,
    ) -> bytes:
        """
        Generate speech via OmniVoice Voice Design.

        Parameters
        ----------
        text       : User text (may contain emotion tags in V3).
        language   : Frontend language code (e.g. 'en-US', 'ta', 'fr').
        gender     : 'male' | 'female' | 'neutral'.
        age        : Speaker age 5–100.
        emotion    : Emotion label — ignored for V3 (tags take precedence).
        expression : Expression label. Always 'none' for V1 (enforced by caller).
        version    : 1 | 2 | 3.
        num_step   : OmniVoice diffusion steps (16 = fast, 32 = quality).

        Returns
        -------
        bytes : WAV bytes at 24 kHz mono PCM-16.
        """
        lang_code = LanguageRouter.resolve(language)
        accent    = LanguageRouter.get_accent(language)

        # Route ALL versions through the robust multi-segment generator.
        # This allows isolated expression generation (Voice Design) for Voice Clone languages!
        final_text = text
        if version < 3:
            final_text = ExpressionEngine.inject_tag(text, expression)
            # Wrap the entire text in the selected emotion tag so TagParser processes it
            final_text = f"<{emotion}>{final_text}</{emotion}>"

        return self._generate_multi_segment(
            final_text, lang_code, gender, age, accent, expression, num_step
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _is_natively_supported(self, lang_code: str) -> bool:
        """Check if the language is natively supported by OmniVoice Voice Design instruct."""
        return lang_code.startswith("en") or lang_code.startswith("zh") or lang_code == "yue"

    def _get_dummy_text(self, gender: str, emotion: str) -> str:
        """Generate a highly expressive text to force OmniVoice zero-shot emotional transfer."""
        emotion = emotion.lower()
        g_prefix = "I am a man speaking. " if gender.lower() == "male" else (
                   "I am a woman speaking. " if gender.lower() == "female" else "")
        
        if emotion in ("happy", "excited"):
            return g_prefix + "I am so incredibly happy and excited today!"
        if emotion in ("sad", "fearful"):
            return g_prefix + "I am feeling very sad and heartbroken today."
        if emotion == "angry":
            return g_prefix + "I am absolutely furious and angry about this!"
        if emotion in ("whisper", "calm"):
            return g_prefix + "I am speaking very softly and calmly."
        if emotion == "surprised":
            return g_prefix + "Wow, I am so incredibly surprised by this!"
        if emotion == "disgusted":
            return g_prefix + "This is absolutely disgusting and awful."
            
        return g_prefix + f"This is my normal voice, speaking with a {emotion} emotion."



    def _generate_multi_segment(
        self,
        text: str,
        lang_code: str,
        gender: str,
        age: int,
        accent: str,
        expression: str,
        num_step: int,
        max_attempts: int = 2
    ) -> bytes:
        """
        V3: parse <emotion> blocks → synthesise each with its emotion instruct
        → concatenate with 200 ms silence between segments.

        Expression tags are parsed strictly inline exactly where the user placed them.
        """
        segments    = TagParser.parse(text)
        audio_parts: list[np.ndarray] = []

        # Pick a single random seed for this entire request.
        # By applying the exact same seed before every segment, we force
        # the model's random noise to be identical, which guarantees the 
        # speaker identity (gender, tone) remains exactly the same person 
        # across different emotion tags!
        # Note: setting np.random.seed and torch.manual_seed identically per segment
        # means diffusion noise is identical, which helps consistency but may cause artifacts.
        request_seed = random.randint(1, 999999)

        # Bug 13 Fix: For non-native languages, generate ONE dummy WAV outside the loop
        # so the reference voice identity remains identical across all segments.
        shared_dummy_wav = None
        if not self._is_natively_supported(lang_code) and segments:
            # Generate the dummy voice based on the FIRST segment's emotion
            first_seg_emotion = segments[0].emotion
            dummy_params = EmotionEngine.resolve(first_seg_emotion, gender, age)
            dummy_instruct = EmotionEngine.build_instruct(gender, age, dummy_params, accent)
            dummy_text = self._get_dummy_text(gender, first_seg_emotion)
            
            shared_dummy_wav = self.engine.synthesize_voice_design(
                text=dummy_text,
                instruct=dummy_instruct,
                speed=dummy_params.speed,
                num_step=50, # Highest quality reference
            )

        for idx, seg in enumerate(segments):
            is_last = (idx == len(segments) - 1)

            # Resolve emotion per segment
            params   = EmotionEngine.resolve(seg.emotion, gender, age)
            instruct = EmotionEngine.build_instruct(gender, age, params, accent)

            # V3 strictly relies on inline manual tags. Do NOT invoke Qwen ExpressionEngine.
            seg_text = seg.text
            
            # Find which expressions the user typed in this segment so we can verify them
            tags_to_verify = []
            if getattr(seg, "is_expression", False):
                tags_to_verify.append(seg.text)

            from audio_verifier import AudioVerifier
            max_attempts = 2
            best_wav_bytes = b""

            for attempt in range(max_attempts):
                if attempt > 0:
                    print(f"🔄 [Verification Loop V3] Expression missing. Retrying '{seg_text}' (Attempt {attempt+1}/{max_attempts})...")
                    take_seed = request_seed + (attempt * 100)
                else:
                    take_seed = request_seed

                # Force exact same speaker characteristics (with slight variance on retries)
                random.seed(take_seed)
                np.random.seed(take_seed)
                torch.manual_seed(take_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(take_seed)

                if getattr(seg, "is_expression", False) or self._is_natively_supported(lang_code):
                    # BUG FIX: OmniVoice's Voice Clone mode does not support non-verbal tags!
                    # We MUST force Voice Design mode for standalone expression segments.
                    # Because the instruct params are identical to the shared_dummy_wav, 
                    # the voice character will perfectly match the cloned foreign speech!
                    wav_bytes = self.engine.synthesize_voice_design(
                        text=seg_text,
                        instruct=instruct,
                        speed=params.speed,
                        num_step=num_step,
                    )
                else:
                    wav_bytes = self.engine.synthesize_voice_clone(
                        text=seg_text,
                        reference_audio_bytes=shared_dummy_wav,
                        ref_text="",
                        speed=params.speed,
                        num_step=num_step,
                    )
                    
                best_wav_bytes = wav_bytes
                
                # Verify if the audio actually contains the expression
                if AudioVerifier.verify_expression(wav_bytes, tags_to_verify):
                    break

            wav_bytes = best_wav_bytes

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

            # 200 ms silence gap between segments (not after last)
            if not is_last:
                silence = np.zeros(int(SAMPLE_RATE * 0.20), dtype=np.float32)
                audio_parts.append(silence)

        if not audio_parts:
            raise ValueError("No valid text segments found.")

        combined = np.concatenate(audio_parts)
        buf      = io.BytesIO()
        sf.write(buf, combined, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
