"""
emotion_engine.py
Maps user-facing emotion labels → OmniVoice-compatible parameters.

Architecture
------------
Emotion + Gender + Age
        ↓
  EmotionParams
  ├─ pitch_hint   : str  → used in Voice Design instruct (e.g. "high pitch")
  ├─ style_hint   : str  → used in Voice Design instruct (e.g. "whisper")
  └─ speed        : float→ passed as model.generate(speed=...) for both modes

OmniVoice Voice Design uses the `instruct` parameter for voice attributes.
OmniVoice Voice Cloning uses `speed` for pacing (cloning does not use instruct).

OmniVoice supported pitch descriptors (instruct):
  "very low pitch" | "low pitch" | "medium pitch" | "high pitch" | "very high pitch"

OmniVoice supported style descriptors (instruct):
  "whisper"

All other fields (pitch_shift, energy_scale) from the old architecture are
REMOVED — they are not part of the OmniVoice API.

User never sees these parameters. The Emotion Layer handles them automatically.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EmotionParams:
    """
    Resolved parameters for a single emotion + speaker combo.

    pitch_hint  : OmniVoice instruct pitch descriptor, or "" for default.
    style_hint  : OmniVoice instruct style descriptor ("whisper" or "").
    speed       : Speaking rate factor for model.generate(speed=...).
    """
    pitch_hint: str    # "very low pitch" | "low pitch" | "medium pitch" | "high pitch" | "very high pitch" | ""
    style_hint: str    # "whisper" | ""
    speed: float       # 0.7 … 1.4
    emotion_name: str = "" # e.g. "sad", "happy"


# ---------------------------------------------------------------------------
# Core emotion map  (pitch_hint, style_hint, speed)
# ---------------------------------------------------------------------------
EMOTION_MAP: Dict[str, EmotionParams] = {
    "neutral":   EmotionParams(pitch_hint="",                style_hint="",        speed=0.85),
    "happy":     EmotionParams(pitch_hint="high pitch",      style_hint="",        speed=0.95),
    "sad":       EmotionParams(pitch_hint="low pitch",       style_hint="",        speed=0.75),
    "calm":      EmotionParams(pitch_hint="",                style_hint="",        speed=0.75),
    "whisper":   EmotionParams(pitch_hint="",                style_hint="whisper", speed=0.75),
}

# ---------------------------------------------------------------------------
# Age → OmniVoice age descriptor (used in instruct for Voice Design)
# ---------------------------------------------------------------------------
def _age_to_descriptor(age: int) -> str:
    """
    Map numeric age to an OmniVoice-compatible age descriptor.

    OmniVoice understands: child, teenager, young adult, middle-aged, elderly.
    """
    if age <= 12:   return "child"
    if age <= 19:   return "teenager"
    if age <= 35:   return "young adult"
    if age <= 50:   return ""          # default adult — no descriptor needed
    if age <= 70:   return "middle-aged"
    return "elderly"


# ---------------------------------------------------------------------------
# Speed adjustment for age (slight scaling on top of emotion speed)
# ---------------------------------------------------------------------------
def _age_speed_factor(age: int) -> float:
    """
    Children and young voices tend to speak slightly faster;
    elderly voices slightly slower.
    """
    if age <= 15:  return 1.10
    if age <= 25:  return 1.05
    if age <= 50:  return 1.00
    if age <= 70:  return 0.97
    return 0.93


# ---------------------------------------------------------------------------
# Gender → OmniVoice instruct string (e.g. "male", "female", or "")
# ---------------------------------------------------------------------------
def _gender_to_descriptor(gender: str) -> str:
    g = gender.lower().strip()
    if g in ("male", "female"):
        return g
    return ""   # neutral → let model pick


# ---------------------------------------------------------------------------
# EmotionEngine
# ---------------------------------------------------------------------------
class EmotionEngine:
    """
    Resolves emotion + gender + age → OmniVoice-compatible EmotionParams.

    For Voice Design:
        Use pitch_hint and style_hint to build the `instruct` string.
    For Voice Cloning:
        Use speed only (OmniVoice cloning ignores instruct).
    """

    @staticmethod
    def resolve(
        emotion: str,
        gender: str = "neutral",
        age: int = 30,
    ) -> EmotionParams:
        """
        Return OmniVoice-compatible emotion parameters.

        Parameters
        ----------
        emotion : Emotion label (case-insensitive). Defaults to 'neutral'.
        gender  : 'male' | 'female' | 'neutral'.
        age     : Speaker age 5–100.

        Returns
        -------
        EmotionParams with pitch_hint, style_hint, and speed.
        """
        key  = emotion.lower().strip()
        base = EMOTION_MAP.get(key, EMOTION_MAP["neutral"])

        # Adjust speed for age
        final_speed = base.speed * _age_speed_factor(age)
        
        # BUG FIX: OmniVoice male voices sound highly robotic and synthetic if slowed down.
        # We enforce a base speed of 1.0 for males, while keeping the 0.85 for females.
        if gender.lower() == "male":
            final_speed = 1.0 * _age_speed_factor(age)

        final_speed = round(max(0.60, min(1.50, final_speed)), 3)

        return EmotionParams(
            pitch_hint=base.pitch_hint,
            style_hint=base.style_hint,
            speed=final_speed,
            emotion_name=key if key not in ("neutral", "whisper") else ""
        )

    @staticmethod
    def build_instruct(
        gender: str,
        age: int,
        emotion_params: EmotionParams,
        accent: str = "",
    ) -> str:
        """
        Build the OmniVoice `instruct` string for Voice Design mode.

        Parameters
        ----------
        gender         : 'male' | 'female' | 'neutral'.
        age            : Speaker age.
        emotion_params : Resolved EmotionParams from resolve().
        accent         : Optional accent string (e.g. 'British accent').

        Returns
        -------
        str  : Comma-separated instruct string, e.g.
               "female, young, high pitch, british accent"
               ""  → auto-voice (OmniVoice chooses voice)

        Notes
        -----
        Voice Design in OmniVoice is primarily trained on Chinese and English.
        It can generalise to other languages but may be less stable for
        low-resource languages.
        """
        parts: list[str] = []

        g = _gender_to_descriptor(gender)
        if g:
            parts.append(g)

        age_desc = _age_to_descriptor(age)
        if age_desc:
            parts.append(age_desc)


        if emotion_params.pitch_hint:
            actual_pitch = emotion_params.pitch_hint
            # Prevent male voices from flipping to female when "happy" or "excited" is chosen
            if g == "male":
                if actual_pitch == "very high pitch":
                    actual_pitch = "high pitch"
                elif actual_pitch == "high pitch":
                    actual_pitch = "moderate pitch"
            parts.append(actual_pitch)

        if emotion_params.style_hint:
            parts.append(emotion_params.style_hint)

        if accent:
            parts.append(accent)

        return ", ".join(parts)

    @staticmethod
    def available_emotions() -> list[str]:
        return sorted(EMOTION_MAP.keys())
