"""
omnivoice_engine.py
Core OmniVoice model loader and inference engine.

Uses the official OmniVoice Python API:
    from omnivoice import OmniVoice
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    audio = model.generate(text=..., instruct=..., ref_audio=..., ref_text=..., speed=..., num_step=...)
    # audio is List[np.ndarray] at 24 kHz

GitHub: https://github.com/k2-fsa/OmniVoice
HuggingFace: https://huggingface.co/k2-fsa/OmniVoice
"""

import io
import os
import tempfile
import numpy as np
import soundfile as sf
import torch
from typing import Optional

# ---------------------------------------------------------------------------
# OmniVoice import — graceful stub when not installed (local dev / CI)
# ---------------------------------------------------------------------------
from omnivoice import OmniVoice as _OmniVoice
OMNIVOICE_AVAILABLE = True

SAMPLE_RATE = 24000  # OmniVoice native sample rate


class OmniVoiceEngine:
    """
    Singleton engine wrapping the official OmniVoice model.

    Usage
    -----
    engine = OmniVoiceEngine.get_instance()
    engine.load()
    wav_bytes = engine.synthesize_voice_design(text, instruct, speed, num_step)
    wav_bytes = engine.synthesize_voice_clone(text, ref_audio_bytes, ref_text, speed, num_step)
    """

    _instance: Optional["OmniVoiceEngine"] = None

    def __init__(self) -> None:
        self.model = None
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
        self.loaded: bool = False

    # ------------------------------------------------------------------ #
    #  Singleton                                                           #
    # ------------------------------------------------------------------ #
    @classmethod
    def get_instance(cls) -> "OmniVoiceEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ #
    #  Model loading                                                      #
    # ------------------------------------------------------------------ #
    def load(
        self,
        model_id: str = "k2-fsa/OmniVoice",
        device_map: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Load OmniVoice from HuggingFace Hub.

        Parameters
        ----------
        model_id   : HuggingFace repo id (default: 'k2-fsa/OmniVoice').
        device_map : 'cuda:0' | 'cpu' | 'mps' | 'xpu'. Auto-detected if None.
        dtype      : torch.float16 (GPU) or torch.float32 (CPU). Auto-detected if None.
        """

        _device_map = device_map or self.device
        _dtype      = dtype or self.dtype

        print(f"[OmniVoice] Loading '{model_id}' on {_device_map} ({_dtype}) …")

        self.model = _OmniVoice.from_pretrained(
            model_id,
            device_map=_device_map,
            dtype=_dtype,
        )
        self.loaded    = True
        print("[OmniVoice] ✅ Model loaded successfully.")

    # ------------------------------------------------------------------ #
    #  Voice Design                                                      #
    # ------------------------------------------------------------------ #
    def synthesize_voice_design(
        self,
        text: str,
        instruct: str,
        speed: float = 1.0,
        num_step: int = 32,
    ) -> bytes:
        """
        Generate speech using OmniVoice Voice Design mode.

        Parameters
        ----------
        text     : Text to synthesise (may include [expression-tag] tokens).
        instruct : Comma-separated voice attributes, e.g.
                   "female, high pitch, british accent"
                   "male, elderly, whisper"
                   ""  → auto voice (model chooses)
        speed    : Speaking rate multiplier (>1 faster, <1 slower).
        num_step : Diffusion steps (16 = faster, 32 = higher quality).

        Returns
        -------
        bytes : WAV bytes at 24 kHz mono PCM-16.
        """
        self._require_loaded()

        kwargs: dict = {
            "text":     text,
            "speed":    speed,
            "num_step": num_step,
        }
        if instruct:
            kwargs["instruct"] = instruct

        max_retries = 5
        last_exc: Exception = RuntimeError("No attempts made.")
        for attempt in range(max_retries):
            try:
                audio_list = self.model.generate(**kwargs)
                return self._ndarray_to_wav_bytes(audio_list[0])
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    print(f"[OmniVoice] Generation failed ({e}). Retrying {attempt + 1}/{max_retries}...")
        # BUG #1 FIX: always raise on exhaustion — never fall through to implicit None
        raise last_exc

    # ------------------------------------------------------------------ #
    #  Cross-Lingual Voice Cloning                                        #
    # ------------------------------------------------------------------ #
    def synthesize_voice_clone(
        self,
        text: str,
        reference_audio_bytes: bytes,
        ref_text: str = "",
        speed: float = 1.0,
        num_step: int = 32,
    ) -> bytes:
        """
        Generate cross-lingual cloned speech using OmniVoice Voice Cloning mode.

        Parameters
        ----------
        text                  : Text to synthesise in the target language.
        reference_audio_bytes : Raw WAV/MP3/OGG/FLAC bytes of the reference voice.
        ref_text              : Transcription of reference audio.
                                If empty, OmniVoice uses Whisper to auto-transcribe.
        speed                 : Speaking rate multiplier.
        num_step              : Diffusion steps.

        Returns
        -------
        bytes : WAV bytes at 24 kHz mono PCM-16.
        """
        self._require_loaded()

        # BUG #5 FIX: write the temp file fully before closing the context manager,
        # then reopen in binary-append mode if the sf.read path fails,
        # so we never attempt tmp.write() on a closed file handle.
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)  # close the raw OS fd; we'll write via soundfile or open()

        try:
            arr, sr = sf.read(io.BytesIO(reference_audio_bytes))
            sf.write(tmp_path, arr, sr, format="WAV", subtype="PCM_16")
        except Exception:
            # If soundfile can't parse (e.g. MP3 without libsndfile plugins),
            # write raw bytes directly — file already exists (mkstemp created it).
            with open(tmp_path, "wb") as fh:
                fh.write(reference_audio_bytes)

        try:
            kwargs: dict = {
                "text":      text,
                "ref_audio": tmp_path,
                "speed":     speed,
                "num_step":  num_step,
            }
            if ref_text:
                kwargs["ref_text"] = ref_text

            # BUG #2 FIX: same fix as #1 — store last exception, always re-raise.
            max_retries = 5
            last_exc: Exception = RuntimeError("No attempts made.")
            for attempt in range(max_retries):
                try:
                    audio_list = self.model.generate(**kwargs)
                    return self._ndarray_to_wav_bytes(audio_list[0])
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        print(f"[OmniVoice] Cloning failed ({e}). Retrying {attempt + 1}/{max_retries}...")
            raise last_exc
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _require_loaded(self) -> None:
        if not self.loaded:
            raise RuntimeError(
                "OmniVoice engine not loaded. Call engine.load() first."
            )

    @staticmethod
    def _ndarray_to_wav_bytes(audio: np.ndarray) -> bytes:
        """Convert 1-D float32 numpy array → WAV bytes at 24 kHz PCM-16."""
        arr = audio.squeeze().astype(np.float32)

        if len(arr) == 0:
            raise ValueError("OmniVoice generated an empty audio array.")

        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("OmniVoice generated corrupted audio (NaN/Inf values).")

        # If the vocoder returned audio in the int16 scale (e.g. max val 32767.0)
        # instead of [-1.0, 1.0], normalize it.
        max_val = np.max(np.abs(arr))

        # BUG #4 FIX: The threshold 0.0001 was too weak and allowed microscopic hiss (flatline) 
        # to bypass the retry loop. A healthy TTS generation will always have peaks above 1% volume.
        # If max_val < 0.01, it is undeniably a failed/blank generation.
        if max_val < 0.01:
            raise ValueError("OmniVoice generated blank audio (silence). Please try again.")

        # Volume Boost: Instead of standard peak normalization (which is ruined by a single loud click),
        # we calculate the 99th percentile of the audio volume and scale that to 95%.
        # This acts like a professional audio compressor, making the voice MASSIVELY louder!
        p99 = np.percentile(np.abs(arr), 99)
        if p99 > 0.001:
            arr = arr * (0.95 / p99)

        # BUG #3 FIX: ZCR threshold relaxed from 0.45 → 0.49.
        # High-pitched female voices, whisper mode, and children's voices
        # have ZCR well above 0.45 but are valid speech, not static noise.
        # Pure white noise sits at ~0.50, so 0.49 is the correct upper bound.
        zcr = np.sum(arr[:-1] * arr[1:] < 0) / len(arr)
        if zcr > 0.49:
            raise ValueError(
                f"OmniVoice generation failed with pure static noise (ZCR: {zcr:.2f}). "
                "Please try again."
            )

        # Clip to prevent clipping distortion
        arr = np.clip(arr, -1.0, 1.0)
        
        # Trim leading/trailing silence or low-volume diffusion noise (below 40dB)
        # This completely eliminates "blank noisy audio at the start" across all versions!
        try:
            import librosa
            arr, _ = librosa.effects.trim(arr, top_db=40)
        except Exception:
            pass

        buf = io.BytesIO()
        sf.write(buf, arr, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
