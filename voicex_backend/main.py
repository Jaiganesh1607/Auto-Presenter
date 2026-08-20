"""
main.py
FastAPI application — OmniVoice Studio backend.

Endpoints
---------
GET  /api/health            → server + model status
GET  /api/languages         → grouped language catalogue
GET  /api/emotions          → available emotion labels
GET  /api/expressions       → available expression options
POST /api/voice-design      → generate speech (voice design)
POST /api/voice-clone       → generate speech (cross-lingual clone)
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from omnivoice_engine import OmniVoiceEngine
from emotion_engine import EmotionEngine
from expression_engine import ExpressionEngine
from language_router import LanguageRouter
from voice_design import VoiceDesignService
from voice_clone import VoiceCloneService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("omnivoice_studio")

# ---------------------------------------------------------------------------
# Lifespan — load model once at startup, share across all requests
# ---------------------------------------------------------------------------
_voice_design_service = None
_voice_clone_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _voice_design_service, _voice_clone_service
    logger.info("Starting OmniVoice Studio backend …")

    engine = OmniVoiceEngine.get_instance()
    engine.load()

    # Services share the pre-loaded engine instance
    _voice_design_service = VoiceDesignService(engine)
    _voice_clone_service  = VoiceCloneService(engine)

    logger.info("Backend ready ✅")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OmniVoice Studio API",
    description="Multilingual TTS powered by OmniVoice — 100+ languages",
    version="2.0.0",
    lifespan=lifespan,
)



# Wide-open CORS for Colab/ngrok usage; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wav_response(wav_bytes: bytes) -> Response:
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
_AUDIO_CONTENT_TYPES = {
    "audio/wav", "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/flac", "audio/mp4",
    "audio/x-m4a", "audio/webm",
    "application/octet-stream",  # some browsers send this for all uploads
}


def _validate_reference_audio(upload: UploadFile) -> None:
    """Raise HTTPException if the uploaded file is not a recognised audio type."""
    content_type = (upload.content_type or "").lower()
    filename_ext = os.path.splitext(upload.filename or "")[1].lower()

    if content_type not in _AUDIO_CONTENT_TYPES and filename_ext not in _AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file: content_type='{content_type}', "
                f"extension='{filename_ext}'. "
                "Please upload WAV, MP3, OGG, FLAC, or M4A."
            ),
        )


def _validate_version(version: int) -> None:
    if version not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="version must be 1, 2, or 3.")


# ---------------------------------------------------------------------------
# Routes — Catalogue
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    engine = OmniVoiceEngine.get_instance()
    return {
        "status":       "ok",
        "model_loaded": engine.loaded,
        "device":       engine.device,
        "stub_mode":    not engine.loaded or getattr(engine, "stub_mode", False),
    }


@app.get("/api/languages")
async def get_languages():
    return LanguageRouter.get_catalogue()


@app.get("/api/emotions")
async def get_emotions():
    return EmotionEngine.available_emotions()


@app.get("/api/expressions")
async def get_expressions():
    return ExpressionEngine.available_expressions()


# ---------------------------------------------------------------------------
# Routes — Voice Design
# ---------------------------------------------------------------------------
@app.post("/api/voice-design")
async def voice_design(
    text:       str = Form(...),
    language:   str = Form("en-US"),
    gender:     str = Form("neutral"),
    age:        int = Form(30),
    emotion:    str = Form("neutral"),
    expression: str = Form("none"),
    version:    int = Form(2),
):
    """
    Generate speech using OmniVoice Voice Design.

    - **version 1**: text + language + gender + age + emotion (no expression)
    - **version 2**: adds expression (injected silently as OmniVoice tag)
    - **version 3**: user writes emotion/expression tags directly in text
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    if not (5 <= age <= 100):
        raise HTTPException(status_code=400, detail="age must be between 5 and 100.")
    if gender.lower() not in ("male", "female", "neutral"):
        raise HTTPException(status_code=400, detail="gender must be male, female, or neutral.")
    if emotion.lower() not in ("neutral", "happy", "sad", "angry", "excited", "calm", "whisper", "fearful", "surprised", "disgusted"):
        raise HTTPException(status_code=400, detail=f"invalid emotion: {emotion}.")
    _validate_version(version)

    # Enforce spec: V1 never uses expressions
    effective_expression = "none" if version == 1 else expression

    logger.info(
        "voice-design | v%d lang=%s gender=%s age=%d emotion=%s expression=%s",
        version, language, gender, age, emotion, effective_expression,
    )

    try:
        wav = _voice_design_service.generate(
            text=text,
            language=language,
            gender=gender,
            age=age,
            emotion=emotion,
            expression=effective_expression,
            version=version,
        )
    except Exception as exc:
        logger.exception("voice-design failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _wav_response(wav)


# ---------------------------------------------------------------------------
# Routes — Raw Voice
# ---------------------------------------------------------------------------
@app.post("/api/raw-voice")
async def raw_voice(
    text: str = Form(...),
    instruct: str = Form("female, professional, clear articulation"),
    speed: float = Form(0.95)
):
    """
    Directly hits the OmniVoice model without TagParser splitting or concatenation.
    This eliminates audio popping/noise and allows fully custom instruct strings.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    
    logger.info("raw-voice | instruct=%s speed=%s", instruct, speed)
    
    engine = OmniVoiceEngine.get_instance()
    try:
        # OmniVoice handles inline tags like [laughter] natively in one pass!
        wav = engine.synthesize_voice_design(
            text=text,
            instruct=instruct,
            speed=speed,
            num_step=32 # High quality
        )
    except Exception as exc:
        logger.exception("raw-voice failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _wav_response(wav)


# ---------------------------------------------------------------------------
# Routes — Voice Clone
# ---------------------------------------------------------------------------
@app.post("/api/voice-clone")
async def voice_clone(
    text:            str        = Form(...),
    target_language: str        = Form("en-US"),
    reference_audio: UploadFile = File(...),
    gender:          str        = Form("neutral"),
    age:             int        = Form(30),
    emotion:         str        = Form("neutral"),
    expression:      str        = Form("none"),
    version:         int        = Form(2),
    ref_text:        str        = Form(""),
):
    """
    Generate cross-lingual cloned speech from a reference voice.

    - **version 1**: clone + emotion (no expression)
    - **version 2**: adds expression injection (silent tag)
    - **version 3**: emotion/expression tags in text are preserved
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    if not (5 <= age <= 100):
        raise HTTPException(status_code=400, detail="age must be between 5 and 100.")
    if gender.lower() not in ("male", "female", "neutral"):
        raise HTTPException(status_code=400, detail="gender must be male, female, or neutral.")
    if emotion.lower() not in ("neutral", "happy", "sad", "angry", "excited", "calm", "whisper", "fearful", "surprised", "disgusted"):
        raise HTTPException(status_code=400, detail=f"invalid emotion: {emotion}.")
    _validate_version(version)
    _validate_reference_audio(reference_audio)

    ref_bytes = await reference_audio.read()
    if not ref_bytes:
        raise HTTPException(status_code=400, detail="Reference audio file is empty.")

    # Enforce spec: V1 never uses expressions
    effective_expression = "none" if version == 1 else expression

    logger.info(
        "voice-clone | v%d lang=%s gender=%s age=%d emotion=%s expression=%s ref_size=%d bytes",
        version, target_language, gender, age, emotion, effective_expression, len(ref_bytes),
    )

    try:
        wav = _voice_clone_service.generate(
            text=text,
            target_language=target_language,
            reference_audio_bytes=ref_bytes,
            gender=gender,
            age=age,
            emotion=emotion,
            expression=effective_expression,
            version=version,
            ref_text=ref_text.strip(),  # empty = Whisper auto-transcribes
        )
    except Exception as exc:
        logger.exception("voice-clone failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _wav_response(wav)


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )
