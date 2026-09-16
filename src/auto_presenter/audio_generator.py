import httpx
import structlog
from pathlib import Path
import os
import asyncio

logger = structlog.get_logger(__name__)

class VoiceXClient:
    def __init__(self):
        # We point this to the local FastAPI microservice you just copied
        self.api_url = os.getenv("VOICEX_API_URL", "http://localhost:8000")
        self.endpoint = f"{self.api_url.rstrip('/')}/api/raw-voice"
        self._lock = asyncio.Lock()

    async def generate_audio(self, text: str, instruct: str, output_path: Path) -> Path:
        """Calls the VoiceX local microservice using raw-voice to bypass noise/popping while enabling expressions."""
        async with self._lock:
            logger.info("calling_voicex_api", url=self.endpoint, instruct=instruct)
            
            payload = {
                "text": text,
                "instruct": instruct,
                "speed": 1.0
            }
            
            # We are calling a local service that runs an extremely heavy AI model (OmniVoice).
            # Depending on GPU power, a 45-second script can easily take >10 minutes to render.
            # We set timeout to None (infinite) so it never crashes just because the GPU is slow.
            timeout = httpx.Timeout(None) 
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    # Use data=payload to send as application/x-www-form-urlencoded (FastAPI Form)
                    response = await client.post(self.endpoint, data=payload)
                    response.raise_for_status()
                    
                    # Write the returned audio byte stream to the .wav file
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                        
                    logger.info("voicex_audio_saved", path=str(output_path))
                    return output_path
                except Exception as e:
                    logger.error("voicex_api_failed", error=repr(e))
                    raise e
