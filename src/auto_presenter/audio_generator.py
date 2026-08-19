import httpx
import structlog
from pathlib import Path
import os

logger = structlog.get_logger(__name__)

class VoiceXClient:
    def __init__(self):
        # We point this to the local FastAPI microservice you just copied
        self.api_url = os.getenv("VOICEX_API_URL", "http://localhost:8000")
        self.endpoint = f"{self.api_url.rstrip('/')}/api/voice-design"

    async def generate_audio(self, text: str, instruct: str, emotion: str, output_path: Path) -> Path:
        """Calls the VoiceX local microservice to generate audio from the refined script."""
        logger.info("calling_voicex_api", url=self.endpoint, emotion=emotion)
        
        payload = {
            "text": text,
            "instruct": instruct,
            "emotion": emotion,
            "language": "en" 
        }
        
        # Give a large timeout for the first run since OmniVoice model takes time to load into VRAM
        timeout = httpx.Timeout(120.0) 
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(self.endpoint, json=payload)
                response.raise_for_status()
                
                # Write the returned audio byte stream to the .wav file
                with open(output_path, "wb") as f:
                    f.write(response.content)
                    
                logger.info("voicex_audio_saved", path=str(output_path))
                return output_path
            except Exception as e:
                logger.error("voicex_api_failed", error=str(e))
                raise e
