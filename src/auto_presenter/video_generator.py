import os
from pathlib import Path
import httpx
import structlog

logger = structlog.get_logger(__name__)

class SadTalkerClient:
    """Client for interfacing with the Kaggle-hosted SadTalker microservice."""
    def __init__(self):
        # Point this to the Cloudflare tunnel URL that exposes the Kaggle FastAPI app
        self.api_url = os.getenv("SADTALKER_API_URL", "")
        self.endpoint = f"{self.api_url.rstrip('/')}/api/generate-video"

    async def generate_video(self, audio_path: Path, avatar_image_path: Path, output_path: Path) -> Path | None:
        """
        Sends the generated audio and the avatar image to Kaggle to generate the lip-synced video.
        """
        if not self.api_url:
            logger.error("sadtalker_api_url_missing", message="SADTALKER_API_URL is not set in .env")
            return None
            
        logger.info("calling_sadtalker_api", url=self.endpoint, audio=audio_path.name, image=avatar_image_path.name)
        
        # We need a large timeout as SadTalker can take 30-60 seconds to render video
        timeout = httpx.Timeout(180.0, connect=60.0)
        
        try:
            with open(audio_path, "rb") as audio_file, open(avatar_image_path, "rb") as image_file:
                files = {
                    "driven_audio": (audio_path.name, audio_file, "audio/wav"),
                    "source_image": (avatar_image_path.name, image_file, "image/png")
                }
                
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(self.endpoint, files=files)
                    
                    if response.status_code != 200:
                        logger.error("sadtalker_api_failed", status=response.status_code, text=response.text)
                        return None
                        
                    # Write the returned .mp4 stream to disk
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                        
            logger.info("sadtalker_video_saved", path=str(output_path))
            return output_path
            
        except httpx.ReadTimeout:
            logger.error("sadtalker_api_timeout", message="The Kaggle API took too long to respond.")
            return None
        except Exception as e:
            logger.exception("sadtalker_client_error", error=str(e))
            return None
