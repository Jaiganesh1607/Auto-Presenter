import httpx
import base64
import structlog
import os
from pathlib import Path
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

class ImageRequest(BaseModel):
    prompt: str
    width: int = 1280
    height: int = 720
    num_inference_steps: int = 6
    guidance_scale: float = 0.0
    seed: int = -1

class ImageGeneratorClient:
    """Client to interface with the Kaggle/Cloudflare Z-Image Turbo API."""
    
    def __init__(self, api_url: str = None):
        # Fallback to env var if not explicitly passed
        self.api_url = api_url or os.environ.get("ZIMAGE_API_URL")
        if not self.api_url:
            raise ValueError("ZIMAGE_API_URL environment variable is not set. Please add the Cloudflare tunnel URL to your .env file.")
        
        # Ensure it doesn't end with a slash for clean concatenation
        self.api_url = self.api_url.rstrip("/")

    async def generate_image(self, prompt: str, output_path: Path) -> Path:
        """
        Sends the prompt to the Kaggle GPU server and saves the resulting base64 image to disk.
        Returns the absolute path to the saved image.
        """
        logger.info("requesting_image_generation", api_url=self.api_url, output_path=str(output_path))
        
        req_data = ImageRequest(prompt=prompt).model_dump()
        
        try:
            import asyncio
            # Increase initial timeout in case Kaggle takes a bit to start the thread
            async with httpx.AsyncClient(timeout=120.0) as client:
                # 1. Post the request to start the job
                response = await client.post(f"{self.api_url}/generate", json=req_data)
                
                if response.status_code != 200:
                    logger.error("image_generation_start_error", status=response.status_code, detail=response.text)
                    raise Exception(f"API Error {response.status_code}: {response.text}")
                
                job_id = response.json()["job_id"]
                logger.info("image_generation_job_started", job_id=job_id)
                
                # 2. Poll the status endpoint until completed
                while True:
                    status_res = await client.get(f"{self.api_url}/status/{job_id}")
                    if status_res.status_code != 200:
                        logger.warning("status_check_failed", status=status_res.status_code)
                        await asyncio.sleep(5)
                        continue
                        
                    data = status_res.json()
                    status = data.get("status")
                    
                    if status == "completed":
                        img_data = base64.b64decode(data["image_base64"])
                        
                        # Ensure the parent directories exist
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(output_path, "wb") as f:
                            f.write(img_data)
                            
                        logger.info("image_generation_success", saved_to=str(output_path), job_id=job_id)
                        return output_path
                        
                    elif status == "failed":
                        error_msg = data.get("error", "Unknown Kaggle Error")
                        logger.error("image_generation_kaggle_error", error=error_msg, job_id=job_id)
                        raise Exception(f"Kaggle GPU Error: {error_msg}")
                        
                    else:
                        # status is "processing", wait 5 seconds and poll again
                        logger.debug("job_processing", job_id=job_id)
                        await asyncio.sleep(5)
                        
        except Exception as e:
            logger.error("image_generation_failed", error=repr(e))
            raise e
