from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import subprocess
import os
import shutil
import uuid
import glob
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SadTalker Microservice")

# Directories for temp processing
TEMP_DIR = "/kaggle/working/temp_sadtalker"
os.makedirs(TEMP_DIR, exist_ok=True)

SADTALKER_DIR = "/kaggle/working/SadTalker"

@app.post("/api/generate-video")
async def generate_video(
    source_image: UploadFile = File(...),
    driven_audio: UploadFile = File(...)
):
    req_id = str(uuid.uuid4())
    logger.info(f"received_video_request req_id={req_id} image={source_image.filename} audio={driven_audio.filename}")
    
    # Generate unique paths for this request
    image_ext = os.path.splitext(source_image.filename)[1] or ".png"
    audio_ext = os.path.splitext(driven_audio.filename)[1] or ".wav"
    
    image_path = os.path.join(TEMP_DIR, f"{req_id}_image{image_ext}")
    audio_path = os.path.join(TEMP_DIR, f"{req_id}_audio{audio_ext}")
    output_video_path = os.path.join(TEMP_DIR, f"{req_id}_output.mp4")
    
    try:
        # 1. Save uploaded files to disk
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(source_image.file, buffer)
            
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(driven_audio.file, buffer)
            
        logger.info(f"files_saved image_path={image_path} audio_path={audio_path}")
        
        # Get list of existing videos before inference
        results_dir = os.path.join(SADTALKER_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)
        before_videos = set(glob.glob(os.path.join(results_dir, "*.mp4")))
        
        # 2. Construct the standard inference command
        cmd = [
            "python", "inference.py",
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", "./results",
            "--preprocess", "full",
            "--batch_size", "2"
        ]
        
        logger.info(f"running_sadtalker cmd={' '.join(cmd)}")
        
        # Run inference in the SadTalker directory
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=SADTALKER_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"sadtalker_failed stdout={stdout.decode()} stderr={stderr.decode()}")
            raise HTTPException(status_code=500, detail=f"SadTalker inference failed: {stderr.decode()}")
            
        # Find the newly created video
        after_videos = set(glob.glob(os.path.join(results_dir, "*.mp4")))
        new_videos = list(after_videos - before_videos)
        
        if not new_videos:
            raise HTTPException(status_code=500, detail="SadTalker succeeded but no new .mp4 was found in results!")
            
        generated_video_path = new_videos[0]
        
        # Move it to our clean output path
        shutil.move(generated_video_path, output_video_path)
        logger.info(f"video_generated_successfully path={output_video_path}")
        
        # 3. Return the generated video
        return FileResponse(
            path=output_video_path, 
            media_type="video/mp4", 
            filename="avatar_output.mp4"
        )
        
    except Exception as e:
        logger.exception("video_generation_error")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# KAGGLE EXECUTION LOGIC (Starts the API in background + Cloudflare Tunnel)
# ==============================================================================
if __name__ == "__main__":
    import threading
    import uvicorn
    import nest_asyncio

    # Allow asyncio to run in Jupyter Notebooks
    nest_asyncio.apply()

    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

    # Start FastAPI in a background thread so the cell isn't blocked
    thread = threading.Thread(target=run_server)
    thread.daemon = True
    thread.start()

    print("✅ FastAPI Server starting in background on port 8000...")
    print("🚀 Starting Cloudflare Tunnel... Look for the '.trycloudflare.com' URL below!")
    
    # Run Cloudflared tunnel
    os.system("cloudflared tunnel --url http://127.0.0.1:8000")
