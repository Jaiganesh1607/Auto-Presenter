# ==============================================================================
# COPY AND PASTE THIS ENTIRE SCRIPT INTO A SINGLE KAGGLE / COLAB CELL
# ==============================================================================

import os
import subprocess
import threading
import time
import base64
import io
import torch
import gc
import numpy as np
import queue
from PIL import Image

# 1. Install dependencies
print("Installing dependencies...")
subprocess.run("pip install -q git+https://github.com/huggingface/diffusers huggingface_hub fastapi uvicorn nest-asyncio pydantic", shell=True)

# 2. Download Cloudflared
if not os.path.exists("cloudflared-linux-amd64"):
    print("Downloading Cloudflared...")
    subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", shell=True)
    subprocess.run("chmod +x cloudflared-linux-amd64", shell=True)

# We must import these AFTER installing them
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import ZImagePipeline
from contextlib import asynccontextmanager

# Request/Response Schemas
class ImageRequest(BaseModel):
    prompt: str
    width: int = 1280
    height: int = 720
    num_inference_steps: int = 6
    guidance_scale: float = 0.0
    seed: int = -1

class JobResponse(BaseModel):
    job_id: str

# Global state
pipe = None
JOBS = {}

print("Loading Z-Image Turbo Model into GPU... This may take a few minutes...")
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.enable_sequential_cpu_offload()
print("Model loaded successfully!")

# 3. Initialize FastAPI and Queue Worker
app = FastAPI(title="Z-Image Turbo API")
job_queue = queue.Queue()

def queue_worker():
    while True:
        job_id, req = job_queue.get()
        _generate_worker(job_id, req)
        job_queue.task_done()

# Start the single background worker that processes jobs one by one
threading.Thread(target=queue_worker, daemon=True).start()

@app.post("/generate", response_model=JobResponse)
async def generate_image(req: ImageRequest):
    import uuid
    global pipe
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "processing", "image_base64": None, "error": None}
    
    # Put the job in the queue to be processed one at a time
    job_queue.put((job_id, req))
    
    return JobResponse(job_id=job_id)

def _generate_worker(job_id: str, req: ImageRequest):
    global pipe
    print(f"[Job {job_id}] Started generation for prompt: {req.prompt[:50]}...")
    try:
        # Handle Seed
        seed = req.seed if req.seed != -1 else np.random.randint(0, 1000000)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Generate Image (Sequential execution is very slow, which is why we poll)
        image = pipe(
            req.prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            generator=generator
        ).images[0]

        # Convert to Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()

        JOBS[job_id]["image_base64"] = img_str
        JOBS[job_id]["status"] = "completed"
        print(f"[Job {job_id}] Generation complete!")
        
    except Exception as e:
        print(f"[Job {job_id}] Error during generation: {str(e)}")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    # Return everything except the massive base64 string if it's not done
    return JOBS[job_id]

# 4. Start Cloudflare Tunnel in the background
def start_tunnel():
    print("Starting Cloudflare Tunnel...")
    process = subprocess.Popen(
        ['./cloudflared-linux-amd64', 'tunnel', '--url', 'http://127.0.0.1:8000'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in process.stdout:
        # Cloudflared prints the public URL to standard output/error
        if "trycloudflare.com" in line:
            url = [word for word in line.split() if "trycloudflare.com" in word][0]
            print("\n" + "="*70)
            print(f"🚀 YOUR API ENDPOINT IS READY!")
            print(f"🔗 COPY THIS URL TO YOUR .env FILE: {url}")
            print("="*70 + "\n")

threading.Thread(target=start_tunnel, daemon=True).start()

# 5. Run FastAPI Server in a Background Thread
def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)

print("Starting FastAPI server on port 8000...")
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# 6. Keep the cell running infinitely
print("Server is running! Waiting for requests...")
while True:
    time.sleep(1)
