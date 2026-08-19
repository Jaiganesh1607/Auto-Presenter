# Auto-Presenter Pipeline

An autonomous, highly concurrent pipeline for generating stunning, fully-scripted PowerPoint presentations using Local/Cloud LLMs and GPU-accelerated Image Generation.

## Current Architecture Status
This repository currently contains the completed code for **Phases 1 through 3** of the pipeline.

### 1. The Presentation Engine (Phase 1 & 1.5)
The brain of the operation. It is responsible for gathering knowledge and structuring the presentation.
- **Context Fallback (`src/content_engine`)**: If a user provides a topic but no context, the engine falls back to a parallel Wikipedia scraping pipeline to gather factual information, eliminating LLM hallucination.
- **Pydantic Structuring (`presentation_engine.py`)**: Uses NVIDIA/OpenRouter LLMs to perfectly parse the context into a rigid `PresentationOutline` schema containing Slide Titles, Bullets (max 4), Conversational Speaker Notes, and raw Visual Concepts.

### 2. The Prompt Generation Engine (Phase 2)
Image models struggle with rendering text and adhering to specific aesthetics. This engine acts as a translator between the raw slide outline and the image model.
- **Batch Processing (`prompt_generator.py`)**: Instead of calling the LLM for every single slide, it packages all slides into a single LLM call to save tokens and minimize API usage.
- **Aesthetic Enforcement**: It uses few-shot prompting to force the LLM to output prompts that adhere strictly to a "highly detailed technical blueprint aesthetic" on a "clean cream background with NO grid lines".
- **Text Limitation**: It forces the LLM to condense 4 bullet points into a 2-word phrase to ensure the image model renders the text flawlessly without typos.

### 3. Pipelined Image Generation (Phase 3)
A custom Client/Server architecture built to interface with external free-tier GPUs (Kaggle/Colab).
- **The Kaggle Server (`kaggle_zimage_server.py`)**: A standalone script that runs on Kaggle/Colab. It hosts the `Z-Image-Turbo` diffusion model in a FastAPI server and exposes it publicly via a Cloudflare Tunnel.
- **The Asynchronous Pipeline (`pipeline.py`)**: A high-performance Producer-Consumer architecture using `asyncio.Queue`. 
    - **Producer**: Generates all the perfect Z-Image prompts in a batch and feeds them into the queue.
    - **Consumer**: Pops prompts from the queue and sends them to the Kaggle server. This ensures that while the LLM is thinking, the remote GPU is constantly rendering, drastically reducing total generation time.

---

## How to Run the Current Pipeline

1. **Spin up the GPU Server**
   - Open Kaggle or Google Colab.
   - Copy the contents of `kaggle_zimage_server.py` into a cell and run it.
   - It will output a Cloudflare URL (e.g. `https://your-tunnel.trycloudflare.com`).
   - Paste that URL into your local `.env` file under `ZIMAGE_API_URL=...`

2. **Run the End-to-End Test**
   - Activate the virtual environment: `env\Scripts\activate`
   - Run the testing pipeline: `python test_pipeline.py`
   - The engine will generate the slides, batch the prompts, and sequentially download the generated `.png` files into the `output_images/` directory!
