import os
import asyncio
import structlog
from dotenv import load_dotenv

# We must adjust sys.path to find 'src' if we run from the root directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from src.content_engine.llm.providers import get_provider
from src.content_engine.llm.client import LLMClient
from src.auto_presenter.presentation_engine import PresentationEngine

def setup_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer()
        ]
    )

async def main():
    setup_logging()
    
    # 1. Load the .env file explicitly
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
    
    # 2. Get the OpenRouter provider 
    provider = get_provider("nvidia")
    
    api_key = os.environ.get(provider.api_key_env_var, "")
    masked_key = f"{api_key[:5]}...{api_key[-5:]}" if api_key else "NOT_FOUND"
    print(f"\nUsing Provider: {provider.name} | Model: {provider.model_name}")
    print(f"Loaded {provider.api_key_env_var}: {masked_key}")
    
    # 3. Initialize the model-agnostic LLM Client
    llm_client = LLMClient(provider=provider)
    
    # 4. Initialize the Presentation Engine
    engine = PresentationEngine(llm_client=llm_client)
    
    # 5. Run the engine (Zero-Context Fallback Mode)
    topic = "Docker and Containerization"
    print(f"\n[STAGE 1] Starting Presentation Engine for Topic: '{topic}'...")
    print("No context provided, so it is automatically searching and building the context first (this takes a minute)...\n")
    
    outline = await engine.generate_outline(topic=topic, model_name=provider.model_name)
    
    # 6. Print the results!
    print("\n[STAGE 2] === PRESENTATION GENERATED SUCCESSFULLY ===")
    print(f"Topic: {outline.topic}")
    print(f"Total Slides Generated: {len(outline.slides)}\n")
    
    print("[STAGE 3] === DEBUGGING OUTPUT FOR FIRST 2 SLIDES ===")
    for slide in outline.slides[:2]:
        print(f"Slide {slide.slide_number}: {slide.title}")
        print("Visual Concept:", slide.visual_concept)
        print("Bullets:")
        for b in slide.bullet_points:
            print(f"  - {b}")
        print("Speaker Notes:\n", slide.speaker_notes)
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
