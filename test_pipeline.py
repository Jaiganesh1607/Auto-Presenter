import asyncio
import os
import structlog
import argparse
from pathlib import Path
from src.content_engine.llm.client import LLMClient
from src.content_engine.llm.providers import get_provider
from src.auto_presenter.prompt_generator import PromptGenerator
from src.auto_presenter.image_generator import ImageGeneratorClient
from src.auto_presenter.pipeline import PipelineOrchestrator
from src.auto_presenter.presentation_engine import PresentationEngine
from src.auto_presenter.script_generator import ScriptGenerator
from src.auto_presenter.audio_generator import VoiceXClient
from src.auto_presenter.video_generator import SadTalkerClient
from src.auto_presenter.pptx_composer import PPTXComposerClient
from src.auto_presenter.video_composer import VideoComposerClient

logger = structlog.get_logger(__name__)

async def main(topic: str, num_slides: int, voice_gender: str):
    # Load configuration
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if user set the Cloudflare URL
    if not os.environ.get("ZIMAGE_API_URL"):
        logger.error("missing_env_var", message="Please set ZIMAGE_API_URL in your .env file to your Cloudflare tunnel link (e.g., https://your-tunnel.trycloudflare.com).")
        return

    # 1. Initialize Engines
    provider = get_provider("nvidia")
    llm_client = LLMClient(provider=provider)
    presentation_engine = PresentationEngine(llm_client)
    prompt_engine = PromptGenerator(llm_client)
    script_engine = ScriptGenerator(llm_client)
    image_client = ImageGeneratorClient()
    audio_client = VoiceXClient()
    video_client = SadTalkerClient()
    
    output_dir = Path("./output_images")
    orchestrator = PipelineOrchestrator(
        prompt_engine=prompt_engine,
        script_engine=script_engine,
        image_client=image_client,
        audio_client=audio_client,
        video_client=video_client,
        output_dir=output_dir
    )
    
    model_name = "poolside/laguna-xs-2.1"
    
    # 2. Get the real outline via the Presentation Engine (Phase 1.5)
    logger.info("generating_presentation_outline", topic=topic)
    presentation = await presentation_engine.generate_outline(topic=topic, model_name=model_name, num_slides=num_slides)
    
    # Save the intermediate presentation outline to output/
    meta_dir = Path("./output")
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "presentation_outline.json", "w") as f:
        f.write(presentation.model_dump_json(indent=4))
    
    # 3. Run the concurrent pipeline using the batched Prompt Engine
    slide_results, final_output_dir = await orchestrator.process_presentation(
        presentation=presentation,
        voice_gender=voice_gender,
        voice_age=30,
        model_name=model_name
    )
    
    # Generate PPTX
    pptx_client = PPTXComposerClient()
    pptx_client.build_presentation(presentation, slide_results, final_output_dir)
    
    # Generate Master MP4 Video
    video_composer = VideoComposerClient()
    video_composer.build_master_video(slide_assets=slide_results, output_dir=final_output_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Presenter Pipeline CLI")
    parser.add_argument("--topic", type=str, default="Docker and Containerization", help="The topic of the presentation")
    parser.add_argument("--slides", type=int, default=3, help="Number of slides to generate")
    parser.add_argument("--gender", type=str, choices=["male", "female"], default="female", help="Voice and Avatar gender")
    
    args = parser.parse_args()
    
    # Run the main pipeline
    asyncio.run(main(topic=args.topic, num_slides=args.slides, voice_gender=args.gender))
