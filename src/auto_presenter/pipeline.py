import asyncio
import structlog
from pathlib import Path
import json

from src.auto_presenter.schemas import PresentationOutline
from src.auto_presenter.prompt_generator import PromptGenerator
from src.auto_presenter.image_generator import ImageGeneratorClient
from src.auto_presenter.video_generator import SadTalkerClient

logger = structlog.get_logger(__name__)

class PipelineOrchestrator:
    def __init__(self, 
                 prompt_engine: PromptGenerator, 
                 image_client: ImageGeneratorClient, 
                 output_dir: Path,
                 script_engine = None,
                 audio_client = None,
                 video_client = None):
        """
        Initializes the fully concurrent pipeline orchestrator.
        """
        self.prompt_engine = prompt_engine
        self.image_client = image_client
        self.script_engine = script_engine
        self.audio_client = audio_client
        self.video_client = video_client
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def process_image_pipeline(self, slide_num: int, final_prompt: str, slide_dir: Path):
        """Swimlane A: Generates the background slide image"""
        if not final_prompt:
            logger.error("image_pipeline_skipped_no_prompt", slide_number=slide_num)
            return None
            
        output_path = slide_dir / f"slide_{slide_num}_bg.png"
        try:
            logger.info("starting_image_generation", slide_number=slide_num)
            await self.image_client.generate_image(prompt=final_prompt, output_path=output_path)
            logger.info("image_generation_complete", slide_number=slide_num)
            return output_path
        except Exception as e:
            logger.error("image_generation_failed", slide_number=slide_num, error=str(e))
            return None

    async def process_video_pipeline(self, slide_num: int, script_data: dict, voice_gender: str, voice_age: int, slide_dir: Path):
        """Swimlane B: Generates the audio and eventually the avatar video from the pre-generated master script"""
        if not self.audio_client or not self.video_client:
            logger.warning("video_pipeline_skipped_missing_engines", slide_number=slide_num)
            return None

        try:
            # 1. Extract script data
            final_script = script_data.get("script_text", "")
            instruct = script_data.get("instruct", f"{voice_gender}, professional")
            
            # 2. Generate Audio
            audio_output_path = slide_dir / f"slide_{slide_num}_voice.wav"
            logger.info("starting_audio_generation", slide_number=slide_num, instruct=instruct)
            await self.audio_client.generate_audio(
                text=final_script, 
                instruct=instruct, 
                output_path=audio_output_path
            )
            
            # 3. Generate Video
            video_output_path = slide_dir / f"slide_{slide_num}_avatar.mp4"
            
            # Select the correct avatar image based on gender
            avatar_dir = Path("avatar_images")
            avatar_image_path = avatar_dir / f"{voice_gender.lower()}.png"
            if not avatar_image_path.exists():
                avatar_image_path = avatar_dir / f"{voice_gender.lower()}.jpg"
                
            if not avatar_image_path.exists():
                logger.warning("avatar_image_missing", expected=str(avatar_image_path))
                return audio_output_path # Fallback to just audio
                
            logger.info("starting_video_generation", slide_number=slide_num, image=avatar_image_path.name)
            await self.video_client.generate_video(audio_output_path, avatar_image_path, video_output_path)
            
            logger.info("video_pipeline_complete", slide_number=slide_num)
            return video_output_path
        except Exception as e:
            logger.error("video_pipeline_failed", slide_number=slide_num, error=repr(e))
            return None

    async def process_single_slide(self, slide, script_data: dict, model_name: str, voice_gender: str, voice_age: int, slide_dir: Path):
        """Orchestrates both swimlanes for a single slide concurrently"""
        logger.info("starting_slide_pipelines", slide_number=slide.slide_number)
        
        # 1. Generate prompt specifically for this slide to maximize creativity
        try:
            final_prompt = await self.prompt_engine.generate_prompt_for_slide(slide, model_name)
            # Save the prompt for debugging
            with open(slide_dir / f"slide_{slide.slide_number}_prompt.txt", "w") as f:
                f.write(final_prompt)
        except Exception as e:
            logger.error("prompt_generation_failed", slide_number=slide.slide_number, error=str(e))
            final_prompt = None
            
        # Fire both swimlanes simultaneously
        results = await asyncio.gather(
            self.process_image_pipeline(slide.slide_number, final_prompt, slide_dir),
            self.process_video_pipeline(slide.slide_number, script_data, voice_gender, voice_age, slide_dir)
        )
        
        image_path, video_path = results
        logger.info("completed_slide_pipelines", slide_number=slide.slide_number)
        
        # We also need to pass the avatar image back so PPTX can use it as a poster frame
        avatar_dir = Path("avatar_images")
        avatar_image_path = avatar_dir / f"{voice_gender}.png"
        
        return slide.slide_number, image_path, video_path, avatar_image_path
        
    async def process_presentation(self, presentation: PresentationOutline, voice_gender: str, voice_age: int, model_name: str = "gpt-4o"):
        """
        Runs the fully concurrent pipelined generation across all slides.
        """
        # Create unique folder for this presentation run
        import datetime
        import re
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r'[^A-Za-z0-9]+', '_', presentation.topic).strip('_').lower()
        presentation_dir = self.output_dir / f"{timestamp}_{slug}"
        presentation_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("starting_presentation_pipeline", total_slides=len(presentation.slides), out_dir=str(presentation_dir))
        
        # Step 0: Generate the cohesive master script for all slides
        try:
            logger.info("generating_master_script_batch")
            master_scripts = await self.script_engine.generate_master_script(
                presentation=presentation,
                voice_gender=voice_gender,
                voice_age=voice_age,
                model_name=model_name
            )
            with open(presentation_dir / "master_script.json", "w") as f:
                json.dump(master_scripts, f, indent=4)
        except Exception as e:
            logger.error("master_script_generation_failed", error=str(e))
            master_scripts = {}
        
        # Step 1: Launch Independent Async Tasks for each Slide
        tasks = []
        for slide in presentation.slides:
            script_data = master_scripts.get(slide.slide_number, {})
            # Create a detached task for this slide's entire lifecycle
            task = asyncio.create_task(
                self.process_single_slide(slide, script_data, model_name, voice_gender, voice_age, presentation_dir)
            )
            tasks.append(task)
            
        # Step 2: Wait for all slide tasks to finish
        slide_results = await asyncio.gather(*tasks)
        
        logger.info("pipeline_complete", output_dir=str(presentation_dir))
        return slide_results, presentation_dir
