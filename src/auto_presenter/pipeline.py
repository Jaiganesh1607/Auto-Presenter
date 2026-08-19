import asyncio
import structlog
from pathlib import Path
import json

from src.auto_presenter.schemas import PresentationOutline
from src.auto_presenter.prompt_generator import PromptGenerator
from src.auto_presenter.image_generator import ImageGeneratorClient

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
        
    async def process_image_pipeline(self, slide_num: int, final_prompt: str):
        """Swimlane A: Generates the background slide image"""
        if not final_prompt:
            logger.error("image_pipeline_skipped_no_prompt", slide_number=slide_num)
            return None
            
        output_path = self.output_dir / f"slide_{slide_num}_bg.png"
        try:
            logger.info("starting_image_generation", slide_number=slide_num)
            await self.image_client.generate_image(prompt=final_prompt, output_path=output_path)
            logger.info("image_generation_complete", slide_number=slide_num)
            return output_path
        except Exception as e:
            logger.error("image_generation_failed", slide_number=slide_num, error=str(e))
            return None

    async def process_video_pipeline(self, slide_num: int, speaker_notes: str, model_name: str, voice_instruct: str):
        """Swimlane B: Generates the script, the audio, and eventually the avatar video"""
        if not self.script_engine or not self.audio_client:
            logger.warning("video_pipeline_skipped_missing_engines", slide_number=slide_num)
            return None

        try:
            logger.info("starting_script_refinement", slide_number=slide_num)
            # 1. Refine Script & get Emotion
            script_data = await self.script_engine.generate_script(speaker_notes, model_name=model_name)
            final_script = script_data.get("script_text", speaker_notes)
            emotion = script_data.get("emotion", "Neutral")
            
            # 2. Generate Audio
            audio_output_path = self.output_dir / f"slide_{slide_num}_voice.wav"
            logger.info("starting_audio_generation", slide_number=slide_num, emotion=emotion)
            await self.audio_client.generate_audio(text=final_script, instruct=voice_instruct, emotion=emotion, output_path=audio_output_path)
            
            # 3. Generate Video (Phase 5 Placeholder)
            # if self.video_client:
            #     video_output_path = self.output_dir / f"slide_{slide_num}_avatar.mp4"
            #     await self.video_client.generate_video(audio_output_path, video_output_path)
            
            logger.info("video_pipeline_audio_complete", slide_number=slide_num)
            return audio_output_path
        except Exception as e:
            logger.error("video_pipeline_failed", slide_number=slide_num, error=str(e))
            return None

    async def process_single_slide(self, slide, final_prompt: str, model_name: str, voice_instruct: str):
        """Orchestrates both swimlanes for a single slide concurrently"""
        logger.info("starting_slide_pipelines", slide_number=slide.slide_number)
        
        # Fire both swimlanes simultaneously
        results = await asyncio.gather(
            self.process_image_pipeline(slide.slide_number, final_prompt),
            self.process_video_pipeline(slide.slide_number, slide.speaker_notes, model_name, voice_instruct)
        )
        
        image_path, video_path = results
        logger.info("completed_slide_pipelines", slide_number=slide.slide_number)
        return image_path, video_path
        
    async def process_presentation(self, presentation: PresentationOutline, voice_instruct: str, model_name: str = "gpt-4o"):
        """
        Runs the fully concurrent pipelined generation across all slides.
        """
        logger.info("starting_presentation_pipeline", total_slides=len(presentation.slides))
        
        # Step 1: Batch generate Z-Image prompts to save LLM latency
        try:
            logger.info("generating_all_prompts_batch")
            prompts_list = await self.prompt_engine.generate_prompts_batch(
                slides=presentation.slides, 
                model_name=model_name
            )
            
            meta_dir = Path("./output")
            meta_dir.mkdir(parents=True, exist_ok=True)
            with open(meta_dir / "zimage_prompts.json", "w") as f:
                json.dump(prompts_list, f, indent=4)
                
        except Exception as e:
            logger.error("prompt_batch_generation_failed", error=str(e))
            prompts_list = [None] * len(presentation.slides)

        # Step 2: Launch Independent Async Tasks for each Slide
        tasks = []
        for i, slide in enumerate(presentation.slides):
            final_prompt = prompts_list[i] if i < len(prompts_list) else None
            # Create a detached task for this slide's entire lifecycle
            task = asyncio.create_task(self.process_single_slide(slide, final_prompt, model_name, voice_instruct))
            tasks.append(task)
            
        # Step 3: Wait for all slide tasks to finish
        await asyncio.gather(*tasks)
        
        logger.info("pipeline_complete", output_dir=str(self.output_dir))
