import asyncio
import structlog
from pathlib import Path
from typing import List
from src.auto_presenter.schemas import PresentationOutline
from src.auto_presenter.prompt_generator import PromptGenerator
from src.auto_presenter.image_generator import ImageGeneratorClient

logger = structlog.get_logger(__name__)

class PipelineOrchestrator:
    def __init__(self, prompt_engine: PromptGenerator, image_client: ImageGeneratorClient, output_dir: Path):
        self.prompt_engine = prompt_engine
        self.image_client = image_client
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def process_presentation(self, presentation: PresentationOutline, model_name: str = "gpt-4o"):
        """
        Runs the pipelined generation.
        Producer: Generates highly detailed Z-Image prompts using the LLM.
        Consumer: Feeds prompts into the Kaggle API to render images sequentially.
        """
        logger.info("starting_concurrent_pipeline", total_slides=len(presentation.slides))
        
        # We use an asyncio Queue to pass prompts from the Producer (LLM) to the Consumer (Image API)
        queue = asyncio.Queue()
        
        # Producer Task: Generates Prompts
        async def prompt_producer():
            try:
                logger.info("generating_all_prompts_batch")
                
                # Make a single LLM call to get all prompts for all slides
                prompts_list = await self.prompt_engine.generate_prompts_batch(
                    slides=presentation.slides, 
                    model_name=model_name
                )
                
                # Save intermediate prompts
                import json
                meta_dir = Path("./output")
                meta_dir.mkdir(parents=True, exist_ok=True)
                with open(meta_dir / "zimage_prompts.json", "w") as f:
                    json.dump(prompts_list, f, indent=4)
                
                # Put the generated prompts into the queue for the consumer
                for i, slide in enumerate(presentation.slides):
                    if i < len(prompts_list):
                        final_prompt = prompts_list[i]
                        await queue.put((slide.slide_number, final_prompt))
                        logger.info("prompt_queued", slide_number=slide.slide_number)
                    else:
                        logger.error("prompt_missing_from_batch", slide_number=slide.slide_number)
                        await queue.put((slide.slide_number, None))
                        
            except Exception as e:
                logger.error("prompt_batch_generation_failed", error=str(e))
                # If the batch fails, send None for all slides so the consumer doesn't hang
                for slide in presentation.slides:
                    await queue.put((slide.slide_number, None))
            
            # Send sentinel value to tell the consumer we are done producing
            await queue.put(None)

        # Consumer Task: Generates Images
        async def image_consumer():
            while True:
                item = await queue.get()
                if item is None:
                    # Sentinel received, shut down consumer
                    queue.task_done()
                    break
                    
                slide_num, final_prompt = item
                
                if final_prompt is None:
                    # Producer failed for this slide, skip it
                    queue.task_done()
                    continue
                
                output_path = self.output_dir / f"slide_{slide_num}.png"
                
                try:
                    logger.info("generating_image_from_queue", slide_number=slide_num)
                    await self.image_client.generate_image(prompt=final_prompt, output_path=output_path)
                except Exception as e:
                    logger.error("image_generation_failed", slide_number=slide_num, error=str(e))
                
                queue.task_done()
                
        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(prompt_producer())
        consumer_task = asyncio.create_task(image_consumer())
        
        # Wait for producer to finish creating prompts
        await producer_task
        # Wait for consumer to finish processing all prompts in the queue
        await queue.join()
        # Wait for the consumer task itself to exit
        await consumer_task
        
        logger.info("pipeline_complete", output_dir=str(self.output_dir))
