import asyncio
import os
import json
from src.content_engine.llm.client import LLMClient
from src.auto_presenter.prompt_generator import PromptGenerator
from dataclasses import dataclass

@dataclass
class SlideOutline:
    title: str
    visual_concept: str
    bullet_points: list

from src.content_engine.llm.providers import get_provider

async def main():
    provider = get_provider("nvidia")
    llm = LLMClient(provider=provider)
    generator = PromptGenerator(llm)
    
    slides = [
        SlideOutline(
            title="What is Docker?",
            visual_concept="A massive cargo ship carrying hundreds of isolated shipping containers across the ocean.",
            bullet_points=[
                "Containers isolate applications.",
                "Run anywhere perfectly.",
                "Eliminates dependency conflicts."
            ]
        ),
        SlideOutline(
            title="The CI/CD Pipeline",
            visual_concept="An automated assembly line where code is built, tested, and deployed at high speed.",
            bullet_points=[
                "Continuous Integration.",
                "Continuous Deployment.",
                "Automated Testing."
            ]
        )
    ]
    
    prompts = await generator.generate_prompts_batch(slides, model_name="meta/llama-3.1-8b-instruct")
    
    print("\n--- GENERATED PROMPTS ---")
    for i, p in enumerate(prompts):
        print(f"\nSlide {i+1}:\n{p}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())
