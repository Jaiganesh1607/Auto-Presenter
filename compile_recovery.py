import os
import asyncio
from pathlib import Path

# Adjust sys.path to allow importing from src
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.auto_presenter.pptx_composer import PPTXComposerClient
from src.auto_presenter.video_composer import VideoComposerClient

class DummySlide:
    def __init__(self, num):
        self.title = f"Slide {num}"
        self.bullet_points = []

class DummyOutline:
    def __init__(self, count):
        self.topic = "Recovered Presentation"
        self.slides = [DummySlide(i) for i in range(1, count + 1)]

def recover_presentation(folder_path_str: str):
    folder = Path(folder_path_str)
    
    # 1. Gather all slide assets
    slide_assets = []
    avatar_image_path = Path("avatar_images/female.png")
    
    # Check max slides by scanning files
    max_slide = 8
    
    for i in range(1, max_slide + 1):
        bg_path = folder / f"slide_{i}_bg.png"
        video_path = folder / f"slide_{i}_avatar.mp4"
        
        # PPTX requires paths to exist, we'll pass them if they do
        slide_assets.append((
            i, 
            bg_path if bg_path.exists() else None, 
            video_path if video_path.exists() else None, 
            avatar_image_path
        ))
        
    outline = DummyOutline(max_slide)
    
    print("Building PPTX...")
    pptx_client = PPTXComposerClient()
    pptx_client.build_presentation(outline, slide_assets, folder)
    print("PPTX Built successfully!")
    
    print("Building Master MP4...")
    try:
        video_composer = VideoComposerClient()
        video_composer.build_master_video(slide_assets, folder)
        print("Master MP4 Built successfully!")
    except Exception as e:
        print(f"Failed to build master MP4: {e}")

if __name__ == "__main__":
    recover_presentation(r"c:\Users\Jaiganesh\Projects\Auto-Presenter\output_images\20260821_140138_blockchain")
