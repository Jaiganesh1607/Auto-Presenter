import asyncio
from pathlib import Path
from dotenv import load_dotenv
from src.auto_presenter.video_generator import SadTalkerClient

# Load environment variables from .env
load_dotenv()

async def test_sadtalker_client():
    client = SadTalkerClient()
    
    # Paths to the test files
    audio_path = Path("./output_images/test_voicex_audio.wav")
    avatar_image = Path("./avatar_images/female.png")
    output_path = Path("./output_images/test_sadtalker_output.mp4")
    
    if not audio_path.exists():
        print(f"Test failed: Audio file {audio_path} not found. Please run test_voicex.py first.")
        return
        
    if not avatar_image.exists():
        print(f"Test failed: Avatar image {avatar_image} not found.")
        print("Please place a 'female.png' inside the 'avatar_images' folder.")
        return
        
    print(f"Sending request to Kaggle SadTalker at: {client.endpoint}")
    print(f"Audio: {audio_path}")
    print(f"Image: {avatar_image}")
    print("-" * 40)
    print("Generating video... (This usually takes 30-60 seconds depending on Kaggle GPU)")
    
    result = await client.generate_video(
        audio_path=audio_path,
        avatar_image_path=avatar_image,
        output_path=output_path
    )
    
    if result:
        print(f"\nSuccess! Avatar video saved to: {result}")
    else:
        print("\nFailed to generate video. Check the logs for details.")

if __name__ == "__main__":
    asyncio.run(test_sadtalker_client())
