import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from src.auto_presenter.audio_generator import VoiceXClient

async def test_voicex_standalone():
    # Load environment variables (VOICEX_API_URL)
    load_dotenv()
    
    client = VoiceXClient()
    
    # 1. Provide a mock refined script WITHOUT expressions (for clean SadTalker animation)
    mock_text = "Welcome to the Auto-Presenter! I am so excited to be here today. Let's get started!"
    
    # 2. Pick an emotion that the VoiceX backend supports (Neutral, Happy, Sad, Calm, Whisper)
    # Neutral has the optimal default speed (0.85). 'Calm' slows it down to 0.75 which causes robotic artifacts.
    emotion = "Neutral"
    
    # 3. Define the voice persona
    gender = "female"
    age = 30
    
    # 4. Set output path
    output_dir = Path("./output_images")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_voicex_audio.wav"
    
    print(f"Sending request to VoiceX Microservice at: {client.endpoint}")
    print(f"Text: {mock_text}")
    print(f"Emotion: {emotion}")
    print(f"Gender: {gender}")
    print(f"Age: {age}")
    print("-" * 40)
    print("Generating audio... (This might take a minute on the first run to load the model into VRAM)")
    
    try:
        # Call the microservice
        result_path = await client.generate_audio(
            text=mock_text,
            gender=gender,
            age=age,
            emotion=emotion,
            output_path=output_path
        )
        print(f"\nSuccess! Audio saved to: {result_path}")
    except Exception as e:
        print(f"\nFailed to generate audio: {e}")
        print("Make sure your VoiceX backend is running (e.g. uvicorn main:app --port 8000)")

if __name__ == "__main__":
    asyncio.run(test_voicex_standalone())
