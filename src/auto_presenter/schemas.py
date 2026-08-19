from pydantic import BaseModel, Field
from typing import List

class SlideOutline(BaseModel):
    """Schema for a single slide's structural outline."""
    slide_number: int = Field(description="The sequential number of the slide.")
    title: str = Field(description="The short, punchy main title of the slide.")
    bullet_points: List[str] = Field(description="A list of 2-4 short bullet points for the slide content.")
    speaker_notes: str = Field(description="The detailed script the avatar will read for this slide.")
    visual_concept: str = Field(description="A brief description of what the visual on the slide should represent (used for prompting).")

class PresentationOutline(BaseModel):
    """Schema for the entire presentation."""
    topic: str = Field(description="The main topic of the presentation.")
    slides: List[SlideOutline] = Field(description="The ordered list of slides in the presentation.")
