import structlog
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from src.auto_presenter.schemas import PresentationOutline
import win32com.client
import pythoncom

logger = structlog.get_logger(__name__)

class PPTXComposerClient:
    def __init__(self):
        pass

    def apply_autoplay_settings(self, pptx_path: Path, slide_assets: list):
        """Uses Windows COM automation to modify the PPTX to auto-advance and auto-play."""
        logger.info("applying_pptx_autoplay_settings", pptx=str(pptx_path))
        try:
            pythoncom.CoInitialize()
            powerpoint = win32com.client.Dispatch("PowerPoint.Application")
            
            # Open without window to prevent flashing if possible
            # args: FileName, ReadOnly, Untitled, WithWindow
            presentation = powerpoint.Presentations.Open(str(pptx_path.absolute()), False, False, False)
            
            # Moviepy to get exact video durations
            try:
                from moviepy import VideoFileClip
            except ImportError:
                from moviepy.editor import VideoFileClip
            
            for idx, slide in enumerate(presentation.Slides):
                # 1-indexed in PowerPoint COM
                slide_number, bg_path, video_path, avatar_image_path = slide_assets[idx]
                
                if video_path and video_path.exists():
                    vclip = VideoFileClip(str(video_path))
                    duration_sec = vclip.duration
                    vclip.close()
                else:
                    duration_sec = 5.0 # fallback
                
                # 1. Slide Auto-Advance
                slide.SlideShowTransition.AdvanceOnTime = True
                slide.SlideShowTransition.AdvanceTime = duration_sec + 0.5 # 0.5s buffer
                
                # 2. Video Auto-Play (msoMedia = 16)
                for shape in slide.Shapes:
                    if shape.Type == 16:
                        shape.AnimationSettings.PlaySettings.PlayOnEntry = True
                        break
                        
            presentation.Save()
            presentation.Close()
            logger.info("autoplay_settings_applied_successfully")
        except Exception as e:
            logger.error("win32com_autoplay_failed", error=str(e))
        finally:
            pythoncom.CoUninitialize()

    def build_presentation(self, outline: PresentationOutline, slide_assets: list, output_dir: Path):
        """
        Builds a final PPTX file from the outline and the generated assets.
        slide_assets is a list of tuples: (slide_number, background_image_path, avatar_video_path)
        """
        logger.info("starting_pptx_composition", topic=outline.topic, output_dir=str(output_dir))
        
        prs = Presentation()
        # Set 16:9 Aspect Ratio (10 inches by 5.625 inches)
        prs.slide_width = Inches(10)
        prs.slide_height = Inches(5.625)
        
        # Sort assets by slide number to ensure correct order
        slide_assets.sort(key=lambda x: x[0])
        
        # We need a blank slide layout
        blank_slide_layout = prs.slide_layouts[6] 
        
        for idx, slide_outline in enumerate(outline.slides):
            slide_number, bg_path, video_path, avatar_image_path = slide_assets[idx]
            
            slide = prs.slides.add_slide(blank_slide_layout)
            
            # 1. Add Background Image
            if bg_path and bg_path.exists():
                slide.shapes.add_picture(str(bg_path), 0, 0, width=Inches(10), height=Inches(5.625))
            else:
                logger.warning("pptx_missing_background", slide=slide_number)
            
            # Note: The AI-generated background image serves as the complete visual slide.
            # We intentionally omit adding any overlapping text boxes to keep the visual clean.
            
            # 5. Add SadTalker Avatar Video (Small talking head in the bottom right corner)
            if video_path and video_path.exists():
                video_width = Inches(1.8)
                video_height = Inches(1.8)
                video_left = Inches(10.0) - video_width - Inches(0.2)
                video_top = Inches(5.625) - video_height - Inches(0.2)
                
                try:
                    # By passing poster_frame_image, PowerPoint will show the avatar image instead of a speaker icon!
                    slide.shapes.add_movie(
                        str(video_path), 
                        video_left, video_top, video_width, video_height,
                        poster_frame_image=str(avatar_image_path),
                        mime_type='video/mp4'
                    )
                except Exception as e:
                    logger.error("pptx_failed_to_add_movie", error=str(e), slide=slide_number)
            else:
                logger.warning("pptx_missing_video", slide=slide_number)

        # Save Presentation
        output_pptx = output_dir / "final_presentation.pptx"
        prs.save(str(output_pptx))
        logger.info("pptx_composition_complete", file=str(output_pptx))
        
        # Apply Auto-Play Settings!
        self.apply_autoplay_settings(output_pptx, slide_assets)
        
        return output_pptx
