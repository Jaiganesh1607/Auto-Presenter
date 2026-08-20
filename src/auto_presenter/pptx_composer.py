import structlog
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from src.auto_presenter.schemas import PresentationOutline

logger = structlog.get_logger(__name__)

class PPTXComposerClient:
    def __init__(self):
        pass

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
            slide_number, bg_path, video_path = slide_assets[idx]
            
            slide = prs.slides.add_slide(blank_slide_layout)
            
            # 1. Add Background Image
            if bg_path and bg_path.exists():
                slide.shapes.add_picture(str(bg_path), 0, 0, width=Inches(10), height=Inches(5.625))
            else:
                logger.warning("pptx_missing_background", slide=slide_number)
            
            # 2. Add Text Backdrop (Semi-transparent black box on the left)
            left = Inches(0.5)
            top = Inches(0.5)
            width = Inches(5.5)
            height = Inches(4.625)
            
            backdrop = slide.shapes.add_shape(
                1, # msoShapeRectangle
                left, top, width, height
            )
            backdrop.fill.solid()
            backdrop.fill.fore_color.rgb = RGBColor(0, 0, 0)
            backdrop.line.fill.background()
            # Note: python-pptx doesn't natively support setting transparency via simple API yet, 
            # so this will be solid black.
            
            # 3. Add Title Text
            txBox = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), width - Inches(0.4), Inches(1))
            tf = txBox.text_frame
            tf.word_wrap = True
            
            p = tf.add_paragraph()
            p.text = slide_outline.title
            p.font.bold = True
            p.font.size = Pt(32)
            p.font.color.rgb = RGBColor(255, 255, 255)
            
            # 4. Add Bullet Points
            bullet_top = top + Inches(1.5)
            bulletBox = slide.shapes.add_textbox(left + Inches(0.2), bullet_top, width - Inches(0.4), height - Inches(1.7))
            btf = bulletBox.text_frame
            btf.word_wrap = True
            
            for point in slide_outline.bullet_points:
                bp = btf.add_paragraph()
                bp.text = f"• {point}"
                bp.font.size = Pt(20)
                bp.font.color.rgb = RGBColor(255, 255, 255)
                # bp.level = 0
            
            # 5. Add SadTalker Avatar Video
            if video_path and video_path.exists():
                video_left = Inches(6.5)
                video_top = Inches(2.125)
                video_width = Inches(3.0)
                video_height = Inches(3.0)
                
                # Add the movie. The poster_frame is required by PowerPoint to show a thumbnail.
                # We can use the avatar_image or the background as a fallback.
                # We'll just leave it None and let PowerPoint use the first frame if possible.
                try:
                    slide.shapes.add_movie(
                        str(video_path), 
                        video_left, video_top, video_width, video_height, 
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
        return output_pptx
