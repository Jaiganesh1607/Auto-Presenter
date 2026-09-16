import structlog
from pathlib import Path
import os
import sys

# Try importing moviepy v2 style, fallback to v1
try:
    from moviepy import ImageClip, VideoFileClip, CompositeVideoClip, concatenate_videoclips
except ImportError:
    from moviepy.editor import ImageClip, VideoFileClip, CompositeVideoClip, concatenate_videoclips

logger = structlog.get_logger(__name__)

class VideoComposerClient:
    def __init__(self):
        pass

    def build_master_video(self, slide_assets: list, output_dir: Path) -> Path:
        """
        Builds a final concatenated MP4 video from all slides.
        slide_assets: list of (slide_number, bg_path, video_path, avatar_image_path)
        """
        logger.info("starting_master_video_composition", total_slides=len(slide_assets), output_dir=str(output_dir))
        
        # Sort by slide number
        slide_assets.sort(key=lambda x: x[0])
        
        slide_clips = []
        
        # 16:9 720p or 1080p? We'll use the background image resolution (usually 1280x720 from Z-Image)
        
        for slide_number, bg_path, video_path, _ in slide_assets:
            if not bg_path or not bg_path.exists():
                logger.warning("video_missing_background", slide=slide_number)
                continue
                
            if not video_path or not video_path.exists():
                logger.warning("video_missing_avatar", slide=slide_number)
                continue
                
            logger.info("processing_video_slide", slide=slide_number)
            
            try:
                # 1. Load Background Image
                bg_clip = ImageClip(str(bg_path))
                
                # 2. Load Avatar Video
                avatar_clip = VideoFileClip(str(video_path))
                
                # Set background duration to precisely match the avatar speaking duration
                bg_clip = bg_clip.with_duration(avatar_clip.duration) if hasattr(bg_clip, 'with_duration') else bg_clip.set_duration(avatar_clip.duration)
                
                # 3. Resize and position the Avatar Video (streamer layout in bottom right)
                # Background is usually 1280x720. 
                # We'll make avatar width around 240px.
                target_width = 240
                
                # Use with_position/resized for moviepy 2.0, fallback to set_pos/resize for 1.x
                if hasattr(avatar_clip, 'resized'):
                    # Moviepy v2.x
                    avatar_resized = avatar_clip.resized(width=target_width)
                    # Position with margin: (x, y) = (1280 - 240 - 20, 720 - height - 20)
                    # In v2, margin is handled by absolute pixel coordinates or strings
                    # 'right', 'bottom' usually aligns it to the edge. We can add a margin by setting absolute pos.
                    # Or simpler: just "right", "bottom"
                    avatar_positioned = avatar_resized.with_position(("right", "bottom"))
                else:
                    # Moviepy v1.x
                    avatar_resized = avatar_clip.resize(width=target_width)
                    # Right bottom alignment with margin can be done via exact coordinates, or just right bottom
                    avatar_positioned = avatar_resized.set_pos(("right", "bottom"))
                
                # 4. Composite them together
                slide_composite = CompositeVideoClip([bg_clip, avatar_positioned])
                # Ensure the duration matches the avatar exactly
                slide_composite = slide_composite.with_duration(avatar_clip.duration) if hasattr(slide_composite, 'with_duration') else slide_composite.set_duration(avatar_clip.duration)
                
                slide_clips.append(slide_composite)
            except Exception as e:
                logger.error("video_slide_composition_failed", slide=slide_number, error=str(e))
                raise e
                
        if not slide_clips:
            raise ValueError("No slide clips were successfully processed.")
            
        # 5. Concatenate all slides seamlessly
        logger.info("concatenating_final_video", total_clips=len(slide_clips))
        final_video = concatenate_videoclips(slide_clips, method="compose")
        
        output_path = output_dir / "final_presentation.mp4"
        
        # 6. Write final video (using h264 for universal compatibility)
        logger.info("writing_final_video", path=str(output_path))
        # Moviepy v2 might use write_videofile or write...
        try:
            final_video.write_videofile(
                str(output_path),
                fps=24,
                codec="libx264",
                audio_codec="aac",
                logger=None # Suppress internal tqdm bar if possible
            )
        except Exception as e:
            logger.error("final_video_writing_failed", error=str(e))
            raise e
            
        # Cleanup
        for clip in slide_clips:
            try: clip.close() 
            except: pass
        try: final_video.close()
        except: pass
        
        logger.info("master_video_composition_complete", path=str(output_path))
        return output_path
