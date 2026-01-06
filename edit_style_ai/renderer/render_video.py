from moviepy.editor import ImageClip, VideoFileClip, concatenate_videoclips, vfx


def _fit_to_portrait(clip, target_w=1080, target_h=1920):
    """Resize and center-crop to 1080x1920 while preserving aspect ratio."""
    aspect = clip.w / clip.h
    target_aspect = target_w / target_h

    if aspect >= target_aspect:
        # Wider than target: fit height, crop width
        clip = clip.resize(height=target_h)
    else:
        # Taller than target: fit width, crop height
        clip = clip.resize(width=target_w)

    return clip.crop(
        width=target_w,
        height=target_h,
        x_center=clip.w / 2,
        y_center=clip.h / 2,
    )


def render(timeline, output="output/final.mp4"):
    """
    Render timeline of video and image clips into a final 1080x1920 video.

    - Images: duration set to shot length + gentle Ken Burns (zoom-in).
    - Videos: trimmed to shot length, resized/cropped to portrait.
    - Concatenates all clips; requires ffmpeg available for moviepy.
    """

    clips = []

    for i, shot in enumerate(timeline):
        try:
            duration = max(float(shot.get("duration", 0)), 0.01)

            if shot["type"] == "image":
                clip = ImageClip(shot["asset"]).set_duration(duration)
                # Gentle zoom-in over the clip duration
                zoom_amount = 0.05
                clip = clip.fx(vfx.resize, lambda t: 1 + zoom_amount * (t / duration))
                clip = _fit_to_portrait(clip)
                clips.append(clip)
            else:
                clip = VideoFileClip(shot["asset"])
                clip = clip.subclip(0, min(duration, clip.duration))
                clip = _fit_to_portrait(clip)
                clips.append(clip)
        except Exception as e:
            print(f"Error processing shot {i} ({shot['asset']}): {e}")
            continue

    if not clips:
        print("No clips were processed successfully")
        return

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(
        output,
        codec="libx264",
        audio_codec="aac",
        fps=30,
        verbose=False,
        logger=None,
    )
    print(f"Video rendered successfully: {output}")
