from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


def detect_cuts(video_path):
    """Detect scene cuts and return cut timestamps including start and end.

    Notes:
    - Includes a 0.0 starting cut and appends the full video duration so
      downstream shot length computation has the final segment.
    - Uses a conservative content threshold; tune as needed for your footage.
    """

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))

    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        scenes = scene_manager.get_scene_list()
        cuts = [max(0.0, scene[0].get_seconds()) for scene in scenes]

        # Ensure timeline starts at 0.0
        if not cuts or cuts[0] > 0.05:
            cuts = [0.0] + cuts

        # Append full duration to close the last shot if available
        duration = video_manager.get_duration()
        duration_sec = None

        # scenedetect may return a Timecode or a tuple; handle both
        if duration:
            if hasattr(duration, "get_seconds"):
                duration_sec = duration.get_seconds()
            elif isinstance(duration, (tuple, list)):
                # First element might be Timecode; otherwise compute from frames/fps
                for d in duration:
                    if hasattr(d, "get_seconds"):
                        duration_sec = d.get_seconds()
                        break
                if duration_sec is None and len(duration) == 2:
                    frames, fps = duration
                    try:
                        duration_sec = frames / float(fps)
                    except Exception:
                        duration_sec = None

        if duration_sec is not None:
            if not cuts or cuts[-1] < duration_sec:
                cuts.append(duration_sec)
    finally:
        video_manager.release()

    return cuts
    video_manager.release()

    return cuts
