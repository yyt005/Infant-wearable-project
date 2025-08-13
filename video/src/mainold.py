import os
from pathlib import Path
from segment_video import segment_video
from object_detect import object_detect

def main():
    videos_dir = "input_videos"
    segment_base_dir = "segments"
    results_base_dir = "results"
    segment_length_sec = 60

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(segment_base_dir, exist_ok=True)
    os.makedirs(results_base_dir, exist_ok=True)

    video_files = sorted(Path(videos_dir).glob("*.mp4"))
    if not video_files:
        print(f"No videos found in {videos_dir}")
        return

    for video_path in video_files:
        print(f"Processing video: {video_path.name}")
        segment_dir = Path(segment_base_dir) / video_path.stem
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_video(str(video_path), str(segment_dir), segment_length_sec)

        result_dir = Path(results_base_dir) / video_path.stem
        object_detect(str(segment_dir), str(result_dir))

if __name__ == "__main__":
    main()
