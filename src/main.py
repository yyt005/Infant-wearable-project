import argparse
from pathlib import Path
import os
import pandas as pd

from preprocess_video import convert_to_5fps, stabilize_video
from segment_video import segment_video
from face_detect import face_detect
from object_detect import object_detect

def merge_csvs(face_csv_paths, object_csv_paths, output_path, original_video_name):
    """Combine all face and object CSVs into one merged CSV."""
    face_dfs = []
    for f in face_csv_paths:
        if f and f.exists():
            df = pd.read_csv(f)
            if not df.empty:
                df['is_face'] = True
                df['original_video'] = original_video_name
                face_dfs.append(df)
    object_dfs = []
    for f in object_csv_paths:
        if f and f.exists():
            df = pd.read_csv(f)
            if not df.empty:
                df['is_face'] = False
                df['original_video'] = original_video_name
                object_dfs.append(df)
    merged = pd.concat(face_dfs + object_dfs, ignore_index=True) if (face_dfs + object_dfs) else None
    if merged is not None:
        merged.to_csv(output_path, index=False)
        print(f"Merged CSV saved to {output_path}")

def run_pipeline(video_path, results_base_dir, use_stab=False, use_seg=False):
    """Process a single video according to stabilization/segmentation options."""
    version_name = (
        ("with_stab" if use_stab else "no_stab") + "_" +
        ("with_seg" if use_seg else "no_seg")
    )
    version_dir = results_base_dir / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    # --- Preprocessing: 5fps conversion (shared) ---
    preprocessed_dir = Path("preprocessed_5fps")
    preprocessed_dir.mkdir(exist_ok=True)
    video_5fps = preprocessed_dir / f"{video_path.stem}_5fps.mp4"
    if not video_5fps.exists():
        print(f"Converting {video_path.name} to 5fps...")
        convert_to_5fps(video_path, video_5fps)
    else:
        print(f"5fps version already exists: {video_5fps.name}")

    # --- Optional stabilization ---
    if use_stab:
        stabilized_video = preprocessed_dir / f"{video_path.stem}_5fps_stab.mp4"
        if not stabilized_video.exists():
            print(f"Stabilizing {video_5fps.name}...")
            stabilize_video(video_5fps, stabilized_video)
        processed_video = stabilized_video
    else:
        processed_video = video_5fps

    # --- Optional segmentation ---
    if use_seg:
        segment_dir = version_dir / "segments" / video_path.stem
        segment_dir.mkdir(parents=True, exist_ok=True)
        print(f"Segmenting {processed_video.name}...")
        segment_video(processed_video, str(segment_dir))
        video_inputs = sorted(segment_dir.glob("*.*"))  # accept any extension
    else:
        video_inputs = [processed_video]

    # --- Run detections ---
    face_csv_paths = []
    object_csv_paths = []
    for v in video_inputs:
        face_dir = version_dir / "detections" / v.stem / "face"
        obj_dir = version_dir / "detections" / v.stem / "object"
        print(f"Running face detection on {v.name}...")
        face_csv = face_detect(str(v), str(face_dir))
        face_csv_paths.append(face_csv)
        print(f"Running object detection on {v.name}...")
        object_csv = object_detect(str(v), str(obj_dir))
        object_csv_paths.append(object_csv)

    # --- Merge CSVs for this video + version ---
    merged_csv_path = version_dir / f"{video_path.stem}_merged_detections.csv"
    merge_csvs(face_csv_paths, object_csv_paths, merged_csv_path, video_path.stem)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stab", action="store_true", help="Apply stabilization")
    parser.add_argument("--seg", action="store_true", help="Apply segmentation")
    parser.add_argument("--all_versions", action="store_true", help="Run all 4 versions automatically")
    args = parser.parse_args()

    # Use absolute paths relative to script
    BASE_DIR = Path(__file__).resolve().parent.parent
    videos_dir = BASE_DIR / "input_videos"
    results_base_dir = BASE_DIR / "results"
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(results_base_dir, exist_ok=True)

    # Accept mp4 + mov
    video_files = sorted(videos_dir.glob("*.*"))
    video_files = [v for v in video_files if v.suffix.lower() in [".mp4", ".mov"]]
    if not video_files:
        print("No videos found in input_videos/")
        return

    for video_path in video_files:
        print(f"\nProcessing {video_path.name}")
        if args.all_versions:
            for stab in [False, True]:
                for seg in [False, True]:
                    run_pipeline(video_path, results_base_dir, use_stab=stab, use_seg=seg)
        else:
            run_pipeline(video_path, results_base_dir, use_stab=args.stab, use_seg=args.seg)

if __name__ == "__main__":
    main()
