# main.py
import os
from pathlib import Path
from segment_video import segment_video
from face_detect import face_detect
from object_detect import object_detect
import pandas as pd
import torch
torch.serialization.add_safe_globals([torch._import_dotted_name('numpy.core.multiarray._reconstruct')])
    
def combine_detections(face_csv_paths, object_csv_paths, output_dir, original_video_name):
    output_dir = Path(output_dir)
    
    # --------------------------
    # Process face detections
    # --------------------------
    face_dfs = []
    for csv_path in face_csv_paths:
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                # Keep required face columns (no class info needed since they're all "face")
                face_df = df[['video_name', 'frame_number', 'face_id', 'x1', 'y1', 'x2', 'y2', 'confidence','class_id']].copy()
                face_df['is_face'] = True
                face_df['original_video'] = original_video_name
                face_df['class_name'] = 'face' 
                face_df['object_class'] = -1
                face_dfs.append(face_df)
    
    combined_face = pd.concat(face_dfs, ignore_index=True) if face_dfs else None

    # --------------------------
    # Process object detections
    # --------------------------
    object_dfs = []
    for csv_path in object_csv_paths:
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                # Include class columns from object detection CSV
                # Assumes object CSV has 'class_id' and 'class_name' (adjust if your columns differ)
                object_df = df[['video_name', 'frame_number', 'x1', 'y1', 'x2', 'y2', 
                                'confidence_score', 'is_face', 'class_name','object_class']].copy()
                object_df = object_df.rename(columns={'confidence_score': 'confidence'})
                object_df['face_id'] = -1  # No face ID for objects
                object_df['class_id'] = -1
                object_df['original_video'] = original_video_name
                object_dfs.append(object_df)
    
    combined_object = pd.concat(object_dfs, ignore_index=True) if object_dfs else None

    # --------------------------
    # Merge face + object data
    # --------------------------
    if combined_face is not None or combined_object is not None:
        merged_df = pd.concat(
            [df for df in [combined_face, combined_object] if df is not None],
            ignore_index=True
        )
        
        # Ensure consistent dtypes
        merged_df['face_id'] = merged_df['face_id'].fillna(-1).astype(int)
        merged_df['class_id'] = merged_df['class_id'].fillna(-1).astype(int)
        merged_df['object_class'] = merged_df['object_class'].fillna(-1).astype(int)
        merged_df['is_face'] = merged_df['is_face'].fillna(False)
        
        # Save outputs
        if combined_face is not None:
            face_output_path = output_dir / "combined_face_detections.csv"
            combined_face.to_csv(face_output_path, index=False)
            print(f"Combined face detections saved to {face_output_path}")
        
        if combined_object is not None:
            object_output_path = output_dir / "combined_object_detections.csv"
            combined_object.to_csv(object_output_path, index=False)
            print(f"Combined object detections saved to {object_output_path}")
        
        merged_output_path = output_dir / "merged_detections.csv"
        merged_df.to_csv(merged_output_path, index=False)
        print(f"Merged detections (with class info) saved to {merged_output_path}")
        return merged_output_path
    else:
        print("No detections to merge")
        return None
    
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
        print(f"\nProcessing video: {video_path.name}")

        # Segment the video
        segment_dir = Path(segment_base_dir) / video_path.stem
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_video(str(video_path), str(segment_dir), segment_length_sec)

        # Prepare result directories
        video_result_dir = Path(results_base_dir) / video_path.stem
        video_result_dir.mkdir(parents=True, exist_ok=True)
        
        face_csv_paths = []
        object_csv_paths = []

        # Process each segment
        for segment_file in sorted(segment_dir.glob("*.mp4")):
            print(f"\nProcessing segment: {segment_file.name}")
            
            # Face detection
            face_result_dir = video_result_dir / "face_detect"
            print(f"Running face detection on {segment_file.name}...")
            face_csv = face_detect(str(segment_file), str(face_result_dir))
            face_csv_paths.append(face_csv)
            
            # Object detection
            object_result_dir = video_result_dir / "object_detect"
            print(f"Running object detection on {segment_file.name}...")
            object_csv = object_detect(str(segment_file), str(object_result_dir))
            object_csv_paths.append(object_csv)
        
        # Combine all detections
        print("\nCombining detection results...")
        combine_detections(face_csv_paths, object_csv_paths, video_result_dir, original_video_name=video_path.stem)

if __name__ == "__main__":
    main()