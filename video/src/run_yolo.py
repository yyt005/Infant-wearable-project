from pathlib import Path
import pandas as pd
from ultralytics import YOLO

def run_yolo_on_segments(segment_dir, output_video_dir):
    Path(output_video_dir).mkdir(parents=True, exist_ok=True)
    model = YOLO('yolov8s.pt')
    all_results = []

    segment_files = sorted(Path(segment_dir).glob("segment_*.mp4"))

    for seg_path in segment_files:
        print(f"Processing {seg_path.name} ...")
        results = model(
            str(seg_path),
            save=True,
            save_txt=False,
            project=output_video_dir,
            name=f"annotated_{seg_path.stem}"  # prefix annotated videos with 'annotated_'
        )

        for frame_idx, result in enumerate(results):
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].item()
                cls_id = int(boxes.cls[i].item())
                all_results.append({
                    "segment": seg_path.name,
                    "frame_number": frame_idx,
                    "object_class": cls_id,
                    "class_name": model.names[cls_id],
                    "confidence_score": conf,
                    "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                    "is_face": model.names[cls_id].lower() == "face"
                })

    df = pd.DataFrame(all_results)
    csv_path = Path(output_video_dir) / "combined_detections.csv"
    df.to_csv(csv_path, index=False)
    print(f"Detection results saved to {csv_path}")