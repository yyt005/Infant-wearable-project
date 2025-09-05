# object_detect.py
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import cv2

def object_detect(input_video_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO('yolov8s.pt')
    video_name = Path(input_video_path).stem
    
    # Prepare output paths
    output_video_path = output_dir / f"object_detected_{video_name}.mp4"
    output_csv_path = output_dir / f"object_detections_{video_name}.csv"
    
    # Open input video
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"Cannot open video {input_video_path}")
        return None
    
    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Process video with YOLO
    results = model.track(
        source=str(input_video_path),
        stream=True,
        save=False,
        project=str(output_dir),
        name=f"object_detected_{video_name}",
        save_txt=False,
        save_conf=True
    )
    
    # Prepare to collect detections
    object_detections = []
    frame_count = 0
    
    # Create VideoWriter for annotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    for result in results:
        frame_count += 1
        annotated_frame = result.plot()  # Get the annotated frame
        out_vid.write(annotated_frame)
        
        if result.boxes is not None:
            for box in result.boxes:
                object_detections.append({
                    'video_name': video_name,
                    'frame_number': frame_count,
                    'object_class': int(box.cls.item()),
                    'class_name': model.names[int(box.cls.item())],
                    'confidence_score': box.conf.item(),
                    'x1': box.xyxy[0][0].item(),
                    'y1': box.xyxy[0][1].item(),
                    'x2': box.xyxy[0][2].item(),
                    'y2': box.xyxy[0][3].item(),
                    'is_face': model.names[int(box.cls.item())].lower() == 'face'
                })
    
    out_vid.release()
    
    # Save object detections to CSV
    if object_detections:
        df = pd.DataFrame(object_detections)
        df.to_csv(output_csv_path, index=False)
        print(f"Object detection results saved to {output_csv_path}")
    else:
        print(f"No objects detected in {video_name}")
    
    print(f"Object detection done for {video_name}, annotated video saved to {output_video_path}")
    return output_csv_path
