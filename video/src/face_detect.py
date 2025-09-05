import cv2
import torch
from pathlib import Path
import sys
import pandas as pd

# Add yolov5-face repo path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'yolov5-face'))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

def face_detect(
    input_video_path, 
    output_dir, 
    conf_thres=0.5,  # Confidence threshold (0-1)
    iou_thres=0.3,   # NMS IoU threshold (0-1)
    img_size=640, 
    device='cpu'
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLOv5-face model
    weights_path = Path(__file__).resolve().parents[1] / 'yolov5-face' / 'weights' / 'yolov5m-face.pt'
    device = select_device(device)
    model = attempt_load(str(weights_path), map_location=device)
    model.eval()

    # Open input video
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"Cannot open video {input_video_path}")
        return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Prepare output paths
    video_name = Path(input_video_path).stem
    output_video_path = output_dir / f"face_detected_{video_name}.mp4"
    output_csv_path = output_dir / f"face_detections_{video_name}.csv"
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    face_detections = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1
        # Preprocess frame for model input
        resized = cv2.resize(frame, (img_size, img_size))
        img = resized[:, :, ::-1].copy()  # Convert BGR to RGB
        img = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0  # Normalize
        img = img.unsqueeze(0)  # Add batch dimension

        # Run model inference with NMS
        with torch.no_grad():
            pred = model(img, augment=False)[0]  # Get predictions
            pred = non_max_suppression(
                pred, 
                conf_thres=conf_thres, 
                iou_thres=iou_thres,
                classes=0  # Filter for face class
            )

        faces_in_frame = 0
        # Process detections for current frame
        if pred[0] is not None and len(pred[0]) > 0:
            # Scale bounding boxes back to original frame size
            pred[0][:, :4] = scale_coords(
                img.shape[2:],  # Model input dimensions (height, width)
                pred[0][:, :4], 
                frame.shape[:2]  # Original frame dimensions (height, width)
            ).round()

            # Iterate over each detected face
            for face_tensor in pred[0]:
                faces_in_frame += 1
                
                # Extract core detection data (first 6 elements)
                core_data = face_tensor[:6].tolist()
                if len(core_data) < 6:
                    print(f"Skipping invalid detection in frame {frame_count} - insufficient core data")
                    continue  # Skip this detection
                
                x1, y1, x2, y2, conf, cls = core_data

                # Extract landmarks with error handling
                landmark_data = face_tensor[6:16].tolist()
                # Check if we have enough landmark values (10 total)
                if len(landmark_data) < 10:
                    print(f"Missing landmarks in frame {frame_count} - using default values")
                    # Fill missing landmarks with -1 (or NaN if preferred)
                    landmark_data = [-1.0] * 10
                
                lm0_x, lm0_y, lm1_x, lm1_y, lm2_x, lm2_y, lm3_x, lm3_y, lm4_x, lm4_y = landmark_data

                # Draw bounding box and label on frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame, 
                    f'Face {conf:.2f}', 
                    (int(x1), max(10, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    1
                )

                # Store detection data
                face_detections.append({
                    'video_name': video_name,
                    'frame_number': frame_count,
                    'face_id': faces_in_frame,
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'confidence': conf,
                    'class_id': int(cls),
                    'landmark_0_x': lm0_x,
                    'landmark_0_y': lm0_y,
                    'landmark_1_x': lm1_x,
                    'landmark_1_y': lm1_y,
                    'landmark_2_x': lm2_x,
                    'landmark_2_y': lm2_y,
                    'landmark_3_x': lm3_x,
                    'landmark_3_y': lm3_y,
                    'landmark_4_x': lm4_x,
                    'landmark_4_y': lm4_y
                })

        # Write annotated frame to output video
        out_vid.write(frame)

    # Clean up resources
    cap.release()
    out_vid.release()
    
    # Save all detections to CSV
    if face_detections:
        df = pd.DataFrame(face_detections)
        df.to_csv(output_csv_path, index=False)
        print(f"Face detection results saved to {output_csv_path}")
    else:
        print(f"No faces detected in {video_name}")

    return output_csv_path
    