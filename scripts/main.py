import os
import cv2
from ultralytics import YOLO

# 1. Resolve paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
detector_path = os.path.join(base_dir, 'final_models', 'detector_best.pt')
classifier_path = os.path.join(base_dir, 'final_models', 'classifier_best.pt')

# Add your Day first, Night second
video_filenames = ['day.mp4', 'night.mp4'] 
# video_filenames = ['occlusion_test.mp4']
input_video_paths = [os.path.join(base_dir, 'videos', v) for v in video_filenames]
output_video_path = os.path.join(base_dir, 'videos', 'bovitrack_demo.mp4')

print("Loading AI Engines...")
detector = YOLO(detector_path)
classifier = YOLO(classifier_path)

# Grab dimensions and FPS from the first video to set up the writer
cap_setup = cv2.VideoCapture(input_video_paths[0])
width = int(cap_setup.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_setup.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap_setup.get(cv2.CAP_PROP_FPS) or 30
cap_setup.release()

# Setup single video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Generating sequential LinkedIn demo at {output_video_path}...")

# Process videos one after the other
for video_path in input_video_paths:
    print(f"Processing {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to ensure consistency if the videos have different native resolutions
        frame = cv2.resize(frame, (width, height))

        # Run inference
        track_results = detector.track(
            frame, 
            persist=True, 
            tracker="custom_tracker.yaml",  # Points to our fixed configuration
            conf=0.6,          
            iou=0.45,          
            verbose=False
        )
        
        if track_results[0].boxes.id is not None:
            boxes = track_results[0].boxes.xyxy.cpu().numpy()
            track_ids = track_results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                cow_crop = frame[y1:y2, x1:x2]
                
                if cow_crop.shape[0] > 10 and cow_crop.shape[1] > 10:
                    cls_results = classifier(cow_crop, verbose=False)
                    confidence = cls_results[0].probs.top1conf.item()
                    
                    if confidence >= 0.55:
                        behavior_name = cls_results[0].names[cls_results[0].probs.top1]
                    else:
                        behavior_name = "analyzing..."

                    label = f"ID #{int(track_id)}: {behavior_name} ({confidence:.2f})"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # UI Polish
                    # Define your new modern UI color (BGR format)
                    ui_color = (255, 255, 0) # Electric Cyan

                    # Draw bounding box with the new color
                    cv2.rectangle(frame, (x1, y1), (x2, y2), ui_color, 2)

                    # UI Polish: Increased font scale and thickness for LinkedIn mobile readability
                    font_scale = 0.7
                    font_thickness = 2

                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

                    # Added more vertical padding (-12 and -5) so the text isn't cramped in the cyan box
                    cv2.rectangle(frame, (x1, max(0, y1 - text_h - 12)), (x1 + text_w, max(0, y1)), ui_color, -1)
                    cv2.putText(frame, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        # Write to the same master file
        out.write(frame)

    cap.release()

out.release()
print("Processing complete! Your final video is ready.")