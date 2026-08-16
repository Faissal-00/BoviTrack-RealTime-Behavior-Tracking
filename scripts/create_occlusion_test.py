import os
import cv2

# 1. Resolve paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
video_filenames = ['day.mp4', 'night.mp4']
input_video_paths = [os.path.join(base_dir, 'videos', v) for v in video_filenames]
output_test_path = os.path.join(base_dir, 'videos', 'occlusion_test.mp4')

# Open first capture to grab global specs
cap_setup = cv2.VideoCapture(input_video_paths[0])
if not cap_setup.isOpened():
    raise FileNotFoundError(f"Error: Could not open {input_video_paths[0]}")

width = int(cap_setup.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_setup.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap_setup.get(cv2.CAP_PROP_FPS) or 30
cap_setup.release()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_test_path, fourcc, fps, (width, height))

# Define occlusion events (time in seconds relative to the global video timeline)
# format: (start_sec, end_sec, side)
occlusion_schedule = [
    (2.5, 4.5, 'left'),    # Day - Left side
    (6.5, 8.5, 'right'),   # Day - Right side
    (12.5, 14.5, 'left'),  # Night - Left side
    (16.5, 18.5, 'right')  # Night - Right side
]

# Convert schedule into frame-based ranges
occlusion_windows = [
    (int(start * fps), int(end * fps), side)
    for start, end, side in occlusion_schedule
]

print("Generating multi-occlusion benchmark video across Day & Night...")

global_frame_idx = 0

for video_path in input_video_paths:
    print(f"Processing {os.path.basename(video_path)} into test stream...")
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        # Check if current global frame falls into any occlusion window
        for start_f, end_f, side in occlusion_windows:
            if start_f <= global_frame_idx <= end_f:
                if side == 'left':
                    barrier_width = int(width * 0.35)
                    cv2.rectangle(frame, (0, 0), (barrier_width, height), (0, 0, 0), -1)
                    cv2.putText(frame, "[SIMULATED OCCLUSION - LEFT]", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif side == 'right':
                    barrier_start_x = int(width * 0.65)
                    cv2.rectangle(frame, (barrier_start_x, 0), (width, height), (0, 0, 0), -1)
                    cv2.putText(frame, "[SIMULATED OCCLUSION - RIGHT]", (barrier_start_x + 20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                break  # Matched current window

        out.write(frame)
        global_frame_idx += 1

    cap.release()

out.release()
print(f"Synthetic multi-occlusion video generated successfully at: {output_test_path}")