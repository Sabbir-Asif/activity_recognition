import cv2
import numpy as np
from pathlib import Path

# Video path
video_path = "../data/working_1.mp4"
output_dir = "../sample_badge"

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Get 5 evenly spaced frames
frame_indices = np.linspace(0, total_frames - 1, 5, dtype=int)

for frame_idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if ret:
        # Step 1: BGR (input image)
        bgr = frame
        
        # Step 2: Convert to LAB
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Step 3: CLAHE on L channel (fix uneven lighting)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Step 4: Convert LAB → BGR
        lab_processed = cv2.merge([l_channel, a_channel, b_channel])
        bgr_processed = cv2.cvtColor(lab_processed, cv2.COLOR_LAB2BGR)
        
        # Step 5: Apply Gamma Correction (fix dark/bright exposure)
        inv_gamma = 1.0 / 1.2
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        bgr_gamma = cv2.LUT(bgr_processed, table)
        
        # # Step 6: Convert to HSV
        # hsv = cv2.cvtColor(bgr_gamma, cv2.COLOR_BGR2HSV)
        
        # Save frame
        output_path = f"{output_dir}/frame_{frame_idx:04d}.jpg"
        cv2.imwrite(output_path, bgr_gamma)
        print(f"Saved: {output_path}")

cap.release()
print("Done!")
