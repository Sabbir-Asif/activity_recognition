import cv2
import os

video_path = "../data/videos/10.160.40.2_IPC_main_20251022165643 (video-converter.com).mp4"
output_dir = "sample_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sampled_frames = 10  # pick 10 frames equally spaced
interval = max(total_frames // sampled_frames, 1)

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if i % interval == 0:
        cv2.imwrite(f"{output_dir}/frame_{i:04d}.jpg", frame)
    i += 1

cap.release()
print(f"Extracted {len(os.listdir(output_dir))} frames to {output_dir}/")
