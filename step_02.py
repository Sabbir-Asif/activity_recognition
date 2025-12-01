import cv2
import numpy as np
import glob
import pandas as pd
import os

# === Configuration ===
FRAME_DIR = "sample_frames"        # Folder containing extracted frames
BADGE_SAVE_DIR = "badge_samples"   # Folder to save cropped badge images
OUTPUT_CSV = "badge_hsv_data.csv"
RESIZE_DIM = (800, 600)

# === Prepare Directories ===
os.makedirs(BADGE_SAVE_DIR, exist_ok=True)

# === Load Frames ===
frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))

if not frame_paths:
    print("❌ No frames found in the given directory.")
    exit()

# To collect all HSV pixels
all_hsv_values = []
records = []

# === Process Each Frame ===
for path in frame_paths:
    frame_name = os.path.basename(path)
    frame = cv2.imread(path)
    frame = cv2.resize(frame, RESIZE_DIM)

    # Let user draw ROI
    roi = cv2.selectROI(
        f"Select Badge ROI (Frame: {frame_name}) - Press ENTER when done, ESC to skip",
        frame,
        fromCenter=False
    )

    if roi == (0, 0, 0, 0):
        continue

    x, y, w, h = map(int, roi)
    badge_region = frame[y:y+h, x:x+w]

    # Save cropped badge for visual documentation
    badge_filename = os.path.join(BADGE_SAVE_DIR, f"badge_{frame_name}")
    cv2.imwrite(badge_filename, badge_region)

    # Convert ROI to HSV
    hsv_roi = cv2.cvtColor(badge_region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_roi)

    # --- Brightness-based thresholding ---
    # Use Otsu's threshold to auto-separate bright badge from darker background
    _, bright_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask to HSV pixels
    bright_pixels = hsv_roi[bright_mask > 0]
    if len(bright_pixels) == 0:
        print(f"⚠️ No bright pixels found in {frame_name}, skipping.")
        continue

    all_hsv_values.append(bright_pixels)

    # Compute stats for bright area only
    hsv_mean = np.mean(bright_pixels, axis=0)
    hsv_std = np.std(bright_pixels, axis=0)
    hsv_min = np.min(bright_pixels, axis=0)
    hsv_max = np.max(bright_pixels, axis=0)

    records.append({
        "frame_name": frame_name,
        "x": x, "y": y, "width": w, "height": h,
        "H_mean": hsv_mean[0], "S_mean": hsv_mean[1], "V_mean": hsv_mean[2],
        "H_std": hsv_std[0], "S_std": hsv_std[1], "V_std": hsv_std[2],
        "H_min": hsv_min[0], "S_min": hsv_min[1], "V_min": hsv_min[2],
        "H_max": hsv_max[0], "S_max": hsv_max[1], "V_max": hsv_max[2],
        "saved_crop": badge_filename
    })

cv2.destroyAllWindows()

# === Combine All HSV Samples and Compute Global Range ===
if len(all_hsv_values) == 0:
    print("❌ No valid ROIs selected. Exiting.")
    exit()

all_hsv_values = np.vstack(all_hsv_values)

lower = np.percentile(all_hsv_values, 5, axis=0).astype(int)
upper = np.percentile(all_hsv_values, 95, axis=0).astype(int)

# === Save Evidence ===
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ Brightness-Filtered Calibration Complete!")
print(f"📄 CSV saved at: {OUTPUT_CSV}")
print(f"🖼️ Cropped badge images saved in: {BADGE_SAVE_DIR}")
print("📊 Global HSV Range (computed from *bright* regions only):")
print("Lower HSV:", lower)
print("Upper HSV:", upper)
