import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path

# === Configuration ===
frame_dir = "../sample_badge"
badge_save_dir = "../badge_samples"
output_csv = "../badge_hsv_data.csv"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# === Prepare Directories ===
os.makedirs(badge_save_dir, exist_ok=True)

# === Load Frames ===
frames = sorted([f for f in Path(frame_dir).glob("*.jpg")])

if not frames:
    print("❌ No frames found!")
    exit()

print(f"✅ Found {len(frames)} frames")

# === Global Variables ===
drawing = False
start_point = None
end_point = None
current_image_orig = None
current_image_display = None
hsv_image = None
bgr_image = None
scale_factor = 1.0
current_frame_idx = 0

# To collect all HSV pixels
all_hsv_values = []
records = []

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point, current_image_orig, current_image_display, hsv_image, bgr_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        print(f"ROI start: ({x}, {y})")
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_image_display = current_image_orig.copy()
            cv2.rectangle(current_image_display, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Badge ROI Selection", current_image_display)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        print(f"ROI end: ({x}, {y})")
        
        process_roi(x, y)

def process_roi(end_x, end_y):
    global current_image_orig, bgr_image, current_frame_idx, all_hsv_values, records, badge_save_dir, frames
    
    # Extract ROI coordinates
    x1, y1 = min(start_point[0], end_x), min(start_point[1], end_y)
    x2, y2 = max(start_point[0], end_x), max(start_point[1], end_y)
    w, h = x2 - x1, y2 - y1
    
    # Validate ROI
    if w < 5 or h < 5:
        print("⚠️ ROI too small, skipping.")
        return
    
    # Extract badge region from BGR
    badge_region_bgr = bgr_image[y1:y2, x1:x2]
    
    # Apply the exact same filtering as step_01
    # Step 2: Convert to LAB
    lab = cv2.cvtColor(badge_region_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Step 3: CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    # Step 4: Convert LAB → BGR
    lab_processed = cv2.merge([l_channel, a_channel, b_channel])
    bgr_processed = cv2.cvtColor(lab_processed, cv2.COLOR_LAB2BGR)
    
    # Step 5: Apply Gamma Correction
    inv_gamma = 1.0 / 1.2
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    bgr_gamma = cv2.LUT(bgr_processed, table)
    
    # Step 6: Convert to HSV
    badge_region_hsv = cv2.cvtColor(bgr_gamma, cv2.COLOR_BGR2HSV)
    
    # Save cropped badge
    frame_name = frames[current_frame_idx].name.replace('.jpg', '')
    badge_filename = os.path.join(badge_save_dir, f"badge_{frame_name}.jpg")
    cv2.imwrite(badge_filename, badge_region_bgr)
    print(f"💾 Saved badge crop: {badge_filename}")
    
    # Split HSV channels
    h, s, v = cv2.split(badge_region_hsv)
    
    # --- Brightness-based thresholding (ROBUST) ---
    _, bright_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply mask to HSV pixels
    bright_pixels = badge_region_hsv[bright_mask > 0]
    
    if len(bright_pixels) == 0:
        print(f"⚠️ No bright pixels found in this ROI, skipping.")
        return
    
    all_hsv_values.append(bright_pixels)
    
    # Compute stats
    hsv_mean = np.mean(bright_pixels, axis=0)
    hsv_std = np.std(bright_pixels, axis=0)
    hsv_min = np.min(bright_pixels, axis=0)
    hsv_max = np.max(bright_pixels, axis=0)
    
    records.append({
        "frame": frame_name,
        "x": x1, "y": y1, "width": w, "height": h,
        "H_mean": hsv_mean[0], "S_mean": hsv_mean[1], "V_mean": hsv_mean[2],
        "H_std": hsv_std[0], "S_std": hsv_std[1], "V_std": hsv_std[2],
        "H_min": hsv_min[0], "S_min": hsv_min[1], "V_min": hsv_min[2],
        "H_max": hsv_max[0], "S_max": hsv_max[1], "V_max": hsv_max[2],
        "saved_crop": badge_filename
    })
    
    print(f"\n📊 ROI Color Statistics (from bright pixels):")
    print(f"  H range: {hsv_min[0]} - {hsv_max[0]} (mean: {hsv_mean[0]:.1f})")
    print(f"  S range: {hsv_min[1]} - {hsv_max[1]} (mean: {hsv_mean[1]:.1f})")
    print(f"  V range: {hsv_min[2]} - {hsv_max[2]} (mean: {hsv_mean[2]:.1f})")

def process_frame(frame_idx):
    global current_image_orig, current_image_display, hsv_image, bgr_image, drawing, start_point, end_point, scale_factor, current_frame_idx
    
    if frame_idx >= len(frames):
        return False
    
    current_frame_idx = frame_idx
    frame_path = frames[frame_idx]
    bgr_image = cv2.imread(str(frame_path))
    current_image_orig = bgr_image.copy()
    
    if current_image_orig is None:
        print(f"❌ Failed to load frame: {frame_path}")
        return False
    
    drawing = False
    start_point = None
    end_point = None
    
    print(f"\n{'='*60}")
    print(f"Frame {frame_idx + 1}/{len(frames)}: {frame_path.name}")
    print(f"{'='*60}")
    print("📌 Draw ROI around the badge (click and drag)")
    print("📌 Press SPACE to continue to next frame")
    
    # Display original image
    cv2.imshow("Badge ROI Selection", current_image_orig)
    
    # Wait for ROI or space key
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space
            break
        elif key == 27:  # ESC
            return None
    
    return True

# === Main Loop - Process All Frames ===
print("\n🚀 Starting automatic ROI selection for all frames...\n")

# Create window first
cv2.namedWindow("Badge ROI Selection", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Badge ROI Selection", mouse_callback)

for frame_idx in range(len(frames)):
    result = process_frame(frame_idx)
    if result is None:
        break
    elif result:
        continue

cv2.destroyAllWindows()

# === Compute Global Range ===
if len(all_hsv_values) == 0:
    print("❌ No valid ROIs selected. Exiting.")
    exit()

all_hsv_values = np.vstack(all_hsv_values)

# Statistical calculations
lower_percentile = np.percentile(all_hsv_values, 5, axis=0).astype(int)
upper_percentile = np.percentile(all_hsv_values, 95, axis=0).astype(int)
mean = np.mean(all_hsv_values, axis=0).astype(int)
std = np.std(all_hsv_values, axis=0).astype(int)

# === IMPROVED RANGE CALCULATION ===
# Instead of using strict percentiles, use mean ± 2*std for robustness
# This accounts for distance variations and lighting changes
lower_robust = np.maximum(0, mean - 2 * std).astype(int)
upper_robust = np.minimum([180, 255, 255], mean + 2 * std).astype(int)

# Further expand to be permissive for different distances
# Hue: ±5 from mean
# Saturation: ±15 from mean (accounts for distant badges with less saturation)
# Value: mean-std to 255 (catches dim distant badges)
lower_permissive = np.array([
    max(0, mean[0] - 5),           # Hue: ±5
    max(0, mean[1] - 15),          # Sat: ±15 (more permissive)
    max(0, mean[2] - std[2])       # Val: mean - std (catch dim badges)
]).astype(int)

upper_permissive = np.array([
    min(180, mean[0] + 5),         # Hue: ±5
    min(255, mean[1] + 15),        # Sat: ±15
    255                             # Val: full brightness range
]).astype(int)

# === Save Results ===
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)

print("\n" + "="*60)
print("✅ Badge Calibration Complete!")
print("="*60)
print(f"📄 CSV saved at: {output_csv}")
print(f"🖼️ Cropped badge images saved in: {badge_save_dir}")
print(f"📊 Total badges analyzed: {len(records)}")
print(f"📊 Total bright pixels: {len(all_hsv_values)}")

print("\n📈 Statistical Analysis:")
print(f"Mean HSV:                    {mean}")
print(f"Std Dev HSV:                 {std}")
print(f"\nPercentile Range (5-95):")
print(f"Lower: {lower_percentile}")
print(f"Upper: {upper_percentile}")

print(f"\n🎯 ROBUST Range (mean ± 2*std):")
print(f"Lower: {lower_robust}")
print(f"Upper: {upper_robust}")

print(f"\n🎯 PERMISSIVE Range (for badges at different distances):")
print(f"Lower: {lower_permissive}")
print(f"Upper: {upper_permissive}")

print("\n💡 RECOMMENDED for step_03.py (use PERMISSIVE range):")
print(f"lower = np.array({list(lower_permissive)})")
print(f"upper = np.array({list(upper_permissive)})")
print(f"mask = cv2.inRange(hsv, lower, upper)")
