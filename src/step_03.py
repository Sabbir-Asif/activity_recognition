import cv2
import numpy as np
from pathlib import Path
import sys

# === Configuration ===
output_dir = "../badge_detection"

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Badge color range (from step_02 calibration) - PERMISSIVE RANGE
# This range accounts for badges at different distances and lighting conditions
# Hue: 35-45 (wide range for color variation)
# Saturation: 58-88 (accounts for dim distant badges with less saturation)
# Value: 219-255 (catches both bright close and dimmer distant badges)
lower_badge = np.array([35, 58, 219])
upper_badge = np.array([45, 88, 255])

# === Image Path (UPDATE THIS) ===
image_path = "../data/image.png"

if not Path(image_path).exists():
    print(f"❌ Image not found: {image_path}")
    exit(1)

print(f"✅ Loading image: {image_path}\n")

# === Load and Process Image ===
frame = cv2.imread(image_path)

if frame is None:
    print(f"❌ Failed to load image: {image_path}")
    exit(1)

frame_name = Path(image_path).stem
print(f"Processing: {frame_name}")

# Apply exact filtering from step_01
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

# Step 6: Convert to HSV
hsv = cv2.cvtColor(bgr_gamma, cv2.COLOR_BGR2HSV)

# Create mask using badge color range
mask = cv2.inRange(hsv, lower_badge, upper_badge)

# Better morphological operations
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Remove small noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
# Fill small holes
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contours detected: {len(contours)}")

# Process contours and draw
bgr_display = bgr.copy()
badges_detected = []

for contour in contours:
    area = cv2.contourArea(contour)
    
    # Filter by area (ignore small noise but allow smaller badges in distance)
    if area > 20:  # Lower threshold to catch distant badges
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            badges_detected.append({
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'area': area
            })
            
            # Draw bounding box (green)
            cv2.rectangle(bgr_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center (red)
            cv2.circle(bgr_display, (cx, cy), 5, (0, 0, 255), -1)
            
            # Add label
            cv2.putText(bgr_display, f"Badge", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"  Badge detected at center: ({cx}, {cy}), area: {area}")

# Save result
output_path = Path(output_dir) / f"detected_{frame_name}.jpg"
cv2.imwrite(str(output_path), bgr_display)
print(f"\n✅ Saved: {output_path}")

if len(badges_detected) > 0:
    print(f"\n📊 Total badges detected: {len(badges_detected)}")
    for i, badge in enumerate(badges_detected):
        print(f"  Badge {i+1}: center={badge['center']}, bbox={badge['bbox']}, area={badge['area']}")
else:
    print("\n⚠️ No badges detected in the image")
