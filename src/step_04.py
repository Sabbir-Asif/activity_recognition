import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path

# === Configuration ===
video_path = "../data/working_1.mp4"         # Input video file
output_video_path = "../working_1_output_with_badges.mp4"
output_excel_path = "../working_1_badge_positions_tracked.xlsx"
resize_dim = (800, 600)
max_distance = 150
merge_distance = 80  # Distance threshold to merge IDs

# Badge color range (from step_02 calibration) - PERMISSIVE RANGE
lower_badge = np.array([35, 58, 219])
upper_badge = np.array([45, 255, 255])

# === PREPROCESSING STEPS FROM STEP_01 ===
def preprocess_frame(bgr_frame):
    """Apply the same preprocessing as step_01"""
    # Step 2: Convert to LAB
    lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
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
    
    return hsv

# === Open Video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎥 Loaded: {video_path} | FPS: {fps} | Total frames: {total_frames}")

# === Prepare Writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, resize_dim)

# === Tracking State ===
hand_positions = {}  # id: (last_x, last_y)
next_id = 1

# === Storage for badge positions ===
records = []

def merge_close_hands(hand_positions, merge_distance):
    """
    Merge hands that are too close to each other, keeping the lower ID
    """
    if len(hand_positions) < 2:
        return hand_positions
    
    hand_ids = sorted(hand_positions.keys())
    merged_positions = hand_positions.copy()
    to_remove = set()
    
    for i in range(len(hand_ids)):
        id1 = hand_ids[i]
        if id1 in to_remove:
            continue
            
        x1, y1 = merged_positions[id1]
        
        for j in range(i + 1, len(hand_ids)):
            id2 = hand_ids[j]
            if id2 in to_remove:
                continue
                
            x2, y2 = merged_positions[id2]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            # If hands are too close, merge them (keep the lower ID)
            if distance < merge_distance:
                print(f"🔄 Merging ID {id2} into ID {id1} (distance: {distance:.1f})")
                to_remove.add(id2)
                # Update the position of the kept ID to the average position
                avg_x = int((x1 + x2) / 2)
                avg_y = int((y1 + y2) / 2)
                merged_positions[id1] = (avg_x, avg_y)
    
    # Remove merged IDs
    for id_to_remove in to_remove:
        del merged_positions[id_to_remove]
    
    return merged_positions

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    time_sec = round(frame_idx / fps, 2)

    # === Resize ===
    frame = cv2.resize(frame, resize_dim)
    
    # === NEW: Apply preprocessing from step_01 ===
    hsv = preprocess_frame(frame)

    # === NEW: Create mask using badge color range ===
    mask = cv2.inRange(hsv, lower_badge, upper_badge)

    # === Better morphological operations ===
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # === Find Contours ===
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_centroids = []
    for contour in contours:
        if cv2.contourArea(contour) > 5:  # Lower threshold for resized frames (catches distant badges)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                current_centroids.append((cx, cy))

    # === Track hands and assign IDs (Original Logic) ===
    used_ids = set()
    updated_hand_positions = {}
    
    # For each detected centroid in current frame, find closest existing hand
    for cx, cy in current_centroids:
        closest_id = None
        min_distance = max_distance
        
        # Find closest existing hand
        for hand_id, (prev_x, prev_y) in hand_positions.items():
            distance = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
            if distance < min_distance and hand_id not in used_ids:
                min_distance = distance
                closest_id = hand_id
        
        if closest_id is not None:
            # Assign to existing hand
            updated_hand_positions[closest_id] = (cx, cy)
            used_ids.add(closest_id)
        else:
            # Assign new ID
            updated_hand_positions[next_id] = (cx, cy)
            used_ids.add(next_id)
            next_id += 1
    
    # Keep hands that weren't detected in this frame (maintain their last positions)
    for hand_id, position in hand_positions.items():
        if hand_id not in used_ids:
            updated_hand_positions[hand_id] = position
    
    # === NEW: Merge close hands to prevent duplicate IDs ===
    hand_positions = merge_close_hands(updated_hand_positions, merge_distance)

    # === Draw bounding boxes and IDs ===
    for hand_id, (cx, cy) in hand_positions.items():
        # Find the contour that corresponds to this centroid
        contour_found = False
        for contour in contours:
            if cv2.contourArea(contour) > 5:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    contour_cx = int(M['m10'] / M['m00'])
                    contour_cy = int(M['m01'] / M['m00'])
                    
                    # If this contour's centroid matches our tracked hand
                    if abs(contour_cx - cx) < 20 and abs(contour_cy - cy) < 20:  # Increased tolerance
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, f"Hand {hand_id}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                        contour_found = True
                        break
        
        # If no contour found, still draw the centroid and ID
        if not contour_found:
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # Yellow for predicted positions
            cv2.putText(frame, f"Hand {hand_id}", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # === Write frame with boxes to output video ===
    out.write(frame)

    # === Record Data ===
    record = {"second": time_sec}
    for hand_id, (cx, cy) in hand_positions.items():
        record[f"hand_{hand_id}_x"] = cx
        record[f"hand_{hand_id}_y"] = cy
    records.append(record)

    if frame_idx % int(fps) == 0:
        detected_count = sum(1 for (cx, cy) in current_centroids)
        tracked_count = len(hand_positions)
        merge_count = len(updated_hand_positions) - len(hand_positions)  # How many were merged
        
        status_msg = f"⏱ Processed {frame_idx}/{total_frames} frames ({time_sec}s) | "
        status_msg += f"Detected: {detected_count} | Tracking: {tracked_count} badges"
        if merge_count > 0:
            status_msg += f" | Merged: {merge_count}"
        status_msg += f" | IDs: {list(hand_positions.keys())}"
        
        print(status_msg)

cap.release()
out.release()
cv2.destroyAllWindows()

# === Save Detected Coordinates to Excel ===
df = pd.DataFrame(records)

# Reorder columns for better readability
base_columns = ['second']
hand_columns = [col for col in df.columns if col.startswith('hand_')]
# Sort by hand ID
hand_columns_sorted = sorted(hand_columns, key=lambda x: int(x.split('_')[1]))

df = df[base_columns + hand_columns_sorted]
df.to_excel(output_excel_path, index=False)

print("\n✅ Done!")
print(f"📹 Output video saved at: {output_video_path}")
print(f"📊 Excel with hand coordinates saved at: {output_excel_path}")
print(f"🔢 Total unique hands tracked: {len(hand_positions)}")
print(f"📈 Maximum simultaneous hands: {max(len(record) - 1 for record in records)}")  # -1 for 'second' column
