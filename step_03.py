import cv2
import numpy as np

# === Input Configuration ===
image_path = "sample_frames/frame_0000.jpg"

# === HSV Range (Replace with your computed values) ===
# Example values — replace with your actual range from CSV computation
lower = np.array([14, 64, 102])    # <-- Your lower bound
upper = np.array([44, 138, 237])  # <-- Your upper bound

# === Load Image ===
frame = cv2.imread(image_path)
frame = cv2.resize(frame, (800, 600))

# === Convert to HSV ===
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# === Create Mask ===
mask = cv2.inRange(hsv, lower, upper)

# === Morphological Cleanup ===
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# === Find Contours ===
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === Draw Detections ===
output = frame.copy()
badge_positions = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:  # minimum area filter to remove noise
        x, y, w, h = cv2.boundingRect(cnt)
        cx = int(x + w/2)
        cy = int(y + h/2)
        badge_positions.append((cx, cy))

        # Draw rectangle and center
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(output, f"({cx},{cy})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# === Show Results ===
stacked = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), output))
cv2.imshow("Badge Detection (Original | Mask | Result)", stacked)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === Print Badge Positions ===
print("Detected badge positions (centroids):", badge_positions)
