import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Step 1: Load Image ---
image_path = '../data/new_video_1.png'  # Change path as needed
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

print("Original image shape:", image.shape)

# --- Step 2: Convert to RGB for display ---
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(6, 6))
plt.title("Original Image (RGB)")
plt.imshow(image_rgb)
plt.axis("off")
plt.show()

# --- Step 3: Convert to HSV ---
hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
plt.figure(figsize=(6, 6))
plt.title("HSV Image")
plt.imshow(hsv)
plt.axis("off")
plt.show()

# --- Step 4: Define glove color range (tune if needed) ---
# These HSV ranges depend on lighting & glove color
lower_glove = np.array([30, 80, 80])
upper_glove = np.array([80, 255, 255])

# --- Step 5: Create mask ---
mask = cv2.inRange(hsv, lower_glove, upper_glove)

# --- Step 6: Remove noise using morphological operations ---
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

plt.figure(figsize=(6, 6))
plt.title("Glove Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")
plt.show()

# --- Step 7: Find contours ---
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours detected: {len(contours)}")

# --- Step 8: Process contours ---
centroids = []
for contour in contours:
    if cv2.contourArea(contour) > 1: # ignore small noise
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image_rgb, "Glove", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
            cv2.circle(image_rgb, (cx, cy), 5, (0, 255, 0), -1)

print("Centroids of detected gloves:", centroids)

# --- Step 9: Display final result ---
plt.figure(figsize=(6, 6))
plt.title("Detected Gloves with Centroids")
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
