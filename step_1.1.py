import cv2
import numpy as np
import pandas as pd
import os

# === Configuration ===
image_path = "badge_samples/badge_frame_2556.jpg"  # Replace with your image
output_csv = "hsv_values.csv"

# === Load Image and Convert to HSV ===
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# === Reshape HSV to list of pixels ===
hsv_array = hsv.reshape(-1, 3)  # shape: (num_pixels, 3)

# === Create DataFrame of all pixels ===
df = pd.DataFrame(hsv_array, columns=['H', 'S', 'V'])

# === Compute Summary Statistics ===
summary = {
    'H_mean': [df['H'].mean()],
    'S_mean': [df['S'].mean()],
    'V_mean': [df['V'].mean()],
    'H_std':  [df['H'].std()],
    'S_std':  [df['S'].std()],
    'V_std':  [df['V'].std()],
    'H_min':  [df['H'].min()],
    'S_min':  [df['S'].min()],
    'V_min':  [df['V'].min()],
    'H_max':  [df['H'].max()],
    'S_max':  [df['S'].max()],
    'V_max':  [df['V'].max()]
}
df_summary = pd.DataFrame(summary)

# === Save both full HSV values and summary in CSV ===
# Option 1: Save HSV pixels first, summary at the end with empty row in between
with open(output_csv, 'w') as f:
    df.to_csv(f, index=False)
    f.write("\n")  # empty row
    f.write("# Summary Statistics\n")
    df_summary.to_csv(f, index=False)

print(f"✅ HSV pixel values and summary saved to: {output_csv}")
