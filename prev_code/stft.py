# ============================================
# STFT Analysis on Hand Tracking Data - FIXED
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# --- Step 1: Load Excel data ---
file_path = "badge_positions_tracked_01.xlsx"
df = pd.read_excel(file_path)

# --- Step 2: Extract time and hand_1_x ---
time = df["second"].values
signal = df["hand_1_x"].values

print(f"Loaded {len(signal)} samples")
print(f"Time range: {time[0]:.2f} to {time[-1]:.2f} seconds")
print(f"Signal range: {np.min(signal):.1f} to {np.max(signal):.1f}")
print(df.head())

# --- Step 3: Remove DC offset and detrend the signal ---
signal_detrended = signal - np.mean(signal)
# Alternative: Use linear detrending for stronger trend removal
# signal_detrended = signal - np.polyval(np.polyfit(time, signal, 1), time)

# --- Step 4: Plot both original and detrended signals ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(time, signal, '-o', color='blue', markersize=2)
ax1.set_title("Original Signal: hand_1_x vs Time")
ax1.set_ylabel("Position")
ax1.grid(True)

ax2.plot(time, signal_detrended, '-o', color='red', markersize=2)
ax2.set_title("Detrended Signal (DC offset removed)")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Position")
ax2.grid(True)

plt.tight_layout()
plt.show()

# --- Step 5: Compute STFT with better parameters ---
fs = 1 / np.mean(np.diff(time))   # Sampling frequency
window_duration_sec = 2.0         # 2-second window
nperseg = int(fs * window_duration_sec)

# Ensure window size is reasonable
if nperseg > len(signal_detrended) // 2:
    nperseg = len(signal_detrended) // 2
    print(f"Adjusted window size to {nperseg} samples")

noverlap = int(nperseg * 0.75)    # 75% overlap for better resolution

print(f"\nSampling Frequency: {fs:.2f} Hz")
print(f"Window Size: {nperseg} samples ({window_duration_sec} sec)")
print(f"Overlap: {noverlap} samples")

# Use the DETRENDED signal for STFT
f, t, Zxx = stft(signal_detrended, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                 window='hann', scaling='spectrum')

# Magnitude of STFT
magnitude = np.abs(Zxx)

# --- Step 6: Plot Spectrogram with proper frequency range ---
plt.figure(figsize=(12, 8))

# Limit frequency axis to meaningful range (0-2 Hz for human motion)
f_max = 2.0  # Maximum frequency to display
f_mask = f <= f_max

plt.pcolormesh(t, f[f_mask], magnitude[f_mask, :], shading='gouraud', cmap='viridis')
plt.title(f"STFT Spectrogram of hand_1_x ({window_duration_sec}-second window)")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.colorbar(label="Magnitude")
plt.ylim(0, f_max)
plt.grid(True, alpha=0.3)
plt.show()

# --- Step 7: Find dominant frequency for each time window (excluding very low frequencies) ---
# Create a copy of magnitude and set very low frequencies to 0 to avoid DC dominance
magnitude_processed = magnitude.copy()
dc_cutoff = 0.1  # Ignore frequencies below 0.1 Hz
dc_mask = f < dc_cutoff
magnitude_processed[dc_mask, :] = 0

# Find dominant frequency for each time window
dominant_freqs = f[np.argmax(magnitude_processed, axis=0)]
dominant_magnitudes = np.max(magnitude_processed, axis=0)

# --- Step 8: Plot dominant frequency over time ---
plt.figure(figsize=(12, 6))

# Plot 1: Dominant frequency
plt.subplot(2, 1, 1)
plt.plot(t, dominant_freqs, 'ro-', markersize=4, linewidth=2)
plt.title("Dominant Frequency Over Time (excluding DC)")
plt.ylabel("Frequency (Hz)")
plt.grid(True)

# Plot 2: Corresponding magnitude
plt.subplot(2, 1, 2)
plt.plot(t, dominant_magnitudes, 'go-', markersize=4, linewidth=2)
plt.title("Magnitude of Dominant Frequency")
plt.xlabel("Time (seconds)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Step 9: Build comprehensive dataset ---
dataset = pd.DataFrame({
    "time_window_center": t,
    "dominant_frequency_Hz": dominant_freqs,
    "dominant_magnitude": dominant_magnitudes,
    "mean_amplitude": magnitude.mean(axis=0),
    "max_amplitude": magnitude.max(axis=0)
})

print("\n=== Frequency Dataset (first few rows) ===")
print(dataset.head(10))

# --- Step 10: Frequency statistics ---
valid_freqs = dominant_freqs[dominant_freqs > dc_cutoff]
if len(valid_freqs) > 0:
    print(f"\n=== Frequency Statistics (excluding DC) ===")
    print(f"Mean dominant frequency: {np.mean(valid_freqs):.3f} Hz")
    print(f"Std of dominant frequency: {np.std(valid_freqs):.3f} Hz")
    print(f"Min dominant frequency: {np.min(valid_freqs):.3f} Hz")
    print(f"Max dominant frequency: {np.max(valid_freqs):.3f} Hz")
    print(f"Median dominant frequency: {np.median(valid_freqs):.3f} Hz")
    
    # Period in seconds
    periods = 1 / valid_freqs
    print(f"\nCorresponding periods: {np.mean(periods):.2f} ± {np.std(periods):.2f} seconds")

# --- Step 11: Save the dataset ---
output_path = "hand1x_stft_dataset_improved.csv"
dataset.to_csv(output_path, index=False)
print(f"\n✅ Dataset saved to {output_path}")

# --- BONUS: Plot all frequency components over time ---
plt.figure(figsize=(14, 10))

# Plot top 5 frequency components over time
for i in range(min(5, len(f))):
    plt.plot(t, magnitude[i, :], label=f'{f[i]:.2f} Hz', linewidth=2)

plt.title("Top Frequency Components Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()

print(f"\n=== Analysis Complete ===")
print(f"Total time windows analyzed: {len(t)}")
print(f"Time resolution: {t[1]-t[0]:.2f} seconds between windows")
print(f"Frequency resolution: {f[1]-f[0]:.3f} Hz")