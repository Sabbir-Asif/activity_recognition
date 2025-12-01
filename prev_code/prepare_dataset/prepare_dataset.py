import pandas as pd
import numpy as np
import json
from scipy.ndimage import gaussian_filter1d

def create_windowed_dataset(csv_file, duration=2.0, fps=25, overlap=0.5, working_threshold=0.7):
    """
    Create a windowed dataset with features from hand position data
    
    Parameters:
    csv_file: path to CSV file
    duration: window duration in seconds
    fps: frames per second
    overlap: overlap between windows (0 to 1)
    working_threshold: minimum proportion of 1s to be labeled as working (default: 0.7 = 70%)
    """
    
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Calculate window parameters
    window_length = int(duration * fps)
    step_size = int(window_length * (1 - overlap))
    
    # Prepare results
    dataset = []
    
    # Process each window
    for start_idx in range(0, len(df) - window_length + 1, step_size):
        end_idx = start_idx + window_length
        window_data = df.iloc[start_idx:end_idx]
        
        # Extract features for this window
        features = extract_features(window_data, fps)
        
        # Add window metadata
        features['window_start'] = float(df.iloc[start_idx]['second'])
        features['window_end'] = float(df.iloc[end_idx-1]['second'])
        features['window_length'] = window_length
        
        # Determine label using 70% threshold
        labels = window_data['state'].values
        proportion_working = np.mean(labels)
        
        # Label as working (1) if at least 70% of the window is working state
        # Label as not working (0) if less than 70% is working state
        features['label'] = 1 if proportion_working >= working_threshold else 0
        features['proportion_working'] = float(proportion_working)  # For reference
        
        dataset.append(features)
    
    return dataset

def extract_features(window_data, fps):
    """
    Extract features from a window of hand position data
    """
    X = window_data['hand_2_x'].values
    Y = window_data['hand_2_y'].values
    seconds = window_data['second'].values
    
    # Calculate magnitude
    magnitude = np.sqrt(X**2 + Y**2)
    avg_magnitude = float(np.mean(magnitude))
    
    # Calculate velocities using numpy.gradient (better for derivatives)
    x_smooth = gaussian_filter1d(X, sigma=1)
    y_smooth = gaussian_filter1d(Y, sigma=1)
    
    # Use the full seconds array for gradient, not the differences
    dx = np.gradient(x_smooth, seconds)
    dy = np.gradient(y_smooth, seconds)
    velocities = np.sqrt(dx**2 + dy**2)
    avg_velocity = float(np.mean(velocities))
    max_velocity = float(np.max(velocities))
    
    # Frequency domain analysis for hand2_x and hand2_y separately
    dominant_freq_x, amplitude_x = calculate_frequency_features(X, fps)
    dominant_freq_y, amplitude_y = calculate_frequency_features(Y, fps)
    
    # Position statistics
    avg_x = float(np.mean(X))
    avg_y = float(np.mean(Y))
    std_x = float(np.std(X))
    std_y = float(np.std(Y))
    
    # Movement statistics
    distances = np.sqrt(np.diff(X)**2 + np.diff(Y)**2)
    total_distance = float(np.sum(distances))
    
    # Additional kinematic features
    movement_smoothness = calculate_movement_smoothness(velocities)
    
    return {
        # Basic position features
        'average_x': avg_x,
        'average_y': avg_y,
        'std_x': std_x,
        'std_y': std_y,
        'average_magnitude': avg_magnitude,
        
        # Frequency features
        'dominant_frequency_x': dominant_freq_x,
        'amplitude_x': amplitude_x,
        'dominant_frequency_y': dominant_freq_y,
        'amplitude_y': amplitude_y,
        
        # Kinematic features (velocity only)
        'average_velocity': avg_velocity,
        'max_velocity': max_velocity,
        
        # Movement statistics
        'total_distance': total_distance,
        'movement_variability': float(np.std(magnitude)),
        'movement_smoothness': movement_smoothness
    }

def calculate_frequency_features(signal_data, fps):
    """
    Calculate dominant frequency and amplitude using FFT
    Uses your exact method without windowing
    """
    T = 1.0 / fps
    X = np.fft.fft(signal_data)
    N = len(X)
    freqs = np.fft.fftfreq(N, T)
    
    positive_freqs = freqs[:N//2]
    magnitude = np.abs(X[:N//2]) / N
    
    # Find the highest magnitude frequency (excluding index 0 - DC component)
    if len(magnitude) > 1:
        # Start from index 1 to ignore DC component
        max_idx = np.argmax(magnitude[1:]) + 1
        dominant_freq = float(positive_freqs[max_idx])
        amplitude = float(magnitude[max_idx])
    else:
        dominant_freq = 0.0
        amplitude = 0.0
    
    return dominant_freq, amplitude

def calculate_movement_smoothness(velocities):
    """
    Calculate movement smoothness as the inverse of velocity variance
    Higher values indicate smoother movement
    """
    if np.std(velocities) > 0:
        return float(1.0 / np.std(velocities))
    else:
        return 0.0

def save_dataset_to_json(dataset, output_file):
    """
    Save the dataset to JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
        
def extract_features_from_array(xy_array, fps=25):
    """
    Extract features from raw (x,y) array (similar to extract_features but without DataFrame)
    """
    x = xy_array[:, 0]  # First column is x
    y = xy_array[:, 1]  # Second column is y
    
    # Create synthetic time array (assuming constant frame rate)
    seconds = np.arange(len(x)) / fps
    
    # Calculate magnitude
    magnitude = np.sqrt(x**2 + y**2)
    avg_magnitude = float(np.mean(magnitude))
    
    # Calculate velocities using numpy.gradient
    from scipy.ndimage import gaussian_filter1d
    x_smooth = gaussian_filter1d(x, sigma=1)
    y_smooth = gaussian_filter1d(y, sigma=1)
    
    dx = np.gradient(x_smooth, seconds)
    dy = np.gradient(y_smooth, seconds)
    velocities = np.sqrt(dx**2 + dy**2)
    avg_velocity = float(np.mean(velocities))
    max_velocity = float(np.max(velocities))
    
    # Frequency domain analysis
    dominant_freq_x, amplitude_x = calculate_frequency_features(x, fps)
    dominant_freq_y, amplitude_y = calculate_frequency_features(y, fps)
    
    # Position statistics
    avg_x = float(np.mean(x))
    avg_y = float(np.mean(y))
    std_x = float(np.std(x))
    std_y = float(np.std(y))
    
    # Movement statistics
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    total_distance = float(np.sum(distances))
    
    # Additional kinematic features
    movement_smoothness = calculate_movement_smoothness(velocities)
    
    return {
        'average_x': avg_x,
        'average_y': avg_y,
        'std_x': std_x,
        'std_y': std_y,
        'average_magnitude': avg_magnitude,
        'dominant_frequency_x': dominant_freq_x,
        'amplitude_x': amplitude_x,
        'dominant_frequency_y': dominant_freq_y,
        'amplitude_y': amplitude_y,
        'average_velocity': avg_velocity,
        'max_velocity': max_velocity,
        'total_distance': total_distance,
        'movement_variability': float(np.std(magnitude)),
        'movement_smoothness': movement_smoothness
    }

# Main execution
if __name__ == "__main__":
    # Parameters
    csv_file = "../../data/positions/badge_positions_tracked_01 - Sheet3.csv"
    duration = 2.0  # 2-second windows
    fps = 25        # 25 frames per second
    overlap = 0.5   # 50% overlap between windows
    working_threshold = 0.7  # 70% threshold for working label
    
    # Create dataset
    dataset = create_windowed_dataset(csv_file, duration, fps, overlap, working_threshold)
    
    # Save to JSON
    output_file = "hand_position_dataset.json"
    save_dataset_to_json(dataset, output_file)
    
    print(f"Dataset created with {len(dataset)} windows")
    print(f"Working threshold: {working_threshold*100}%")
    print(f"Saved to {output_file}")
    
    # Print sample window
    if dataset:
        print("\nSample window features:")
        for key, value in dataset[0].items():
            print(f"{key}: {value}")
        
        # Print dataset statistics
        working_windows = sum(1 for d in dataset if d['label'] == 1)
        non_working_windows = sum(1 for d in dataset if d['label'] == 0)
        
        print(f"\nDataset statistics:")
        print(f"Total windows: {len(dataset)}")
        print(f"Working windows (label 1): {working_windows} ({working_windows/len(dataset)*100:.1f}%)")
        print(f"Non-working windows (label 0): {non_working_windows} ({non_working_windows/len(dataset)*100:.1f}%)")
        
        # Print distribution of working proportions
        proportions = [d['proportion_working'] for d in dataset]
        print(f"\nWorking proportion statistics:")
        print(f"Min: {min(proportions):.3f}")
        print(f"Max: {max(proportions):.3f}")
        print(f"Mean: {np.mean(proportions):.3f}")
        print(f"Windows with 100% working: {sum(1 for p in proportions if p == 1.0)}")
        print(f"Windows with 0% working: {sum(1 for p in proportions if p == 0.0)}")