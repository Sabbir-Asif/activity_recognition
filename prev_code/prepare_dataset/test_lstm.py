import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sys
sys.path.append('src/prepare_dataset')
from prepare_dataset import extract_features

def test_lstm_correctly(csv_file, model_path="lstm_hand_movement_classifier.h5", confidence_threshold=0.8):
    """
    CORRECT way to test LSTM model - matches training data processing
    """
    # Load model
    model = load_model(model_path)
    
    # Load CSV and process EXACTLY like training data
    df = pd.read_csv(csv_file)
    
    # Parameters must match training
    window_size = 50  # 2 seconds at 25fps
    step_size = 25    # 50% overlap
    sequence_length = 5  # Must match training sequence_length
    
    # First: Create feature windows (same as training)
    feature_windows = []
    labels = []
    
    for i in range(0, len(df) - window_size + 1, step_size):
        window_data = df.iloc[i:i+window_size]
        
        # Extract features EXACTLY like during training
        features = extract_features(window_data, fps=25)
        
        # Convert to feature vector in correct order
        feature_columns = [
            'average_x', 'average_y', 'std_x', 'std_y', 'average_magnitude',
            'dominant_frequency_x', 'amplitude_x', 'dominant_frequency_y', 'amplitude_y',
            'average_velocity', 'max_velocity', 'total_distance', 
            'movement_variability', 'movement_smoothness'
        ]
        
        feature_vector = [features[col] for col in feature_columns]
        feature_windows.append(feature_vector)
        
        # Get label (for reference)
        window_labels = window_data['state'].values
        proportion_working = np.mean(window_labels)
        label = 1 if proportion_working >= 0.7 else 0
        labels.append(label)
    
    feature_windows = np.array(feature_windows)
    labels = np.array(labels)
    
    print(f"Created {len(feature_windows)} feature windows")
    print(f"True working windows: {sum(labels == 1)}")
    print(f"True not working windows: {sum(labels == 0)}")
    
    # Second: Create sequences for LSTM (same as training)
    sequences = []
    sequence_labels = []
    
    for i in range(len(feature_windows) - sequence_length + 1):
        sequence = feature_windows[i:i+sequence_length]
        sequences.append(sequence)
        # Use label from last window in sequence (same as training)
        sequence_labels.append(labels[i+sequence_length-1])
    
    sequences = np.array(sequences)
    sequence_labels = np.array(sequence_labels)
    
    print(f"Created {len(sequences)} sequences for LSTM")
    
    # Scale features (you should use the same scaler from training)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Fit scaler on all feature data (in practice, use the trained scaler)
    all_features = feature_windows.reshape(-1, feature_windows.shape[-1])
    scaler.fit(all_features)
    
    # Scale sequences
    sequences_scaled = []
    for seq in sequences:
        seq_scaled = scaler.transform(seq)
        sequences_scaled.append(seq_scaled)
    
    sequences_scaled = np.array(sequences_scaled)
    
    # Make predictions
    predictions_proba = model.predict(sequences_scaled, verbose=0)
    predictions = np.argmax(predictions_proba, axis=1)
    confidences = np.max(predictions_proba, axis=1)
    
    # Apply confidence threshold
    final_classifications = []
    for pred, conf in zip(predictions, confidences):
        if conf > confidence_threshold and pred == 1:
            final_classifications.append(1)  # Working
        else:
            final_classifications.append(0)  # Not Working
    
    final_classifications = np.array(final_classifications)
    
    # Calculate REAL accuracy
    accuracy = np.mean(final_classifications == sequence_labels)
    print(f"\nREAL Accuracy (vs true labels): {accuracy:.4f}")
    
    # Plot the timeline graph with hand positions colored by predictions
    plot_timeline_with_predictions(df, final_classifications, confidences, 
                                  window_size, step_size, sequence_length, confidence_threshold)
    
    return final_classifications, confidences, sequence_labels

def plot_timeline_with_predictions(df, predictions, confidences, window_size, step_size, 
                                  sequence_length, confidence_threshold):
    """
    Plot hand positions over time colored by working/non-working predictions
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    
    time = df['second'].values if 'second' in df.columns else np.arange(len(df))
    x_positions = df['hand_2_x'].values
    y_positions = df['hand_2_y'].values
    
    # Create a color array for each time point based on predictions
    point_colors = ['gray'] * len(df)  # Initialize with gray
    point_alphas = [0.3] * len(df)     # Initialize with low alpha
    
    # Color the points based on which window they belong to
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        # Calculate the time range for this sequence's prediction
        # Each sequence corresponds to a specific time window
        start_idx = i * step_size
        end_idx = start_idx + window_size + (sequence_length - 1) * step_size
        end_idx = min(end_idx, len(df))  # Don't exceed data length
        
        # Determine color based on prediction
        color = 'green' if pred == 1 else 'red'
        alpha = min(0.8, 0.3 + conf * 0.5)  # Alpha based on confidence
        
        # Color all points in this time window
        for j in range(start_idx, end_idx):
            point_colors[j] = color
            point_alphas[j] = alpha
    
    # Plot 1: X and Y positions over time with colored background
    for i in range(len(predictions)):
        start_idx = i * step_size
        end_idx = start_idx + window_size + (sequence_length - 1) * step_size
        end_idx = min(end_idx, len(df))
        
        color = 'green' if predictions[i] == 1 else 'red'
        alpha = min(0.2, confidences[i] * 0.3)
        
        ax1.axvspan(time[start_idx], time[end_idx-1], alpha=alpha, color=color)
    
    ax1.plot(time, x_positions, 'b-', linewidth=1, label='Hand X', alpha=0.8)
    ax1.plot(time, y_positions, 'r-', linewidth=1, label='Hand Y', alpha=0.8)
    ax1.set_ylabel('Position (pixels)')
    ax1.set_title(f'Hand Positions Over Time with LSTM Classification\n(Green: Working, Red: Not Working | Threshold: {confidence_threshold})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of positions colored by classification
    scatter = ax2.scatter(time, x_positions, c=point_colors, alpha=point_alphas, 
                         s=30, label='X position')
    ax2.scatter(time, y_positions, c=point_colors, alpha=point_alphas, 
               s=30, marker='x', label='Y position')
    ax2.set_ylabel('Position (pixels)')
    ax2.set_title('Hand Positions Colored by Classification')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Create custom legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Working'),
        Patch(facecolor='red', alpha=0.7, label='Not Working'),
        Patch(facecolor='gray', alpha=0.3, label='Unclassified')
    ]
    ax2.legend(handles=legend_elements)
    
    # Plot 3: Confidence over time with classification regions
    confidence_times = []
    confidence_values = []
    confidence_colors = []
    
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        start_idx = i * step_size
        end_idx = start_idx + window_size + (sequence_length - 1) * step_size
        end_idx = min(end_idx, len(df))
        
        mid_time = (time[start_idx] + time[end_idx-1]) / 2
        confidence_times.append(mid_time)
        confidence_values.append(conf)
        confidence_colors.append('green' if pred == 1 else 'red')
    
    # Plot confidence bars
    bars = ax3.bar(confidence_times, confidence_values, 
                  width=[(time[min(i*step_size+window_size+(sequence_length-1)*step_size, len(time)-1)] - time[i*step_size]) for i in range(len(predictions))],
                  color=confidence_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add threshold line
    ax3.axhline(y=confidence_threshold, color='blue', linestyle='--', linewidth=2,
                label=f'Confidence Threshold ({confidence_threshold})')
    
    # Fill working region
    ax3.axhspan(confidence_threshold, 1.0, alpha=0.1, color='green', label='Working Region')
    
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Confidence')
    ax3.set_title('Classification Confidence Over Time')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Also create a separate detailed trajectory plot
    plot_colored_trajectory(df, predictions, confidences, window_size, step_size, 
                           sequence_length, confidence_threshold)

def plot_colored_trajectory(df, predictions, confidences, window_size, step_size,
                          sequence_length, confidence_threshold):
    """
    Plot hand movement trajectory colored by classification
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_positions = df['hand_2_x'].values
    y_positions = df['hand_2_y'].values
    
    # Plot the overall trajectory in light gray
    ax.plot(x_positions, y_positions, 'gray', alpha=0.2, linewidth=1, label='Overall Trajectory')
    
    # Plot colored segments based on classification
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        end_idx = min(end_idx, len(df))
        
        # Get the segment for this window
        segment_x = x_positions[start_idx:end_idx]
        segment_y = y_positions[start_idx:end_idx]
        
        # Determine color and style
        color = 'green' if pred == 1 else 'red'
        linewidth = 1 + (conf * 2)  # Thicker lines for higher confidence
        alpha = min(0.8, 0.3 + conf * 0.5)
        
        # Plot this segment
        ax.plot(segment_x, segment_y, color=color, linewidth=linewidth, alpha=alpha,
                label=f"{'Working' if pred == 1 else 'Not Working'}" if i == 0 else "")
        
        # Add confidence text for high confidence segments
        if conf > 0.7:
            center_x = np.mean(segment_x)
            center_y = np.mean(segment_y)
            ax.text(center_x, center_y, f'{conf:.2f}', fontsize=8,
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(f'Hand Movement Trajectory Colored by LSTM Classification\n(Threshold: {confidence_threshold})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    plt.show()

def print_summary(predictions, true_labels, confidences, confidence_threshold):
    """
    Print performance summary
    """
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    accuracy = np.mean(predictions == true_labels)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Working predictions: {sum(predictions == 1)}/{len(predictions)}")
    print(f"Not working predictions: {sum(predictions == 0)}/{len(predictions)}")
    print(f"Average confidence: {np.mean(confidences):.4f}")

# Usage
if __name__ == "__main__":
    csv_path = "../../data/positions/badge_positions_tracked_01 - Sheet3.csv"
    confidence_threshold = 0.8
    
    print(f"Testing LSTM model with timeline visualization...")
    predictions, confidences, true_labels = test_lstm_correctly(csv_path, confidence_threshold=confidence_threshold)