import numpy as np
import pandas as pd
from dtaidistance import dtw
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import joblib
import os

def create_sequences_from_csv(csv_file, window_size=50, step_size=25):
    """Create sequences from your hand position CSV data"""
    df = pd.read_csv(csv_file)
    
    sequences = []
    labels = []
    
    for i in range(0, len(df) - window_size + 1, step_size):
        window_data = df.iloc[i:i+window_size]
        
        # Create sequence of (x,y) coordinates
        sequence = window_data[['hand_2_x', 'hand_2_y']].values
        sequences.append(sequence)
        
        # Label (using 70% threshold)
        window_labels = window_data['state'].values
        proportion_working = np.mean(window_labels)
        label = 1 if proportion_working >= 0.7 else 0
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

def extract_dtw_features(sequences, labels, n_reference_patterns=5):
    """Extract DTW-based features for neural network"""
    
    # Separate sequences by class
    working_seqs = sequences[labels == 1]
    not_working_seqs = sequences[labels == 0]
    
    # Select reference patterns (centroids or diverse samples)
    ref_working = working_seqs[:n_reference_patterns]
    ref_not_working = not_working_seqs[:n_reference_patterns]
    
    features = []
    
    for seq in sequences:
        feature_vector = []
        
        # DTW distances to working reference patterns
        for ref_seq in ref_working:
            distance = dtw.distance(seq.flatten(), ref_seq.flatten())
            feature_vector.append(distance)
        
        # DTW distances to not-working reference patterns
        for ref_seq in ref_not_working:
            distance = dtw.distance(seq.flatten(), ref_seq.flatten())
            feature_vector.append(distance)
        
        # Statistical features
        feature_vector.append(np.mean(feature_vector))  # Mean distance
        feature_vector.append(np.std(feature_vector))   # Distance variability
        feature_vector.append(np.min(feature_vector))   # Min distance to any pattern
        
        features.append(feature_vector)
    
    return np.array(features), ref_working, ref_not_working

def build_dtw_classifier(input_dim):
    """Build the neural network classifier for DTW features"""
    model = Sequential([
        # Input: DTW distance features
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.1),
        
        # Output: Working (1) vs Not Working (0)
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model_artifacts(model, scaler, ref_working, ref_not_working, model_name="dtw_nn_classifier"):
    """
    Save all model artifacts for future use
    """
    # Create models directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    # Save the neural network model
    model.save(f"saved_models/{model_name}.h5")
    
    # Save the scaler
    joblib.dump(scaler, f"saved_models/{model_name}_scaler.pkl")
    
    # Save reference patterns
    np.save(f"saved_models/{model_name}_ref_working.npy", ref_working)
    np.save(f"saved_models/{model_name}_ref_not_working.npy", ref_not_working)
    
    print(f"✓ Model artifacts saved in 'saved_models' directory:")
    print(f"  - {model_name}.h5 (Neural Network)")
    print(f"  - {model_name}_scaler.pkl (Feature Scaler)")
    print(f"  - {model_name}_ref_working.npy (Working Reference Patterns)")
    print(f"  - {model_name}_ref_not_working.npy (Not Working Reference Patterns)")

def load_model_artifacts(model_name="dtw_nn_classifier"):
    """
    Load all model artifacts for prediction
    """
    from tensorflow.keras.models import load_model
    
    model = load_model(f"saved_models/{model_name}.h5")
    scaler = joblib.load(f"saved_models/{model_name}_scaler.pkl")
    ref_working = np.load(f"saved_models/{model_name}_ref_working.npy", allow_pickle=True)
    ref_not_working = np.load(f"saved_models/{model_name}_ref_not_working.npy", allow_pickle=True)
    
    return model, scaler, ref_working, ref_not_working

def main():
    # Load and prepare data
    print("Loading data and creating sequences...")
    sequences, labels = create_sequences_from_csv(
        "../../data/positions/badge_positions_tracked_01 - Sheet3.csv",
        window_size=50,
        step_size=25
    )
    
    print(f"Created {len(sequences)} sequences")
    print(f"Working sequences: {sum(labels == 1)}")
    print(f"Not working sequences: {sum(labels == 0)}")
    
    # Extract DTW features
    print("Extracting DTW features...")
    X, ref_working, ref_not_working = extract_dtw_features(sequences, labels, n_reference_patterns=5)
    y = labels
    
    print(f"DTW feature shape: {X.shape}")
    print(f"Reference patterns - Working: {len(ref_working)}, Not Working: {len(ref_not_working)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and train model
    print("Building and training neural network...")
    model = build_dtw_classifier(X_train_scaled.shape[1])
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"DTW + Neural Network Results")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and all artifacts
    print(f"\n{'='*50}")
    print("Saving Model Artifacts...")
    print(f"{'='*50}")
    save_model_artifacts(model, scaler, ref_working, ref_not_working, "dtw_hand_movement_classifier")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('saved_models/training_history.png')
    plt.show()
    
    return model, scaler, ref_working, ref_not_working

# Function to test the saved model on new data
def test_saved_model_on_new_data(csv_file, model_name="dtw_hand_movement_classifier"):
    """
    Test the saved model on new CSV data and plot results with color coding
    """
    print(f"Loading saved model: {model_name}")
    model, scaler, ref_working, ref_not_working = load_model_artifacts(model_name)
    
    # Read the original CSV data for plotting
    df = pd.read_csv(csv_file)
    
    # Create sequences from new data
    sequences, _ = create_sequences_from_csv(csv_file, window_size=50, step_size=50)
    
    # Extract DTW features using the same reference patterns
    features = []
    for seq in sequences:
        feature_vector = []
        
        # DTW distances to working reference patterns
        for ref_seq in ref_working:
            distance = dtw.distance(seq.flatten(), ref_seq.flatten())
            feature_vector.append(distance)
        
        # DTW distances to not-working reference patterns
        for ref_seq in ref_not_working:
            distance = dtw.distance(seq.flatten(), ref_seq.flatten())
            feature_vector.append(distance)
        
        # Statistical features
        feature_vector.append(np.mean(feature_vector))
        feature_vector.append(np.std(feature_vector))
        feature_vector.append(np.min(feature_vector))
        
        features.append(feature_vector)
    
    X_new = np.array(features)
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    predictions_proba = model.predict(X_new_scaled)
    predictions = np.argmax(predictions_proba, axis=1)
    confidences = np.max(predictions_proba, axis=1)
    
    print(f"\nPredictions for new data:")
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        status = "WORKING" if pred == 1 else "NOT WORKING"
        print(f"Sequence {i+1}: {status} (confidence: {conf:.3f})")
    
    # Plot the results
    plot_predictions_with_timeline(df, sequences, predictions, confidences)
    
    return predictions, confidences

def plot_predictions_with_timeline(df, sequences, predictions, confidences):
    """
    Plot hand positions with timeline colored by working/non-working status
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Get time information
    time = df['second'].values if 'second' in df.columns else np.arange(len(df))
    
    # Plot 1: Hand positions over time with classification coloring
    window_size = len(sequences[0]) if len(sequences) > 0 else 50
    step_size = 50  # Since we used step_size=50 in create_sequences_from_csv
    
    # Create a color array for the entire timeline
    colors = []
    current_idx = 0
    
    for i, pred in enumerate(predictions):
        # Determine color based on prediction
        color = 'green' if pred == 1 else 'red'
        
        # Assign color to this window's time range
        if i == len(predictions) - 1:
            # Last window - extend to end of data
            window_end = min(current_idx + window_size, len(df))
        else:
            window_end = current_idx + step_size
        
        # Add colors for this window
        colors.extend([color] * (window_end - current_idx))
        current_idx = window_end
    
    # Pad colors if needed
    if len(colors) < len(df):
        colors.extend(['gray'] * (len(df) - len(colors)))
    
    # Plot hand positions with colored background
    ax1.scatter(time, df['hand_2_x'].values, c=colors, alpha=0.6, s=20, label='X position')
    ax1.scatter(time, df['hand_2_y'].values, c=colors, alpha=0.6, s=20, label='Y position', marker='x')
    ax1.set_ylabel('Position (pixels)')
    ax1.set_title('Hand Positions Over Time (Colored by Classification)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add confidence as transparency
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, len(df))
        
        if pred == 1:  # Working
            ax1.axvspan(time[start_idx], time[end_idx-1], alpha=conf*0.2, color='green')
        else:  # Not working
            ax1.axvspan(time[start_idx], time[end_idx-1], alpha=conf*0.2, color='red')
    
    # Plot 2: X and Y coordinates separately with classification regions
    ax2.plot(time, df['hand_2_x'].values, 'b-', linewidth=1, label='Hand X', alpha=0.7)
    ax2.plot(time, df['hand_2_y'].values, 'r-', linewidth=1, label='Hand Y', alpha=0.7)
    
    # Highlight classification regions
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, len(df))
        
        color = 'green' if pred == 1 else 'red'
        ax2.axvspan(time[start_idx], time[end_idx-1], alpha=0.3, color=color, 
                   label=f"{'Working' if pred == 1 else 'Not Working'} (conf: {conf:.2f})" if i == 0 else "")
    
    ax2.set_ylabel('Position (pixels)')
    ax2.set_title('Hand Coordinates with Classification Regions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Classification confidence over time
    confidence_timeline = []
    time_windows = []
    
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, len(df))
        
        # Use midpoint of window for x-axis
        mid_time = (time[start_idx] + time[end_idx-1]) / 2
        confidence_timeline.append(conf)
        time_windows.append(mid_time)
        
        # Add colored bars for classification
        color = 'green' if pred == 1 else 'red'
        ax3.bar(mid_time, conf, width=time[end_idx-1]-time[start_idx], 
                color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Confidence')
    ax3.set_title('Classification Confidence Over Time')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Add confidence threshold line
    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total windows analyzed: {len(predictions)}")
    print(f"Working windows: {sum(predictions == 1)} ({sum(predictions == 1)/len(predictions)*100:.1f}%)")
    print(f"Not working windows: {sum(predictions == 0)} ({sum(predictions == 0)/len(predictions)*100:.1f}%)")
    print(f"Average confidence: {np.mean(confidences):.3f}")
    print(f"High confidence predictions (>0.8): {sum(confidences > 0.8)}")
    
    # Create a simplified timeline plot
    plot_simplified_timeline(df, sequences, predictions, confidences)

def plot_simplified_timeline(df, sequences, predictions, confidences):
    """
    Create a simplified timeline visualization
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    time = df['second'].values if 'second' in df.columns else np.arange(len(df))
    window_size = len(sequences[0]) if len(sequences) > 0 else 50
    step_size = 50
    
    # Plot the hand movement trajectory
    ax.plot(df['hand_2_x'].values, df['hand_2_y'].values, 'k-', alpha=0.3, linewidth=1, label='Hand Trajectory')
    
    # Color code points based on classification
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, len(df))
        
        # Get the data points for this window
        window_x = df['hand_2_x'].iloc[start_idx:end_idx].values
        window_y = df['hand_2_y'].iloc[start_idx:end_idx].values
        
        # Plot with classification color
        color = 'green' if pred == 1 else 'red'
        alpha = max(0.3, conf)  # Higher confidence = more opaque
        
        ax.scatter(window_x, window_y, c=color, alpha=alpha, s=30, 
                  label=f"{'Working' if pred == 1 else 'Not Working'}" if i == 0 else "")
        
        # Add confidence as text for high-confidence predictions
        if conf > 0.8:
            center_x = np.mean(window_x)
            center_y = np.mean(window_y)
            ax.text(center_x, center_y, f'{conf:.2f}', fontsize=8, 
                   ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='white', alpha=0.7))
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('Hand Movement Trajectory Colored by Classification\n(Green: Working, Red: Not Working)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train and save the model
    # model, scaler, ref_working, ref_not_working = main()
    
    # Example: Test on new data (uncomment to use)
    print(f"\n{'='*50}")
    print("Testing on New Data...")
    print(f"{'='*50}")
    test_saved_model_on_new_data("../../data/positions/badge_positions_tracked_01 - Sheet3.csv")