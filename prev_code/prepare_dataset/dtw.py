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
    
    return np.array(features)

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
    X = extract_dtw_features(sequences, labels, n_reference_patterns=5)
    y = labels
    
    print(f"DTW feature shape: {X.shape}")
    
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
    plt.show()

if __name__ == "__main__":
    main()