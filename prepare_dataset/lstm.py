import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

def load_json_data(json_file):
    """
    Load the JSON data and prepare sequences for LSTM
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    return df

def create_sequences_from_json(df, sequence_length=10):
    """
    Create sequential data from the JSON features for LSTM
    Each sequence contains multiple consecutive windows
    """
    sequences = []
    labels = []
    
    # Feature columns to use for LSTM
    feature_columns = [
        'average_x', 'average_y', 'std_x', 'std_y', 'average_magnitude',
        'dominant_frequency_x', 'amplitude_x', 'dominant_frequency_y', 'amplitude_y',
        'average_velocity', 'max_velocity', 'total_distance', 
        'movement_variability', 'movement_smoothness'
    ]
    
    # Create sequences of consecutive windows
    for i in range(len(df) - sequence_length + 1):
        sequence = df[feature_columns].iloc[i:i+sequence_length].values
        # Use the label from the last window in the sequence
        label = df['label'].iloc[i+sequence_length-1]
        
        sequences.append(sequence)
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

def create_sliding_window_sequences(df, window_size=5):
    """
    Alternative: Create sequences using sliding window on features
    """
    sequences = []
    labels = []
    
    feature_columns = [
        'average_x', 'average_y', 'std_x', 'std_y', 'average_magnitude',
        'dominant_frequency_x', 'amplitude_x', 'dominant_frequency_y', 'amplitude_y',
        'average_velocity', 'max_velocity', 'total_distance', 
        'movement_variability', 'movement_smoothness'
    ]
    
    features = df[feature_columns].values
    target = df['label'].values
    
    for i in range(len(features) - window_size + 1):
        sequences.append(features[i:i+window_size])
        labels.append(target[i+window_size-1])
    
    return np.array(sequences), np.array(labels)

def build_lstm_model(input_shape, num_classes=2):
    """
    Build LSTM model architecture
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names=['Not Working', 'Working']):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('LSTM - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def main():
    # Load JSON data
    json_file = "hand_position_dataset.json"
    print("Loading JSON data...")
    df = load_json_data(json_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:")
    print(df['label'].value_counts())
    print(f"Working samples (1): {sum(df['label'] == 1)}")
    print(f"Not working samples (0): {sum(df['label'] == 0)}")
    
    # Create sequences for LSTM
    sequence_length = 5  # Use 5 consecutive windows as sequence
    print(f"\nCreating sequences with length {sequence_length}...")
    
    # Choose sequence creation method
    X, y = create_sliding_window_sequences(df, sequence_length)
    # X, y = create_sequences_from_json(df, sequence_length)  # Alternative method
    
    print(f"Sequences shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Sequence class distribution: {np.bincount(y)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale the features
    scaler = StandardScaler()
    
    # Reshape for scaling (temporarily flatten the sequence dimension)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    # Fit scaler on training data
    scaler.fit(X_train_reshaped)
    
    # Transform both training and test data
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Build LSTM model
    print("\nBuilding LSTM model...")
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    model = build_lstm_model(input_shape)
    
    print("Model Summary:")
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
    ]
    
    # Train the model
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train_scaled, y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating LSTM model...")
    
    # Make predictions
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\nLSTM Model Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(report)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)
    
    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    
    # Save the model
    model.save('lstm_hand_movement_classifier.h5')
    print("\nModel saved as 'lstm_hand_movement_classifier.h5'")
    
    return model, history, accuracy, cm

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run the main function
    model, history, accuracy, cm = main()