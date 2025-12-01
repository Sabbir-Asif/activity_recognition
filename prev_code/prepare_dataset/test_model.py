import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import sys
sys.path.append('src/prepare_dataset')
from prepare_dataset import extract_features_from_array

def test_csv_classifier(csv_file, model_path="lstm_hand_movement_classifier.h5"):
    """
    Ultra-simple function to test classifier on CSV
    """
    # Load model
    model = load_model(model_path)
    
    # Load CSV
    df = pd.read_csv(csv_file)
    x = df['hand_2_x'].values[1350:1400]
    y = df['hand_2_y'].values[1350:1400]
    # x = df['hand_2_x'].values[:50]
    # y = df['hand_2_y'].values[:50]
    xy_array = np.column_stack([x, y])
    
    # Extract features
    features = extract_features_from_array(xy_array)
    
    # Create feature vector and sequence
    feature_columns = ['average_x', 'average_y', 'std_x', 'std_y', 'average_magnitude',
                      'dominant_frequency_x', 'amplitude_x', 'dominant_frequency_y', 'amplitude_y',
                      'average_velocity', 'max_velocity', 'total_distance', 
                      'movement_variability', 'movement_smoothness']
    
    feature_vector = np.array([[features[col] for col in feature_columns]])
    sequence = np.repeat(feature_vector, 5, axis=0).reshape(1, 5, len(feature_columns))
    
    # Predict
    prediction = model.predict(sequence, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction, axis=1)[0]
    
    return predicted_class, confidence

# Usage
if __name__ == "__main__":
    import sys
    csv_path = "../../data/positions/badge_positions_tracked_01 - Sheet3.csv"
    
    pred, conf = test_csv_classifier(csv_path)
    print(f"Result: {'WORKING' if pred == 1 and conf > 0.8 else 'NOT WORKING'} ({conf:.2f})")