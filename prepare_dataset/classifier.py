import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(json_file):
    """
    Load the JSON data and prepare features and labels
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Separate features and target
    feature_columns = [
        'average_x', 'average_y', 'std_x', 'std_y', 'average_magnitude',
        'dominant_frequency_x', 'amplitude_x', 'dominant_frequency_y', 'amplitude_y',
        'average_velocity', 'max_velocity', 'total_distance', 
        'movement_variability', 'movement_smoothness'
    ]
    
    X = df[feature_columns]
    y = df['label']
    
    return X, y, df

def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple classifiers and evaluate their performance
    """
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    return results

def plot_results(results, y_test):
    """
    Plot comparison of model performances
    """
    # Accuracy comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Accuracy comparison
    plt.subplot(1, 3, 1)
    plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # Plot 2: Confusion matrices
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(1, 3, i + 1)
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Working', 'Working'],
                   yticklabels=['Not Working', 'Working'])
        plt.title(f'{name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

def feature_importance_analysis(model, feature_names):
    """
    Analyze and plot feature importance for Random Forest
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances - Random Forest")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("\nTop 5 Most Important Features:")
        for i in range(5):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def main():
    # Load your data
    json_file = "hand_position_dataset.json"
    
    print("Loading data...")
    X, y, df = load_and_prepare_data(json_file)
    
    # Print dataset info
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:")
    print(y.value_counts())
    print(f"Working samples: {sum(y == 1)}")
    print(f"Not working samples: {sum(y == 0)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\nTraining classification models...")
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"{'='*50}")
    
    # Plot results
    plot_results(results, y_test)
    
    # Feature importance analysis
    if 'Random Forest' in results:
        print("\nFeature Importance Analysis:")
        feature_importance_analysis(results['Random Forest']['model'], X.columns.tolist())
    
    # Detailed classification report for best model
    print(f"\nDetailed Classification Report for {best_model_name}:")
    print(results[best_model_name]['classification_report'])
    
    # Save the best model (optional)
    import joblib
    joblib.dump(results[best_model_name]['model'], 'best_classifier_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\nBest model and scaler saved to disk.")

# Alternative: If you want to try different approaches
def advanced_model_training():
    """
    Alternative approach with more models and hyperparameter tuning
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Load data
    json_file = "hand_position_dataset.json"
    X, y, _ = load_and_prepare_data(json_file)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models for cross-validation
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Perform cross-validation
    print("Cross-validation results (5-fold):")
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

if __name__ == "__main__":
    # Run main training and evaluation
    main()
    
    # Uncomment to run advanced cross-validation
    print("\n" + "="*60)
    print("ADVANCED CROSS-VALIDATION RESULTS")
    print("="*60)
    advanced_model_training()