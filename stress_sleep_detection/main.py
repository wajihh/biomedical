from data_loader import WESADDataLoader
from model import MultiModalModel
from sklearn.model_selection import train_test_split
import numpy as np
from plotter import plot_signals, plot_accuracy

def main():
    # Initialize data loader
    loader = WESADDataLoader()
    
    # Load data for subject S2
    ecg, eda, resp, labels = loader.load_data(subject_id=2)
    
    # Preprocess data into segments
    segments = loader.preprocess(ecg, eda, resp)
    
    # Ensure labels match segment count
    if len(labels) > segments.shape[0]:
        labels = labels[:segments.shape[0]]
    elif len(labels) < segments.shape[0]:
        segments = segments[:len(labels)]
    
    # Convert labels to integer type for model compatibility
    labels = labels.astype(int)
    
    # Print class distribution to check balance
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique, counts))}")
    
    # Split data into train, validation, and test sets with stratification
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        segments, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
    )
    
    # Initialize and train the model
    model = MultiModalModel(input_shape=(3500, 3), num_classes=3)
    history = model.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)
    
    # Evaluate on test set
    loss, accuracy = model.model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")
    
    # Save model in native Keras format
    model.model.save('stress_sleep_model.keras')
    
    # Plot signals and accuracy
    plot_signals(ecg, eda, resp, labels)
    plot_accuracy(history)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")