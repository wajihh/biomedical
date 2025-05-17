import numpy as np
from sklearn.model_selection import train_test_split
from ecg_data_loader import ECGDataLoader
from ecg_cnn_model import ECGCNNModel
from ecg_plotter import ECGPlotter

def main():
    # Initialize components
    data_loader = ECGDataLoader(data_path='data', record_names=['100', '101','203'])
    cnn_model = ECGCNNModel()
    plotter = ECGPlotter()

    # Load and preprocess data
    signal, r_peaks, labels = data_loader.load_ecg_data(samples=40000)
    segments, segment_labels = data_loader.segment_beats(signal, r_peaks, labels)
    segments = data_loader.preprocess_data(segments)
    print(f"Initial segments: {segments.shape}, segment_labels: {len(segment_labels)}")

    # Check class distribution and adjust stratification
    unique, counts = np.unique(segment_labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    stratify = segment_labels if all(count >= 2 for count in class_counts.values()) else None
    print(f"Class distribution: {dict(zip(unique, counts))}, stratify: {stratify is not None}")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        segments, segment_labels, test_size=0.3, random_state=42, stratify=stratify
    )
    print(f"After first split - X_train: {X_train.shape}, y_train: {len(y_train)}, X_temp: {X_temp.shape}, y_temp: {len(y_temp)}")

    # Check class distribution in y_temp for second split
    unique_temp, counts_temp = np.unique(y_temp, return_counts=True)
    stratify_second = y_temp if all(count >= 2 for count in counts_temp) else None
    print(f"y_temp class distribution: {dict(zip(unique_temp, counts_temp))}, stratify_second: {stratify_second is not None}")

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=stratify_second
    )
    print(f"After second split - X_val: {X_val.shape}, y_val: {len(y_val)}, X_test: {X_test.shape}, y_test: {len(y_test)}")

    # Plot sample segments
    for i in range(min(3, len(segments))):
        plotter.plot_ecg_segment(segments[i, :, 0], segment_labels[i], i)

    # Train model
    history = cnn_model.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=16)

    # Evaluate model
    results = cnn_model.evaluate(X_test, y_test)
    print("\nCNN Results:")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.2f}")

    # Plot confusion matrix
    y_pred = (cnn_model.model.predict(X_test) > 0.5).astype(int)
    plotter.plot_confusion_matrix(y_test, y_pred)

    # Save model
    cnn_model.save_model()

if __name__ == "__main__":
    main()