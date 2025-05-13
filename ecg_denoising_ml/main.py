from ecg_data_loader import ECGDataLoader
from ecg_denoiser import ECGDenoiser
from ecg_feature_extractor import ECGFeatureExtractor
from ecg_classifier import ECGClassifier
from ecg_plotter import ECGPlotter
import numpy as np

def main():
    """Main function for ECG denoising, feature extraction, and classification."""
    # Load data
    loader = ECGDataLoader(record_name='100', data_path='./data')
    loader.load_data()
    
    # Plot raw signal
    plotter = ECGPlotter(loader.signal, loader.time)
    plotter.plot_signal(title="Raw ECG Signal", save_path='raw_ecg.png')
    
    # Denoise signal
    denoiser = ECGDenoiser(loader.signal)
    denoised_signal = denoiser.wavelet_denoise()
    
    # Plot denoised signal
    plotter = ECGPlotter(denoised_signal, loader.time)
    plotter.plot_signal(title="Denoised ECG Signal", save_path='denoised_ecg.png')
    
    # Extract features
    extractor = ECGFeatureExtractor(denoised_signal, loader.fs)
    peaks = extractor.detect_r_peaks(height=0.3, distance=180)
    
    # Plot signal with R-peaks
    plotter.plot_signal(peaks=peaks, title="Denoised ECG with R-peaks", save_path='r_peaks.png')
    print(f"Detected {len(peaks)} R-peaks")
    
    # Extract HRV and RR intervals
    features, rr_intervals = extractor.extract_features()
    
    # Plot HRV
    plotter.plot_hrv(rr_intervals, title="HRV (RR Intervals)", save_path='hrv.png')
    
    # Simulate labels (1: normal, 0: arrhythmic) for each feature set
    labels = np.ones(len(features))
    labels[::2] = 0  # Simulate 50% arrhythmic
    
    # Train and evaluate classifiers
    classifier = ECGClassifier()
    X_train, X_test, y_train, y_test = classifier.prepare_data(features, labels)
    results = classifier.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Print results
    for name, metrics in results.items():
        print(f"\n{name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"F1-Score: {metrics['f1']:.2f}")

if __name__ == "__main__":
    main()