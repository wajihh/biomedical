from ecg_data_loader import ECGDataLoader
from ecg_filter import ECGFilter
from ecg_peak_detector import ECGPeakDetector
from ecg_plotter import ECGPlotter

def main():
    """Main function to process, analyze ECG data, and calculate heart rate."""
    # Load data from local path
    loader = ECGDataLoader(record_name='100', data_path='./data')
    loader.load_data()
    
    # Plot raw signal
    plotter = ECGPlotter(loader.signal, loader.time)
    plotter.plot_signal(title="Raw ECG Signal", save_path='raw_ecg.png')
    
    # Apply bandpass filter
    filter = ECGFilter(loader.signal, loader.fs)
    filtered_signal = filter.apply_bandpass_filter()
    
    # Plot filtered signal
    plotter = ECGPlotter(filtered_signal, loader.time)
    plotter.plot_signal(title="Filtered ECG Signal", save_path='filtered_ecg.png')
    
    # Detect R-peaks
    detector = ECGPeakDetector(filtered_signal, loader.fs)
    peaks = detector.detect_r_peaks(height=0.5, distance=200)
    
    # Plot ECG with R-peaks
    plotter.plot_signal(peaks=peaks, title="ECG with R-peaks", save_path='r_peaks.png')
    print(f"Detected {len(peaks)} R-peaks")
    
    # Calculate and display heart rate
    try:
        hr_data = detector.calculate_heart_rate()
        print(f"Average Heart Rate: {hr_data['avg_heart_rate']:.2f} BPM")
        
        # Plot instantaneous heart rate
        plotter.plot_heart_rate(
            hr_data['inst_times'],
            hr_data['inst_heart_rates'],
            title="Instantaneous Heart Rate",
            save_path='heart_rate.png'
        )
    except ValueError as e:
        print(f"Heart rate calculation failed: {e}")

if __name__ == "__main__":
    main()