import wfdb
import numpy as np
import pandas as pd

class ECGDataLoader:
    def __init__(self, data_path, record_names=['100', '101','203'], window_size=256):
        self.data_path = data_path
        self.record_names = record_names
        self.window_size = window_size

    def load_ecg_data(self, samples=20000):
        """Load ECG signals from multiple CSV files and annotations from MIT-BIH with offset adjustment."""
        all_signals = []
        all_r_peaks = []
        all_labels = []
        offset = 0

        for record_name in self.record_names:
            # Load signal from CSV
            df = pd.read_csv(f"{self.data_path}/{record_name}.csv", nrows=samples)
            signal = df['MLII'].values
            all_signals.append(signal)

            # Load annotations from .atr
            annotation = wfdb.rdann(f"{self.data_path}/{record_name}", 'atr', sampfrom=0, sampto=samples)
            r_peaks = annotation.sample
            labels = np.array(['N' if sym == 'N' else 'A' for sym in annotation.symbol])
            # Adjust R-peaks with cumulative offset
            r_peaks = r_peaks + offset
            all_r_peaks.append(r_peaks)
            all_labels.append(labels)
            offset += len(signal)  # Update offset for the next record

        # Concatenate all data
        signal = np.concatenate(all_signals)
        r_peaks = np.concatenate(all_r_peaks)
        labels = np.concatenate(all_labels)

        return signal, r_peaks, labels

    def segment_beats(self, signal, r_peaks, labels):
        """Segment ECG into fixed-size windows around R-peaks with boundary checks."""
        half_window = self.window_size // 2
        segments = []
        segment_labels = []
        valid_indices = []

        # First pass: Identify valid segments
        for i, r_peak in enumerate(r_peaks):
            start = r_peak - half_window
            end = r_peak + half_window
            if 0 <= start and end < len(signal):
                valid_indices.append(i)

        # Second pass: Create segments and labels for valid indices
        for i in valid_indices:
            r_peak = r_peaks[i]
            start = r_peak - half_window
            end = r_peak + half_window
            segment = signal[start:end]
            segments.append(segment)
            segment_labels.append(1 if labels[i] == 'A' else 0)

        return np.array(segments), np.array(segment_labels)

    def preprocess_data(self, segments):
        """Normalize segments to [0, 1] and add channel dimension."""
        segments = (segments - np.min(segments)) / (np.max(segments) - np.min(segments))
        return segments.reshape(segments.shape[0], segments.shape[1], 1)