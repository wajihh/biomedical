from scipy import signal
import numpy as np

class ECGFeatureExtractor:
    """Class to extract HRV and RR intervals from ECG."""
    
    def __init__(self, signal, fs=360):
        """Initialize with signal and sampling frequency."""
        self.signal = signal
        self.fs = fs
        self.peaks = None
        
    def detect_r_peaks(self, height=0.5, distance=200):
        """Detect R-peaks."""
        self.peaks, _ = signal.find_peaks(self.signal, height=height, distance=distance)
        return self.peaks
    
    def extract_features(self):
        """Extract features per RR interval."""
        if self.peaks is None or len(self.peaks) < 2:
            raise ValueError("Not enough peaks for feature extraction")
        
        # RR intervals in seconds
        rr_intervals = np.diff(self.peaks) / self.fs
        
        # Compute features for each RR interval
        features = []
        for i in range(len(rr_intervals) - 1):
            segment = rr_intervals[i:i+2]  # Consider pairs for local features
            if len(segment) < 2:
                continue
            mean_rr = np.mean(segment)
            sdnn = np.std(segment) if len(segment) > 1 else 0
            rmssd = np.sqrt(np.mean(np.diff(segment)**2)) if len(segment) > 1 else 0
            pnn50 = 100 * np.sum(np.abs(np.diff(segment)) > 0.05) / len(segment) if len(segment) > 1 else 0
            
            features.append({
                'mean_rr': mean_rr,
                'sdnn': sdnn,
                'rmssd': rmssd,
                'pnn50': pnn50
            })
        
        return features, rr_intervals