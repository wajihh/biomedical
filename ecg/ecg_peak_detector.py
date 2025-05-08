from scipy import signal
import numpy as np

class ECGPeakDetector:
    """Class to detect R-peaks and calculate heart rate in ECG signals."""
    
    def __init__(self, signal, fs=360):
        """Initialize with ECG signal and sampling frequency."""
        self.signal = signal
        self.fs = fs
        self.peaks = None
        
    def detect_r_peaks(self, height=0.5, distance=200):
        """Detect R-peaks in ECG signal."""
        self.peaks, _ = signal.find_peaks(self.signal, height=height, distance=distance)
        return self.peaks
    
    def calculate_heart_rate(self):
        """Calculate average and instantaneous heart rates from R-peaks."""
        if self.peaks is None or len(self.peaks) < 2:
            raise ValueError("Not enough peaks detected for heart rate calculation")
        
        # Calculate RR intervals in seconds
        rr_intervals = np.diff(self.peaks) / self.fs
        
        # Average heart rate (BPM) = 60 / mean(RR interval in seconds)
        avg_heart_rate = 60 / np.mean(rr_intervals)
        
        # Instantaneous heart rates (BPM) for each RR interval
        inst_heart_rates = 60 / rr_intervals
        
        # Times for instantaneous heart rates (midpoint of each RR interval)
        inst_times = (self.peaks[:-1] + self.peaks[1:]) / 2 / self.fs
        
        return {
            'avg_heart_rate': avg_heart_rate,
            'inst_heart_rates': inst_heart_rates,
            'inst_times': inst_times
        }