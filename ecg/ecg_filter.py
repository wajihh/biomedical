from scipy import signal

class ECGFilter:
    """Class to apply bandpass filtering to ECG signals."""
    
    def __init__(self, signal, fs):
        """Initialize with signal and sampling frequency."""
        self.signal = signal
        self.fs = fs
        
    def apply_bandpass_filter(self, lowcut=0.5, highcut=40.0, order=4):
        """Apply bandpass filter to ECG signal."""
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        self.signal = signal.filtfilt(b, a, self.signal)
        return self.signal