import pywt
import numpy as np

class ECGDenoiser:
    """Class to denoise ECG signals using wavelet transforms."""
    
    def __init__(self, signal):
        """Initialize with ECG signal."""
        self.signal = signal
        
    def wavelet_denoise(self, wavelet='db4', level=4, threshold_type='soft'):
        """Apply wavelet denoising."""
        # Decompose signal
        coeffs = pywt.wavedec(self.signal, wavelet, level=level)
        
        # Threshold coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(self.signal)))
        
        coeffs = [pywt.threshold(c, threshold, mode=threshold_type) if i > 0 else c 
                  for i, c in enumerate(coeffs)]
        
        # Reconstruct signal
        denoised_signal = pywt.waverec(coeffs, wavelet)
        
        # Ensure output length matches input
        return denoised_signal[:len(self.signal)]