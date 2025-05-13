import matplotlib.pyplot as plt

class ECGPlotter:
    """Class to plot ECG signals and features."""
    
    def __init__(self, signal, time):
        """Initialize with signal and time."""
        self.signal = signal
        self.time = time
        
    def plot_signal(self, peaks=None, title="ECG Signal", save_path=None):
        """Plot ECG signal with optional R-peaks."""
        plt.figure(figsize=(12, 4))
        plt.plot(self.time, self.signal, 'b-', label='ECG Signal')
        if peaks is not None:
            plt.plot(self.time[peaks], self.signal[peaks], 'ro', label='R-peaks')
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (mV)')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_hrv(self, rr_intervals, title="HRV (RR Intervals)", save_path=None):
        """Plot RR intervals."""
        plt.figure(figsize=(12, 4))
        plt.plot(rr_intervals, 'g-', label='RR Intervals')
        plt.title(title)
        plt.xlabel('Interval Index')
        plt.ylabel('RR Interval (s)')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()