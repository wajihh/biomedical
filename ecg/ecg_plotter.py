import matplotlib.pyplot as plt

class ECGPlotter:
    """Class to plot ECG signals, R-peaks, and heart rates."""
    
    def __init__(self, signal, time):
        """Initialize with signal and time array."""
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
        
    def plot_heart_rate(self, inst_times, inst_heart_rates, title="Instantaneous Heart Rate", save_path=None):
        """Plot instantaneous heart rate over time."""
        plt.figure(figsize=(12, 4))
        plt.plot(inst_times, inst_heart_rates, 'g-', label='Instantaneous Heart Rate')
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Heart Rate (BPM)')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()