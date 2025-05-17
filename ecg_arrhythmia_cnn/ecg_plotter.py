import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

class ECGPlotter:
    def plot_ecg_segment(self, segment, label, index):
        """Plot and save a single ECG segment."""
        plt.figure(figsize=(10, 4))
        plt.plot(segment)
        plt.title(f"ECG Segment {index} - {'Arrhythmic' if label else 'Normal'}")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig(f"segment_{index}.png")
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix."""
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=[0, 1], display_labels=['Normal', 'Arrhythmic']
        )
        disp.plot(cmap='Blues')
        plt.savefig("confusion_matrix.png")
        plt.close()