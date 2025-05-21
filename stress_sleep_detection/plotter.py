import matplotlib.pyplot as plt

def plot_signals(ecg, eda, resp, labels):
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(ecg, label='ECG')
    plt.title('ECG Signal')
    plt.subplot(3, 1, 2)
    plt.plot(eda, label='EDA')
    plt.title('EDA Signal')
    plt.subplot(3, 1, 3)
    plt.plot(resp, label='Respiration')
    plt.title('Respiration Signal')
    plt.tight_layout()
    plt.savefig('signals.png')

def plot_accuracy(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')