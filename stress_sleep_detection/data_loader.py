import numpy as np
import pandas as pd
import pickle
from scipy.signal import resample
from scipy.stats import mode

class WESADDataLoader:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.vcc = 3
        self.chan_bit = 2**16

    def load_data(self, subject_id):
        # Load RespiBAN data (ECG, respiration)
        respiban_path = f'{self.data_path}/S{subject_id}/S{subject_id}_respiban.txt'
        respiban_data = np.loadtxt(respiban_path, skiprows=1)  # Skip header
        ecg_raw = respiban_data[:, 2]
        resp_raw = respiban_data[:, 5]

        # Convert to SI units
        ecg = ((ecg_raw / self.chan_bit - 0.5) * self.vcc)
        resp = ((resp_raw / self.chan_bit - 0.5) * 100)

        # Load EDA from Empatica E4
        eda_path = f'{self.data_path}/S{subject_id}/S{subject_id}_E4_Data/EDA.csv'
        eda_data = pd.read_csv(eda_path, header=None).values
        eda = eda_data[2:, 0]

        # Load labels from SX.pkl
        pkl_path = f'{self.data_path}/S{subject_id}/S{subject_id}.pkl'
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
        labels = pkl_data['label']

        # Debug: Print raw label distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Raw label distribution for S{subject_id}: {dict(zip(unique, counts))}")

        # Align lengths
        min_length = min(len(ecg), len(resp), len(labels))
        print(f"Lengths before alignment - ECG: {len(ecg)}, Resp: {len(resp)}, Labels: {len(labels)}, Min: {min_length}")
        ecg = ecg[:min_length]
        resp = resp[:min_length]
        labels = labels[:min_length]

        # Debug: Print label distribution after alignment
        unique_aligned, counts_aligned = np.unique(labels, return_counts=True)
        print(f"Label distribution after alignment: {dict(zip(unique_aligned, counts_aligned))}")

        # Filter labels to focus on baseline, stress, amusement
        valid_indices = np.isin(labels, [1, 2, 3])
        print(f"Valid indices count: {np.sum(valid_indices)} out of {len(labels)}")
        ecg = ecg[valid_indices]
        resp = resp[valid_indices]
        labels = labels[valid_indices]
        # Map labels: 1->0 (baseline), 2->1 (stress), 3->2 (amusement)
        labels = np.where(labels == 1, 0, labels)
        labels = np.where(labels == 2, 1, labels)
        labels = np.where(labels == 3, 2, labels)

        # Debug: Print label distribution after filtering
        unique_filtered, counts_filtered = np.unique(labels, return_counts=True)
        print(f"Label distribution after filtering: {dict(zip(unique_filtered, counts_filtered))}")

        # Resample EDA to match the aligned length
        num_samples = len(ecg)
        eda_resampled = resample(eda, num_samples)

        return ecg, eda_resampled, resp, labels

    def preprocess(self, ecg, eda, resp, labels):
        # Normalize signals to [0, 1]
        ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min())
        eda = (eda - eda.min()) / (eda.max() - eda.min())
        resp = (resp - resp.min()) / (resp.max() - resp.min())

        # Segment into windows (5 seconds)
        window_size = 3500  # 700 Hz * 5s
        step_size = 1750    # 50% overlap
        segments = []
        segment_labels = []
        for i in range(0, len(ecg) - window_size + 1, step_size):
            segment = np.stack([ecg[i:i+window_size], eda[i:i+window_size], resp[i:i+window_size]], axis=-1)
            # Get the majority label for this window
            window_labels = labels[i:i+window_size]
            majority_label = mode(window_labels, axis=0, keepdims=False)[0]
            segments.append(segment)
            segment_labels.append(majority_label)
        return np.array(segments), np.array(segment_labels)