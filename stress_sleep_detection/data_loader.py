import numpy as np
import pandas as pd
import pickle
from scipy.signal import resample

class WESADDataLoader:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.vcc = 3
        self.chan_bit = 2**16

    def load_data(self, subject_id):
        # Load RespiBAN data (ECG, respiration)
        respiban_path = f'{self.data_path}/S{subject_id}/S{subject_id}_respiban.txt'
        respiban_data = np.loadtxt(respiban_path, skiprows=1)  # Skip header
        # Columns: 0=seq, 1=ignore, 2=ECG, 3=EDA, 4=EMG, 5=Resp, 6=Temp, 7-9=ACC XYZ
        ecg_raw = respiban_data[:, 2]
        resp_raw = respiban_data[:, 5]

        # Convert to SI units (per WESAD documentation)
        ecg = ((ecg_raw / self.chan_bit - 0.5) * self.vcc)  # ECG in mV
        resp = ((resp_raw / self.chan_bit - 0.5) * 100)     # Respiration in %

        # Load EDA from Empatica E4 (4 Hz, in μS)
        eda_path = f'{self.data_path}/S{subject_id}/S{subject_id}_E4_Data/EDA.csv'
        eda_data = pd.read_csv(eda_path, header=None).values
        eda_timestamp = eda_data[0, 0]  # Start timestamp
        eda_freq = eda_data[1, 0]       # Sampling rate (4 Hz)
        eda = eda_data[2:, 0]           # Raw EDA data in μS

        # Load labels from SX.pkl
        pkl_path = f'{self.data_path}/S{subject_id}/S{subject_id}.pkl'
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
        labels = pkl_data['label']  # 700 Hz labels (0=transient, 1=baseline, 2=stress, 3=amusement)

        # Align lengths by truncating to the shortest array
        min_length = min(len(ecg), len(resp), len(labels))
        ecg = ecg[:min_length]
        resp = resp[:min_length]
        labels = labels[:min_length]

        # Filter labels to focus on baseline, stress, amusement (ignore 0, 4, 5, 6, 7)
        valid_indices = np.isin(labels, [1, 2, 3])
        ecg = ecg[valid_indices]
        resp = resp[valid_indices]
        labels = labels[valid_indices]
        # Map labels: 1->0 (baseline), 2->1 (stress), 3->2 (amusement)
        labels = np.where(labels == 1, 0, labels)
        labels = np.where(labels == 2, 1, labels)
        labels = np.where(labels == 3, 2, labels)

        # Resample EDA to match the aligned length
        num_samples = len(ecg)  # Target length after alignment
        eda_resampled = resample(eda, num_samples)

        return ecg, eda_resampled, resp, labels

    def preprocess(self, ecg, eda, resp):
        # Normalize signals to [0, 1]
        ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min())
        eda = (eda - eda.min()) / (eda.max() - eda.min())
        resp = (resp - resp.min()) / (resp.max() - resp.min())

        # Segment into windows (5 seconds)
        window_size = 3500  # 700 Hz * 5s
        step_size = 1750    # 50% overlap
        segments = []
        for i in range(0, len(ecg) - window_size + 1, step_size):
            segment = np.stack([ecg[i:i+window_size], eda[i:i+window_size], resp[i:i+window_size]], axis=-1)
            segments.append(segment)
        return np.array(segments)