import pandas as pd
import numpy as np
import os

class ECGDataLoader:
    """Class to load ECG data from CSV."""
    
    def __init__(self, record_name='100', data_path='./data', fs=360):
        """Initialize with record name, data path, and sampling frequency."""
        self.record_name = record_name
        self.data_path = data_path
        self.fs = fs
        self.signal = None
        self.time = None
        
    def load_data(self):
        """Load ECG data from CSV."""
        csv_path = os.path.join(self.data_path, f'{self.record_name}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            self.signal = df['MLII'].values[:10000]  # Use MLII, limit to 10000 samples
            self.time = np.arange(len(self.signal)) / self.fs
            print(f"Loaded record {self.record_name} with {len(self.signal)} samples")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise