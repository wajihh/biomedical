# CNN-Based Arrhythmia Detection from Raw ECG
This project implements a convolutional neural network (CNN) to classify ECG beats as normal or arrhythmic using raw signals from the MIT-BIH Arrhythmia Database. Unlike traditional methods requiring feature extraction (e.g., HRV), the CNN processes 256-sample windows around R-peaks directly.
# Setup

Clone the repository:
git clone https://github.com/wajihh/biomedical.git
cd biomedical-projects/ecg_arrhythmia_cnn


# Set up virtual environment:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Upload 110.csv to data/.


3 Usage
Run the pipeline:
python3 main.py

# Outputs:

segment_*.png: Sample ECG segments.
confusion_matrix.png: Classification performance.
models/cnn_model.h5: Trained model.

# Results
The CNN achieves improved accuracy over classical ML (e.g., KNN: 0.50 from prior work), leveraging real annotations.
Blog Post
