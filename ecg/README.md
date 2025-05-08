ECG Signal Processing Project
Overview
This project processes ECG signals from the MIT-BIH Arrhythmia Database (Kaggle dataset), loaded from a local CSV file. It performs bandpass filtering, R-peak detection, heart rate calculation, and visualization, designed for Python environments like VS Code on Ubuntu.
Setup on Ubuntu with VS Code

Prerequisites:

Ensure Python 3.7+ is installed (python3 --version).
Copy the CSV file (100.csv) to ecg/data/100.csv in your project directory.
Ensure the CSV has an 'MLII' column for the ECG signal.


Install Dependencies:
sudo apt update
sudo apt install python3-pip python3-venv
cd ecg
python3 -m venv venv
source venv/bin/activate
pip install scipy numpy pandas matplotlib


Project Structure:

Place all files in the ecg folder:ecg/
├── data/
│   └── 100.csv
├── ecg_data_loader.py
├── ecg_filter.py
├── ecg_peak_detector.py
├── ecg_plotter.py
├── main.py
├── .gitignore
└── README.md




Run in VS Code:

Open the ecg folder in VS Code.
Select the Python interpreter from venv (Ctrl+Shift+P, "Python: Select Interpreter", choose ./venv/bin/python).
Open main.py, click the "Run" button or use the terminal:source venv/bin/activate
python main.py





Dependencies

Python 3.7+
numpy
pandas
scipy
matplotlib

Usage

The script processes record '100' from data/100.csv.
It performs:
Bandpass filtering (0.5-40 Hz).
R-peak detection.
Heart rate calculation (average and instantaneous).
Visualization of raw signal, filtered signal, R-peaks, and heart rate.


Outputs are saved as raw_ecg.png, filtered_ecg.png, r_peaks.png, heart_rate.png.

GitHub Repository Setup

Initialize a Git repository:git init
git add .
git commit -m "Initial commit"


Create a GitHub repository.
Push to GitHub:git remote add origin <your-repo-url>
git branch -M main
git push -u origin main



Blog Post Outline
ECG Signal Processing and Visualization

Introduction
Importance of ECG analysis
MIT-BIH dataset from Kaggle


Data Loading
Using local CSV
CSV format benefits


Preprocessing
Bandpass filtering (0.5-40 Hz)
Butterworth filter details


R-peak Detection
scipy.signal.find_peaks
Parameter tuning


Heart Rate Calculation
Average and instantaneous heart rates from RR intervals
Visualization of heart rate trends


Visualization
Raw and filtered signal plots
R-peak annotations
Heart rate plot


Conclusion
Results summary
Future improvements



License
MIT License
