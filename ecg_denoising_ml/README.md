# ECG Denoising and Arrhythmia Classification

## Overview

This project denoises ECG signals using wavelet transforms, extracts HRV and RR interval features, and classifies arrhythmias using SVM, KNN, and Decision Tree models. It uses the MIT-BIH Arrhythmia Database (record 100.csv) and runs in VS Code on Ubuntu.

## Setup

Prerequisites:
Python 3.7+
100.csv in ecg_denoising_ml/data/100.csv with 'MLII' column.


Install Dependencies:cd ecg_denoising_ml
python3 -m venv venv
source venv/bin/activate
pip install pywavelets scikit-learn scipy numpy pandas matplotlib


Run:python main.py


Outputs:
Plots: raw_ecg.png, denoised_ecg.png, r_peaks.png, hrv.png
Console: R-peak count, classifier metrics
PDF report: report.pdf (compile report.tex with latexmk)



## Project Structure

ecg_data_loader.py: Loads ECG data
ecg_denoiser.py: Wavelet denoising
ecg_feature_extractor.py: HRV, RR intervals
ecg_classifier.py: SVM, KNN, Decision Tree
ecg_plotter.py: Visualization
main.py: Pipeline
report.tex: LaTeX report

Notes

Labels are simulated. Use MIT-BIH annotations for real classification.
Adjust R-peak parameters (height, distance) if needed.

License
MIT License
