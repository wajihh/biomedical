# Case Study: Stress and Sleep Detection with WESAD

## Introduction
This case study explores stress and sleep stage detection using the WESAD dataset, leveraging ECG, EDA, and respiration signals with a CNN-LSTM model.

## Methodology

Dataset: WESAD provides multimodal data from 15 subjects, with ECG (700 Hz), EDA, and respiration signals recorded during neutral, stress, and amusement states.
Preprocessing: Signals were normalized, resampled (EDA to 700 Hz), and segmented into 5-second windows with 50% overlap.
Model: A CNN-LSTM architecture processed multi-sensor data, trained for 20 epochs with stratification.

## Results

Achieved a test accuracy of 1.00 for subject S2, with validation accuracy also at 1.00, indicating potential overfitting or data leakage.
Multi-modal integration improved feature representation over single-sensor models.
Plots generated (signals.png, accuracy.png) confirmed signal quality and training trends.

## Challenges

Handling imbalanced classes required stratification.
High-frequency data processing demanded memory optimization (8 GB RAM).
Perfect accuracy suggests overfitting; further validation with other subjects (e.g., S3) and regularization (dropout added) is underway.

## Conclusion
Multi-modal analysis with WESAD shows promise for stress/sleep detection, but results need validation across subjects to ensure generalizability.
Author: Wajih Humayun Hashmi, May 21, 2025
