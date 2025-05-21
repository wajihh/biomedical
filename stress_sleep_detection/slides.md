Slide Deck: Stress/Sleep Detection with WESAD
Slide 1: Title

Stress and Sleep Detection using WESAD
Presented by: Wajih Humayun Hashmi
Date: May 21, 2025

Slide 2: Introduction

Objective: Detect stress or sleep stages from ECG, EDA, and respiration.
Dataset: WESAD (15 subjects, multimodal signals).
Tools: Python, TensorFlow, Scikit-learn.

Slide 3: Methodology

Data Prep: Normalize, resample (EDA to 700 Hz), segment into 5s windows.
Model: CNN-LSTM with dropout for multi-modal classification.
Training: 20 epochs, stratified splits.

Slide 4: Results

Test Accuracy for S2: 1.00 (potential overfitting).
Multi-modal advantage confirmed; plots generated.
Testing with S3 for validation.

Slide 5: Challenges & Future Work

Challenges: Overfitting, memory usage.
Future: Cross-subject validation, real-time deployment.

Slide 6: Conclusion

WESAD enables robust stress/sleep detection.
Requires further validation for generalizability.
