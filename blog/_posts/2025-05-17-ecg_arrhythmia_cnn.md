# Journey to Building an ECG Arrhythmia CNN

# Introduction

On May 17, 2025, I embarked on an exciting project to develop a Convolutional Neural Network (CNN) for detecting arrhythmias in ECG signals, using the MIT-BIH Arrhythmia Database. This journey, guided by Grok 3 from xAI, transformed a challenging dataset into a robust model achieving a balanced accuracy of ~0.88.

# Challenges and Solutions

Initial Imbalance: Starting with records 100 and 101, the dataset had only 2 arrhythmic beats out of 135 segments, leading to poor metrics (Precision: 0.00, Recall: 0.00).
Data Expansion: Increasing samples to 40,000 per record improved segment count to 266 but kept arrhythmic beats at 2, highlighting the need for diverse data.
Adding Record 203: Including record 203, rich in ventricular tachycardia and PVCs, boosted arrhythmic beats to 51 out of 475 segments, significantly improving balance.
Splitting Issues: Early mismatches (e.g., 20 vs. 66 samples) were resolved by adjusting stratification logic, ensuring consistent splits.
Training Success: After 20 epochs, the model achieved Accuracy: 0.97, Precision: 0.88, Recall: 0.88, and F1: 0.88, exceeding the ~0.85 target.

# Lessons Learned

Data diversity is critical for imbalanced classification tasks.
Debugging with print statements and iterative testing is key to resolving complex errors.
Incremental improvements (e.g., adding records) can lead to substantial gains.

# Future Directions

Consider adding records 207 and 208 or applying class weights for further enhancement. A separate project focusing on high-arrhythmia records could also be explored.
Written by Wajih Humayun, May 17, 2025
