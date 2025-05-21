# Case Study: Stress and Sleep Detection with WESAD

## Introduction
This case study explores stress and sleep stage detection using the WESAD dataset, leveraging ECG, EDA, and respiration signals to classify states (baseline, stress, amusement) with a CNN-LSTM model.
Methodology
Data Collection

Dataset: WESAD, collected from 15 subjects (S1 and S12 excluded due to sensor issues).
Signals:
ECG (700 Hz, mV) and respiration (700 Hz, %) from RespiBAN (SX_respiban.txt).
EDA (4 Hz, μS) from Empatica E4 (SX_E4_Data/EDA.csv).
Labels (700 Hz) from SX.pkl (0=transient, 1=baseline, 2=stress, 3=amusement, 4-7=other).



## Preprocessing

Aligned signal lengths by truncating to the shortest array.
Resampled EDA to 700 Hz using scipy.signal.resample.
Filtered labels to include only baseline (1), stress (2), and amusement (3), mapping to 0, 1, 2.
Normalized signals to [0, 1].
Segmented into 5-second windows (3500 samples at 700 Hz) with 50% overlap, using majority voting for segment labels.

## Model Architecture

Input Shape: (3500, 3) for ECG, EDA, respiration.
Layers:
Conv1D (32 filters, kernel size 5, ReLU).
MaxPooling1D (pool size 2).
LSTM (64 units, return sequences), Dropout (0.3).
LSTM (32 units), Dropout (0.3).
Dense (16 units, ReLU), Dropout (0.3).
Dense (3 units, softmax).


Optimizer: Adam.
Loss: Sparse categorical crossentropy.
Training: 20 epochs, batch size 32, 80-20 train-test split (with 75-25 train-validation split).

## Results
Performance Comparison



Subject
Gender
Raw Label Distribution
Filtered Label Distribution
Final Label Distribution
Test Accuracy
Notes



S2
Male
TBD
300:400:300 (estimated)
TBD
1.00
Potential overfitting; validation 1.00 from Epoch 1.


S8
Female
{0: x, 1: y, 2: z, 3: w, 4: a, 5: b, 6: c, 7: d}
882:0:0
882:0:0
1.00
Single-class after filtering, invalid result.


S4
Male
{0: 2314199, 1: 810601, 2: 444500, 3: 260400, 4: 563500, 5: 35699, 6: 30800, 7: 36401}
810601:444500:260400
463:253:149
0.82
Successful multi-class classification after segmentation fix.



S2 Results:
Test accuracy 1.00, training loss from 0.8055 to 0.00019632.
Overfitting suspected due to segment overlap.


S8 Results:
Filtered to 882:0:0, training loss from 0.8464 to 0.00030067.
Accuracy 1.00 invalid due to single-class output.


S4 Results:
Raw labels diverse, filtered to 810601:444500:260400, final segments 463:253:149.
Test accuracy 0.82, training loss from 1.0096 to 0.4543.
Segmentation fix with majority voting resolved single-class issue.


## Plots:
signals.png: Visualized raw signals.
accuracy.png: Showed improving accuracy curves, peaking at 0.8825.



## Challenges

Label Segmentation: Initial issue resolved with majority voting.
Overfitting: Still present, likely due to segment overlap or model complexity.
Data Validation: Required manual alignment and debugging.
Memory Constraints: Managed with 8 GB RAM via batch size optimization.

## Conclusion
The CNN-LSTM model successfully classifies stress and sleep states for S4 with a test accuracy of 0.82 after fixing label segmentation. Overfitting remains a challenge, suggesting further optimization (e.g., reducing overlap, tuning dropout). Future work includes validating with S2, S8, and other subjects.
Author: Wajih Humayun Hashmi, May 21, 2025
