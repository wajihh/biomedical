# Slide Deck: Stress/Sleep Detection with WESAD
## Slide 1: Title

Stress and Sleep Detection using WESAD
Presented by: Wajih Humayun Hashmi
Date: May 21, 2025

## Slide 2: Introduction

Objective: Classify baseline, stress, and amusement states.
Dataset: WESAD (15 subjects, S1/S12 excluded).
Tools: Python, TensorFlow, Scikit-learn.

## Slide 3: Methodology

Data Prep:
Signals: ECG, EDA, respiration (700 Hz).
Resample EDA, normalize, segment (5s, 50% overlap).


Model:
CNN-LSTM with dropout (0.3).
20 epochs, batch size 32, stratified splits.



## Slide 4: Results

Performance Comparison:
S2 (Male): ~300:400:300 labels, Accuracy: 1.00 (overfitting).
S8 (Female): 882:0:0 filtered, Accuracy: 1.00 (single-class).
S4 (Male): 810601:444500:260400 filtered, 463:253:149 final, Accuracy: 0.82.


Plots: signals.png, accuracy.png generated.

## Slide 5: Challenges & Future Work

Challenges:
Overfitting due to overlap.
Initial segmentation issues resolved.


Future Work:
Validate with S2, S8, others.
Reduce overlap, tune model.



Slide 6: Conclusion

CNN-LSTM achieves 0.82 accuracy for S4.
Next steps: Optimize and expand validation.

