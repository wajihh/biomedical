# Stress and Sleep Detection using WESAD Dataset
## Wajih Humayun Hashmi
### May 21, 2025

---

## Introduction

- **Objective**: Classify physiological states (baseline, stress, amusement) using WESAD dataset
- **Dataset**: WESAD (15 subjects, S1/S12 excluded due to sensor issues)
- **Tools**: Python, TensorFlow, Scikit-learn
- **Signals**: ECG, EDA, and respiration data

---

## Methodology

### Data Preprocessing
- **Signal Processing**:
  - ECG & Respiration: 700 Hz sampling rate
  - EDA: Resampled from 4 Hz to 700 Hz
  - Normalized signals to [0, 1] range
- **Segmentation**:
  - 5-second windows (3500 samples)
  - 50% overlap between segments
  - Majority voting for segment labels

---

## Model Architecture

- **Input**: (3500, 3) for ECG, EDA, respiration
- **Layers**:
  1. Conv1D (32 filters, kernel size 5, ReLU)
  2. MaxPooling1D (pool size 2)
  3. LSTM (64 units, return sequences), Dropout (0.3)
  4. LSTM (32 units), Dropout (0.3)
  5. Dense (16 units, ReLU), Dropout (0.3)
  6. Dense (3 units, softmax)
- **Training**:
  - Optimizer: Adam
  - Loss: Sparse categorical crossentropy
  - 20 epochs, batch size 32
  - 80-20 train-test split (75-25 train-validation)

---

## Results

### Performance Comparison

| Subject | Gender | Label Distribution | Test Accuracy | Notes |
|---------|--------|-------------------|---------------|-------|
| S2 | Male | ~300:400:300 | 1.00 | Overfitting suspected |
| S8 | Female | 882:0:0 | 1.00 | Single-class (invalid) |
| S4 | Male | 463:253:149 | 0.82 | Successful multi-class |

### Key Findings
- S4 achieved 0.82 accuracy with balanced classes
- Training loss improved from 1.0096 to 0.4543
- Visualizations available in `signals.png` and `accuracy.png`

---

## Challenges & Future Work

### Challenges
1. **Overfitting**: Due to segment overlap
2. **Data Validation**: Required manual alignment
3. **Memory Constraints**: Managed with batch optimization
4. **Label Segmentation**: Resolved with majority voting

### Future Work
1. Validate model with S2, S8, and other subjects
2. Reduce segment overlap to minimize overfitting
3. Tune model architecture and hyperparameters
4. Explore alternative preprocessing techniques

---

## Conclusion

- CNN-LSTM model successfully classifies states with 0.82 accuracy
- Segmentation fix resolved single-class issues
- Further optimization needed for better generalization
- Promising results for stress/sleep detection applications

---

## References & Resources

### Dataset
- WESAD Dataset: [WESAD: A Multimodal Dataset for Wearable Stress and Affect Detection](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)

### Tools & Libraries
- TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)
- Python: [https://www.python.org/](https://www.python.org/)

### Related Research
- Schmidt, P., et al. (2018). "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection"
- Deep Learning for Physiological Signal Analysis: Recent Advances and Future Directions

---

*Thank You*
*Questions?*

