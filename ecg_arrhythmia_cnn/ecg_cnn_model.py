import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class ECGCNNModel:
    def __init__(self, input_shape=(256, 1)):
        self.model = self.build_model(input_shape)
        self.results = {}

    def build_model(self, input_shape):
        """Define 1D CNN architecture."""
        model = Sequential([
            Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Train the CNN."""
        history = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        self.results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        return self.results

    def save_model(self, path='models/cnn_model.h5'):
        """Save trained model."""
        self.model.save(path)