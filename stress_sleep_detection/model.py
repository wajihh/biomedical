import tensorflow as tf
from tensorflow.keras import layers, models

class MultiModalModel:
    def __init__(self, input_shape=(3500, 3), num_classes=3):
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape, num_classes):
        model = models.Sequential([
            layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dense(16, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y, X_val, y_val, epochs=20, batch_size=32):
        history = self.model.fit(X, y, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
        return history