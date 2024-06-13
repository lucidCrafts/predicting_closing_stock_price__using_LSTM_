import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

class StockModel:
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.model = self.build_model((seq_length, 1))
    
    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(120, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(60, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, X_train, y_train, batch_size=1, epochs=20):
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    
    def save_model(self, file_path):
        self.model.save(file_path)
    
    @staticmethod
    def load_model(file_path):
        return load_model(file_path)
    
    @classmethod
    def load_from_file(cls, file_path, seq_length):
        model = load_model(file_path)
        return cls(seq_length)
