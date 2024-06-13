import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
    
    def download_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return data
    
    def preprocess_data(self, data):
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        return scaled_data, scaler
    
    def create_sequences(self, data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
