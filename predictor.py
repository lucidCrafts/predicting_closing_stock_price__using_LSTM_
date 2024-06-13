import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class Predictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return self.scaler.inverse_transform(predictions)
    
    def evaluate(self, y_test, predictions):
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return rmse
    
    def plot_results(self, data, split, seq_length, predictions):
        train = data[:split + seq_length]
        valid = data[split + seq_length:]
        valid['Predictions'] = predictions

        plt.figure(figsize=(16, 8))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD')
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()
