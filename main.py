from data_loader import DataLoader
from model import StockModel
from predictor import Predictor

    
    

    
def main():
    # Parameters
    ticker = 'TSLA'
    start_date = '2010-01-01'
    end_date = '2023-12-31'
    seq_length = 11
    split_ratio = 0.8
    model_file_path = 'saved_models/stock_model.h5'
    
    # Step 1: Download and preprocess data
    data_loader = DataLoader(ticker, start_date, end_date)
    data = data_loader.download_data()
    scaled_data, scaler = data_loader.preprocess_data(data)
    
    # Step 2: Create sequences
    X, y = data_loader.create_sequences(scaled_data, seq_length)
    
    # Step 3: Train-test split
    split = int(split_ratio * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Step 4: Build, train, and save the model
    stock_model = StockModel(seq_length)
    stock_model.train(X_train, y_train, batch_size=1, epochs=4)
    stock_model.save_model(model_file_path)
    
    # Step 5: Load the model and make predictions
    loaded_model = StockModel.load_from_file(model_file_path, seq_length)
    predictor = Predictor(loaded_model.model, scaler)
    scaled_predictions = predictor.predict(X_test)
    
    original_predictions = scaler.inverse_transform(scaled_predictions)
    y_test_augmented= scaler.inverse_transform(y_test)
    
        
    
    # Step 6: Evaluate and plot results
    rmse = predictor.evaluate(y_test_augmented, original_predictions)
    print(f'RMSE: {rmse}')
    predictor.plot_results(data, split, seq_length, original_predictions)

if __name__ == "__main__":
    main()
