# Stock Market Prediction using LSTM

This project aims to predict stock market performance using Long Short-Term Memory (LSTM) networks.

## Overview

Stock market prediction is a challenging task due to its inherent volatility and complexity. LSTM networks, a type of recurrent neural network (RNN), are well-suited for sequential data like stock prices. This project utilizes LSTM networks to predict future stock prices based on historical data.

## Features

- **Data Loading**: Utilizes Yahoo Finance API to download historical stock data.
- **Preprocessing**: Scales the data using Min-Max scaling to prepare it for LSTM model input.
- **Model Building**: Constructs an LSTM model architecture using TensorFlow and Keras.
- **Training**: Trains the LSTM model on historical stock price data.
- **Saving and Loading**: Saves trained models to disk for future use and loads them for prediction.
- **Prediction**: Makes predictions on unseen data using the trained model.
- **Evaluation**: Evaluates model performance using metrics such as Root Mean Squared Error (RMSE).

## Directory Structure

stock_prediction/
├── src/
│ ├── data_loader.py
│ ├── model.py
│ ├── predictor.py
│ └── main.py
├── saved_models/
│ └── stock_model.h5
├── config/
│ └── config.yaml (if needed for configuration)
├── requirements.txt
└── README.md


## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/lucidCrafts/predicting_closing_stock_price__using_LSTM_.git
    cd stock_prediction
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the main script**:
    ```bash
    python src/main.py
    ```

## Usage

1. **Data Preparation**:
    - Set the `ticker`, `start_date`, and `end_date` parameters in `main.py`.
    - Adjust other parameters such as `seq_length` and `split_ratio` as needed.

2. **Training**:
    - Run `main.py` to download data, preprocess it, train the model, and save it to disk.

3. **Prediction**:
    - Load the trained model using the `load_from_file` method in `StockModel`.
    - Make predictions on new data using the loaded model.

## Dependencies

- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow
- yfinance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI](https://openai.com) for providing the GPT model used for generating this README template.

