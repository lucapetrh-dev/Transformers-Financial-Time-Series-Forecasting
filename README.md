# Transformers for Financial Time Series Forecasting in Cryptocurrency Markets

This Master Thesis project explores the application of Transformer models for financial time series forecasting, with a focus on cryptocurrency markets. The goal is to evaluate the effectiveness of Transformers in capturing complex patterns in highly volatile cryptocurrency price movements. In addition, Transformer models are compared against traditional time series models such as Long Short-Term Memory (LSTM) networks and LSTM-Convolutional Neural Network (LSTM-CNN) hybrids.

A significant feature of this research is the integration of sentiment analysis into the forecasting models. The idea is to assess how sentiment analysis metrics (e.g., social media activity and tweet volumes) impact the predictive accuracy of cryptocurrency price forecasts.

## Methodology

1. **Data Collection and Preprocessing**  
   Gather financial and sentiment data, followed by preprocessing steps for feature engineering and model input preparation.

2. **Feature Engineering**  
   Extract and transform raw financial and sentiment data into model-friendly features.

3. **Model Development**  
   Develop and implement the following models:
   - LSTM
   - LSTM-CNN
   - Transformer

4. **Parallel Experiments**  
   Conduct experiments using:
   - Financial data only
   - Financial data combined with sentiment analysis features

5. **Model Validation on Bitcoin (BTC)**  
   Validate the models by testing on Bitcoin (BTC) data.

6. **Application to Other Cryptocurrencies**  
   Apply the models to additional cryptocurrencies including AAVE, ADA, DOGE, ETH, XMR, and XRP.

7. **Transfer Learning Implementation**  
   Explore transfer learning techniques to improve model generalization across different cryptocurrencies.

## Key Findings

- **Transformer models** incorporating sentiment analysis tend to outperform models based solely on financial data features.
- **Sentiment data augmentation** does not significantly improve the performance of LSTM or LSTM-CNN models.
- These findings can help develop more accurate forecasting tools in speculative financial markets such as cryptocurrencies.

## Repository Structure

```
├── data/                            # Raw financial time series data
│
├── models/                          # Model implementations and experiments
│   ├── scripts/                     # Utility scripts
│   │   └── utils.py                 # Script with helper functions
│   ├── sentiment_scores/            # Models using sentiment data
│   │   ├── LSTM_CNN_sentiment.ipynb # LSTM-CNN model with sentiment data
│   │   ├── LSTM_sentiment.ipynb     # LSTM model with sentiment data
│   │   └── Transformer_sentiment.ipynb # Transformer model with sentiment data
│   ├── ARIMA.ipynb                  # ARIMA model experiment
│   ├── LSTM_CNN.ipynb               # LSTM-CNN hybrid model
│   ├── LSTM.ipynb                   # LSTM model
│   ├── run_all_sentiment.ipynb      # Run all models with sentiment data
│   └── run_all.ipynb                # Run all models with financial data only
│   └── Transformer.ipynb            # Transformer model experiment
│
├── results/                         # Results from model experiments
│
├── feature_engineering.ipynb        # Notebook for feature engineering
├── moving_average.ipynb             # Moving average calculations and tests
├── preprocessing.ipynb              # Data preprocessing and cleaning
└── stationarity.ipynb               # Stationarity tests and analysis
```

## Usage

Run and compare results across different models:
   - Use `run_all.ipynb` for financial data-only experiments.
   - Use `run_all_sentiment.ipynb` for financial + sentiment data experiments.

## Results

Detailed results, analysis, and performance metrics can be found within the respective notebook files. Key comparisons between the different models are summarized in the results section of each notebook.

### Comparison of Final Results

The following table compares the final results of all the model combinations, focusing on the MAE metric across different cryptocurrencies. The baseline model, predicting prices of tomorrow using today's price, is used for benchmarking purposes.

| **Model**               | **ADA**   | **BTC**      | **DOGE**  | **ETH**   | **XMR**    | **XRP**   | **AAVE**  |
|-------------------------|-----------|--------------|-----------|-----------|------------|-----------|-----------|
| **Baseline**             | 0.004711  | 121.214873   | 0.000804  | 8.774213  | 1.039195   | 0.003622  | 17.229896 |
| **LSTM**                 | 0.121723  | 0.058811     | 0.023134  | 0.039746  | 0.018086   | 0.018221  | 0.540885  |
| **LSTM-CNN**             | 0.081877  | 0.044432     | 0.050316  | 0.039123  | 0.046871   | 0.026126  | 0.442550  |
| **Transformer**          | 0.581803  | 0.016008     | 0.023030  | 0.090027  | 0.016065   | 0.017376  | 0.102130  |
| **LSTM-Sentiment**       | 0.050398  | 0.052252     | 0.013051  | 0.034302  | 0.012693   | 0.022688  | 0.470003  |
| **LSTM-CNN-Sentiment**   | 0.121398  | 0.072749     | 0.056103  | 0.066412  | 0.186515   | 0.102465  | 0.219814  |
| **Transformer-Sentiment**| **0.005927** | **0.013554**  | **0.005782** | **0.008844** | **0.009335** | **0.013617** | **0.483078** |

**Transformer-Sentiment** outperforms other models across almost all cryptocurrencies, achieving the lowest MAE values.
