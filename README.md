# Runguard AI

## Multi-Horizon Cryptocurrency Forecasting with a Neural Network

Runguard AI is a web application that uses a neural network built from scratch with NumPy to forecast cryptocurrency prices across multiple timeframes. It provides live price data, interactive charts, and prediction tools for up to 20 cryptocurrencies.

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Tech Stack](#tech-stack)
5. [Neural Network Architecture](#neural-network-architecture)
6. [Data Pipeline](#data-pipeline)
7. [Backtesting](#backtesting)
8. [Live Deployment](#live-deployment)
9. [Future Improvements](#future-improvements)

---

## Overview

Cryptocurrency markets are notoriously volatile and difficult to predict. Runguard AI attempts to address this challenge by applying a **feedforward neural network** to historical price data, aiming to forecast future price movements with a **multi-horizon approach** (from 1 hour to 30 days).

The project is designed as a **website-first MVP**, allowing users to:
- Select from a curated list of cryptocurrencies
- Choose between hourly or daily candle data
- Predict future prices across various timeframes
- View live prices and interactive charts

---

## Features

- **Live Price Tracking** – Real-time prices for multiple cryptocurrencies via CoinGecko API
- **Neural Network Predictions** – Forecast prices using a custom NumPy implementation
- **Multi-Horizon Forecasting** – Predict from 1 hour to 30 days ahead
- **Interactive Charts** – Visualize price history and predictions
- **Live Price Table** – See all coins at a glance on the home page
- **Theme Switcher** – Choose between Dark, Light, Rose, and Amber themes
- **Responsive Design** – Works on desktop, tablet, and mobile
- **Backtesting** – Validate model accuracy with walk-forward testing
- **Prediction History** – Track past predictions and their outcomes

---

## How It Works

1. **Data Collection** – The app fetches historical market data from CoinGecko's API
2. **Feature Engineering** – Raw price data is transformed into technical indicators
3. **Model Training** – A neural network is trained on historical data
4. **Live Prediction** – The model uses the latest live price to make forecasts
5. **Result Display** – Predictions are anchored to the actual live market price

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python, FastAPI, Uvicorn |
| **Frontend** | HTML, CSS, JavaScript |
| **Machine Learning** | NumPy (from scratch) |
| **Data Source** | CoinGecko API |
| **Database** | Local file-based cache (CSV/JSON) |
| **Deployment** | Railway |
| **Styling** | Custom CSS |

---

## Neural Network Architecture

The neural network is implemented **from scratch** using only NumPy, with no external ML libraries.

### Structure:
- **Input Layer**: 12 features (OHLCV, returns, SMA, volatility, etc.)
- **Hidden Layers**: 32 neurons with ReLU activation
- **Output Layer**: 1 neuron (predicted price)

### Training:
- **Optimizer**: Gradient Descent
- **Loss Function**: Mean Squared Error (MSE)
- **Validation**: Chronological split (80/20)

### Ensemble Strategy:
The model compares its predictions against a **simple recent-return baseline** and selects the better-performing strategy on validation data, ensuring robustness.

---

## Data Pipeline

1. **API Call** – Fetch historical market data from CoinGecko
2. **Caching** – Store data locally to reduce API calls
3. **Feature Engineering** – Calculate returns, moving averages, volatility
4. **Model Training** – Train a horizon-specific model
5. **Live Anchoring** – Combine historical predictions with live prices

---

## Backtesting

The backtesting module performs **walk-forward validation** to assess model performance on unseen data.

### Metrics:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **Directional Accuracy**

### Results:
Backtest results are saved to `data/backtests/runguard_backtest.csv` for analysis.

---

## Live Deployment

The application is deployed on **Railway** and accessible publicly.

**Live URL**: `https://web-production-9fd1b.up.railway.app`

---

## Future Improvements

- Add more cryptocurrencies
- Implement LSTM/GRU architectures
- Incorporate sentiment analysis
- Add portfolio management features
- Integrate real-time WebSocket updates

---

## License

This project is for educational purposes only. Use at your own risk.