import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
\
import joblib
import yfinance as yf
import os

LOOKBACK = 10
CONFIDENCE_THRESH = 0.60
MODEL_DIR = "models"

st.set_page_config(page_title="Stock Predictor", layout="centered")

st.title("Selective Strategy Stock Predictor")
st.write("Hybrid LSTM & Random Forest Ensemble Model")

tickers = ['GOOG', 'MSFT', 'AAPL', 'NVDA']
selected_ticker = st.selectbox("Select a Ticker", tickers)

@st.cache_resource
def load_models(ticker):
    lstm_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    rf_path = os.path.join(MODEL_DIR, f"{ticker}_rf.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.joblib")
    
    if os.path.exists(lstm_path) and os.path.exists(rf_path) and os.path.exists(scaler_path):
        lstm = tf.keras.models.load_model(lstm_path)
        rf = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)
        return lstm, rf, scaler
    return None, None, None

def calculate_rsi(data, window=14):
    diff = data.diff(1).dropna()
    gain = 0 * diff
    loss = 0 * diff
    gain[diff > 0] = diff[diff > 0]
    loss[diff < 0] = -diff[diff < 0]
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_and_preprocess(ticker, scaler):
    tk = yf.Ticker(ticker)
    data = tk.history(period="1y")
    
    if data.empty:
        st.error(f"Yahoo Finance API failed to return data for {ticker}. The server may be temporarily rate-limited.")
        return None, None, None
        
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    df = data[['Close']].copy()
    
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    sma10 = df['Close'].rolling(window=10).mean()
    sma50 = df['Close'].rolling(window=50).mean()
    
    df['SMA10_Dist'] = (df['Close'] - sma10) / sma10
    df['SMA50_Dist'] = (df['Close'] - sma50) / sma50
    
    df['Volatility'] = df['Log_Returns'].rolling(window=20).std()
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    
    df['Golden_Cross'] = (sma10 > sma50).astype(int)
    df['Death_Cross'] = (sma10 < sma50).astype(int)
    
    df.dropna(inplace=True)
    
    features = ['Log_Returns', 'SMA10_Dist', 'SMA50_Dist', 'Volatility', 'RSI', 'Golden_Cross', 'Death_Cross']
    feature_data = df[features]
    
    if feature_data.empty:
        st.error("Not enough historical data available to calculate technical indicators.")
        return None, None, None
    scaled_data = scaler.transform(feature_data.values)
    
    if len(scaled_data) >= LOOKBACK:
        X_seq = np.array([scaled_data[-LOOKBACK:]])
        X_flat = np.array([scaled_data[-1]])
        return X_seq, X_flat, float(df.iloc[-1]['Close'])
    return None, None, None

lstm_model, rf_model, trained_scaler = load_models(selected_ticker)

if st.button("Generate Prediction"):
    if lstm_model is None or rf_model is None or trained_scaler is None:
        st.error(f"Models or Scaler for {selected_ticker} not found in the {MODEL_DIR} directory.")
    else:
        with st.spinner("Fetching live data and running inference..."):
            X_seq, X_flat, latest_price = fetch_and_preprocess(selected_ticker, trained_scaler)
            
            if X_seq is not None:
                lstm_prob = lstm_model.predict(X_seq, verbose=0)[0][0]
                rf_prob = rf_model.predict_proba(X_flat)[0][1]
                
                buy_mask = (lstm_prob >= CONFIDENCE_THRESH) or (rf_prob >= CONFIDENCE_THRESH)
                sell_mask = (lstm_prob <= (1 - CONFIDENCE_THRESH)) or (rf_prob <= (1 - CONFIDENCE_THRESH))
                
                st.subheader(f"Latest Close Price: ${latest_price:.2f}")
                
                col1, col2 = st.columns(2)
                col1.metric("LSTM Confidence (Buy)", f"{lstm_prob * 100:.1f}%")
                col2.metric("RF Confidence (Buy)", f"{rf_prob * 100:.1f}%")
                
                if buy_mask and not sell_mask:
                    st.success("SIGNAL: BUY")
                elif sell_mask and not buy_mask:
                    st.error("SIGNAL: SELL")
                else:
                    st.warning("SIGNAL: HOLD (Insufficient Confidence / Conflicting Signals)")