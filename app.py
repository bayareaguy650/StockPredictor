import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from textblob import TextBlob
import requests
import plotly.graph_objects as go

st.title("📊 Professional Stock Forecast App with Sentiment & Downloadable Forecasts")

# --- Inputs ---
ticker = st.text_input("Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

ma_short = st.slider("Short-term MA", 5, 30, 10)
ma_long = st.slider("Long-term MA", 30, 200, 50)
rf_trees = st.slider("Random Forest Trees", 10, 500, 100)
lstm_epochs = st.slider("LSTM Epochs", 5, 50, 20)
lstm_batch = st.slider("LSTM Batch Size", 8, 64, 16)
forecast_days = st.slider("Forecast Days", 1, 30, 5)

use_sentiment = st.checkbox("Include News Sentiment", True)
news_api_key = st.text_input("NewsAPI Key (optional)", "")

# --- Fetch stock data ---
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
   st.error("No data found.")
else:
   st.subheader("Historical Closing Prices")

   # Interactive Plotly plot
   fig_hist = go.Figure()
   fig_hist.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
   st.plotly_chart(fig_hist, use_container_width=True)

   # Feature Engineering
   data[f"MA_{ma_short}"] = data['Close'].rolling(ma_short).mean()
   data[f"MA_{ma_long}"] = data['Close'].rolling(ma_long).mean()

   feature_cols = [f"MA_{ma_short}", f"MA_{ma_long}"]

   # --- News Sentiment ---
   if use_sentiment and news_api_key:
       try:
           url = f"https://nam06.safelinks.protection.outlook.com/?url=https%3A%2F%2Fnewsapi.org%2Fv2%2Feverything%3Fq%3D&data=05%7C02%7Cadodeja%40linkedin.com%7Cdea16d3946464c6378c908ddec53df26%7C72f988bf86f141af91ab2d7cd011db47%7C0%7C0%7C638926567142528386%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=8%2FhUTB%2Fcwh1xVri5t5OQdeiXZCMyYvWiQq5eI3Mh%2FWs%3D&reserved=0{ticker}&sortBy=publishedAt&apiKey={news_api_key}"
           response = requests.get(url).json()
           headlines = [a['title'] for a in response.get('articles', [])[:10]]
           if headlines:
               sentiments = [TextBlob(title).sentiment.polarity for title in headlines]
               avg_sentiment = sum(sentiments)/len(sentiments)
           else:
               avg_sentiment = 0
       except:
           avg_sentiment = 0
       st.write(f"Average News Sentiment: {avg_sentiment:.2f}")
       data['Sentiment'] = avg_sentiment
       feature_cols.append('Sentiment')

   data = data.dropna()
   X = data[feature_cols]
   y = data['Close']

   split = int(len(X)*0.8)
   X_train, X_test = X.iloc[:split], X.iloc[split:]
   y_train, y_test = y.iloc[:split], y.iloc[split:]

   results = {}
   colors = ["blue", "red", "green"]

   # --- Models ---
   lr = LinearRegression(); lr.fit(X_train, y_train)
   y_pred_lr = lr.predict(X_test); mse_lr = mean_squared_error(y_test, y_pred_lr)
   lr_std = np.std(y_test - y_pred_lr)
   results["Linear Regression"] = (y_pred_lr, mse_lr, lr_std)

   rf = RandomForestRegressor(n_estimators=rf_trees, random_state=42)
   rf.fit(X_train, y_train)
   y_pred_rf = rf.predict(X_test); mse_rf = mean_squared_error(y_test, y_pred_rf)
   rf_std = np.std(y_test.values.ravel() - y_pred_rf.ravel())
  # rf_std = np.std(y_test - y_pred_rf)
   results["Random Forest"] = (y_pred_rf, mse_rf, rf_std)

   X_lstm = np.array(X).reshape((len(X),1,X.shape[1])); y_lstm = np.array(y)
   X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
   y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]
   lstm_model = Sequential()
   lstm_model.add(LSTM(50, return_sequences=False, input_shape=(1,X.shape[1])))
   lstm_model.add(Dense(1))
   lstm_model.compile(optimizer='adam', loss='mse')
   lstm_model.fit(X_train_lstm, y_train_lstm, epochs=lstm_epochs, batch_size=lstm_batch, verbose=0)
   y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
   mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
   lstm_std = np.std(y_test_lstm - y_pred_lstm)
   results["LSTM"] = (y_pred_lstm, mse_lstm, lstm_std)

   st.subheader("Model Comparison (MSE & CI)")
  
   for model_name, (pred, mse, std) in results.items():
        print(f"DEBUG: model_name={model_name}, mse={mse}, std={std}")
        st.write(f"{model_name}: MSE={float(mse):.2f}, CI ±{float(std):.2f}")

   # --- Interactive Predictions Plot ---
   fig_pred = go.Figure()
   fig_pred.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual', line=dict(color='black')))
   for i, (model_name, (pred, _, std)) in enumerate(results.items()):
       fig_pred.add_trace(go.Scatter(x=y_test.index, y=pred, mode='lines', name=model_name, line=dict(color=colors[i])))
       fig_pred.add_trace(go.Scatter(x=y_test.index, y=np.array(pred) + float(std), mode='lines', name=f"{model_name} +CI", line=dict(color=colors[i], dash='dash')))
       fig_pred.add_trace(go.Scatter(x=y_test.index, y=np.array(pred) - float(std), mode='lines', name=f"{model_name} -CI", line=dict(color=colors[i], dash='dash')))
   st.plotly_chart(fig_pred, use_container_width=True)

   # --- Forecast Future ---
   last_row = X.iloc[-1:].copy()
   last_date = pd.to_datetime(data.index[-1])
   future_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
   forecasts = {}
   for model_name, model_info in zip(["Linear Regression", "Random Forest", "LSTM"], [lr, rf, lstm_model]):
    future_preds = []
    current_features = last_row.values.astype(np.float32)  # ensure float32
    std = results[model_name][2]

    for _ in range(forecast_days):
        if model_name == "LSTM":
            inp = current_features.reshape((1, 1, X.shape[1]))
            pred = float(model_info.predict(inp)[0][0])
        else:
            inp = current_features.reshape(1, -1)
            pred = float(model_info.predict(inp)[0])

        future_preds.append(pred)

        # update feature row
        ma_short_val = (current_features[0][0]*(ma_short-1)+pred)/ma_short
        ma_long_val = (current_features[0][1]*(ma_long-1)+pred)/ma_long

        if 'Sentiment' in feature_cols:
            sentiment_val = current_features[0][2]
            current_features = np.array([[ma_short_val, ma_long_val, sentiment_val]], dtype=np.float32)
        else:
            current_features = np.array([[ma_short_val, ma_long_val]], dtype=np.float32)

    forecasts[model_name] = (future_preds, std)


   # --- Interactive Forecast Plot ---
   fig_forecast = go.Figure()
   fig_forecast.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical', line=dict(color='black')))
   for i, (model_name, (preds, std)) in enumerate(forecasts.items()):
       fig_forecast.add_trace(go.Scatter(x=future_index, y=preds, mode='lines', name=f"{model_name} Forecast", line=dict(color=colors[i])))
       fig_forecast.add_trace(go.Scatter(x=future_index, y=np.array(preds)+std, mode='lines', name=f"{model_name}+CI", line=dict(color=colors[i], dash='dash')))
       fig_forecast.add_trace(go.Scatter(x=future_index, y=np.array(preds)-std, mode='lines', name=f"{model_name}-CI", line=dict(color=colors[i], dash='dash')))
   st.plotly_chart(fig_forecast, use_container_width=True)

   # --- Downloadable Forecast CSV ---
   download_df = pd.DataFrame({model_name: forecasts[model_name][0] for model_name in forecasts}, index=future_index)
   st.download_button("Download Forecast CSV", download_df.to_csv().encode('utf-8'), file_name=f"{ticker}_forecast.csv")