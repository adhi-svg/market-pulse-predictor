import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.title("📈 AI-Driven Stock Price Prediction")

uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = data[['Date', 'Close']].dropna()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    data.set_index('Date', inplace=True)

    st.subheader("Raw Data")
    st.line_chart(data['Close'])

    # Preprocessing
    close_data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    test_data = scaled_data[int(len(scaled_data) * 0.8):]

    X_test = []
    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Load model
    model = load_model("model.keras")

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    real_prices = scaler.inverse_transform(test_data[60:])

    # Plot
    st.subheader("Prediction vs Actual")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(real_prices, label='Real Prices', color='green')
    ax.plot(predictions, label='Predicted Prices', color='orange')
    ax.legend()
    st.pyplot(fig)
