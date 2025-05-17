import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import io

st.title("📈 AI-Driven Stock Price Prediction")

uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        st.stop()

    if 'Date' not in data.columns or 'Close' not in data.columns:
        st.error("CSV must contain 'Date' and 'Close' columns.")
        st.stop()

    data = data[['Date', 'Close']].dropna()
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data = data.sort_values('Date')
    data.set_index('Date', inplace=True)


    st.subheader("Raw Data")
    st.line_chart(data['Close'])

    st.subheader("Summary Statistics")
    st.write(data.describe())

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

    # Load model with error handling
    try:
        model = load_model("model.keras")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    real_prices = scaler.inverse_transform(test_data[60:])

    # Evaluation metrics
    mse = mean_squared_error(real_prices, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_prices, predictions)
    r2 = r2_score(real_prices, predictions)

    st.subheader("Evaluation Metrics")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"R² Score: {r2:.4f}")
   
    # Prediction vs Actual plot
    st.subheader("Prediction vs Actual Stock Prices")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(real_prices, label='Real Prices', color='green')
    ax.plot(predictions, label='Predicted Prices', color='orange')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    st.pyplot(fig)

    # Residual errors plot
    residuals = real_prices.flatten() - predictions.flatten()
    st.subheader("Residual Errors Plot")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(residuals, label='Residuals', color='red')
    ax2.axhline(y=0, color='black', linestyle='--')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Error')
    ax2.legend()
    st.pyplot(fig2)

    # Download button for results
    st.subheader("Download Predictions")
    result_df = pd.DataFrame({
        'Date': data.index[-len(predictions):],
        'Real Price': real_prices.flatten(),
        'Predicted Price': predictions.flatten(),
        'Residual Error': residuals
    })
    csv = result_df.to_csv(index=False).encode()
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name='stock_price_predictions.csv',
        mime='text/csv',
    )
else:
    st.info("Please upload a CSV file to proceed.")
