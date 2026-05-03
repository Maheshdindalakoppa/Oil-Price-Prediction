import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import forecast_future

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="AI Oil Forecast", layout="wide")

st.title("🛢️ AI Oil Price Intelligence Dashboard")

st.markdown("Powered by LSTM Deep Learning Model")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\Projects\Oil price prediction\Crude oil - Crude oil.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

df = load_data()

# Standardize column
if "Close/Last" in df.columns:
    df.rename(columns={"Close/Last": "Price"}, inplace=True)

target = "Price"

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Controls")

forecast_days = st.sidebar.slider("Forecast Days", 1, 60, 30)
show_data = st.sidebar.checkbox("Show Dataset")

# -----------------------------
# CURRENT PRICE
# -----------------------------
current_price = df[target].iloc[-1]
previous_price = df[target].iloc[-2]

price_change = current_price - previous_price
price_change_pct = (price_change / previous_price) * 100

# -----------------------------
# AI TREND LOGIC
# -----------------------------
def get_trend(pred_values):
    if pred_values[-1] > pred_values[0]:
        return "📈 Uptrend"
    elif pred_values[-1] < pred_values[0]:
        return "📉 Downtrend"
    else:
        return "➡ Sideways"

# -----------------------------
# DASHBOARD METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current Price", f"${current_price:.2f}")

with col2:
    st.metric("Change", f"{price_change:.2f}", f"{price_change_pct:.2f}%")

# -----------------------------
# DATA VIEW
# -----------------------------
if show_data:
    st.subheader("📊 Dataset")
    st.dataframe(df.tail())

# -----------------------------
# HISTORICAL CHART
# -----------------------------
st.subheader("📈 Historical Trend")

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df['Date'], df[target], color="blue")
ax1.set_title("Oil Price History")
st.pyplot(fig1)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔮 Generate AI Forecast"):

    try:
        last_60 = df[target].values[-60:]

        forecast = forecast_future(last_60, days=forecast_days)
        forecast = forecast.flatten()

        future_dates = pd.date_range(df['Date'].iloc[-1], periods=forecast_days + 1)[1:]

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast
        })

        # -----------------------------
        # AI INSIGHTS
        # -----------------------------
        trend = get_trend(forecast)

        predicted_avg = np.mean(forecast)
        predicted_max = np.max(forecast)
        predicted_min = np.min(forecast)

        st.success("AI Prediction Completed")

        st.subheader("🧠 AI Insights")

        st.info(f"""
        📊 Predicted Trend: {trend}

        🔮 Average Forecast Price: ${predicted_avg:.2f}

        📈 Expected High: ${predicted_max:.2f}

        📉 Expected Low: ${predicted_min:.2f}
        """)

        # -----------------------------
        # PLOT
        # -----------------------------
        fig2, ax2 = plt.subplots(figsize=(12, 5))

        ax2.plot(df['Date'].tail(100), df[target].tail(100),
                 label="Historical", color="black")

        ax2.plot(forecast_df["Date"], forecast_df["Forecast"],
                 label="Forecast", color="orange")

        ax2.set_title("AI Forecast (LSTM)")
        ax2.legend()

        st.pyplot(fig2)

        # -----------------------------
        # TABLE
        # -----------------------------
        st.subheader("📋 Forecast Table")
        st.dataframe(forecast_df)

        # -----------------------------
        # DOWNLOAD
        # -----------------------------
        csv = forecast_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇ Download Forecast Report",
            data=csv,
            file_name="ai_oil_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")