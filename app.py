import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import timedelta

st.set_page_config(page_title="Coal Production Forecasting", layout="wide")
st.title("Coal Production Forecasting Dashboard")

# Sidebar Inputs
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with Date and Production columns", type="csv")

# Main Content
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip().lower() for col in df.columns]
    
    if 'date' not in df.columns or 'production' not in df.columns:
        st.error("CSV must contain 'date' and 'production' columns")
    else:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        st.subheader("Raw Data Preview")
        st.dataframe(df.tail())

        # Feature engineering
        df['days'] = (df['date'] - df['date'].min()).dt.days

        # Train/test split
        split_ratio = st.sidebar.slider("Training Ratio", 0.5, 0.95, 0.8)
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # Model
        model = LinearRegression()
        model.fit(train_df[['days']], train_df['production'])

        test_df['prediction'] = model.predict(test_df[['days']])

        # Metrics
        rmse = np.sqrt(mean_squared_error(test_df['production'], test_df['prediction']))
        r2 = r2_score(test_df['production'], test_df['prediction'])

        st.subheader("📊 Forecast vs Actual")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['date'], df['production'], label='Actual Production')
        ax.plot(test_df['date'], test_df['prediction'], label='Predicted Production')
        ax.set_xlabel("Date")
        ax.set_ylabel("Production")
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"*RMSE:* {rmse:.2f} | *R² Score:* {r2:.2f}")

        # Future Forecast
        st.subheader("🔮 Future Forecast")
        future_months = st.slider("Months to Forecast", 1, 12, 6)
        last_date = df['date'].max()
        last_day = (last_date - df['date'].min()).days

        future_days = np.array([last_day + 30*i for i in range(1, future_months + 1)]).reshape(-1, 1)
        future_dates = [last_date + timedelta(days=30*i) for i in range(1, future_months + 1)]
        future_preds = model.predict(future_days)

        future_df = pd.DataFrame({
            'date': future_dates,
            'predicted_production': future_preds
        })

        st.dataframe(future_df)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(future_df['date'], future_df['predicted_production'], marker='o', color='orange')
        ax2.set_title("Future Coal Production Forecast")
        st.pyplot(fig2)

else:
    st.info("Please upload a CSV file to get started. Example columns: 'date', 'production'")