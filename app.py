# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# ---------- Page Setup ----------
st.set_page_config(page_title="Retail Sales Forecast Dashboard", layout="wide", page_icon="📊")
st.title("📊 Retail Sales Forecast Dashboard")

# ---------- Helper Functions ----------
def add_time_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['dow'] = df['Date'].dt.dayofweek
    return df

def compute_lags_rolls(df, lag_list=[1,2,3,4], roll_windows=[4,8]):
    df = df.sort_values(['Store','Date']).reset_index(drop=True).copy()
    for l in lag_list:
        df[f'lag_{l}'] = df.groupby('Store')['Weekly_Sales'].shift(l)
    for w in roll_windows:
        df[f'roll_{w}'] = (
            df.groupby('Store')['Weekly_Sales'].shift(1)
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    fill_cols = [f'lag_{l}' for l in lag_list] + [f'roll_{w}' for w in roll_windows]
    df[fill_cols] = df[fill_cols].fillna(0)
    return df

def build_pipeline(feature_cols):
    ohe = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer([('store_ohe', ohe, ['Store'])], remainder='passthrough')
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    return Pipeline([('prep', preprocessor), ('model', rf)])

def prepare_features_for_prediction(history_df, predict_dates_df, feature_cols):
    combined = pd.concat([history_df, predict_dates_df], sort=False).sort_values(['Store','Date']).reset_index(drop=True)
    combined = add_time_features(combined)
    combined = compute_lags_rolls(combined)
    for c in feature_cols:
        if c not in combined.columns:
            combined[c] = 0
    mask = combined['Date'].isin(predict_dates_df['Date'])
    return combined.loc[mask, feature_cols]

def generate_forecast(store_df, n_weeks, model, feature_cols):
    last_date = store_df['Date'].max()
    med_temp = store_df['Temperature'].median()
    med_fp = store_df['Fuel_Price'].median()
    med_cpi = store_df['CPI'].median()
    med_unemp = store_df['Unemployment'].median()

    temp_hist = store_df.copy()
    fut_rows = []

    for i in range(1, n_weeks + 1):
        pred_date = last_date + timedelta(weeks=i)
        fut_df = pd.DataFrame({
            'Store': [store_df['Store'].iloc[0]],
            'Date': [pred_date],
            'Weekly_Sales': [np.nan],
            'Temperature': [med_temp],
            'Fuel_Price': [med_fp],
            'CPI': [med_cpi],
            'Unemployment': [med_unemp],
            'Holiday_Flag': [0]
        })
        X_pred = prepare_features_for_prediction(temp_hist, fut_df, feature_cols)
        y_pred = model.predict(X_pred)[0]
        fut_df['Weekly_Sales'] = y_pred
        temp_hist = pd.concat([temp_hist, fut_df]).sort_values(['Store','Date']).reset_index(drop=True)
        fut_rows.append(fut_df)

    forecast_df = pd.concat(fut_rows).reset_index(drop=True)
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    return forecast_df

# ---------- Load Data ----------
DATA_PATH = "sales_data.csv"
MODEL_PATH = "sales_pipeline.pkl"

if not os.path.exists(DATA_PATH):
    st.error("❌ 'sales_data.csv' not found.")
    st.stop()

data = pd.read_csv(DATA_PATH)
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data = data.sort_values(['Store','Date']).reset_index(drop=True)

feature_cols = [
    'Store','year','month','week','dow','Holiday_Flag',
    'Temperature','Fuel_Price','CPI','Unemployment',
    'lag_1','lag_2','lag_3','lag_4','roll_4','roll_8'
]

# ---------- Load or Train Model ----------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    warnings.warn(f"Could not load model: {e}. Training a new model in the cloud...")
    data_feat = add_time_features(data)
    data_feat = compute_lags_rolls(data_feat)
    model = build_pipeline(feature_cols)
    model.fit(data_feat[feature_cols], data_feat['Weekly_Sales'])
    joblib.dump(model, MODEL_PATH)

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Forecast Settings")
stores_selected = st.sidebar.multiselect("Select Store(s)", options=sorted(data['Store'].unique()), default=[data['Store'].iloc[0]])
n_weeks = st.sidebar.slider("Forecast Weeks", 1, 52, 4)
run_forecast = st.sidebar.button("Run Forecast")

# ---------- KPI Card Colors ----------
card_colors = ["#FFB3BA","#BAE1FF","#BAFFC9","#FFFFBA","#FFDFBA","#E2BAFF","#FFC2E2","#C2FFD6"]

# ---------- Main Dashboard ----------
if run_forecast and stores_selected:
    all_forecasts = []
    st.info(f"Running {n_weeks}-week forecast for store(s): {', '.join(map(str, stores_selected))}")

    # KPI Cards
    kpi_cols = st.columns(len(stores_selected))
    for i, store in enumerate(stores_selected):
        store_df = data[data['Store']==store].copy()
        forecast_df = generate_forecast(store_df, n_weeks, model, feature_cols)
        all_forecasts.append((store, forecast_df))

        last_week = store_df['Weekly_Sales'].iloc[-1]
        next_week = forecast_df['Weekly_Sales'].iloc[0]
        pct_change = ((next_week - last_week)/last_week)*100

        arrow = "🔺" if pct_change >= 0 else "🔻"
        arrow_color = "green" if pct_change >= 0 else "red"
        color = card_colors[i % len(card_colors)]

        kpi_cols[i].markdown(f"""
        <div style='background-color:{color};padding:20px;border-radius:15px;text-align:center;
                            box-shadow:2px 2px 12px rgba(0,0,0,0.2)'>
            <h4>Store {store} 💰</h4>
            <h2>{next_week:.0f}</h2>
            <h4 style='color:{arrow_color}'>{arrow} {pct_change:.2f}% vs last week</h4>
        </div>
        """, unsafe_allow_html=True)

    # Forecast Chart
    fig = make_subplots(rows=1, cols=1)
    for idx, (store, forecast_df) in enumerate(all_forecasts):
        hist_df = data[data['Store']==store]
        color_line = card_colors[idx % len(card_colors)]
        fig.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Weekly_Sales'], mode='lines+markers',
                                 name=f"Store {store} History", line=dict(color=color_line)))
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Weekly_Sales'], mode='lines+markers',
                                 name=f"Store {store} Forecast", line=dict(color=color_line, dash='dash')))
        fig.add_vrect(x0=forecast_df['Date'].min(), x1=forecast_df['Date'].max(),
                      fillcolor=color_line, opacity=0.1, line_width=0)

    fig.update_layout(title="📈 Weekly Sales Forecast",
                      xaxis_title="Date", yaxis_title="Weekly Sales",
                      template="plotly_white", hovermode="x unified")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 KPIs & Chart", "📋 Forecast Table", "📥 Download Reports"])
    with tab1:
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        for store, forecast_df in all_forecasts:
            st.subheader(f"Store {store}")
            st.dataframe(forecast_df[['Date','Weekly_Sales']].reset_index(drop=True)
                         .style.background_gradient(cmap='Blues'))
    with tab3:
        for store, forecast_df in all_forecasts:
            csv_data = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(f"📥 Download CSV for Store {store}", data=csv_data,
                               file_name=f"forecast_store_{store}.csv")

    # Dynamic Summary
    for store, forecast_df in all_forecasts:
        first_forecast = forecast_df['Weekly_Sales'].iloc[0]
        last_hist = data[data['Store']==store]['Weekly_Sales'].iloc[-1]
        arrow = "🔺" if first_forecast >= last_hist else "🔻"
        st.info(f"{arrow} Store {store}: Expected change {last_hist:.0f} → {first_forecast:.0f} "
                f"({((first_forecast - last_hist)/last_hist)*100:.2f}%) next week")

st.info("This dashboard forecasts weekly retail sales using a RandomForest model trained on historical data. 🚀")
