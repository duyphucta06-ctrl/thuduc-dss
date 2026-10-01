import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Hệ thống Tưới TP. Thủ Đức", page_icon="🌱", layout="wide")

LAT, LON = 10.8494, 106.7537

@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv("clean_dataset.csv")
        df["date"] = pd.to_datetime(df["date"])
        features = ["temp_max", "temp_min", "temp_mean", "humidity_mean", "radiation_sum", "precipitation_sum", "month", "is_rainy_season"]
        X = df[features]
        y = df["et0_gee"]
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return df, model, features
    except Exception as e:
        return None, None, None

def fetch_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT, "longitude": LON,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", 
                  "relative_humidity_2m_mean", "precipitation_sum", 
                  "precipitation_probability_max", "shortwave_radiation_sum"],
        "timezone": "Asia/Bangkok", "forecast_days": 1
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            d = res.json()["daily"]
            return {
                "temp_max": d["temperature_2m_max"][0], "temp_min": d["temperature_2m_min"][0],
                "temp_mean": d["temperature_2m_mean"][0], "humidity_mean": d["relative_humidity_2m_mean"][0],
                "precipitation_sum": d["precipitation_sum"][0], "pop_max": d["precipitation_probability_max"][0],
                "radiation_sum": d["shortwave_radiation_sum"][0]
            }
    except Exception:
        pass
    return None

st.title("🌱 Dashboard Hỗ trợ Quyết định Tưới tiêu - TP. Thủ Đức")

df_hist, model, features = load_and_train()
weather_live = fetch_weather()

col1, col2, col3, col4 = st.columns(4)
if weather_live:
    col1.metric("🌧️ % Mưa dự báo", f"{weather_live['pop_max']} %")
    col2.metric("💧 Lượng mưa", f"{weather_live['precipitation_sum']} mm")
    col3.metric("🌡️ Nhiệt độ TB", f"{weather_live['temp_mean']} °C")
    col4.metric("☀️ Bức xạ mặt trời", f"{weather_live['radiation_sum']} MJ/m²")

st.markdown("---")
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("🤖 Dự báo ET₀ ML & Nhu cầu tưới")
    kc = st.slider("Hệ số cây trồng (Kc):", 0.4, 1.3, 1.0, 0.05)
    if weather_live and model:
        now = datetime.datetime.now()
        input_data = pd.DataFrame([{
            "temp_max": weather_live["temp_max"], "temp_min": weather_live["temp_min"],
            "temp_mean": weather_live["temp_mean"], "humidity_mean": weather_live["humidity_mean"],
            "radiation_sum": weather_live["radiation_sum"], "precipitation_sum": weather_live["precipitation_sum"],
            "month": now.month, "is_rainy_season": 1 if 5 <= now.month <= 11 else 0
        }])
        et0_pred = model.predict(input_data)[0]
        etc = et0_pred * kc
        rain_eff = weather_live["precipitation_sum"] * 0.8
        i_rec = max(0.0, etc - rain_eff)

        st.success(f"**ET₀ Dự báo ML:** {et0_pred:.2f} mm/ngày")
        st.info(f"**Nhu cầu nước (ETc):** {etc:.2f} mm/ngày")
        if i_rec > 0:
            st.warning(f"💧 **Cần tưới bù:** {i_rec:.2f} mm/ngày ({i_rec*10:.1f} m³/ha)")
        else:
            st.success("🌧️ **Không cần tưới:** Lượng mưa đã đủ!")

with col_right:
    st.subheader("📈 So sánh ET₀ GEE vs ET₀ ML")
    if df_hist is not None and model is not None:
        scale_option = st.radio("Độ chia thời gian:", ["Tối đa (10 năm)", "1 Năm (12 Tháng)", "10 Ngày gần nhất"], horizontal=True)
        df_plot = df_hist.copy()
        df_plot["et0_ml"] = model.predict(df_plot[features])

        fig, ax = plt.subplots(figsize=(8, 4))
        if scale_option == "10 Ngày gần nhất":
            df_sub = df_plot.tail(10)
            ax.plot(df_sub["date"].dt.strftime('%m-%d'), df_sub["et0_gee"], marker='o', label="ET0 GEE", color="green")
            ax.plot(df_sub["date"].dt.strftime('%m-%d'), df_sub["et0_ml"], marker='x', linestyle="--", label="ET0 ML", color="orange")
        elif scale_option == "1 Năm (12 Tháng)":
            df_sub = df_plot[df_plot["date"].dt.year == 2024].groupby(df_plot["date"].dt.month)[["et0_gee", "et0_ml"]].mean().reset_index()
            ax.plot(df_sub["date"], df_sub["et0_gee"], marker='o', label="ET0 GEE", color="green")
            ax.plot(df_sub["date"], df_sub["et0_ml"], marker='x', linestyle="--", label="ET0 ML", color="orange")
        else:
            df_sub = df_plot.groupby(df_plot["date"].dt.year)[["et0_gee", "et0_ml"]].mean().reset_index()
            ax.plot(df_sub["date"], df_sub["et0_gee"], marker='o', label="ET0 GEE", color="green")
            ax.plot(df_sub["date"], df_sub["et0_ml"], marker='x', linestyle="--", label="ET0 ML", color="orange")

        ax.set_ylabel("ET0 (mm/ngày)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
