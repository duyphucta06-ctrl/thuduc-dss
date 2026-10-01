import datetime
import requests
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Hệ thống Hỗ trợ Quyết định Tưới tiêu - TP. Thủ Đức",
    page_icon="🌱",
    layout="wide"
)

LAT, LON = 10.8494, 106.7537
MODEL_FILE = "rf_et0_gee.pkl"
DATA_FILE = "clean_dataset.csv"
FEATURES = ["temp_max", "temp_min", "temp_mean", "humidity_mean", "radiation_sum", "precipitation_sum", "month", "is_rainy_season"]

# --- 1. TẢI MÔ HÌNH VÀ DỮ LIỆU LỊCH SỬ ---
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return None

@st.cache_resource
def load_ml_model(df_hist):
    try:
        return joblib.load(MODEL_FILE)
    except Exception:
        if df_hist is not None:
            from sklearn.ensemble import RandomForestRegressor
            X = df_hist[FEATURES]
            y = df_hist["et0_gee"]
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            return model
        return None

# --- 2. CẬP NHẬT DỮ LIỆU THỜI TIẾT THỜI GIAN THỰC ĐẾN HÔM NAY ---
@st.cache_data(ttl=3600)
def fetch_realtime_weather_series():
    """Tải chuỗi dữ liệu thời tiết cập nhật tới ngày hôm nay"""
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=30) # Lấy 30 ngày gần nhất
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": today.strftime("%Y-%m-%d"),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "precipitation_probability_max",
            "shortwave_radiation_sum",
        ],
        "timezone": "Asia/Bangkok",
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()["daily"]
            df_recent = pd.DataFrame({
                "date": pd.to_datetime(data["time"]),
                "temp_max": data["temperature_2m_max"],
                "temp_min": data["temperature_2m_min"],
                "temp_mean": data["temperature_2m_mean"],
                "humidity_mean": data["relative_humidity_2m_mean"],
                "precipitation_sum": data["precipitation_sum"],
                "pop_max": data["precipitation_probability_max"],
                "radiation_sum": data["shortwave_radiation_sum"]
            })
            df_recent["month"] = df_recent["date"].dt.month
            df_recent["is_rainy_season"] = df_recent["month"].apply(lambda x: 1 if 5 <= x <= 11 else 0)
            return df_recent
    except Exception as e:
        st.error(f"Không thể cập nhật API Open-Meteo: {e}")
    return None

# --- GIAO DIỆN APP ---
st.title("🌱 Dashboard Hỗ trợ Quyết định Tưới tiêu - TP. Thủ Đức")
st.caption("Dữ liệu tự động cập nhật thời gian thực từ Open-Meteo API & Mô hình ML dự báo ET₀")

df_hist = load_historical_data()
model = load_ml_model(df_hist)
df_recent = fetch_realtime_weather_series()

# 1. Hiển thị chỉ số dự báo hôm nay
if df_recent is not None and not df_recent.empty:
    today_data = df_recent.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌧️ % Mưa hôm nay", f"{today_data['pop_max']} %")
    col2.metric("💧 Lượng mưa dự báo", f"{today_data['precipitation_sum']} mm")
    col3.metric("🌡️ Nhiệt độ TB", f"{today_data['temp_mean']} °C")
    col4.metric("☀️ Bức xạ mặt trời", f"{today_data['radiation_sum']} MJ/m²")

st.markdown("---")

col_left, col_right = st.columns([1, 1.2])

# 2. Cột trái: Khuyến nghị tưới
with col_left:
    st.subheader("🤖 Dự báo ET₀ ML & Nhu cầu tưới")
    kc = st.slider("Hệ số cây trồng (Kc):", 0.4, 1.3, 1.0, 0.05)
    
    if df_recent is not None and model is not None:
        today_data = df_recent.iloc[-1]
        input_data = pd.DataFrame([today_data[FEATURES]])
        
        et0_pred = model.predict(input_data)[0]
        etc = et0_pred * kc
        rain_eff = today_data["precipitation_sum"] * 0.8
        i_rec = max(0.0, etc - rain_eff)
        
        st.success(f"**ET₀ Dự báo ML hôm nay:** {et0_pred:.2f} mm/ngày")
        st.info(f"**Nhu cầu nước cây trồng (ETc):** {etc:.2f} mm/ngày")
        
        if i_rec > 0:
            st.warning(f"💧 **Nước cần tưới bù:** {i_rec:.2f} mm/ngày ({i_rec*10:.1f} m³/ha)")
        else:
            st.success("🌧️ **Không cần tưới:** Lượng mưa hôm nay đã đủ!")

# 3. Cột phải: Biểu đồ kết hợp chuỗi thời gian thực
with col_right:
    st.subheader("📈 So sánh ET₀ GEE vs ET₀ ML")
    
    if df_hist is not None and model is not None:
        # Gộp dữ liệu lịch sử và chuỗi thời gian thực mới nhất
        df_combined = df_hist.copy()
        
        if df_recent is not None:
            # Thêm các dòng mới từ API vào chuỗi dữ liệu
            new_rows = df_recent[~df_recent["date"].isin(df_combined["date"])].copy()
            if not new_rows.empty:
                # Dữ liệu GEE vệ tinh cho ngày mới sẽ trễ 1 ngày (chèn giá trị nội suy/trễ)
                new_rows["et0_gee"] = np.nan
                df_combined = pd.concat([df_combined, new_rows], ignore_index=True)
                df_combined["et0_gee"] = df_combined["et0_gee"].interpolate(method="linear").bfill()

        # Dự báo ET0 bằng ML cho toàn bộ chuỗi
        df_combined["et0_ml"] = model.predict(df_combined[FEATURES])
        
        scale_option = st.radio(
            "Độ chia thời gian:",
            ["10 Ngày gần nhất (Real-time)", "1 Năm (12 Tháng)", "Tối đa (Theo Toàn bộ các năm)"],
            horizontal=True
        )
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        if scale_option == "10 Ngày gần nhất (Real-time)":
            df_sub = df_combined.tail(10)
            x_dates = df_sub["date"].dt.strftime('%m-%d')
            
            # ET0 GEE chỉ vẽ tới ngày T-1 (trễ 1 ngày)
            ax.plot(x_dates[:-1], df_sub["et0_gee"][:-1], marker='o', label="ET0 GEE (Trễ 1 ngày)", color="green")
            # ET0 ML vẽ cập nhật đến tận hôm nay
            ax.plot(x_dates, df_sub["et0_ml"], marker='x', linestyle="--", label="ET0 ML (Thời gian thực)", color="orange")
            ax.set_xlabel("Ngày (Tháng-Ngày)")
            
        elif scale_option == "1 Năm (12 Tháng)":
            available_years = sorted(df_combined["date"].dt.year.unique(), reverse=True)
            selected_year = st.selectbox("Chọn năm hiển thị:", available_years)
            
            df_year = df_combined[df_combined["date"].dt.year == selected_year]
            df_sub = df_year.groupby(df_year["date"].dt.month)[["et0_gee", "et0_ml"]].mean().reset_index()
            
            ax.plot(df_sub["date"], df_sub["et0_gee"], marker='o', label="ET0 GEE", color="green")
            ax.plot(df_sub["date"], df_sub["et0_ml"], marker='x', linestyle="--", label="ET0 ML", color="orange")
            ax.set_xlabel(f"Tháng (Năm {selected_year})")
            ax.set_xticks(range(1, 13))
            
        else:
            df_sub = df_combined.groupby(df_combined["date"].dt.year)[["et0_gee", "et0_ml"]].mean().reset_index()
            ax.plot(df_sub["date"], df_sub["et0_gee"], marker='o', label="ET0 GEE", color="green")
            ax.plot(df_sub["date"], df_sub["et0_ml"], marker='x', linestyle="--", label="ET0 ML", color="orange")
            ax.set_xlabel("Năm")
            ax.set_xticks(df_sub["date"])

        ax.set_ylabel("ET0 (mm/ngày)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
