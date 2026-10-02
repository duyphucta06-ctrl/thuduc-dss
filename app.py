import datetime
from datetime import timedelta
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ==========================================
# CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="DSS Tưới tiêu thông minh - TP. Thủ Đức", page_icon="🌱", layout="wide"
)

st.markdown(
    """
    <h2 style='color: #2e7d32;'>🌱 Dashboard Hỗ trợ Quyết định Tưới tiêu - TP. Thủ Đức</h2>
    <p style='color: gray;'>Dữ liệu tự động cập nhật thời gian thực từ Open-Meteo API & Mô hình Machine Learning dự báo $ET_0$</p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ==========================================
# LOAD MÔ HÌNH MACHINE LEARNING
# ==========================================
@st.cache_data(ttl=3600)
def fetch_live_weather_data():
  try:
    # 1. Lấy dữ liệu lịch sử từ 2015 đến hôm qua
    end_archive = (datetime.date.today() - timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    params_archive = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": "2015-01-01",
        "end_date": end_archive,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "shortwave_radiation_sum",
            "precipitation_sum",
            "windgusts_10m_max",
            "et0_fao_evapotranspiration",
        ],
        "timezone": "Asia/Bangkok",
    }

    res_arc = requests.get(archive_url, params=params_archive).json()
    if "daily" not in res_arc:
      st.error(f"Lỗi phản hồi từ Open-Meteo Archive API: {res_arc}")
      return pd.DataFrame()

    df_arc = pd.DataFrame(res_arc["daily"])
    df_arc["date"] = pd.to_datetime(df_arc["time"])

    df_arc = df_arc.rename(
        columns={
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "temperature_2m_mean": "temp_mean",
            "relative_humidity_2m_mean": "humidity_mean",
            "shortwave_radiation_sum": "radiation_sum",
            "precipitation_sum": "precipitation_sum",
            "windgusts_10m_max": "wind_speed_10m_max",
            "et0_fao_evapotranspiration": "et0_actual",
        }
    )

    # 2. Lấy dữ liệu dự báo (Forecast)
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    params_fc = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "shortwave_radiation_sum",
            "precipitation_sum",
            "windgusts_10m_max",
        ],
        "timezone": "Asia/Bangkok",
        "forecast_days": 3,
    }

    res_fc = requests.get(forecast_url, params=params_fc).json()
    if "daily" in res_fc:
      df_fc = pd.DataFrame(res_fc["daily"])
      df_fc["date"] = pd.to_datetime(df_fc["time"])
      df_fc = df_fc.rename(
          columns={
              "temperature_2m_max": "temp_max",
              "temperature_2m_min": "temp_min",
              "temperature_2m_mean": "temp_mean",
              "relative_humidity_2m_mean": "humidity_mean",
              "shortwave_radiation_sum": "radiation_sum",
              "precipitation_sum": "precipitation_sum",
              "windgusts_10m_max": "wind_speed_10m_max",
          }
      )
      df_fc["et0_actual"] = np.nan
      df_full = (
          pd.concat([df_arc, df_fc])
          .drop_duplicates(subset=["date"])
          .sort_values("date")
          .reset_index(drop=True)
      )
    else:
      df_full = df_arc

    # Làm sạch khuyết thiếu
    df_full = df_full.interpolate(method="linear").bfill().ffill()

    # Dự báo ET0 bằng mô hình Machine Learning
    if model is not None:
      features = [
          "temp_max",
          "temp_min",
          "temp_mean",
          "humidity_mean",
          "radiation_sum",
          "precipitation_sum",
          "wind_speed_10m_max",
      ]
      df_full["et0_ml"] = model.predict(df_full[features])
    else:
      df_full["et0_ml"] = df_full["et0_actual"]

    return df_full
  except Exception as e:
    st.error(f"Lỗi khi tải dữ liệu thời tiết: {e}")
    return pd.DataFrame()

# Tải dữ liệu
# Tải dữ liệu
df_data = fetch_live_weather_data()

# Lấy thông tin ngày gần nhất để hiển thị các chỉ số trên cùng (Metrics)
latest_row = df_data.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nhiệt độ trung bình", f"{latest_row['temp_mean']:.1f} °C")
col2.metric("Độ ẩm trung bình", f"{latest_row['humidity_mean']:.1f} %")
col3.metric(
    "Lượng mưa (Hôm nay/Dự báo)", f"{latest_row['precipitation_sum']:.1f} mm"
)
col4.metric("Bức xạ mặt trời", f"{latest_row['radiation_sum']:.2f} MJ/m²")

st.markdown("---")

# ==========================================
# BỐ CỤC GIAO DIỆN CHÍNH (2 CỘT)
# ==========================================
left_col, right_col = st.columns([1, 1.4])

# --- CỘT TRÁI: HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH (DSS) ---
with left_col:
  st.subheader("💧 Dự báo $ET_0$ ML & Lượng nước tưới")
  kc = st.slider("Hệ số cây trồng ($K_c$):", 0.4, 1.3, 1.0, 0.05)

  et0_today = latest_row["et0_ml"]
  etc = et0_today * kc
  peff = latest_row["precipitation_sum"] * 0.7  # Lượng mưa hiệu quả giả định
  net_water = max(0, etc - peff)

  st.info(f"**$ET_0$ Dự báo ML hôm nay:** {et0_today:.2f} mm/ngày")
  st.success(f"**Nhu cầu nước cây trồng ($ET_c$):** {etc:.2f} mm/ngày")

  if net_water > 0:
    st.warning(
        f"⚠️ **CẦN TƯỚI:** Khuyến nghị tưới bù **{net_water:.2f} mm** (tương"
        f" đương {net_water * 1:.1f} lít/m²)"
    )
  else:
    st.success(
        "✅ **KHÔNG CẦN TƯỚI:** Lượng mưa hiện tại đã đáp ứng đủ nhu cầu nước!"
    )

# --- CỘT PHẢI: BIỂU ĐỒ SO SÁNH ---
with right_col:
  st.subheader("📊 So sánh $ET_0$ Open-Meteo vs $ET_0$ ML")

  # Tùy chọn độ chia thời gian theo yêu cầu
  time_option = st.radio(
      "Độ chia thời gian:",
      ["10 ngày gần nhất", "12 tháng gần nhất", "10 năm gần nhất"],
      horizontal=True,
  )

  # Lọc dữ liệu dựa trên lựa chọn của người dùng
  today = pd.Timestamp.today().normalize()
  if time_option == "10 ngày gần nhất":
    filtered_df = df_data[df_data["date"] >= (today - timedelta(days=9))].copy()
    # Theo yêu cầu: Ngày dự báo (ngày cuối cùng/ngày thứ 10) chỉ có đường ET0 của ML (ET0 Open-Meteo = NaN)
    if len(filtered_df) > 0:
      filtered_df.iloc[-1, filtered_df.columns.get_loc("et0_actual")] = np.nan
  elif time_option == "12 tháng gần nhất":
    filtered_df = df_data[
        df_data["date"] >= (today - timedelta(days=365))
    ].copy()
  else:  # 10 năm gần nhất
    filtered_df = df_data[
        df_data["date"] >= (today - timedelta(days=365 * 10))
    ].copy()

  # Vẽ biểu đồ tương tác bằng Plotly
  fig = go.Figure()

  # Đường ET0 thực tế tải từ Open-Meteo
  fig.add_trace(
      go.Scatter(
          x=filtered_df["date"],
          y=filtered_df["et0_actual"],
          mode="lines+markers",
          name="$ET_0$ Open-Meteo (Thực tế)",
          line=dict(color="#2e7d32", width=2),
      )
  )

  # Đường ET0 dự báo từ mô hình Machine Learning
  fig.add_trace(
      go.Scatter(
          x=filtered_df["date"],
          y=filtered_df["et0_ml"],
          mode="lines+markers",
          name="$ET_0$ Dự báo (ML)",
          line=dict(color="#ff9800", width=2, dash="dash"),
      )
  )

  fig.update_layout(
      xaxis_title="Thời gian",
      yaxis_title="$ET_0$ (mm/ngày)",
      legend=dict(
          orientation="horizontal",
          yanchor="bottom",
          y=1.02,
          xanchor="right",
          x=1,
      ),
      margin=dict(l=20, r=20, t=30, b=20),
      height=400,
  )

  st.plotly_chart(fig, use_container_width=True)
