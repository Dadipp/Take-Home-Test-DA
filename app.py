# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Retail Customer Segmentation Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/merged_data.csv", parse_dates=["Sale_Date"])  # ✅ relative path from repo root
    return df

# Load data
df = load_data()

# --- Sidebar Navigasi ---
st.sidebar.header("Navigasi")
page = st.sidebar.radio("Pilih Halaman", [
    "Overview Data", "Analisis RFM", "Customer Clustering",
    "Demographic Segmentation"
])

# --- Sidebar Filter ---
st.sidebar.markdown("### Filter Data")
df['Year'] = df['Sale_Date'].dt.year
year_options = ['All'] + sorted(df['Year'].unique().tolist())
selected_year = st.sidebar.selectbox("Pilih Tahun", year_options, index=0)
selected_city = st.sidebar.selectbox("Pilih City", ['All'] + sorted(df['City'].unique()))

filtered_df = df.copy()
if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['Year'] == selected_year]
if selected_city != 'All':
    filtered_df = filtered_df[filtered_df['City'] == selected_city]

# --- Hitung RFM ---
current_date = datetime(2025, 8, 14)  # Use current date from context
rfm = filtered_df.groupby('Customer_ID').agg({
    'Sale_Date': lambda x: (current_date - x.max()).days,  # Recency
    'Sale_ID': 'count',  # Frequency
    'Monetary': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']

# --- Normalisasi untuk segmentasi heuristik ---
if len(rfm) > 0 and rfm['Recency'].nunique() > 1:  # Pastikan ada variasi dalam Recency
    rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 4, labels=[4, 3, 2, 1], duplicates='drop')  # 4=baik (rendah), 1=buruk (tinggi)
else:
    rfm['R_Score'] = 1  # Default skor jika tidak ada variasi
    st.warning("Data Recency tidak cukup bervariasi untuk segmentasi. Silakan ubah filter.")

if len(rfm) > 0 and rfm['Frequency'].nunique() > 1:  # Pastikan ada variasi dalam Frequency
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')  # 4=baik (tinggi), 1=buruk (rendah)
else:
    rfm['F_Score'] = 1  # Default skor jika tidak ada variasi
    st.warning("Data Frequency tidak cukup bervariasi untuk segmentasi. Silakan ubah filter.")

if len(rfm) > 0 and rfm['Monetary'].nunique() > 1:  # Pastikan ada variasi dalam Monetary
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')  # 4=baik (tinggi), 1=buruk (rendah)
else:
    rfm['M_Score'] = 1  # Default skor jika tidak ada variasi
    st.warning("Data Monetary tidak cukup bervariasi untuk segmentasi. Silakan ubah filter.")

# --- Segmentasi Heuristik Berdasarkan RFM ---
def assign_segment(row):
    if row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
        return 'Champions'
    elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] < 3:
        return 'Loyal Customers'
    elif row['R_Score'] >= 3 and row['F_Score'] < 3 and row['M_Score'] < 3:
        return 'New Customers'
    elif row['R_Score'] < 3 and row['F_Score'] >= 3:
        return 'Hibernating'
    elif row['R_Score'] < 3 and row['F_Score'] < 3:
        return 'Lost Customers'
    return 'Undefined'

rfm['Segment'] = rfm.apply(assign_segment, axis=1)

# --- Overview Data ---
if page == "Overview Data":
    st.title("📦 Overview Customer Order Data")
    st.markdown("Tinjauan umum terhadap data transaksi pelanggan.")

    st.dataframe(filtered_df.head(), use_container_width=True)
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📟 Jumlah Transaksi", filtered_df['Sale_ID'].nunique())
    with col2:
        st.metric("👥 Jumlah Pelanggan", filtered_df['Customer_ID'].nunique())
    with col3:
        st.metric("💰 Total Penjualan", f"{filtered_df['Monetary'].sum():,.2f}")
    st.markdown("---")
    filtered_df['YearMonth'] = filtered_df['Sale_Date'].dt.to_period('M').astype(str)
    sales_trend = filtered_df.groupby('YearMonth')['Monetary'].sum().reset_index()
    fig = px.line(sales_trend, x='YearMonth', y='Monetary',
                  title="📈 Tren Penjualan Bulanan",
                  labels={'Monetary': 'Total Penjualan', 'YearMonth': 'Bulan'})
    st.plotly_chart(fig, use_container_width=True)

# --- Analisis RFM ---
elif page == "Analisis RFM":
    st.title("📊 RFM Analysis")
    st.markdown("Analisis pelanggan berdasarkan Recency, Frequency, dan Monetary.")

    st.dataframe(rfm.head(), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(px.histogram(rfm, x='Recency', nbins=30, title="Distribusi Recency"), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(rfm, x='Frequency', nbins=30, title="Distribusi Frequency"), use_container_width=True)
    with col3:
        st.plotly_chart(px.histogram(rfm, x='Monetary', nbins=30, title="Distribusi Monetary"), use_container_width=True)

    st.markdown("### 🌟 Segmentasi Heuristik Berdasarkan RFM")
    fig = px.box(rfm, x='Segment', y='Monetary', color='Segment',
                 title="Distribusi Monetary per Segment RFM",
                 category_orders={'Segment': ['Champions', 'Loyal Customers', 'New Customers', 'Hibernating', 'Lost Customers']})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Insight Segmentasi RFM")
    st.markdown("""
        - **Champions**: Pelanggan terbaik dengan skor tinggi di semua metrik (Recency, Frequency, Monetary).
        - **Loyal Customers**: Pelanggan dengan Recency dan Frequency baik, tetapi Monetary rendah.
        - **New Customers**: Pelanggan baru dengan frekuensi dan nilai belanja rendah.
        - **Hibernating**: Pelanggan yang lama tidak transaksi tetapi memiliki Frequency baik di masa lalu.
        - **Lost Customers**: Pelanggan yang lama tidak transaksi dengan frekuensi rendah.
            """)

# --- Customer Clustering ---
elif page == "Customer Clustering":
    st.title("🧩 Customer Segmentation via K-Means Clustering")
    st.markdown("Cluster pelanggan berdasarkan nilai RFM.")

    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    # Pengecekan jumlah sampel sebelum clustering
    n_samples = len(rfm)
    n_clusters = 5
    if n_samples >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    else:
        rfm['Cluster'] = -1  # Nilai default jika clustering tidak bisa dilakukan
        st.warning(f"Jumlah pelanggan ({n_samples}) kurang dari jumlah cluster ({n_clusters}). Clustering tidak dapat dilakukan. Silakan ubah filter untuk mendapatkan lebih banyak data.")

    fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                        color=rfm['Cluster'].astype(str),
                        title="📍 Visualisasi Klaster Pelanggan (3D)",
                        labels={'Cluster': 'Klaster'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Statistik RFM per Klaster")
    if n_samples >= n_clusters:
        st.dataframe(rfm.groupby('Cluster')[features].mean().round(1), use_container_width=True)
    else:
        st.write("Tidak ada statistik klaster karena clustering tidak dapat dilakukan.")

    # --- Insight Otomatis ---
    if n_samples >= n_clusters:
        st.markdown("### 🔐 Insight Otomatis dari Klaster")
        cluster_stats = rfm.groupby('Cluster')[features].mean()
        best_cluster = cluster_stats['Monetary'].idxmax()
        low_recency_cluster = cluster_stats['Recency'].idxmin()

        st.success(f"💡 Cluster {best_cluster} memiliki nilai *Monetary* tertinggi ({cluster_stats.loc[best_cluster, 'Monetary']:.2f}).")
        st.info(f"📉 Cluster {low_recency_cluster} memiliki *Recency* terendah ({cluster_stats.loc[low_recency_cluster, 'Recency']:.0f} hari), menandakan pelanggan paling aktif.")
    else:
        st.markdown("### 🔐 Insight Otomatis dari Klaster")
        st.write("Tidak ada insight karena clustering tidak dapat dilakukan.")

    # --- Download CSV ---
    csv = rfm.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📅 Download Hasil Klaster CSV",
        data=csv,
        file_name='customer_cluster_result.csv',
        mime='text/csv'
    )

# --- Demographic Segmentation ---
elif page == "Demographic Segmentation":
    st.title("🌍 Demographic Segmentation")
    st.markdown("Analisis berdasarkan atribut demografis pelanggan.")

    fig1 = px.pie(filtered_df, names='Gender', title="Distribusi Gender Pelanggan")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.treemap(filtered_df, path=['City'], values='Monetary',
                      title="Penjualan Berdasarkan City")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### 🧠 Insight Demografi")
    st.markdown("""
    - Mayoritas pelanggan berdasarkan gender dan kota.
    - Kontribusi penjualan signifikan dari kota-kota tertentu.
    - Segmentasi ini dapat dimanfaatkan untuk menyusun strategi pemasaran yang lebih relevan secara geografis maupun demografis.
    """)