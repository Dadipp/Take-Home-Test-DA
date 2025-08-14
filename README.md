# 🧠 Retail Customer Segmentation Dashboard

An interactive **Streamlit** application for analyzing and segmenting customers based on transaction data retail.  
This dashboard provides insights through **RFM Analysis**, **K-Means Clustering**, and segmentation by **Demographics**.

---

## 📌 Link Streamlit
🔗 [Open Streamlit App](https://take-home-test-da-4tjxn42utrjusmvbhkvgst.streamlit.app)

---

## 📖 Project Overview

**Objective:**  
To help businesses understand their customers better by segmenting them into meaningful groups.  
This segmentation enables more targeted marketing strategies, improved customer retention, and better resource allocation.

**Main Features:**
- **Overview Data**: Summary of transactions, customers, and sales trends.
- **RFM Analysis**: Segmentation based on Recency, Frequency, and Monetary value.
- **Customer Clustering**: K-Means clustering with interactive 3D visualization.
- **Demographic Segmentation**: Insights based on customer gender and city.

---

## 📂 Dataset

- **File**: `merged_data.csv`  
- **Size**: ~19,833 rows  
- **Key Columns**:
  - `Sale_Date`
  - `City`
  - `Customer_ID`
  - `Sale_ID`
  - `Monetary`
  - `Product_Name`

---

## 📊 Features in Detail

### 1. Overview Data
- Table preview of transaction data.
- KPIs: Total Transactions, Number of Customers, Total Sales.
- Monthly sales trend chart.

### 2. RFM Analysis
- Calculation of **Recency**, **Frequency**, **Monetary**.
- Histograms for each RFM metric.
- Segmentation into: `Champions`, `Loyal Customers`, `New Customers`, `Hibernating`, `Lost Customers`.
- Boxplot to compare segments.

### 3. Customer Clustering
- Standardization of RFM features.
- K-Means clustering model.
- 3D scatter plot visualization.
- Downloadable clustering results.

### 4. Demographic Segmentation
- Pie chart of customer gender distribution.
- Treemap of sales by city.
- Geographic and demographic insights.

---

## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **Pandas**, **NumPy**
- **Plotly**
- **scikit-learn**
