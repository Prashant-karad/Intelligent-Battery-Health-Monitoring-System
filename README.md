---
title: Battery Health Prediction System
emoji: 🔋
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: streamlit_app.py
pinned: false
license: mit
---

# 🔋 Battery Health Prediction System

Advanced ML-based battery State of Health (SoH) and Remaining Useful Life (RUL) prediction system with anomaly detection.

## Features

- **SoH Prediction**: Predict battery health percentage using Random Forest ML model
- **RUL Estimation**: Estimate remaining useful life in cycles
- **Anomaly Detection**: Identify unusual patterns using Isolation Forest
- **Interactive Visualizations**: Time series analysis with Plotly charts
- **Real-time Analysis**: Upload CSV and get instant predictions

## How to Use

1. Upload a CSV file containing battery measurement data
2. Required column: `Voltage_measured`
3. Optional columns: `Time`, `Current_measured`, `Temperature_measured`
4. View predictions, anomalies, and interactive charts

## Model Information

- **SoH Model**: Random Forest Regressor (200 estimators)
- **RUL Model**: Random Forest Regressor (300 estimators)
- **Features**: V_mean, V_min, V_std, V_area (voltage-based)
- **Anomaly Detection**: Isolation Forest algorithm

## Technology Stack

- Streamlit for UI
- scikit-learn for ML models
- Plotly for interactive visualizations
- Pandas/NumPy for data processing
