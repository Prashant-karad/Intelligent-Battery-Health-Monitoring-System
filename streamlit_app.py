import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.ensemble import IsolationForest

# Configure Plotly for Streamlit cloud deployment
pio.renderers.default = 'browser'

# Page configuration
st.set_page_config(
    page_title="Battery Health Prediction",
    page_icon="🔋",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
    .stApp {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
    div[data-testid="stMetricValue"] {font-size: 2.5rem; font-weight: bold;}
    h1 {color: white; text-align: center; padding: 20px;}
    h2, h3 {color: #667eea;}
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load models (cached)
@st.cache_resource
def load_models():
    """Load ML models"""
    soh_model = joblib.load('soh_random_forest_model.joblib')
    rul_model = joblib.load('rul_random_forest_model.joblib')
    return soh_model, rul_model

soh_model, rul_model = load_models()

# Feature extraction
def extract_features(df):
    """Extract the 4 features the model expects"""
    if 'Voltage_measured' not in df.columns:
        return None
    
    v = df['Voltage_measured'].values
    t = df['Time'].values if 'Time' in df.columns else np.arange(len(v))
    
    features = {
        'V_mean': v.mean(),
        'V_min': v.min(),
        'V_std': v.std(),
        'V_area': np.trapz(v, t)
    }
    
    return pd.DataFrame([features], columns=['V_mean', 'V_min', 'V_std', 'V_area'])

# Anomaly detection
def detect_anomalies(df, contamination=0.1):
    """Detect anomalies using Isolation Forest"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Time' in numeric_cols:
        numeric_cols.remove('Time')
    
    if len(numeric_cols) == 0 or len(df) < 10:
        return {'anomaly_indices': [], 'total_anomalies': 0, 'percentage': 0}
    
    try:
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        data = df[numeric_cols].fillna(df[numeric_cols].mean())
        predictions = iso_forest.fit_predict(data)
        anomaly_indices = np.where(predictions == -1)[0].tolist()
        
        return {
            'anomaly_indices': anomaly_indices,
            'total_anomalies': len(anomaly_indices),
            'percentage': (len(anomaly_indices) / len(df)) * 100
        }
    except Exception as e:
        st.error(f"Anomaly detection error: {e}")
        return {'anomaly_indices': [], 'total_anomalies': 0, 'percentage': 0}

# Predictions
def predict_battery_health(features_df):
    """Make SoH and RUL predictions"""
    soh = soh_model.predict(features_df.values)[0]
    rul = rul_model.predict(features_df.values)[0]
    return np.clip(soh, 0, 1.05), max(0, int(rul))

# SIMPLE, CLEAR VISUALIZATIONS
def create_voltage_chart(df, anomaly_indices):
    """Simple voltage chart with clear anomaly markers"""
    x_data = df['Time'] if 'Time' in df.columns else list(range(len(df)))
    x_label = 'Time' if 'Time' in df.columns else 'Sample'
    
    fig = go.Figure()
    
    # Main voltage line - bold and clear
    fig.add_trace(go.Scatter(
        x=x_data,
        y=df['Voltage_measured'],
        mode='lines+markers',
        name='Voltage',
        line=dict(color='#667eea', width=3),
        marker=dict(size=5, color='#667eea')
    ))
    
    # Anomalies - big red X marks
    if len(anomaly_indices) > 0:
        fig.add_trace(go.Scatter(
            x=x_data.iloc[anomaly_indices] if 'Time' in df.columns else anomaly_indices,
            y=df['Voltage_measured'].iloc[anomaly_indices],
            mode='markers',
            name='⚠️ Anomaly',
            marker=dict(color='red', size=15, symbol='x', line=dict(width=3, color='darkred'))
        ))
    
    fig.update_layout(
        title={'text': '📊 Battery Voltage', 'font': {'size': 22}},
        xaxis_title=x_label,
        yaxis_title='Voltage (V)',
        template='plotly_white',
        height=450,
        font=dict(size=14),
        hovermode='x unified'
    )
    
    return fig

def create_current_chart(df, anomaly_indices):
    """Simple current chart"""
    if 'Current_measured' not in df.columns:
        return None
        
    x_data = df['Time'] if 'Time' in df.columns else list(range(len(df)))
    x_label = 'Time' if 'Time' in df.columns else 'Sample'
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=df['Current_measured'],
        mode='lines+markers',
        name='Current',
        line=dict(color='#f59e0b', width=3),
        marker=dict(size=5, color='#f59e0b')
    ))
    
    if len(anomaly_indices) > 0:
        fig.add_trace(go.Scatter(
            x=x_data.iloc[anomaly_indices] if 'Time' in df.columns else anomaly_indices,
            y=df['Current_measured'].iloc[anomaly_indices],
            mode='markers',
            name='⚠️ Anomaly',
            marker=dict(color='red', size=15, symbol='x', line=dict(width=3))
        ))
    
    fig.update_layout(
        title={'text': '⚡ Battery Current', 'font': {'size': 22}},
        xaxis_title=x_label,
        yaxis_title='Current (A)',
        template='plotly_white',
        height=450,
        font=dict(size=14),
        hovermode='x unified'
    )
    
    return fig

def create_soh_chart(soh):
    """Simple SoH bar with health zones"""
    # Determine status
    if soh >= 0.85:
        color = '#51cf66'
        status = '✅ Excellent'
    elif soh >= 0.70:
        color = '#ffd43b'
        status = '⚠️ Fair'
    else:
        color = '#ff6b6b'
        status = '❌ Replace Soon'
    
    fig = go.Figure()
    
    # Horizontal bar
    fig.add_trace(go.Bar(
        y=['Battery Health'],
        x=[soh * 100],
        orientation='h',
        marker=dict(color=color),
        text=f'{soh*100:.1f}%  {status}',
        textposition='inside',
        textfont=dict(size=20, color='white'),
        hovertemplate='%{x:.1f}%<extra></extra>'
    ))
    
    # Add warning line at 80%
    fig.add_vline(x=80, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="⚠️ Replace at 80%", annotation_position="top left")
    
    fig.update_layout(
        title={'text': '🔋 State of Health (SoH)', 'font': {'size': 22}},
        xaxis=dict(title='Health %', range=[0, 105], tickmode='linear', dtick=10),
        yaxis=dict(showticklabels=False),
        template='plotly_white',
        height=250,
        font=dict(size=14)
    )
    
    return fig

def create_rul_chart(rul):
    """Simple RUL bar with thresholds"""
    # Determine status color
    if rul > 50:
        color = '#51cf66'
    elif rul > 20:
        color = '#ffd43b'
    else:
        color = '#ff6b6b'
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Cycles Remaining'],
        y=[rul],
        marker=dict(color=color),
        text=f'{rul} cycles',
        textposition='outside',
        textfont=dict(size=20),
        hovertemplate='%{y} cycles<extra></extra>'
    ))
    
    # Warning line at 20 cycles
    if rul < 100:
        fig.add_hline(y=20, line_dash="dash", line_color="red", line_width=2,
                      annotation_text="⚠️ Plan replacement", annotation_position="right")
    
    fig.update_layout(
        title={'text': '🔄 Remaining Life (RUL)', 'font': {'size': 22}},
        yaxis=dict(title='Cycles'),
        xaxis=dict(showticklabels=False),
        template='plotly_white',
        height=350,
        font=dict(size=14)
    )
    
    return fig

# Main App
def main():
    st.markdown("<h1>🔋 Battery Health Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>Simple, Clear Battery Analysis</h3>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("Upload battery CSV data to get instant health predictions and anomaly detection")
        
        st.header("📁 Required")
        st.code("Voltage_measured")
        
        st.header("📁 Optional")
        st.code("Time, Current_measured")
        
        st.header("⚙️ Settings")
        contamination = st.slider("Anomaly sensitivity (%)", 1, 30, 10) / 100
    
    # File upload
    st.markdown("### 📤 Upload Data")
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows")
            
            with st.expander("📋 Preview"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Process
            features_df = extract_features(df)
            if features_df is None:
                st.error("❌ Missing 'Voltage_measured' column")
                return
            
            soh, rul = predict_battery_health(features_df)
            
            # Detect anomalies
            with st.spinner("🔍 Analyzing..."):
                anomalies = detect_anomalies(df, contamination)
            
            # RESULTS
            st.markdown("---")
            st.markdown("## 📊 Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Health (SoH)", f"{soh*100:.1f}%")
            with col2:
                st.metric("Life Remaining", f"{rul} cycles")
            with col3:
                st.metric("Anomalies", f"{anomalies['total_anomalies']}")
            
            # Anomaly alert
            if anomalies['total_anomalies'] > 0:
                pct = anomalies['percentage']
                if pct < 5:
                    st.success(f"🟢 {anomalies['total_anomalies']} minor anomalies ({pct:.1f}%)")
                elif pct < 15:
                    st.warning(f"🟡 {anomalies['total_anomalies']} anomalies detected ({pct:.1f}%)")
                else:
                    st.error(f"🔴 {anomalies['total_anomalies']} significant anomalies ({pct:.1f}%)")
                
                with st.expander("View anomalies"):
                    st.dataframe(df.iloc[anomalies['anomaly_indices']], use_container_width=True)
            
            # VISUALIZATIONS
            st.markdown("---")
            st.markdown("## 📈 Charts")
            
            tab1, tab2 = st.tabs(["📊 Time Series", "🎯 Health Status"])
            
            with tab1:
                # Voltage chart
                if 'Voltage_measured' in df.columns:
                    st.plotly_chart(create_voltage_chart(df, anomalies['anomaly_indices']), use_container_width=True)
                
                # Current chart
                current_chart = create_current_chart(df, anomalies['anomaly_indices'])
                if current_chart:
                    st.plotly_chart(current_chart, use_container_width=True)
            
            with tab2:
                # SoH and RUL side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_soh_chart(soh), use_container_width=True)
                with col2:
                    st.plotly_chart(create_rul_chart(rul), use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.info("👆 Upload CSV to start")

if __name__ == "__main__":
    main()
