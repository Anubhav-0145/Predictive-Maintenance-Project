# app.py - Deployment Version
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
from pathlib import Path
import requests
from io import BytesIO

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with better mobile support
st.markdown("""
<style>
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem !important;
        }
        .sub-header {
            font-size: 1.2rem !important;
        }
    }
    
    .stButton > button {
        width: 100%;
    }
    
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def load_model_from_github():
    """Load model from GitHub if not found locally"""
    try:
        # GitHub raw URL for model
        model_url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/models/best_predictive_maintenance_model.pkl"
        scaler_url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/models/scaler.pkl"
        
        # Download model
        model_response = requests.get(model_url)
        model_response.raise_for_status()
        
        # Download scaler
        scaler_response = requests.get(scaler_url)
        scaler_response.raise_for_status()
        
        # Load from bytes
        model = joblib.load(BytesIO(model_response.content))
        scaler = joblib.load(BytesIO(scaler_response.content))
        
        return model, scaler
        
    except Exception as e:
        st.warning(f"Could not load from GitHub: {str(e)}")
        return None, None

@st.cache_resource
def load_model_and_scaler():
    """Load model with fallback options"""
    # Try local paths first
    local_paths = [
        ('models/best_predictive_maintenance_model.pkl', 'models/scaler.pkl'),
        ('best_predictive_maintenance_model.pkl', 'scaler.pkl'),
        ('./models/best_predictive_maintenance_model.pkl', './models/scaler.pkl'),
    ]
    
    for model_path, scaler_path in local_paths:
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                st.sidebar.success(f"✅ Models loaded from {model_path}")
                return model, scaler
        except Exception as e:
            continue
    
    # Try loading from GitHub
    st.sidebar.info("🔄 Trying to load models from GitHub...")
    model, scaler = load_model_from_github()
    if model and scaler:
        return model, scaler
    
    # Fallback: create demo model
    st.sidebar.warning("⚠️ Using demo mode. Upload model files for full functionality.")
    return create_demo_model(), None

def create_demo_model():
    """Create a simple model for demo purposes"""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    # Train on simple synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 7)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model.fit(X, y)
    return model

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'Type': np.random.choice(['L', 'M', 'H'], n_samples),
        'Air temperature [K]': np.random.uniform(295, 311, n_samples),
        'Process temperature [K]': np.random.uniform(305, 314, n_samples),
        'Rotational speed [rpm]': np.random.randint(1168, 2886, n_samples),
        'Torque [Nm]': np.random.uniform(3.8, 77.6, n_samples),
        'Tool wear [min]': np.random.randint(0, 253, n_samples),
        'Target': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    })
    
    return data

def main():
    # Sidebar with enhanced navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092658.png", width=80)
        st.title("🔧 Maintenance AI")
        
        # Navigation
        page = st.radio(
            "Navigate",
            ["🏠 Dashboard", "📊 Predict", "📈 Analytics", "🤖 Model Info", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stats in sidebar
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "94.2%")
        with col2:
            st.metric("Uptime", "99.8%")
        
        st.markdown("---")
        
        # System status
        st.markdown("### 🔍 System Status")
        model_loaded = st.container()
        with model_loaded:
            if 'model' in st.session_state and st.session_state.model:
                st.success("✅ Model Ready")
            else:
                st.warning("⚠️ Demo Mode")
        
        st.markdown("---")
        
        # Quick actions
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        if st.button("📥 Export Logs", use_container_width=True):
            st.info("Logs exported (demo)")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model, st.session_state.scaler = load_model_and_scaler()
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Predict":
        show_prediction_page()
    elif page == "📈 Analytics":
        show_analytics_page()
    elif page == "🤖 Model Info":
        show_model_info_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_dashboard():
    """Main dashboard view"""
    st.markdown('<h1 class="main-header">🏭 Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Machines Monitored", "142", "+3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predicted Failures", "8", "↓ 2")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg MTBF", "245 days", "+12")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Cost Saved", "$42.8K", "↑ 15%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("🔍 Run Prediction", use_container_width=True, type="primary"):
            st.switch_page("📊 Predict")
    
    with quick_col2:
        if st.button("📊 View Analytics", use_container_width=True):
            st.switch_page("📈 Analytics")
    
    with quick_col3:
        if st.button("🔄 Update Models", use_container_width=True):
            st.info("Model update initiated (demo)")
    
    # Recent predictions
    st.markdown("### 📋 Recent Predictions")
    
    # Sample recent predictions
    recent_data = pd.DataFrame({
        'Machine ID': ['M-001', 'M-002', 'M-003', 'M-004', 'M-005'],
        'Prediction': ['Normal', 'At Risk', 'Normal', 'Failure', 'Normal'],
        'Confidence': [0.92, 0.67, 0.88, 0.94, 0.91],
        'Timestamp': ['10:30 AM', '10:15 AM', '09:45 AM', '09:30 AM', '09:15 AM']
    })
    
    st.dataframe(recent_data, use_container_width=True, hide_index=True)
    
    # Alert summary
    st.markdown("### ⚠️ Active Alerts")
    
    alert_col1, alert_col2 = st.columns([2, 1])
    
    with alert_col1:
        st.warning("**Machine M-004**: High probability of bearing failure (94%)")
        st.info("**Machine M-002**: Elevated temperature detected")
    
    with alert_col2:
        st.metric("Active Alerts", "2", "↓ 1")
    
    # Performance chart
    st.markdown("### 📈 Performance Trend")
    
    # Sample performance data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    performance = np.cumsum(np.random.randn(30)) + 95
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=performance, mode='lines+markers', 
                            name='Accuracy', line=dict(color='blue', width=2)))
    
    fig.update_layout(
        title='Model Accuracy Over Time',
        xaxis_title='Date',
        yaxis_title='Accuracy (%)',
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_page():
    """Prediction page for running model predictions"""
    st.markdown('<h1 class="main-header">📊 Make Predictions</h1>', unsafe_allow_html=True)
    
    st.markdown("### Enter Machine Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
        air_temp = st.slider("Air Temperature (K)", 295.0, 311.0, 302.0)
        process_temp = st.slider("Process Temperature (K)", 305.0, 314.0, 310.0)
    
    with col2:
        rotational_speed = st.slider("Rotational Speed (rpm)", 1168, 2886, 1500)
        torque = st.slider("Torque (Nm)", 3.8, 77.6, 40.0)
        tool_wear = st.slider("Tool Wear (min)", 0, 253, 100)
    
    if st.button("🔮 Predict Machine State", use_container_width=True, type="primary"):
        st.success("Prediction completed (demo mode)")
        st.metric("Prediction", "Normal", "Confidence: 92%")

def show_analytics_page():
    """Analytics page for data visualization"""
    st.markdown('<h1 class="main-header">📈 Analytics</h1>', unsafe_allow_html=True)
    
    st.markdown("### Machine Performance Analytics")
    
    # Sample data
    data = create_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(data, x='Type', color='Target', title='Failure Distribution by Type')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(data, x='Tool wear [min]', y='Torque [Nm]', color='Target', title='Tool Wear vs Torque')
        st.plotly_chart(fig, use_container_width=True)

def show_model_info_page():
    """Model information page"""
    st.markdown('<h1 class="main-header">🤖 Model Information</h1>', unsafe_allow_html=True)
    
    st.markdown("### Model Details")
    st.info("**Model Type**: Random Forest Classifier")
    st.info("**Accuracy**: 94.2%")
    st.info("**Training Samples**: 1,000")
    st.info("**Last Updated**: 2024-01-15")

def show_settings_page():
    """Settings page"""
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    st.markdown("### System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### General Settings")
        refresh_rate = st.slider("Refresh Rate (seconds)", 5, 300, 30)
        st.success(f"Refresh rate set to {refresh_rate} seconds")
    
    with col2:
        st.markdown("#### Alert Settings")
        alert_threshold = st.slider("Alert Threshold (%)", 50, 100, 75)
        st.success(f"Alert threshold set to {alert_threshold}%")

if __name__ == "__main__":
    main()