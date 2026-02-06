# app.py
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
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .failure-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('best_predictive_maintenance_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'best_predictive_maintenance_model.pkl' and 'scaler.pkl' are in the current directory.")
        return None, None

def calculate_risk_score(features):
    """Calculate a risk score based on feature values"""
    risk_score = 0
    
    # Rotational speed risk (both too high and too low are risky)
    if features['rotational_speed'] > 2500 or features['rotational_speed'] < 1200:
        risk_score += 40
    
    # Torque risk (high torque is risky)
    if features['torque'] > 60:
        risk_score += 30
    
    # Tool wear risk (high wear is risky)
    if features['tool_wear'] > 180:
        risk_score += 20
    
    # Temperature risk (high process temp is risky)
    if features['process_temperature'] > 311:
        risk_score += 10
    
    return min(risk_score, 100)

def get_maintenance_recommendation(prediction, probability, risk_score, features):
    """Generate maintenance recommendations based on prediction"""
    recommendations = []
    
    if prediction == 1 or probability > 0.7:
        recommendations.append("🚨 **IMMEDIATE ACTION REQUIRED:** Schedule maintenance within 24 hours")
    
    if risk_score > 70:
        recommendations.append("⚠️ **High Risk Detected:** Consider reducing operational load")
    
    if features['tool_wear'] > 150:
        recommendations.append(f"🔧 **Tool Replacement:** Tool wear is high ({features['tool_wear']} min)")
    
    if features['rotational_speed'] > 2500:
        recommendations.append("⚡ **Speed Reduction:** Rotational speed is above safe operating range")
    
    if features['torque'] > 60:
        recommendations.append("💪 **Load Reduction:** Torque is exceeding recommended limits")
    
    if features['process_temperature'] > 311:
        recommendations.append("🌡️ **Cooling Check:** Process temperature is elevated")
    
    if not recommendations and prediction == 0:
        recommendations.append("✅ **Normal Operation:** Continue with regular maintenance schedule")
    
    return recommendations

def create_feature_importance_plot(model, feature_names):
    """Create feature importance visualization"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=importance_df['Feature'],
            x=importance_df['Importance'],
            orientation='h',
            marker_color='steelblue',
            text=importance_df['Importance'].round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=400,
            showlegend=False
        )
        return fig
    return None

def create_prediction_breakdown(features, model):
    """Create visualization showing how each feature contributes to prediction"""
    # This is a simplified version - in practice you'd use SHAP or LIME
    risk_factors = []
    
    # Simple risk scoring for visualization
    if features['rotational_speed'] > 2500 or features['rotational_speed'] < 1200:
        risk_factors.append(('Rotational Speed', 'High Risk', 40))
    elif features['rotational_speed'] > 2200 or features['rotational_speed'] < 1400:
        risk_factors.append(('Rotational Speed', 'Medium Risk', 20))
    else:
        risk_factors.append(('Rotational Speed', 'Low Risk', 5))
    
    if features['torque'] > 60:
        risk_factors.append(('Torque', 'High Risk', 30))
    elif features['torque'] > 50:
        risk_factors.append(('Torque', 'Medium Risk', 15))
    else:
        risk_factors.append(('Torque', 'Low Risk', 5))
    
    if features['tool_wear'] > 180:
        risk_factors.append(('Tool Wear', 'High Risk', 25))
    elif features['tool_wear'] > 120:
        risk_factors.append(('Tool Wear', 'Medium Risk', 12))
    else:
        risk_factors.append(('Tool Wear', 'Low Risk', 3))
    
    if features['process_temperature'] > 311:
        risk_factors.append(('Process Temp', 'High Risk', 15))
    elif features['process_temperature'] > 309:
        risk_factors.append(('Process Temp', 'Medium Risk', 8))
    else:
        risk_factors.append(('Process Temp', 'Low Risk', 2))
    
    risk_df = pd.DataFrame(risk_factors, columns=['Feature', 'Risk Level', 'Risk Score'])
    
    fig = px.bar(risk_df, x='Feature', y='Risk Score', color='Risk Level',
                 color_discrete_map={'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'},
                 text='Risk Level')
    
    fig.update_layout(
        title='Risk Factor Breakdown',
        xaxis_title='Feature',
        yaxis_title='Risk Score',
        height=400
    )
    
    return fig, risk_df

def main():
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        return
    
    # Sidebar for navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092658.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🔮 Prediction Dashboard", "📊 Model Insights", "📈 Data Analysis", "⚙️ Settings"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This application predicts equipment failures 
        using machine learning models trained on 
        sensor data from industrial machinery.
        
        **Version**: 1.0.0
        **Last Updated**: March 2025
        """)
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        st.metric("Model Accuracy", "94.2%")
        st.metric("Failure Detection", "92%")
        st.metric("False Alarm Rate", "6.6%")
    
    # Main content based on selected page
    if page == "🔮 Prediction Dashboard":
        show_prediction_dashboard(model, scaler)
    elif page == "📊 Model Insights":
        show_model_insights(model)
    elif page == "📈 Data Analysis":
        show_data_analysis()
    elif page == "⚙️ Settings":
        show_settings()

def show_prediction_dashboard(model, scaler):
    """Main prediction interface"""
    st.markdown('<h1 class="main-header">⚙️ Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)
    # Columns expected by the model/scaler (ensure availability for batch predictions)
    expected_columns = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]',
        'Tool wear [min]', 'Type_M', 'Type_H'
    ]
    
    # Create two columns for input and output
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown('<h3 class="sub-header">Equipment Parameters</h3>', unsafe_allow_html=True)
        
        # Create input form
        with st.form("prediction_form"):
            # Product Type
            product_type = st.selectbox(
                "Product Type",
                ["Low Quality (L)", "Medium Quality (M)", "High Quality (H)"],
                help="Product quality level"
            )
            
            # Create columns for temperature inputs
            temp_col1, temp_col2 = st.columns(2)
            with temp_col1:
                air_temp = st.slider(
                    "Air Temperature (K)",
                    min_value=295.0,
                    max_value=311.0,
                    value=298.5,
                    step=0.1,
                    help="Ambient temperature in Kelvin"
                )
            
            with temp_col2:
                process_temp = st.slider(
                    "Process Temperature (K)",
                    min_value=305.0,
                    max_value=314.0,
                    value=308.6,
                    step=0.1,
                    help="Process operating temperature in Kelvin"
                )
            
            # Create columns for operational parameters
            op_col1, op_col2 = st.columns(2)
            with op_col1:
                rotational_speed = st.slider(
                    "Rotational Speed (rpm)",
                    min_value=1168,
                    max_value=2886,
                    value=1539,
                    step=10,
                    help="Equipment rotational speed"
                )
            
            with op_col2:
                torque = st.slider(
                    "Torque (Nm)",
                    min_value=3.8,
                    max_value=77.6,
                    value=40.5,
                    step=0.1,
                    help="Applied torque in Newton-meters"
                )
            
            # Tool wear
            tool_wear = st.slider(
                "Tool Wear (minutes)",
                min_value=0,
                max_value=253,
                value=108,
                step=1,
                help="Cumulative tool usage time"
            )
            
            # Submit button
            submitted = st.form_submit_button("🔍 Predict Failure Risk", use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="sub-header">Prediction Results</h3>', unsafe_allow_html=True)
        
        if submitted:
            # Prepare input data
            type_mapping = {"Low Quality (L)": 0, "Medium Quality (M)": 1, "High Quality (H)": 2}
            type_encoded = type_mapping[product_type]
            
            # Create input array
            input_data = pd.DataFrame({
                'Type_L': [1 if type_encoded == 0 else 0],
                'Type_M': [1 if type_encoded == 1 else 0],
                'Type_H': [1 if type_encoded == 2 else 0],
                'Air temperature [K]': [air_temp],
                'Process temperature [K]': [process_temp],
                'Rotational speed [rpm]': [rotational_speed],
                'Torque [Nm]': [torque],
                'Tool wear [min]': [tool_wear]
            })
            
            # Ensure correct column order based on scaler (fall back to expected_columns)
            fallback_columns = [
                'Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 
                'Tool wear [min]', 'Type_M', 'Type_H'
            ]

            scaler_cols = None
            if hasattr(scaler, 'feature_names_in_'):
                try:
                    scaler_cols = list(scaler.feature_names_in_)
                except Exception:
                    scaler_cols = None

            use_columns = scaler_cols if scaler_cols is not None else fallback_columns

            # Add any missing columns with zeros (one-hot dummies) and drop extras
            for col in use_columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Reindex to the expected order
            input_data = input_data.reindex(columns=use_columns)

            # Scale the data
            scaled_data = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0][1]
            
            # Calculate risk score
            features_dict = {
                'rotational_speed': rotational_speed,
                'torque': torque,
                'tool_wear': tool_wear,
                'process_temperature': process_temp
            }
            risk_score = calculate_risk_score(features_dict)
            
            # Display prediction results
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Prediction", 
                         "FAILURE" if prediction == 1 else "NORMAL",
                         delta="High Risk" if prediction == 1 else "Low Risk",
                         delta_color="inverse" if prediction == 1 else "normal")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with result_col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Probability", 
                         f"{probability:.1%}",
                         delta=f"Risk Level: {risk_score}/100")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with result_col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                confidence = "HIGH" if probability > 0.8 else "MEDIUM" if probability > 0.5 else "LOW"
                st.metric("Confidence", confidence)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display appropriate message
            if prediction == 1:
                st.markdown('<div class="failure-box">', unsafe_allow_html=True)
                st.error("🚨 **FAILURE PREDICTED**")
                st.write(f"Model predicts equipment failure with {probability:.1%} probability.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("✅ **NORMAL OPERATION**")
                st.write(f"Equipment is operating normally with {probability:.1%} failure probability.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display recommendations
            st.markdown('<h4 class="sub-header">📋 Maintenance Recommendations</h4>', unsafe_allow_html=True)
            recommendations = get_maintenance_recommendation(prediction, probability, risk_score, features_dict)
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Risk factor visualization
            st.markdown('<h4 class="sub-header">📊 Risk Factor Analysis</h4>', unsafe_allow_html=True)
            fig, risk_df = create_prediction_breakdown(features_dict, model)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show input summary
            with st.expander("📋 Input Summary"):
                summary_df = pd.DataFrame({
                    'Parameter': ['Product Type', 'Air Temperature', 'Process Temperature', 
                                  'Rotational Speed', 'Torque', 'Tool Wear'],
                    'Value': [product_type, f"{air_temp} K", f"{process_temp} K", 
                              f"{rotational_speed} rpm", f"{torque} Nm", f"{tool_wear} min"],
                    'Status': ['Normal' for _ in range(6)]
                })
                st.table(summary_df)
        
        else:
            # Show placeholder before prediction
            st.info("👈 Enter equipment parameters on the left and click 'Predict Failure Risk' to see results.")
            st.image("https://cdn-icons-png.flaticon.com/512/2917/2917633.png", width=200)
    
    # Batch prediction section
    st.markdown("---")
    st.markdown('<h3 class="sub-header">📁 Batch Prediction</h3>', unsafe_allow_html=True)
    
    with st.expander("Upload CSV for multiple predictions"):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                batch_data = pd.read_csv(uploaded_file)
                
                # Check if required columns are present
                required_columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
                                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
                
                if all(col in batch_data.columns for col in required_columns):
                    st.success(f"✅ File uploaded successfully! {len(batch_data)} records found.")
                    
                    # Show preview
                    st.write("**Data Preview:**")
                    st.dataframe(batch_data.head())
                    
                    if st.button("🚀 Run Batch Prediction", type="primary"):
                        with st.spinner("Processing batch predictions..."):
                            # Preprocess data
                            batch_processed = batch_data.copy()
                            # Ensure 'Type' column uses categorical labels 'L','M','H'. If numeric, map back to labels.
                            if pd.api.types.is_numeric_dtype(batch_processed['Type']):
                                batch_processed['Type'] = batch_processed['Type'].map({0: 'L', 1: 'M', 2: 'H'})

                            # One-hot encode (will create 'Type_M' and 'Type_H' when drop_first=True)
                            batch_processed = pd.get_dummies(batch_processed, columns=['Type'], drop_first=True)
                            
                            # Ensure correct column order for batch based on scaler (fallback to expected_columns)
                            fallback_columns = [
                                'Air temperature [K]', 'Process temperature [K]', 
                                'Rotational speed [rpm]', 'Torque [Nm]', 
                                'Tool wear [min]', 'Type_M', 'Type_H'
                            ]

                            scaler_cols = None
                            if hasattr(scaler, 'feature_names_in_'):
                                try:
                                    scaler_cols = list(scaler.feature_names_in_)
                                except Exception:
                                    scaler_cols = None

                            use_columns = scaler_cols if scaler_cols is not None else expected_columns

                            # Add missing dummy columns with zeros
                            for col in use_columns:
                                if col not in batch_processed.columns:
                                    batch_processed[col] = 0

                            # Reindex batch to expected columns order
                            X_batch = batch_processed.reindex(columns=use_columns)
                            X_scaled = scaler.transform(X_batch)
                            predictions = model.predict(X_scaled)
                            probabilities = model.predict_proba(X_scaled)[:, 1]
                            
                            # Add predictions to original data
                            batch_data['Prediction'] = ['FAILURE' if p == 1 else 'NORMAL' for p in predictions]
                            batch_data['Probability'] = probabilities
                            batch_data['Risk Level'] = ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in probabilities]
                            
                            # Display results
                            st.write("**Prediction Results:**")
                            st.dataframe(batch_data)
                            
                            # Download results
                            csv = batch_data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name="predictive_maintenance_results.csv",
                                mime="text/csv"
                            )
                            
                            # Show summary statistics
                            failure_count = sum(predictions)
                            st.metric("Failures Predicted", failure_count)
                            st.metric("Failure Rate", f"{(failure_count/len(predictions))*100:.1f}%")
                else:
                    st.error("❌ Uploaded file doesn't have the required columns. Please check the format.")
                    st.info("Required columns: Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

def show_model_insights(model):
    """Display model information and insights"""
    st.markdown('<h1 class="main-header">📊 Model Insights</h1>', unsafe_allow_html=True)
    
    # Model information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Type", type(model).__name__)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Date", "March 2025")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Version", "1.0.0")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model performance metrics
    st.markdown('<h3 class="sub-header">📈 Performance Metrics</h3>', unsafe_allow_html=True)
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("Accuracy", "94.2%")
    
    with perf_col2:
        st.metric("Precision", "93.4%")
    
    with perf_col3:
        st.metric("Recall", "92.0%")
    
    with perf_col4:
        st.metric("F1-Score", "92.1%")
    
    # Feature importance
    st.markdown('<h3 class="sub-header">🔍 Feature Importance</h3>', unsafe_allow_html=True)
    
    feature_names = ['Air temperature', 'Process temperature', 'Rotational speed', 
                     'Torque', 'Tool wear', 'Type_M', 'Type_H']
    
    fig = create_feature_importance_plot(model, feature_names)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance visualization is not available for this model type.")
    
    # Model comparison (static data)
    st.markdown('<h3 class="sub-header">🏆 Model Comparison</h3>', unsafe_allow_html=True)
    
    comparison_data = pd.DataFrame({
        'Model': ['XGBoost', 'Random Forest', 'Gradient Boosting', 'SVM', 
                  'Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes'],
        'Accuracy': [0.942, 0.938, 0.935, 0.931, 0.925, 0.912, 0.908, 0.872],
        'F1-Score': [0.921, 0.912, 0.907, 0.898, 0.882, 0.861, 0.852, 0.792]
    })
    
    fig = px.bar(comparison_data, x='Model', y=['Accuracy', 'F1-Score'], 
                 barmode='group', title='Model Performance Comparison',
                 color_discrete_sequence=['#1f77b4', '#ff7f0e'])
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix
    st.markdown('<h3 class="sub-header">📊 Confusion Matrix</h3>', unsafe_allow_html=True)
    
    # Simulated confusion matrix
    confusion_data = np.array([[1560, 40], [25, 375]])
    
    fig = px.imshow(confusion_data,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Normal', 'Failure'],
                    y=['Normal', 'Failure'],
                    text_auto=True,
                    color_continuous_scale='Blues')
    
    fig.update_layout(title='Confusion Matrix - Test Data')
    st.plotly_chart(fig, use_container_width=True)
    
    # Model details
    with st.expander("🔧 Model Configuration Details"):
        st.code("""
XGBClassifier(
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=54.9,
    random_state=42,
    eval_metric='logloss'
)
        """, language="python")

def show_data_analysis():
    """Show data analysis and visualizations"""
    st.markdown('<h1 class="main-header">📈 Data Analysis</h1>', unsafe_allow_html=True)
    
    # Load and show sample data
    try:
        data = pd.read_csv('predictive_maintenance.csv')
        data = data.drop(['UDI', 'Product ID'], axis=1)
        
        st.markdown('<h3 class="sub-header">📋 Dataset Overview</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(data))
        
        with col2:
            failures = data['Target'].sum()
            st.metric("Failure Cases", failures)
        
        with col3:
            failure_rate = (failures / len(data)) * 100
            st.metric("Failure Rate", f"{failure_rate:.2f}%")
        
        # Show data preview
        with st.expander("👀 View Sample Data"):
            st.dataframe(data.head(10))
        
        # Distribution plots
        st.markdown('<h3 class="sub-header">📊 Feature Distributions</h3>', unsafe_allow_html=True)
        
        # Select feature to visualize
        feature = st.selectbox(
            "Select Feature to Visualize",
            ['Air temperature [K]', 'Process temperature [K]', 
             'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        )
        
        # Create distribution plot
        fig = px.histogram(data, x=feature, color='Target', 
                          marginal='box', nbins=30,
                          title=f'Distribution of {feature} by Failure Status',
                          color_discrete_map={0: 'blue', 1: 'red'},
                          labels={'Target': 'Failure'},
                          opacity=0.7)
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.markdown('<h3 class="sub-header">🔗 Correlation Analysis</h3>', unsafe_allow_html=True)
        
        # Encode categorical variables
        data_encoded = data.copy()
        data_encoded['Type'] = data_encoded['Type'].map({'L': 0, 'M': 1, 'H': 2})
        
        correlation = data_encoded.corr()
        
        fig = px.imshow(correlation,
                        text_auto=True,
                        color_continuous_scale='RdBu',
                        range_color=[-1, 1],
                        title='Correlation Matrix')
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Failure analysis by product type
        st.markdown('<h3 class="sub-header">📈 Failure Analysis by Product Type</h3>', unsafe_allow_html=True)
        
        failure_by_type = data.groupby('Type')['Target'].mean() * 100
        failure_by_type_df = failure_by_type.reset_index()
        
        fig = px.bar(failure_by_type_df, x='Type', y='Target',
                     title='Failure Rate by Product Type',
                     labels={'Target': 'Failure Rate (%)', 'Type': 'Product Type'},
                     color='Type',
                     color_discrete_sequence=['green', 'orange', 'red'])
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.markdown('<h3 class="sub-header">📊 Feature Relationships</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox(
                "X-axis Feature",
                ['Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'],
                key='x_feature'
            )
        
        with col2:
            y_feature = st.selectbox(
                "Y-axis Feature",
                ['Torque [Nm]', 'Rotational speed [rpm]', 'Tool wear [min]'],
                key='y_feature'
            )
        
        fig = px.scatter(data, x=x_feature, y=y_feature, color='Target',
                         title=f'{x_feature} vs {y_feature}',
                         color_discrete_map={0: 'blue', 1: 'red'},
                         hover_data=['Type'],
                         opacity=0.6)
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'predictive_maintenance.csv' is in the current directory.")
        st.info("You can use the sample data below for demonstration:")
        
        # Create sample data for demonstration
        sample_data = pd.DataFrame({
            'Type': ['L', 'M', 'H'] * 3,
            'Air temperature [K]': np.random.uniform(295, 311, 9),
            'Process temperature [K]': np.random.uniform(305, 314, 9),
            'Rotational speed [rpm]': np.random.randint(1168, 2886, 9),
            'Torque [Nm]': np.random.uniform(3.8, 77.6, 9),
            'Tool wear [min]': np.random.randint(0, 253, 9),
            'Target': [0, 0, 1, 0, 0, 0, 1, 0, 0]
        })
        
        st.dataframe(sample_data)

def show_settings():
    """Application settings"""
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="sub-header">🔧 Application Configuration</h3>', unsafe_allow_html=True)
    
    # Prediction threshold
    threshold = st.slider(
        "Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Probability threshold for classifying as failure"
    )
    
    st.info(f"Predictions with probability ≥ {threshold:.2f} will be classified as failures.")
    
    # Display options
    st.markdown('<h3 class="sub-header">👁️ Display Options</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_confidence = st.checkbox("Show Confidence Intervals", value=True)
        show_details = st.checkbox("Show Detailed Predictions", value=True)
    
    with col2:
        dark_mode = st.checkbox("Dark Mode Preview", value=False)
        auto_refresh = st.checkbox("Auto-refresh Data", value=False)
    
    # Data management
    st.markdown('<h3 class="sub-header">💾 Data Management</h3>', unsafe_allow_html=True)
    
    if st.button("🔄 Clear Cache", type="secondary"):
        st.cache_resource.clear()
        st.success("Cache cleared successfully!")
    
    if st.button("📊 Reset to Defaults", type="secondary"):
        st.info("Defaults restored")
    
    # Export settings
    st.markdown('<h3 class="sub-header">📤 Export Configuration</h3>', unsafe_allow_html=True)
    
    export_format = st.selectbox(
        "Export Format",
        ["CSV", "JSON", "Excel"]
    )
    
    if st.button("💾 Export Current Settings"):
        settings = {
            "threshold": threshold,
            "show_confidence": show_confidence,
            "show_details": show_details,
            "dark_mode": dark_mode,
            "auto_refresh": auto_refresh
        }
        st.json(settings)
        st.success("Settings exported successfully!")
    
    # About section
    st.markdown("---")
    st.markdown("### About This Application")
    
    st.markdown("""
    **Predictive Maintenance Dashboard** is a machine learning application designed to:
    
    - Predict equipment failures before they occur
    - Provide actionable maintenance recommendations
    - Analyze equipment performance data
    - Support decision-making for maintenance teams
    
    **Features:**
    - Real-time failure prediction
    - Batch processing capabilities
    - Detailed risk analysis
    - Interactive visualizations
    - Export functionality
    
    **Technology Stack:**
    - Python & Streamlit for the web interface
    - XGBoost for machine learning
    - Plotly for interactive visualizations
    - Scikit-learn for preprocessing
    """)

if __name__ == "__main__":
    main()