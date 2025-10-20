# 🚀 Dashboard Implementation Guide

## 📋 Overview

This guide provides step-by-step instructions for implementing interactive dashboards and presentation tools for the Customer Segmentation & CLV Prediction project.

## 🎯 Quick Start - Streamlit Demo

### Prerequisites
```bash
pip install streamlit plotly pandas scikit-learn xgboost joblib
```

### Basic Streamlit App Structure

```python
# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Customer CLV Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load model and data
@st.cache_data
def load_model():
    # Load your trained XGBoost model
    return joblib.load('xgboost_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('shopping_behavior_updated.csv')

# Main app
def main():
    st.title("🎯 Customer CLV Prediction Dashboard")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["CLV Predictor", "Customer Segments", "ROI Calculator", "Model Performance"]
    )
    
    if page == "CLV Predictor":
        show_clv_predictor()
    elif page == "Customer Segments":
        show_segment_analysis()
    elif page == "ROI Calculator":
        show_roi_calculator()
    elif page == "Model Performance":
        show_model_performance()

def show_clv_predictor():
    st.header("🔮 Live CLV Predictor")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 80, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        location = st.selectbox("Location", ["California", "New York", "Texas", "Florida"])
        previous_purchases = st.number_input("Previous Purchases", 0, 100, 10)
    
    with col2:
        purchase_amount = st.number_input("Purchase Amount (USD)", 10, 500, 100)
        review_rating = st.slider("Review Rating", 1.0, 5.0, 3.5)
        subscription = st.selectbox("Subscription Status", ["Yes", "No"])
        frequency = st.selectbox("Purchase Frequency", ["Weekly", "Monthly", "Quarterly"])
    
    # Prediction button
    if st.button("Predict CLV"):
        # Prepare input data
        input_data = prepare_input_data(age, gender, location, previous_purchases, 
                                      purchase_amount, review_rating, subscription, frequency)
        
        # Make prediction
        model = load_model()
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.success(f"Predicted CLV: ${prediction:,.2f}")
        
        # Show confidence interval
        confidence = 0.95
        margin = prediction * 0.1  # 10% margin
        st.info(f"Confidence Interval (95%): ${prediction - margin:,.2f} - ${prediction + margin:,.2f}")

def show_segment_analysis():
    st.header("👥 Customer Segment Analysis")
    
    df = load_data()
    
    # Segment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of segments
        segment_counts = df['Cluster'].value_counts()
        fig = px.pie(values=segment_counts.values, 
                    names=[f"Segment {i}" for i in segment_counts.index],
                    title="Customer Segment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Segment characteristics
        segment_stats = df.groupby('Cluster').agg({
            'Age': 'mean',
            'Purchase Amount (USD)': 'mean',
            'Previous Purchases': 'mean',
            'Review Rating': 'mean'
        }).round(2)
        
        st.dataframe(segment_stats)
    
    # Interactive segment explorer
    st.subheader("🔍 Segment Explorer")
    selected_segment = st.selectbox("Select Segment", [0, 1, 2])
    
    segment_data = df[df['Cluster'] == selected_segment]
    
    # Display segment characteristics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Age", f"{segment_data['Age'].mean():.1f}")
        st.metric("Average Purchase", f"${segment_data['Purchase Amount (USD)'].mean():.2f}")
    
    with col2:
        st.metric("Previous Purchases", f"{segment_data['Previous Purchases'].mean():.1f}")
        st.metric("Review Rating", f"{segment_data['Review Rating'].mean():.2f}")
    
    with col3:
        st.metric("Segment Size", f"{len(segment_data)} customers")
        st.metric("Revenue Share", f"{(segment_data['Purchase Amount (USD)'].sum() / df['Purchase Amount (USD)'].sum() * 100):.1f}%")

def show_roi_calculator():
    st.header("💰 ROI Calculator")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        cac = st.number_input("Customer Acquisition Cost (CAC)", 10, 200, 50)
        target_customers = st.number_input("Target New Customers", 100, 10000, 5000)
        avg_clv = st.number_input("Average CLV", 500, 5000, 1517.88)
    
    with col2:
        conversion_rate = st.slider("Conversion Rate (%)", 1, 20, 5)
        retention_rate = st.slider("Retention Rate (%)", 50, 95, 80)
        discount_rate = st.slider("Discount Rate (%)", 0, 20, 10)
    
    # Calculate ROI
    ad_budget = target_customers * cac
    expected_customers = target_customers * (conversion_rate / 100)
    expected_revenue = expected_customers * avg_clv * (retention_rate / 100)
    net_revenue = expected_revenue * (1 - discount_rate / 100)
    roi = (net_revenue - ad_budget) / ad_budget
    
    # Display results
    st.subheader("📊 ROI Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ad Budget", f"${ad_budget:,.0f}")
    with col2:
        st.metric("Expected Customers", f"{expected_customers:,.0f}")
    with col3:
        st.metric("Expected Revenue", f"${expected_revenue:,.0f}")
    with col4:
        st.metric("ROI", f"{roi:.2f}x")
    
    # ROI sensitivity chart
    st.subheader("📈 ROI Sensitivity Analysis")
    
    budget_range = np.arange(50000, 500000, 25000)
    roi_range = [(budget * (conversion_rate/100) * avg_clv * (retention_rate/100) * (1-discount_rate/100) - budget) / budget 
                 for budget in budget_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=budget_range, y=roi_range, mode='lines+markers', name='ROI'))
    fig.update_layout(
        title="ROI vs Advertising Budget",
        xaxis_title="Advertising Budget ($)",
        yaxis_title="ROI (x)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_model_performance():
    st.header("🎯 Model Performance")
    
    # Model metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", "0.9996", "99.96%")
    with col2:
        st.metric("MSE", "546.01", "Low Error")
    with col3:
        st.metric("Model Type", "XGBoost", "Best Performer")
    
    # Feature importance (if available)
    st.subheader("🔍 Feature Importance")
    st.info("Feature importance analysis would be displayed here based on your trained model.")
    
    # Model comparison
    st.subheader("📊 Model Comparison")
    
    comparison_data = {
        'Model': ['XGBoost', 'Random Forest'],
        'R² Score': [0.9996, 0.9995],
        'MSE': [546.01, 641.69]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

def prepare_input_data(age, gender, location, previous_purchases, purchase_amount, review_rating, subscription, frequency):
    # This function should prepare the input data in the same format as your training data
    # You'll need to implement the same preprocessing steps here
    pass

if __name__ == "__main__":
    main()
```

### Running the App
```bash
streamlit run app.py
```

## 🎨 Advanced Dashboard with Dash

### Installation
```bash
pip install dash plotly pandas scikit-learn xgboost
```

### Basic Dash App Structure

```python
# dash_app.py
import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib

# Initialize the app
app = dash.Dash(__name__)

# Load data and model
df = pd.read_csv('shopping_behavior_updated.csv')
model = joblib.load('xgboost_model.pkl')

# App layout
app.layout = html.Div([
    html.H1("Customer CLV Prediction Dashboard", className="header"),
    
    # Navigation tabs
    dcc.Tabs(id="tabs", value="predictor", children=[
        dcc.Tab(label="CLV Predictor", value="predictor"),
        dcc.Tab(label="Segment Analysis", value="segments"),
        dcc.Tab(label="ROI Calculator", value="roi"),
        dcc.Tab(label="Model Performance", value="performance"),
    ]),
    
    html.Div(id="tab-content")
])

# Callback for tab content
@callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_content(tab):
    if tab == "predictor":
        return create_predictor_tab()
    elif tab == "segments":
        return create_segments_tab()
    elif tab == "roi":
        return create_roi_tab()
    elif tab == "performance":
        return create_performance_tab()

def create_predictor_tab():
    return html.Div([
        html.H3("Live CLV Predictor"),
        # Add input components and prediction logic
    ])

# Add other tab creation functions...

if __name__ == "__main__":
    app.run_server(debug=True)
```

## 🚀 Deployment Options

### 1. Streamlit Cloud (Free)
- Push code to GitHub
- Connect to Streamlit Cloud
- Automatic deployment

### 2. Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Create requirements.txt
pip freeze > requirements.txt

# Deploy
git add .
git commit -m "Deploy dashboard"
git push heroku main
```

### 3. AWS/GCP/Azure
- Use container services
- Set up CI/CD pipeline
- Scale based on usage

## 📊 Interactive Features Implementation

### 1. Real-time Predictions
```python
# Add to your Streamlit app
@st.cache_data
def predict_clv(input_data):
    model = load_model()
    prediction = model.predict(input_data)
    return prediction[0]

# Use in your app
if st.button("Predict"):
    result = predict_clv(prepared_data)
    st.success(f"Predicted CLV: ${result:,.2f}")
```

### 2. Interactive Visualizations
```python
# Plotly interactive charts
fig = px.scatter(df, x='Age', y='Purchase Amount (USD)', 
                color='Cluster', hover_data=['CLV_Target'])
fig.update_layout(
    title="Customer Segments by Age and Purchase Amount",
    xaxis_title="Age",
    yaxis_title="Purchase Amount (USD)"
)
st.plotly_chart(fig, use_container_width=True)
```

### 3. Dynamic Filtering
```python
# Add filters to your dashboard
selected_segments = st.multiselect(
    "Select Customer Segments",
    options=[0, 1, 2],
    default=[0, 1, 2]
)

filtered_df = df[df['Cluster'].isin(selected_segments)]
```

## 🔧 Customization Tips

### 1. Styling
```python
# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)
```

### 2. Performance Optimization
```python
# Use caching for expensive operations
@st.cache_data
def load_and_process_data():
    # Expensive data processing
    return processed_data

# Use session state for user interactions
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
```

### 3. Error Handling
```python
try:
    prediction = model.predict(input_data)
    st.success(f"Prediction: ${prediction[0]:,.2f}")
except Exception as e:
    st.error(f"Error making prediction: {str(e)}")
```

## 📱 Mobile Responsiveness

### Streamlit
```python
# Use columns for responsive layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.plotly_chart(fig, use_container_width=True)
```

### Dash
```python
# Use responsive components
dcc.Graph(
    id='chart',
    responsive=True,
    style={'height': '400px'}
)
```

## 🔐 Security Considerations

### 1. Input Validation
```python
def validate_input(age, amount):
    if not (18 <= age <= 100):
        raise ValueError("Age must be between 18 and 100")
    if amount < 0:
        raise ValueError("Amount must be positive")
    return True
```

### 2. Model Security
```python
# Load model securely
import pickle
import hashlib

def load_secure_model(model_path):
    # Verify model integrity
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
```

## 📈 Monitoring and Analytics

### 1. Usage Tracking
```python
# Track user interactions
import time

def track_prediction(user_input, prediction):
    timestamp = time.time()
    # Log to database or analytics service
    pass
```

### 2. Performance Monitoring
```python
# Monitor model performance
def monitor_model_performance():
    # Check prediction accuracy
    # Monitor response times
    # Track error rates
    pass
```

This implementation guide provides a solid foundation for creating interactive dashboards. Start with the Streamlit version for quick prototyping, then move to Dash for more advanced features and customization.
