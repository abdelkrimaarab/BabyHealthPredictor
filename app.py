import streamlit as st
import pandas as pd
import numpy as np
from utils.data_preprocessing import DataPreprocessor
from utils.model_training import ModelTrainer
from utils.visualization import DataVisualizer

# Configure page
st.set_page_config(
    page_title="Newborn Morbidity Prediction",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# Main page
st.title("🏥 Newborn Morbidity Predictive Modeling")
st.markdown("---")

st.markdown("""
### Welcome to the Newborn Morbidity Prediction System

This application provides comprehensive tools for analyzing newborn health data and building predictive models for morbidity assessment.

**Features:**
- 📊 **Data Upload & Validation**: Upload Excel files with newborn health data
- 🔍 **Exploratory Data Analysis**: Comprehensive data visualization and statistical analysis
- 🤖 **Machine Learning Models**: Train and evaluate multiple predictive models
- 📈 **Model Performance**: Cross-validation and detailed performance metrics
- 🎯 **Predictions**: Interactive interface for new patient predictions
- 📤 **Export Results**: Download model results and predictions

**Supported Models:**
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine
- Gradient Boosting Classifier
- Neural Network (MLP)

### Getting Started
1. Navigate to **Data Upload** to upload your Excel file
2. Explore your data in **Exploratory Analysis**
3. Train models in **Model Training**
4. Make predictions in **Predictions**
5. Export results in **Export Results**
""")

# Sidebar navigation info
st.sidebar.markdown("## Navigation")
st.sidebar.markdown("Use the pages in the sidebar to navigate through the application workflow.")

# Display current data status
st.sidebar.markdown("## Current Status")
if st.session_state.data is not None:
    st.sidebar.success(f"✅ Data loaded: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
else:
    st.sidebar.warning("⚠️ No data loaded")

if st.session_state.preprocessed_data is not None:
    st.sidebar.success("✅ Data preprocessed")
else:
    st.sidebar.info("ℹ️ Data preprocessing pending")

if st.session_state.models:
    st.sidebar.success(f"✅ {len(st.session_state.models)} models trained")
else:
    st.sidebar.info("ℹ️ No models trained yet")

# Quick stats if data is available
if st.session_state.data is not None:
    st.markdown("### Quick Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", st.session_state.data.shape[0])
    
    with col2:
        st.metric("Features", st.session_state.data.shape[1])
    
    with col3:
        missing_percentage = (st.session_state.data.isnull().sum().sum() / 
                            (st.session_state.data.shape[0] * st.session_state.data.shape[1])) * 100
        st.metric("Missing Data %", f"{missing_percentage:.1f}%")
    
    with col4:
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).shape[1]
        st.metric("Numeric Features", numeric_cols)
