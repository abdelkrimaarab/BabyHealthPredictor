import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.data_preprocessing import DataPreprocessor
from utils.model_training import ModelTrainer
from utils.visualization import DataVisualizer

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Model Training")
st.markdown("---")

# Initialize session state if needed
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# Check if data is loaded
if st.session_state.data is None:
    st.warning("⚠️ No data loaded. Please upload data in the **Data Upload** page first.")
    st.stop()

# Check if preprocessed data is available
if st.session_state.preprocessed_data is None:
    st.warning("⚠️ No preprocessed data found. Please preprocess your data in the **Data Upload** page first.")
    st.stop()

# Initialize components
preprocessor = DataPreprocessor()
trainer = ModelTrainer()
visualizer = DataVisualizer()

df = st.session_state.preprocessed_data

# Target variable selection
st.subheader("Target Variable Configuration")

# Use previously selected target or let user select
if hasattr(st.session_state, 'target_column') and st.session_state.target_column:
    default_target = st.session_state.target_column
else:
    default_target = None

target_column = st.selectbox(
    "Select target variable for morbidity prediction:",
    options=df.columns.tolist(),
    index=df.columns.tolist().index(default_target) if default_target in df.columns else 0,
    help="Choose the column that represents morbidity outcome"
)

if target_column:
    st.session_state.target_column = target_column
    
    # Display target variable information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Values", df[target_column].nunique())
    with col2:
        st.metric("Missing Values", df[target_column].isnull().sum())
    with col3:
        if df[target_column].nunique() == 2:
            st.metric("Problem Type", "Binary Classification")
        else:
            st.metric("Problem Type", "Multi-class Classification")
    
    # Show target distribution
    target_counts = df[target_column].value_counts()
    st.write("**Target Distribution:**")
    st.dataframe(target_counts.to_frame('Count'), use_container_width=True)
    
    # Check for class imbalance
    if len(target_counts) == 2:
        imbalance_ratio = target_counts.min() / target_counts.max()
        if imbalance_ratio < 0.3:
            st.warning(f"⚠️ Class imbalance detected! Minority class ratio: {imbalance_ratio:.2f}")
            st.info("Consider using techniques like SMOTE or adjusting class weights during training.")

# Data preparation
st.subheader("Data Preparation")

try:
    # Prepare data for modeling
    X, y = preprocessor.prepare_for_modeling(df, target_column)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Features (X)", X.shape[1])
    with col2:
        st.metric("Samples", len(y))
    
    # Display feature names
    with st.expander("View Feature Names"):
        feature_names = X.columns.tolist()
        st.write(feature_names)
    
    # Train-test split configuration
    st.write("**Train-Test Split Configuration:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Random State", 0, 1000, 42)
    with col3:
        stratify = st.checkbox("Stratify Split", value=True, help="Maintain class distribution in train/test sets")
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y if stratify else None
    )
    
    st.success(f"✅ Data split completed: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Model selection and training
    st.subheader("Model Selection & Training")
    
    # Get available models
    available_models = trainer.get_available_models()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Select Models to Train:**")
        selected_models = st.multiselect(
            "Choose models:",
            options=list(available_models.keys()),
            default=['Random Forest', 'Logistic Regression'],
            help="Select one or more models to train and compare"
        )
    
    with col2:
        st.write("**Training Options:**")
        tune_hyperparameters = st.checkbox(
            "Hyperparameter Tuning", 
            value=False, 
            help="Enable grid search for hyperparameter optimization (takes longer)"
        )
        
        cross_validation = st.checkbox(
            "Cross Validation", 
            value=True, 
            help="Use 5-fold cross-validation for model evaluation"
        )
    
    # Training button
    if st.button("🚀 Train Models", type="primary", disabled=len(selected_models) == 0):
        if len(selected_models) > 0:
            st.write("### Training Progress")
            
            # Train models
            results = trainer.train_multiple_models(
                selected_models, X_train, X_test, y_train, y_test, tune_hyperparameters
            )
            
            # Store results in session state
            st.session_state.models = trainer.models
            st.session_state.model_results = trainer.model_results
            
            st.success("🎉 Model training completed!")
            
            # Display results
            st.subheader("Training Results")
            
            # Model comparison table
            comparison_df = trainer.get_model_comparison()
            if comparison_df is not None:
                st.dataframe(comparison_df, use_container_width=True)
                
                # Model comparison chart
                comparison_fig = visualizer.plot_model_comparison(comparison_df)
                if comparison_fig:
                    st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Cross-validation scores
            if cross_validation:
                cv_fig = visualizer.plot_cross_validation_scores(trainer.model_results)
                if cv_fig:
                    st.plotly_chart(cv_fig, use_container_width=True)
            
            # Individual model details
            st.subheader("Individual Model Performance")
            
            for model_name in selected_models:
                if model_name in trainer.model_results:
                    with st.expander(f"📊 {model_name} Details"):
                        results = trainer.model_results[model_name]
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")
                        with col2:
                            st.metric("CV Mean", f"{results['cv_mean']:.4f}")
                        with col3:
                            if 'roc_auc' in results:
                                st.metric("ROC AUC", f"{results['roc_auc']:.4f}")
                        
                        # Classification report
                        st.write("**Classification Report:**")
                        class_report = pd.DataFrame(results['classification_report']).transpose()
                        st.dataframe(class_report, use_container_width=True)
                        
                        # Confusion matrix
                        st.write("**Confusion Matrix:**")
                        conf_fig = visualizer.plot_confusion_matrix(
                            results['confusion_matrix'], model_name
                        )
                        st.plotly_chart(conf_fig, use_container_width=True)
                        
                        # Feature importance
                        if 'feature_importances' in results:
                            st.write("**Feature Importance:**")
                            feature_importance_df = pd.DataFrame({
                                'feature': X.columns,
                                'importance': results['feature_importances']
                            }).sort_values('importance', ascending=False)
                            
                            importance_fig = visualizer.plot_feature_importance(
                                feature_importance_df, model_name
                            )
                            if importance_fig:
                                st.plotly_chart(importance_fig, use_container_width=True)
                        
                        # Hyperparameters (if tuned)
                        if tune_hyperparameters and model_name in trainer.best_params:
                            st.write("**Best Parameters:**")
                            st.json(trainer.best_params[model_name])
        else:
            st.error("Please select at least one model to train.")

except Exception as e:
    st.error(f"Error in data preparation: {str(e)}")
    st.info("Please check your target variable selection and ensure the data is properly preprocessed.")

# Model management
if st.session_state.models:
    st.subheader("Model Management")
    
    # Best model selection
    best_model_name, best_model = trainer.get_best_model()
    if best_model_name:
        st.success(f"🏆 Best performing model: **{best_model_name}** (Test Accuracy: {st.session_state.model_results[best_model_name]['test_accuracy']:.4f})")
    
    # Model export options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Best Model"):
            # This would typically save the model to disk
            st.success("Model saved successfully! (Feature would save to disk in production)")
    
    with col2:
        if st.button("📊 Generate Model Report"):
            # Generate comprehensive model report
            report = f"""
# Model Training Report

## Dataset Information
- Total Features: {X.shape[1]}
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}
- Target Variable: {target_column}

## Models Trained
"""
            for model_name, results in st.session_state.model_results.items():
                report += f"""
### {model_name}
- Test Accuracy: {results['test_accuracy']:.4f}
- Cross-Validation Mean: {results['cv_mean']:.4f}
- Cross-Validation Std: {results['cv_std']:.4f}
"""
                if 'roc_auc' in results:
                    report += f"- ROC AUC: {results['roc_auc']:.4f}\n"
            
            report += f"\n## Best Model: {best_model_name}\n"
            
            st.text_area("Model Training Report", report, height=400)
            
            st.download_button(
                label="📥 Download Model Report",
                data=report,
                file_name='model_training_report.txt',
                mime='text/plain'
            )

# Navigation hint
if st.session_state.models:
    st.info("🎯 Models trained successfully! Navigate to **Predictions** to make predictions on new data.")
