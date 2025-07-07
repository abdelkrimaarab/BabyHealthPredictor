import streamlit as st
import pandas as pd
import numpy as np
from utils.model_training import ModelTrainer
from utils.data_preprocessing import DataPreprocessor

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")

st.title("🎯 Morbidity Predictions")
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

# Check if models are trained
if not st.session_state.models:
    st.warning("⚠️ No trained models found. Please train models in the **Model Training** page first.")
    st.stop()

# Initialize components
trainer = ModelTrainer()
trainer.models = st.session_state.models
trainer.model_results = st.session_state.model_results

preprocessor = DataPreprocessor()

# Get data information
df = st.session_state.preprocessed_data
target_column = st.session_state.target_column

# Get feature names (excluding target)
feature_names = [col for col in df.columns if col != target_column]

st.subheader("Available Models")

# Display available models
model_comparison = []
for model_name, results in st.session_state.model_results.items():
    model_comparison.append({
        'Model': model_name,
        'Test Accuracy': f"{results['test_accuracy']:.4f}",
        'CV Mean': f"{results['cv_mean']:.4f}",
        'Status': '✅ Ready'
    })

comparison_df = pd.DataFrame(model_comparison)
st.dataframe(comparison_df, use_container_width=True)

# Model selection for predictions
selected_model = st.selectbox(
    "Select model for predictions:",
    options=list(st.session_state.models.keys()),
    help="Choose which trained model to use for making predictions"
)

if selected_model:
    st.success(f"Selected model: **{selected_model}**")
    
    # Prediction methods
    st.subheader("Prediction Methods")
    
    prediction_method = st.radio(
        "Choose prediction method:",
        options=["Single Patient Prediction", "Batch Prediction", "File Upload Prediction"],
        help="Select how you want to input data for predictions"
    )
    
    if prediction_method == "Single Patient Prediction":
        st.write("### Single Patient Input")
        st.write("Enter patient information below:")
        
        # Create input form for each feature
        input_data = {}
        
        # Get original data if available for categorical mapping
        original_data = st.session_state.data if st.session_state.data is not None else df
        
        # Create input fields in columns
        n_cols = 3
        cols = st.columns(n_cols)
        
        for i, feature in enumerate(feature_names):
            col_idx = i % n_cols
            
            with cols[col_idx]:
                # Check if feature exists in original data and is categorical
                if (feature in original_data.columns and 
                    original_data[feature].dtype == 'object' and 
                    df[feature].dtype in ['int64', 'float64']):
                    
                    # This is an encoded categorical variable
                    # Get unique original values
                    original_values = sorted(original_data[feature].dropna().unique())
                    
                    # Create mapping from original to encoded values
                    encoded_values = sorted(df[feature].unique())
                    
                    # Display original values for selection
                    selected_original = st.selectbox(
                        f"{feature}",
                        options=original_values,
                        help=f"Select from available categories"
                    )
                    
                    # Map back to encoded value
                    # Create mapping by matching sorted original to sorted encoded
                    if len(original_values) == len(encoded_values):
                        value_mapping = dict(zip(original_values, encoded_values))
                        input_data[feature] = value_mapping[selected_original]
                    else:
                        # Fallback: try to find the encoded value
                        temp_df = original_data[[feature]].copy()
                        temp_df['encoded'] = df[feature]
                        mapping = temp_df.groupby(feature)['encoded'].first().to_dict()
                        input_data[feature] = mapping.get(selected_original, encoded_values[0])
                
                elif df[feature].dtype in ['int64', 'float64']:
                    # Numeric feature
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        help=f"Range: {min_val:.2f} - {max_val:.2f}"
                    )
                else:
                    # Already categorical or other type
                    unique_vals = sorted(df[feature].unique())
                    input_data[feature] = st.selectbox(
                        f"{feature}",
                        options=unique_vals,
                        help=f"Possible values: {unique_vals}"
                    )
        
        # Show data info section
        with st.expander("ℹ️ Field Information"):
            st.write("**Data Types and Ranges:**")
            field_info = []
            
            for feature in feature_names:
                if (feature in original_data.columns and 
                    original_data[feature].dtype == 'object'):
                    field_type = "Categorical"
                    field_range = f"Options: {', '.join(map(str, sorted(original_data[feature].dropna().unique())))}"
                else:
                    field_type = "Numeric"
                    field_range = f"Range: {df[feature].min():.2f} - {df[feature].max():.2f}"
                
                field_info.append({
                    'Field': feature,
                    'Type': field_type,
                    'Details': field_range
                })
            
            info_df = pd.DataFrame(field_info)
            st.dataframe(info_df, use_container_width=True)
        
        # Make prediction button
        if st.button("🔮 Make Prediction", type="primary"):
            try:
                # Prepare input data
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                prediction, probabilities = trainer.predict_new_data(selected_model, input_df)
                
                # Display results
                st.subheader("Prediction Results")
                
                # Decode prediction to original value if possible
                predicted_value = prediction[0]
                if (target_column in preprocessor.label_encoders and 
                    hasattr(preprocessor.label_encoders[target_column], 'inverse_transform')):
                    try:
                        predicted_value = preprocessor.label_encoders[target_column].inverse_transform([prediction[0]])[0]
                    except:
                        # Fallback: try to get from original data mapping
                        original_data = st.session_state.data
                        if original_data is not None and target_column in original_data.columns:
                            unique_original = sorted(original_data[target_column].unique())
                            if prediction[0] < len(unique_original):
                                predicted_value = unique_original[int(prediction[0])]
                            else:
                                predicted_value = f"Class_{prediction[0]}"
                        else:
                            predicted_value = f"Class_{prediction[0]}"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Class", predicted_value)
                
                with col2:
                    if probabilities is not None:
                        confidence = max(probabilities[0]) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
                
                # Show prediction probabilities
                if probabilities is not None:
                    st.write("**Prediction Probabilities:**")
                    
                    # Create class labels (original values if available)
                    class_labels = list(range(len(probabilities[0])))
                    
                    # Try to get original class names from the original data
                    original_data = st.session_state.data
                    if original_data is not None and target_column in original_data.columns:
                        # Get unique original values sorted by their encoded values
                        original_values = sorted(original_data[target_column].unique())
                        if len(original_values) == len(probabilities[0]):
                            class_labels = original_values
                        else:
                            # Create mapping from preprocessed data if available
                            if st.session_state.preprocessed_data is not None:
                                temp_original = original_data[target_column].dropna()
                                temp_encoded = st.session_state.preprocessed_data[target_column].dropna()
                                if len(temp_original) == len(temp_encoded):
                                    # Create mapping
                                    mapping_df = pd.DataFrame({
                                        'original': temp_original,
                                        'encoded': temp_encoded
                                    }).drop_duplicates()
                                    mapping_dict = dict(zip(mapping_df['encoded'], mapping_df['original']))
                                    class_labels = [mapping_dict.get(i, f'Class_{i}') for i in range(len(probabilities[0]))]
                    
                    # Fallback: try label encoder
                    if all(isinstance(label, (int, float)) for label in class_labels):
                        if (target_column in preprocessor.label_encoders and 
                            hasattr(preprocessor.label_encoders[target_column], 'classes_')):
                            try:
                                class_labels = preprocessor.label_encoders[target_column].classes_.tolist()
                            except:
                                pass
                    
                    prob_df = pd.DataFrame({
                        'Class': class_labels,
                        'Probability': probabilities[0]
                    })
                    st.dataframe(prob_df, use_container_width=True)
                    
                    # Probability visualization
                    import plotly.express as px
                    prob_fig = px.bar(
                        prob_df, 
                        x='Class', 
                        y='Probability',
                        title='Prediction Probabilities'
                    )
                    st.plotly_chart(prob_fig, use_container_width=True)
                
                # Input data summary
                with st.expander("Input Data Summary"):
                    st.dataframe(input_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    elif prediction_method == "Batch Prediction":
        st.write("### Batch Prediction")
        st.write("Enter multiple patients' data separated by commas or new lines:")
        
        # Show expected format
        st.write("**Expected format:**")
        st.code(f"Feature order: {', '.join(feature_names)}")
        
        # Text area for batch input
        batch_input = st.text_area(
            "Enter patient data (one patient per line, values separated by commas):",
            height=200,
            help="Example: 1.2,0,3.4,1,0.5 (for each feature in order)"
        )
        
        if st.button("🔮 Make Batch Predictions", type="primary"):
            if batch_input.strip():
                try:
                    # Parse batch input
                    lines = batch_input.strip().split('\n')
                    batch_data = []
                    
                    for line in lines:
                        if line.strip():
                            values = [float(x.strip()) for x in line.split(',')]
                            if len(values) == len(feature_names):
                                batch_data.append(values)
                            else:
                                st.error(f"Row has {len(values)} values, expected {len(feature_names)}")
                                break
                    
                    if batch_data:
                        # Create DataFrame
                        batch_df = pd.DataFrame(batch_data, columns=feature_names)
                        
                        # Make predictions
                        predictions, probabilities = trainer.predict_new_data(selected_model, batch_df)
                        
                        # Decode predictions to original values if possible
                        decoded_predictions = predictions.copy()
                        if (target_column in preprocessor.label_encoders and 
                            hasattr(preprocessor.label_encoders[target_column], 'inverse_transform')):
                            try:
                                decoded_predictions = preprocessor.label_encoders[target_column].inverse_transform(predictions)
                            except:
                                # Fallback: try to get from original data mapping
                                original_data = st.session_state.data
                                if original_data is not None and target_column in original_data.columns:
                                    unique_original = sorted(original_data[target_column].unique())
                                    decoded_predictions = [unique_original[int(pred)] if int(pred) < len(unique_original) 
                                                         else f"Class_{pred}" for pred in predictions]
                                else:
                                    decoded_predictions = [f"Class_{pred}" for pred in predictions]
                        
                        # Display results
                        st.subheader("Batch Prediction Results")
                        
                        results_df = batch_df.copy()
                        results_df['Predicted_Class'] = decoded_predictions
                        
                        if probabilities is not None:
                            results_df['Confidence'] = [max(prob) * 100 for prob in probabilities]
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        st.write("**Prediction Summary:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Predictions", len(predictions))
                        
                        with col2:
                            pred_counts = pd.Series(decoded_predictions).value_counts()
                            most_common = pred_counts.index[0]
                            st.metric("Most Common Prediction", str(most_common))
                        
                        with col3:
                            if probabilities is not None:
                                avg_confidence = np.mean([max(prob) for prob in probabilities]) * 100
                                st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                        
                        # Download results
                        csv_data = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv_data,
                            file_name='batch_predictions.csv',
                            mime='text/csv'
                        )
                        
                except Exception as e:
                    st.error(f"Error in batch prediction: {str(e)}")
            else:
                st.warning("Please enter batch data for predictions.")
    
    elif prediction_method == "File Upload Prediction":
        st.write("### File Upload Prediction")
        st.write("Upload a CSV or Excel file with patient data for prediction:")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file containing patient data with the required features"
        )
        
        if uploaded_file is not None:
            try:
                # Load the file
                if uploaded_file.name.endswith('.csv'):
                    pred_df = pd.read_csv(uploaded_file)
                else:
                    pred_df = pd.read_excel(uploaded_file)
                
                st.success(f"File loaded successfully! Shape: {pred_df.shape}")
                
                # Show data preview
                st.write("**Data Preview:**")
                st.dataframe(pred_df.head(), use_container_width=True)
                
                # Check if all required features are present
                missing_features = set(feature_names) - set(pred_df.columns)
                extra_features = set(pred_df.columns) - set(feature_names)
                
                if missing_features:
                    st.error(f"Missing required features: {missing_features}")
                    st.info(f"Required features: {feature_names}")
                else:
                    if extra_features:
                        st.warning(f"Extra columns found (will be ignored): {extra_features}")
                    
                    # Select only required features
                    pred_df_selected = pred_df[feature_names]
                    
                    # Check for missing values
                    if pred_df_selected.isnull().sum().sum() > 0:
                        st.warning("Missing values detected. Please handle them before prediction.")
                        st.write("Missing values per column:")
                        st.write(pred_df_selected.isnull().sum())
                        
                        # Option to handle missing values
                        handle_missing = st.checkbox("Fill missing values with column means")
                        
                        if handle_missing:
                            pred_df_selected = pred_df_selected.fillna(pred_df_selected.mean())
                            st.success("Missing values filled with column means.")
                    
                    # Make predictions button
                    if st.button("🔮 Make File Predictions", type="primary"):
                        try:
                            # Make predictions
                            predictions, probabilities = trainer.predict_new_data(selected_model, pred_df_selected)
                            
                            # Decode predictions to original values if possible
                            decoded_predictions = predictions.copy()
                            if (target_column in preprocessor.label_encoders and 
                                hasattr(preprocessor.label_encoders[target_column], 'inverse_transform')):
                                try:
                                    decoded_predictions = preprocessor.label_encoders[target_column].inverse_transform(predictions)
                                except:
                                    # Fallback: try to get from original data mapping
                                    original_data = st.session_state.data
                                    if original_data is not None and target_column in original_data.columns:
                                        unique_original = sorted(original_data[target_column].unique())
                                        decoded_predictions = [unique_original[int(pred)] if int(pred) < len(unique_original) 
                                                             else f"Class_{pred}" for pred in predictions]
                                    else:
                                        decoded_predictions = [f"Class_{pred}" for pred in predictions]
                            
                            # Create results DataFrame
                            results_df = pred_df.copy()
                            results_df['Predicted_Class'] = decoded_predictions
                            
                            if probabilities is not None:
                                results_df['Confidence'] = [max(prob) * 100 for prob in probabilities]
                                
                                # Add individual class probabilities with original class names
                                class_labels = list(range(probabilities.shape[1]))
                                if (target_column in preprocessor.label_encoders and 
                                    hasattr(preprocessor.label_encoders[target_column], 'classes_')):
                                    try:
                                        class_labels = preprocessor.label_encoders[target_column].classes_.tolist()
                                    except:
                                        # Fallback: try to get from original data
                                        original_data = st.session_state.data
                                        if original_data is not None and target_column in original_data.columns:
                                            class_labels = sorted(original_data[target_column].unique())
                                
                                for i in range(probabilities.shape[1]):
                                    class_name = str(class_labels[i]) if i < len(class_labels) else f'Class_{i}'
                                    results_df[f'Probability_{class_name}'] = probabilities[:, i]
                            
                            # Display results
                            st.subheader("File Prediction Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Summary statistics
                            st.write("**Prediction Summary:**")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Predictions", len(predictions))
                            
                            with col2:
                                pred_counts = pd.Series(decoded_predictions).value_counts()
                                st.write("**Prediction Distribution:**")
                                st.dataframe(pred_counts.to_frame('Count'))
                            
                            with col3:
                                if probabilities is not None:
                                    avg_confidence = np.mean([max(prob) for prob in probabilities]) * 100
                                    st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                            
                            # Visualization
                            if len(pred_counts) <= 10:
                                import plotly.express as px
                                pred_fig = px.pie(
                                    values=pred_counts.values,
                                    names=pred_counts.index,
                                    title='Prediction Distribution'
                                )
                                st.plotly_chart(pred_fig, use_container_width=True)
                            
                            # Download results
                            csv_data = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Predictions with Results",
                                data=csv_data,
                                file_name='file_predictions_with_results.csv',
                                mime='text/csv'
                            )
                            
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")
                            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

# Model interpretation
if selected_model:
    st.subheader("Model Interpretation")
    
    # Feature importance
    feature_importance_df = trainer.get_feature_importance(selected_model, feature_names)
    
    if feature_importance_df is not None:
        st.write("**Feature Importance:**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Feature importance plot
            from utils.visualization import DataVisualizer
            visualizer = DataVisualizer()
            importance_fig = visualizer.plot_feature_importance(
                feature_importance_df, selected_model, top_n=10
            )
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)
        
        with col2:
            st.write("**Top 10 Important Features:**")
            st.dataframe(
                feature_importance_df.head(10),
                use_container_width=True
            )
    
    # Model performance summary
    st.write("**Model Performance Summary:**")
    model_results = st.session_state.model_results[selected_model]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Accuracy", f"{model_results['test_accuracy']:.4f}")
    
    with col2:
        st.metric("CV Mean", f"{model_results['cv_mean']:.4f}")
    
    with col3:
        st.metric("CV Std", f"{model_results['cv_std']:.4f}")
    
    with col4:
        if 'roc_auc' in model_results:
            st.metric("ROC AUC", f"{model_results['roc_auc']:.4f}")

# Prediction guidelines
with st.expander("📋 Prediction Guidelines"):
    st.markdown("""
    ### How to Use the Prediction System
    
    1. **Single Patient Prediction**: Enter individual patient data using the form fields
    2. **Batch Prediction**: Enter multiple patients' data in text format
    3. **File Upload**: Upload CSV/Excel files with patient data
    
    ### Important Notes
    - Ensure all required features are provided
    - Values should be in the same format as the training data
    - Check prediction confidence levels
    - Higher confidence indicates more reliable predictions
    
    ### Feature Requirements
    All predictions require the following features in order:
    """)
    
    for i, feature in enumerate(feature_names, 1):
        st.write(f"{i}. {feature}")

# Navigation hint
st.info("📊 Generate comprehensive reports in the **Export Results** page!")
