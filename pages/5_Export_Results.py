import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import io
import base64

st.set_page_config(page_title="Export Results", page_icon="📤", layout="wide")

st.title("📤 Export Results & Reports")
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

# Check if we have necessary data
if st.session_state.data is None:
    st.warning("⚠️ No data available. Please upload data first.")
    st.stop()

# Get all available data
original_data = st.session_state.data
preprocessed_data = st.session_state.preprocessed_data
models = st.session_state.models
model_results = st.session_state.model_results
target_column = getattr(st.session_state, 'target_column', None)

st.subheader("Available Data & Results")

# Show what's available for export
col1, col2, col3 = st.columns(3)

with col1:
    st.write("**📊 Data:**")
    st.write(f"✅ Original data: {original_data.shape}")
    if preprocessed_data is not None:
        st.write(f"✅ Preprocessed data: {preprocessed_data.shape}")
    else:
        st.write("❌ No preprocessed data")

with col2:
    st.write("**🤖 Models:**")
    if models:
        st.write(f"✅ Trained models: {len(models)}")
        for model_name in models.keys():
            st.write(f"  - {model_name}")
    else:
        st.write("❌ No trained models")

with col3:
    st.write("**🎯 Target:**")
    if target_column:
        st.write(f"✅ Target variable: {target_column}")
    else:
        st.write("❌ No target variable selected")

# Export options
st.subheader("Export Options")

export_type = st.selectbox(
    "Select what to export:",
    options=[
        "Complete Analysis Report",
        "Data Export",
        "Model Performance Report",
        "Prediction Template",
        "Custom Export"
    ]
)

if export_type == "Complete Analysis Report":
    st.write("### Complete Analysis Report")
    st.write("Generate a comprehensive report including all analysis results.")
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        include_data_summary = st.checkbox("Include Data Summary", value=True)
        include_eda_results = st.checkbox("Include EDA Results", value=True)
        include_preprocessing = st.checkbox("Include Preprocessing Details", value=True)
    
    with col2:
        include_model_results = st.checkbox("Include Model Results", value=True)
        include_feature_importance = st.checkbox("Include Feature Importance", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
    
    if st.button("📋 Generate Complete Report", type="primary"):
        # Generate comprehensive report
        report_content = f"""
# Newborn Morbidity Prediction Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report presents a comprehensive analysis of newborn morbidity prediction using machine learning techniques.

"""
        
        if include_data_summary:
            report_content += f"""
## Data Summary
- **Dataset Shape**: {original_data.shape[0]} records, {original_data.shape[1]} features
- **Target Variable**: {target_column if target_column else 'Not specified'}
- **Missing Data**: {original_data.isnull().sum().sum()} missing values ({(original_data.isnull().sum().sum() / (original_data.shape[0] * original_data.shape[1]) * 100):.2f}%)
- **Numeric Features**: {len(original_data.select_dtypes(include=[np.number]).columns)}
- **Categorical Features**: {len(original_data.select_dtypes(include=['object']).columns)}

### Feature List
{chr(10).join([f"- {col}" for col in original_data.columns])}
"""
        
        if include_preprocessing and preprocessed_data is not None:
            report_content += f"""
## Data Preprocessing
- **Original Shape**: {original_data.shape}
- **Processed Shape**: {preprocessed_data.shape}
- **Preprocessing Steps Applied**:
  - Missing value handling
  - Categorical encoding
  - Data type conversions
"""
        
        if include_model_results and model_results:
            report_content += """
## Model Performance Results

"""
            for model_name, results in model_results.items():
                report_content += f"""
### {model_name}
- **Test Accuracy**: {results['test_accuracy']:.4f}
- **Cross-Validation Mean**: {results['cv_mean']:.4f}
- **Cross-Validation Std**: {results['cv_std']:.4f}
"""
                if 'roc_auc' in results:
                    report_content += f"- **ROC AUC**: {results['roc_auc']:.4f}\n"
                
                report_content += "\n"
            
            # Best model
            best_model = max(model_results.keys(), key=lambda x: model_results[x]['test_accuracy'])
            report_content += f"""
### Best Performing Model
**{best_model}** achieved the highest test accuracy of {model_results[best_model]['test_accuracy']:.4f}
"""
        
        if include_feature_importance and model_results:
            report_content += """
## Feature Importance Analysis

"""
            for model_name, results in model_results.items():
                if 'feature_importances' in results:
                    feature_names = [col for col in preprocessed_data.columns if col != target_column]
                    importances = results['feature_importances']
                    
                    # Sort features by importance
                    feature_importance_pairs = list(zip(feature_names, importances))
                    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                    
                    report_content += f"""
### {model_name} - Top 10 Important Features
"""
                    for i, (feature, importance) in enumerate(feature_importance_pairs[:10], 1):
                        report_content += f"{i}. {feature}: {importance:.4f}\n"
                    
                    report_content += "\n"
        
        if include_recommendations:
            report_content += """
## Recommendations

### Model Deployment
- Deploy the best performing model for production use
- Implement model monitoring and retraining pipeline
- Set up data validation for incoming predictions

### Data Quality Improvements
- Address missing data issues in data collection
- Standardize data entry procedures
- Implement data validation rules

### Future Enhancements
- Collect additional relevant features
- Experiment with ensemble methods
- Consider deep learning approaches for larger datasets
- Implement automated model updates

### Clinical Integration
- Integrate predictions into clinical workflow
- Provide uncertainty estimates with predictions
- Establish feedback loop for model improvement
"""
        
        # Display report
        st.text_area("Complete Analysis Report", report_content, height=600)
        
        # Download button
        st.download_button(
            label="📥 Download Complete Report",
            data=report_content,
            file_name=f'complete_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            mime='text/plain'
        )

elif export_type == "Data Export":
    st.write("### Data Export")
    
    # Data selection
    data_to_export = st.selectbox(
        "Select data to export:",
        options=["Original Data", "Preprocessed Data", "Both"]
    )
    
    # Format selection
    export_format = st.selectbox(
        "Select export format:",
        options=["CSV", "Excel", "JSON"]
    )
    
    if st.button("💾 Export Data", type="primary"):
        if data_to_export == "Original Data":
            data = original_data
            filename = "original_data"
        elif data_to_export == "Preprocessed Data":
            if preprocessed_data is not None:
                data = preprocessed_data
                filename = "preprocessed_data"
            else:
                st.error("No preprocessed data available")
                st.stop()
        else:  # Both
            # Combine both datasets
            data = {
                'original': original_data,
                'preprocessed': preprocessed_data if preprocessed_data is not None else pd.DataFrame()
            }
            filename = "combined_data"
        
        # Export based on format
        if export_format == "CSV":
            if isinstance(data, dict):
                # Multiple sheets - combine or export separately
                st.warning("Multiple datasets - exporting as separate CSV files")
                for sheet_name, df in data.items():
                    if not df.empty:
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"📥 Download {sheet_name} CSV",
                            data=csv_data,
                            file_name=f'{filename}_{sheet_name}.csv',
                            mime='text/csv'
                        )
            else:
                csv_data = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f'{filename}.csv',
                    mime='text/csv'
                )
        
        elif export_format == "Excel":
            # Create Excel file with multiple sheets if needed
            buffer = io.BytesIO()
            
            if isinstance(data, dict):
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    for sheet_name, df in data.items():
                        if not df.empty:
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                data.to_excel(buffer, index=False)
            
            st.download_button(
                label="📥 Download Excel",
                data=buffer.getvalue(),
                file_name=f'{filename}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        
        elif export_format == "JSON":
            if isinstance(data, dict):
                json_data = {}
                for sheet_name, df in data.items():
                    if not df.empty:
                        json_data[sheet_name] = df.to_dict('records')
                json_str = json.dumps(json_data, indent=2, default=str)
            else:
                json_str = data.to_json(orient='records', indent=2)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f'{filename}.json',
                mime='application/json'
            )

elif export_type == "Model Performance Report":
    st.write("### Model Performance Report")
    
    if not model_results:
        st.warning("No model results available. Please train models first.")
    else:
        # Model selection
        selected_models = st.multiselect(
            "Select models to include in report:",
            options=list(model_results.keys()),
            default=list(model_results.keys())
        )
        
        if selected_models and st.button("📊 Generate Model Report", type="primary"):
            # Create model performance report
            report_content = f"""
# Model Performance Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Comparison Summary

| Model | Test Accuracy | CV Mean | CV Std | ROC AUC |
|-------|---------------|---------|--------|---------|
"""
            
            for model_name in selected_models:
                results = model_results[model_name]
                roc_auc = results.get('roc_auc', 'N/A')
                report_content += f"| {model_name} | {results['test_accuracy']:.4f} | {results['cv_mean']:.4f} | {results['cv_std']:.4f} | {roc_auc if roc_auc == 'N/A' else f'{roc_auc:.4f}'} |\n"
            
            # Detailed results for each model
            report_content += "\n## Detailed Model Results\n\n"
            
            for model_name in selected_models:
                results = model_results[model_name]
                report_content += f"""
### {model_name}

**Performance Metrics:**
- Test Accuracy: {results['test_accuracy']:.4f}
- Training Accuracy: {results['train_accuracy']:.4f}
- Cross-Validation Mean: {results['cv_mean']:.4f}
- Cross-Validation Std: {results['cv_std']:.4f}
"""
                
                if 'roc_auc' in results:
                    report_content += f"- ROC AUC: {results['roc_auc']:.4f}\n"
                
                # Classification report
                if 'classification_report' in results:
                    report_content += "\n**Classification Report:**\n"
                    class_report = results['classification_report']
                    for class_name, metrics in class_report.items():
                        if isinstance(metrics, dict):
                            report_content += f"- {class_name}: precision={metrics.get('precision', 'N/A'):.3f}, recall={metrics.get('recall', 'N/A'):.3f}, f1-score={metrics.get('f1-score', 'N/A'):.3f}\n"
                
                report_content += "\n"
            
            # Best model recommendation
            best_model = max(selected_models, key=lambda x: model_results[x]['test_accuracy'])
            report_content += f"""
## Recommendation

Based on test accuracy, **{best_model}** is the best performing model with an accuracy of {model_results[best_model]['test_accuracy']:.4f}.
"""
            
            # Display and download
            st.text_area("Model Performance Report", report_content, height=500)
            
            st.download_button(
                label="📥 Download Model Report",
                data=report_content,
                file_name=f'model_performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                mime='text/plain'
            )

elif export_type == "Prediction Template":
    st.write("### Prediction Template")
    st.write("Generate a template file for making predictions with your trained models.")
    
    if preprocessed_data is None or target_column is None:
        st.warning("Need preprocessed data and target column to generate template.")
    else:
        # Get feature names
        feature_names = [col for col in preprocessed_data.columns if col != target_column]
        
        # Template format
        template_format = st.selectbox(
            "Select template format:",
            options=["CSV", "Excel"]
        )
        
        # Create template
        if st.button("📋 Generate Template", type="primary"):
            # Create template with sample data and instructions
            template_data = {
                'Instructions': [
                    'Fill in the values below for each patient',
                    'Each row represents one patient',
                    'Remove this instructions column before uploading',
                    'Ensure all required features are provided',
                    'Use the same data format as training data'
                ]
            }
            
            # Add feature columns with sample/default values
            for feature in feature_names:
                feature_data = preprocessed_data[feature]
                if feature_data.dtype in ['int64', 'float64']:
                    # Use mean as default for numeric features
                    default_val = feature_data.mean()
                    template_data[feature] = [default_val] * 5
                else:
                    # Use most common value for categorical
                    default_val = feature_data.mode().iloc[0] if not feature_data.mode().empty else 0
                    template_data[feature] = [default_val] * 5
            
            template_df = pd.DataFrame(template_data)
            
            # Show template
            st.write("**Template Preview:**")
            st.dataframe(template_df, use_container_width=True)
            
            # Export template
            if template_format == "CSV":
                csv_data = template_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV Template",
                    data=csv_data,
                    file_name='prediction_template.csv',
                    mime='text/csv'
                )
            else:  # Excel
                buffer = io.BytesIO()
                template_df.to_excel(buffer, index=False)
                st.download_button(
                    label="📥 Download Excel Template",
                    data=buffer.getvalue(),
                    file_name='prediction_template.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            
            # Feature information
            st.write("**Feature Information:**")
            feature_info = []
            for feature in feature_names:
                feature_data = preprocessed_data[feature]
                feature_info.append({
                    'Feature': feature,
                    'Type': str(feature_data.dtype),
                    'Min': feature_data.min() if feature_data.dtype in ['int64', 'float64'] else 'N/A',
                    'Max': feature_data.max() if feature_data.dtype in ['int64', 'float64'] else 'N/A',
                    'Unique_Values': feature_data.nunique()
                })
            
            feature_info_df = pd.DataFrame(feature_info)
            st.dataframe(feature_info_df, use_container_width=True)

elif export_type == "Custom Export":
    st.write("### Custom Export")
    st.write("Create a custom export with selected components.")
    
    # Component selection
    st.write("**Select components to include:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_raw_data = st.checkbox("Raw Data")
        include_processed_data = st.checkbox("Processed Data")
        include_data_summary = st.checkbox("Data Summary Statistics")
        include_missing_analysis = st.checkbox("Missing Data Analysis")
    
    with col2:
        include_model_comparison = st.checkbox("Model Comparison")
        include_best_model = st.checkbox("Best Model Details")
        include_feature_importance = st.checkbox("Feature Importance")
        include_predictions = st.checkbox("Sample Predictions")
    
    # Export format
    export_format = st.selectbox(
        "Export format:",
        options=["Text Report", "JSON", "Excel Workbook"]
    )
    
    if st.button("🎯 Create Custom Export", type="primary"):
        if export_format == "Text Report":
            # Create custom text report
            custom_report = f"""
# Custom Analysis Export
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
            
            if include_data_summary:
                custom_report += f"""
## Data Summary
- Total Records: {original_data.shape[0]}
- Total Features: {original_data.shape[1]}
- Target Variable: {target_column if target_column else 'Not specified'}
"""
            
            if include_missing_analysis:
                missing_count = original_data.isnull().sum().sum()
                custom_report += f"""
## Missing Data Analysis
- Total Missing Values: {missing_count}
- Missing Percentage: {(missing_count / (original_data.shape[0] * original_data.shape[1]) * 100):.2f}%
"""
            
            if include_model_comparison and model_results:
                custom_report += """
## Model Comparison
"""
                for model_name, results in model_results.items():
                    custom_report += f"- {model_name}: {results['test_accuracy']:.4f}\n"
            
            if include_best_model and model_results:
                best_model = max(model_results.keys(), key=lambda x: model_results[x]['test_accuracy'])
                custom_report += f"""
## Best Model: {best_model}
- Test Accuracy: {model_results[best_model]['test_accuracy']:.4f}
- CV Mean: {model_results[best_model]['cv_mean']:.4f}
"""
            
            # Display and download
            st.text_area("Custom Report", custom_report, height=400)
            
            st.download_button(
                label="📥 Download Custom Report",
                data=custom_report,
                file_name=f'custom_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                mime='text/plain'
            )
        
        elif export_format == "JSON":
            # Create custom JSON export
            custom_data = {}
            
            if include_data_summary:
                custom_data['data_summary'] = {
                    'total_records': int(original_data.shape[0]),
                    'total_features': int(original_data.shape[1]),
                    'target_variable': target_column,
                    'numeric_features': len(original_data.select_dtypes(include=[np.number]).columns),
                    'categorical_features': len(original_data.select_dtypes(include=['object']).columns)
                }
            
            if include_model_comparison and model_results:
                custom_data['model_results'] = {}
                for model_name, results in model_results.items():
                    custom_data['model_results'][model_name] = {
                        'test_accuracy': float(results['test_accuracy']),
                        'cv_mean': float(results['cv_mean']),
                        'cv_std': float(results['cv_std'])
                    }
            
            if include_raw_data:
                custom_data['raw_data'] = original_data.to_dict('records')
            
            if include_processed_data and preprocessed_data is not None:
                custom_data['processed_data'] = preprocessed_data.to_dict('records')
            
            # Convert to JSON string
            json_str = json.dumps(custom_data, indent=2, default=str)
            
            st.download_button(
                label="📥 Download Custom JSON",
                data=json_str,
                file_name=f'custom_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json'
            )
        
        elif export_format == "Excel Workbook":
            # Create Excel workbook with multiple sheets
            buffer = io.BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                if include_raw_data:
                    original_data.to_excel(writer, sheet_name='Raw_Data', index=False)
                
                if include_processed_data and preprocessed_data is not None:
                    preprocessed_data.to_excel(writer, sheet_name='Processed_Data', index=False)
                
                if include_data_summary:
                    summary_data = {
                        'Metric': ['Total Records', 'Total Features', 'Target Variable', 'Missing Values'],
                        'Value': [
                            original_data.shape[0],
                            original_data.shape[1],
                            target_column if target_column else 'Not specified',
                            original_data.isnull().sum().sum()
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                if include_model_comparison and model_results:
                    model_comparison_data = []
                    for model_name, results in model_results.items():
                        model_comparison_data.append({
                            'Model': model_name,
                            'Test_Accuracy': results['test_accuracy'],
                            'CV_Mean': results['cv_mean'],
                            'CV_Std': results['cv_std']
                        })
                    pd.DataFrame(model_comparison_data).to_excel(writer, sheet_name='Model_Results', index=False)
            
            st.download_button(
                label="📥 Download Custom Excel",
                data=buffer.getvalue(),
                file_name=f'custom_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

# Quick export buttons
st.subheader("Quick Exports")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Quick Data Export"):
        if preprocessed_data is not None:
            csv_data = preprocessed_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Processed Data",
                data=csv_data,
                file_name=f'processed_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )

with col2:
    if st.button("🤖 Quick Model Summary") and model_results:
        summary = "Model Performance Summary\n\n"
        for model_name, results in model_results.items():
            summary += f"{model_name}: {results['test_accuracy']:.4f}\n"
        
        st.download_button(
            label="📥 Download Model Summary",
            data=summary,
            file_name=f'model_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            mime='text/plain'
        )

with col3:
    if st.button("📋 Quick Report"):
        quick_report = f"""
Quick Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset: {original_data.shape[0]} records, {original_data.shape[1]} features
Target: {target_column if target_column else 'Not specified'}
Models Trained: {len(model_results) if model_results else 0}
Best Model: {max(model_results.keys(), key=lambda x: model_results[x]['test_accuracy']) if model_results else 'None'}
"""
        
        st.download_button(
            label="📥 Download Quick Report",
            data=quick_report,
            file_name=f'quick_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            mime='text/plain'
        )

# Export history/log
st.subheader("Export Guidelines")

st.info("""
### Export Tips:
- **Complete Analysis Report**: Best for comprehensive documentation
- **Data Export**: Use for sharing processed datasets
- **Model Performance Report**: Focus on model comparison and metrics
- **Prediction Template**: Create templates for future predictions
- **Custom Export**: Tailor exports to specific needs

### File Formats:
- **CSV**: Best for data compatibility
- **Excel**: Good for structured reports with multiple sheets
- **JSON**: Ideal for programmatic access
- **Text**: Human-readable reports and documentation
""")
