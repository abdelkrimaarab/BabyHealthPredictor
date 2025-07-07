import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class DataVisualizer:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_missing_values(self, df):
        """Create visualization for missing values"""
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        # Create DataFrame for plotting
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Count', ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(
            missing_df, 
            y='Column', 
            x='Missing_Count',
            title='Missing Values by Column',
            labels={'Missing_Count': 'Number of Missing Values'},
            color='Missing_Percentage',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(height=max(400, len(missing_df) * 25))
        return fig
    
    def plot_data_types(self, df):
        """Create visualization for data types distribution"""
        dtype_counts = df.dtypes.value_counts()
        
        # Convert dtype names to strings to avoid JSON serialization issues
        dtype_names = [str(dtype) for dtype in dtype_counts.index]
        
        fig = px.pie(
            values=dtype_counts.values,
            names=dtype_names,
            title='Distribution of Data Types'
        )
        
        return fig
    
    def plot_numeric_distributions(self, df, max_cols=4):
        """Create distribution plots for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        n_cols = min(max_cols, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        # Calculate appropriate vertical spacing based on number of rows
        vertical_spacing = max(0.02, min(0.1, 1.0 / (n_rows + 1))) if n_rows > 1 else 0.1
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=list(numeric_cols),
            vertical_spacing=vertical_spacing
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row,
                col=col_pos
            )
        
        fig.update_layout(
            title='Distribution of Numeric Variables',
            height=300 * n_rows,
            showlegend=False
        )
        
        return fig
    
    def plot_categorical_distributions(self, df, max_cols=4):
        """Create bar plots for categorical columns"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            return None
        
        n_cols = min(max_cols, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        # Calculate appropriate vertical spacing based on number of rows
        vertical_spacing = max(0.02, min(0.1, 1.0 / (n_rows + 1))) if n_rows > 1 else 0.1
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=list(categorical_cols),
            vertical_spacing=vertical_spacing
        )
        
        for i, col in enumerate(categorical_cols):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1
            
            value_counts = df[col].value_counts().head(10)  # Top 10 categories
            
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values, name=col, showlegend=False),
                row=row,
                col=col_pos
            )
        
        fig.update_layout(
            title='Distribution of Categorical Variables',
            height=300 * n_rows,
            showlegend=False
        )
        
        return fig
    
    def plot_correlation_matrix(self, df, target_column=None):
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return None
        
        correlation_matrix = numeric_df.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        
        fig.update_layout(height=max(400, len(correlation_matrix) * 30))
        return fig
    
    def plot_target_distribution(self, df, target_column):
        """Create distribution plot for target variable"""
        if target_column not in df.columns:
            return None
        
        target_counts = df[target_column].value_counts()
        
        fig = px.bar(
            x=target_counts.index,
            y=target_counts.values,
            title=f'Distribution of {target_column}',
            labels={'x': target_column, 'y': 'Count'}
        )
        
        return fig
    
    def plot_feature_target_relationship(self, df, feature_column, target_column):
        """Create plots showing relationship between features and target"""
        if feature_column not in df.columns or target_column not in df.columns:
            return None
        
        if df[feature_column].dtype in ['object']:
            # Categorical feature
            fig = px.box(
                df, 
                x=feature_column, 
                y=target_column,
                title=f'{feature_column} vs {target_column}'
            )
        else:
            # Numeric feature - simple scatter plot without trendline to avoid statsmodels dependency
            fig = px.scatter(
                df,
                x=feature_column,
                y=target_column,
                title=f'{feature_column} vs {target_column}'
            )
        
        return fig
    
    def plot_model_comparison(self, comparison_df):
        """Create model comparison visualization"""
        if comparison_df is None or comparison_df.empty:
            return None
        
        fig = px.bar(
            comparison_df,
            x='Model',
            y='Test Accuracy',
            title='Model Performance Comparison',
            color='Test Accuracy',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix, model_name):
        """Create confusion matrix heatmap"""
        fig = px.imshow(
            confusion_matrix,
            text_auto=True,
            aspect="auto",
            title=f'Confusion Matrix - {model_name}',
            labels=dict(x="Predicted", y="Actual"),
            color_continuous_scale='Blues'
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance_df, model_name, top_n=10):
        """Create feature importance plot"""
        if feature_importance_df is None or feature_importance_df.empty:
            return None
        
        # Get top N features
        top_features = feature_importance_df.head(top_n)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importance - {model_name}',
            labels={'importance': 'Importance Score', 'feature': 'Features'}
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        return fig
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """Create ROC curve plot"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc:.3f})',
            line=dict(width=2)
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_cross_validation_scores(self, model_results):
        """Create cross-validation scores comparison"""
        cv_data = []
        for model_name, results in model_results.items():
            cv_data.append({
                'Model': model_name,
                'CV Mean': results['cv_mean'],
                'CV Std': results['cv_std']
            })
        
        cv_df = pd.DataFrame(cv_data)
        
        fig = px.bar(
            cv_df,
            x='Model',
            y='CV Mean',
            error_y='CV Std',
            title='Cross-Validation Scores Comparison',
            labels={'CV Mean': 'Cross-Validation Mean Score'}
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        return fig
