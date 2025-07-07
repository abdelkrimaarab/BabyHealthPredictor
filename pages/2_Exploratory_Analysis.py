import streamlit as st
import pandas as pd
import numpy as np
from utils.data_preprocessing import DataPreprocessor
from utils.visualization import DataVisualizer

st.set_page_config(page_title="Exploratory Analysis", page_icon="🔍", layout="wide")

st.title("🔍 Exploratory Data Analysis")
st.markdown("---")

# Initialize session state if needed
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# Check if data is loaded
if st.session_state.data is None:
    st.warning("⚠️ No data loaded. Please upload data in the **Data Upload** page first.")
    st.stop()

# Initialize visualizer
visualizer = DataVisualizer()
df = st.session_state.data

# Use original data for display, but keep preprocessed data available for calculations
df_analysis = df  # Always use original data for display
df_processed = st.session_state.preprocessed_data  # Keep preprocessed data for calculations if needed

if st.session_state.preprocessed_data is not None:
    st.info("ℹ️ Displaying original data values for better readability")
else:
    st.info("ℹ️ Using original data for analysis")

st.subheader("Dataset Summary")

# Basic statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", df_analysis.shape[0])
with col2:
    st.metric("Total Features", df_analysis.shape[1])
with col3:
    numeric_features = df_analysis.select_dtypes(include=[np.number]).shape[1]
    st.metric("Numeric Features", numeric_features)
with col4:
    categorical_features = df_analysis.select_dtypes(include=['object']).shape[1]
    st.metric("Categorical Features", categorical_features)

# Descriptive statistics
st.subheader("Descriptive Statistics")

tab1, tab2 = st.tabs(["📊 Numeric Variables", "📋 Categorical Variables"])

with tab1:
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df_analysis[numeric_cols].describe(), use_container_width=True)
        
        # Distribution plots
        st.write("**Distribution of Numeric Variables:**")
        dist_fig = visualizer.plot_numeric_distributions(df_analysis)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)
    else:
        st.info("No numeric variables found in the dataset.")

with tab2:
    categorical_cols = df_analysis.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        # Show categorical summary using original data
        cat_summary = []
        for col in categorical_cols:
            cat_summary.append({
                'Column': col,
                'Unique Values': df_analysis[col].nunique(),
                'Most Frequent': df_analysis[col].mode().iloc[0] if not df_analysis[col].mode().empty else 'N/A',
                'Most Frequent Count': df_analysis[col].value_counts().iloc[0] if len(df_analysis[col].value_counts()) > 0 else 0
            })
        
        cat_df = pd.DataFrame(cat_summary)
        st.dataframe(cat_df, use_container_width=True)
        
        # Distribution plots using original data
        st.write("**Distribution of Categorical Variables:**")
        cat_fig = visualizer.plot_categorical_distributions(df_analysis)
        if cat_fig:
            st.plotly_chart(cat_fig, use_container_width=True)
            
        # Show detailed value counts for each categorical variable
        st.write("**Detailed Value Counts:**")
        selected_cat_col = st.selectbox(
            "Select categorical variable to see detailed counts:",
            options=categorical_cols
        )
        
        if selected_cat_col:
            value_counts = df_analysis[selected_cat_col].value_counts()
            st.dataframe(value_counts.to_frame('Count'), use_container_width=True)
    else:
        st.info("No categorical variables found in the dataset.")

# Correlation analysis
st.subheader("Correlation Analysis")
numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns

if len(numeric_cols) >= 2:
    corr_fig = visualizer.plot_correlation_matrix(df_analysis)
    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True)
    
    # High correlations
    corr_matrix = df_analysis[numeric_cols].corr()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # High correlation threshold
                high_corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if high_corr_pairs:
        st.write("**High Correlations (|r| > 0.7):**")
        high_corr_df = pd.DataFrame(high_corr_pairs)
        st.dataframe(high_corr_df, use_container_width=True)
    else:
        st.info("No high correlations found between variables.")
else:
    st.info("Need at least 2 numeric variables for correlation analysis.")

# Target variable analysis
st.subheader("Target Variable Analysis")

# Let user select target variable
all_columns = df_analysis.columns.tolist()
target_column = st.selectbox(
    "Select target variable for morbidity prediction:",
    options=[None] + all_columns,
    help="Choose the column that represents morbidity outcome"
)

if target_column:
    st.write(f"**Analysis of target variable: {target_column}**")
    
    # Target distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Target Variable Summary:**")
        target_summary = {
            'Unique Values': df_analysis[target_column].nunique(),
            'Missing Values': df_analysis[target_column].isnull().sum(),
            'Data Type': str(df_analysis[target_column].dtype)
        }
        
        for key, value in target_summary.items():
            st.write(f"- {key}: {value}")
        
        # Value counts
        st.write("**Value Distribution:**")
        value_counts = df_analysis[target_column].value_counts()
        st.dataframe(value_counts.to_frame('Count'), use_container_width=True)
    
    with col2:
        # Target distribution plot
        target_fig = visualizer.plot_target_distribution(df_analysis, target_column)
        if target_fig:
            st.plotly_chart(target_fig, use_container_width=True)
    
    # Feature-target relationships
    st.write("**Feature-Target Relationships:**")
    
    # Select features to analyze
    feature_columns = [col for col in df_analysis.columns if col != target_column]
    selected_features = st.multiselect(
        "Select features to analyze relationship with target:",
        options=feature_columns,
        default=feature_columns[:5] if len(feature_columns) >= 5 else feature_columns,
        help="Select features to see their relationship with the target variable"
    )
    
    if selected_features:
        # Create tabs for different features
        if len(selected_features) <= 4:
            tabs = st.tabs([f"📊 {feature}" for feature in selected_features])
            
            for i, feature in enumerate(selected_features):
                with tabs[i]:
                    relationship_fig = visualizer.plot_feature_target_relationship(
                        df_analysis, feature, target_column
                    )
                    if relationship_fig:
                        st.plotly_chart(relationship_fig, use_container_width=True)
                    
                    # Statistical summary using original data
                    if df_analysis[feature].dtype in ['object']:
                        # Categorical feature - show cross-tabulation with original values
                        crosstab = pd.crosstab(df_analysis[feature], df_analysis[target_column])
                        st.write("**Cross-tabulation:**")
                        st.dataframe(crosstab, use_container_width=True)
                        
                        # Show percentage distribution
                        crosstab_pct = pd.crosstab(df_analysis[feature], df_analysis[target_column], normalize='index') * 100
                        st.write("**Percentage Distribution:**")
                        st.dataframe(crosstab_pct.round(2), use_container_width=True)
                    else:
                        # Numeric feature
                        grouped_stats = df_analysis.groupby(target_column)[feature].describe()
                        st.write("**Grouped Statistics:**")
                        st.dataframe(grouped_stats, use_container_width=True)
        else:
            # For many features, use a dropdown
            selected_feature = st.selectbox(
                "Select a feature to analyze:",
                options=selected_features
            )
            
            if selected_feature:
                relationship_fig = visualizer.plot_feature_target_relationship(
                    df_analysis, selected_feature, target_column
                )
                if relationship_fig:
                    st.plotly_chart(relationship_fig, use_container_width=True)

# Data quality insights
st.subheader("Data Quality Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("**Missing Data Analysis:**")
    missing_data = df_analysis.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(df_analysis) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)
        
        # Show which columns have missing data
        st.write("**Columns with Missing Data:**")
        for col in missing_data.index:
            missing_count = missing_data[col]
            missing_pct = (missing_count / len(df_analysis) * 100)
            st.write(f"• **{col}**: {missing_count} values ({missing_pct:.1f}%)")
    else:
        st.success("✅ No missing values found!")

with col2:
    st.write("**Data Type Analysis:**")
    dtype_counts = df_analysis.dtypes.value_counts()
    dtype_df = pd.DataFrame({
        'Data Type': dtype_counts.index,
        'Count': dtype_counts.values
    })
    st.dataframe(dtype_df, use_container_width=True)

# Outlier detection for numeric variables
st.subheader("Outlier Detection")

numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    selected_numeric = st.selectbox(
        "Select numeric variable for outlier analysis:",
        options=numeric_cols
    )
    
    if selected_numeric:
        # Calculate outliers using IQR method
        Q1 = df_analysis[selected_numeric].quantile(0.25)
        Q3 = df_analysis[selected_numeric].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_analysis[(df_analysis[selected_numeric] < lower_bound) | 
                              (df_analysis[selected_numeric] > upper_bound)]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Outliers", len(outliers))
        with col2:
            st.metric("Outlier %", f"{len(outliers)/len(df_analysis)*100:.1f}%")
        with col3:
            st.metric("IQR", f"{IQR:.2f}")
        
        # Box plot
        import plotly.express as px
        box_fig = px.box(df_analysis, y=selected_numeric, title=f"Box Plot - {selected_numeric}")
        st.plotly_chart(box_fig, use_container_width=True)
        
        if len(outliers) > 0:
            st.write("**Outlier Details:**")
            st.dataframe(outliers[[selected_numeric]], use_container_width=True)

# Export analysis summary
st.subheader("Export Analysis Summary")

if st.button("📋 Generate Analysis Report"):
    # Create analysis summary
    summary_report = f"""
# Exploratory Data Analysis Report

## Dataset Overview
- Total Records: {df_analysis.shape[0]}
- Total Features: {df_analysis.shape[1]}
- Numeric Features: {len(df_analysis.select_dtypes(include=[np.number]).columns)}
- Categorical Features: {len(df_analysis.select_dtypes(include=['object']).columns)}

## Data Quality
- Missing Values: {df_analysis.isnull().sum().sum()}
- Missing Percentage: {(df_analysis.isnull().sum().sum() / (df_analysis.shape[0] * df_analysis.shape[1]) * 100):.2f}%

## Target Variable: {target_column if target_column else 'Not Selected'}
"""
    
    if target_column:
        summary_report += f"""
- Unique Values: {df_analysis[target_column].nunique()}
- Missing Values: {df_analysis[target_column].isnull().sum()}
"""
    
    st.text_area("Analysis Summary", summary_report, height=300)
    
    # Download button for the report
    st.download_button(
        label="📥 Download Analysis Report",
        data=summary_report,
        file_name='eda_report.txt',
        mime='text/plain'
    )

# Store target column in session state for model training
if target_column:
    st.session_state.target_column = target_column
    st.success(f"✅ Target variable '{target_column}' saved for model training!")

# Navigation hint
if target_column:
    st.info("🚀 Ready for model training! Navigate to **Model Training** to build predictive models.")
