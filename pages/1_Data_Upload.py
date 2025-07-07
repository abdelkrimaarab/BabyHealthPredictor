import streamlit as st
import pandas as pd
import numpy as np
from utils.data_preprocessing import DataPreprocessor
from utils.visualization import DataVisualizer

st.set_page_config(page_title="Data Upload", page_icon="📊", layout="wide")

st.title("📊 Data Upload & Validation")
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

# Initialize preprocessor and visualizer
preprocessor = DataPreprocessor()
visualizer = DataVisualizer()

# File upload
st.subheader("Upload Excel File")
uploaded_file = st.file_uploader(
    "Choose an Excel file containing newborn health data",
    type=['xlsx', 'xls'],
    help="Upload an Excel file with newborn morbidity data"
)

if uploaded_file is not None:
    with st.spinner("Loading and processing file..."):
        # Load the file
        df, error = preprocessor.load_excel_file(uploaded_file)
        
        if error:
            st.error(f"Error loading file: {error}")
            st.info("Please ensure your file is a valid Excel format (.xlsx or .xls)")
        else:
            # Store in session state
            st.session_state.data = df
            st.success(f"✅ File loaded successfully! Shape: {df.shape}")
            
            # Display basic information
            st.subheader("Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", df.shape[0])
            with col2:
                st.metric("Total Features", df.shape[1])
            with col3:
                missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing Data %", f"{missing_percentage:.1f}%")
            with col4:
                numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
                st.metric("Numeric Features", numeric_cols)
            
            # Display first few rows
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column information
            st.subheader("Column Information")
            info = preprocessor.basic_info(df)
            
            col_info_df = pd.DataFrame({
                'Column': info['columns'],
                'Data Type': [str(dtype) for dtype in info['dtypes'].values()],
                'Missing Values': [info['missing_values'][col] for col in info['columns']],
                'Missing %': [f"{info['missing_percentage'][col]:.1f}%" for col in info['columns']],
                'Unique Values': [info['unique_values'][col] for col in info['columns']]
            })
            
            st.dataframe(col_info_df, use_container_width=True)
            
            # Data quality checks
            st.subheader("Data Quality Assessment")
            
            # Missing values visualization
            if df.isnull().sum().sum() > 0:
                st.write("**Missing Values Analysis:**")
                missing_fig = visualizer.plot_missing_values(df)
                st.plotly_chart(missing_fig, use_container_width=True)
            else:
                st.success("✅ No missing values found in the dataset!")
            
            # Data types distribution
            st.write("**Data Types Distribution:**")
            dtype_fig = visualizer.plot_data_types(df)
            st.plotly_chart(dtype_fig, use_container_width=True)
            
            # Potential target column detection
            st.subheader("Target Variable Detection")
            potential_targets = preprocessor.detect_target_column(df)
            
            if potential_targets:
                st.write("**Potential target columns detected:**")
                for target in potential_targets:
                    with st.expander(f"📊 {target}"):
                        unique_vals = df[target].value_counts()
                        st.write(f"**Unique values:** {df[target].nunique()}")
                        st.write("**Value distribution:**")
                        st.dataframe(unique_vals.head(10))
                        
                        if df[target].nunique() <= 10:
                            target_fig = visualizer.plot_target_distribution(df, target)
                            st.plotly_chart(target_fig, use_container_width=True)
            else:
                st.warning("⚠️ No obvious target columns detected. You may need to specify the target manually.")
            
            # Data cleaning options
            st.subheader("Data Preprocessing Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Cleaning Options:**")
                clean_data = st.checkbox("Clean data (remove empty rows/columns)", value=True)
                handle_missing = st.checkbox("Handle missing values", value=True)
                remove_high_missing = st.checkbox("Remove columns with >50% missing values", value=True)
            
            with col2:
                st.write("**Encoding Options:**")
                encode_categorical = st.checkbox("Encode categorical variables", value=True)
                scale_features = st.checkbox("Scale numeric features", value=False)
            
            # Apply preprocessing
            if st.button("🔄 Apply Preprocessing", type="primary"):
                with st.spinner("Applying preprocessing..."):
                    processed_df = df.copy()
                    
                    # Clean data
                    if clean_data:
                        processed_df, dropped_cols = preprocessor.clean_data(processed_df)
                        if dropped_cols:
                            st.warning(f"Dropped columns with high missing values: {dropped_cols}")
                    
                    # Encode categorical variables
                    if encode_categorical:
                        processed_df = preprocessor.encode_categorical_variables(processed_df)
                    
                    # Scale features
                    if scale_features:
                        processed_df = preprocessor.scale_features(processed_df)
                    
                    # Store preprocessed data
                    st.session_state.preprocessed_data = processed_df
                    
                    st.success("✅ Preprocessing completed!")
                    
                    # Show preprocessing results
                    st.subheader("Preprocessing Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Before Preprocessing:**")
                        st.write(f"Shape: {df.shape}")
                        st.write(f"Missing values: {df.isnull().sum().sum()}")
                    
                    with col2:
                        st.write("**After Preprocessing:**")
                        st.write(f"Shape: {processed_df.shape}")
                        st.write(f"Missing values: {processed_df.isnull().sum().sum()}")
                    
                    # Show processed data preview
                    st.write("**Processed Data Preview:**")
                    st.dataframe(processed_df.head(10), use_container_width=True)

# Data download option
if st.session_state.preprocessed_data is not None:
    st.subheader("Download Processed Data")
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv_data = convert_df_to_csv(st.session_state.preprocessed_data)
    
    st.download_button(
        label="📥 Download processed data as CSV",
        data=csv_data,
        file_name='processed_newborn_data.csv',
        mime='text/csv'
    )

# Navigation hint
if st.session_state.data is not None:
    st.info("✨ Data loaded successfully! Navigate to **Exploratory Analysis** to explore your data further.")
