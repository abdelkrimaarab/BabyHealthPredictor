import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import streamlit as st

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        
    def load_excel_file(self, file):
        """Load Excel file with error handling"""
        try:
            # Try different encodings and sheet detection
            df = pd.read_excel(file, engine='openpyxl')
            
            # Convert datetime columns to strings to avoid serialization issues
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)
                # Handle other problematic object types
                elif df[col].dtype == 'object':
                    # Try to convert problematic object types to strings
                    df[col] = df[col].astype(str)
            
            return df, None
        except Exception as e:
            try:
                # Try with xlrd engine
                df = pd.read_excel(file, engine='xlrd')
                
                # Apply same conversions
                for col in df.columns:
                    if df[col].dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].astype(str)
                    elif df[col].dtype == 'object':
                        df[col] = df[col].astype(str)
                
                return df, None
            except Exception as e2:
                error_msg = f"Error reading Excel file: {str(e)}\nAlternative attempt: {str(e2)}"
                return None, error_msg
    
    def basic_info(self, df):
        """Get basic information about the dataset"""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        return info
    
    def detect_target_column(self, df):
        """Attempt to detect potential target columns for morbidity prediction"""
        potential_targets = []
        
        # Common morbidity-related column names (French and English)
        morbidity_keywords = [
            'morbidité', 'morbidity', 'morbidit', 'outcome', 'resultat', 'résultat',
            'complication', 'pathologie', 'pathology', 'disease', 'maladie',
            'diagnostic', 'diagnosis', 'condition', 'etat', 'état', 'status',
            'healthy', 'sain', 'normal', 'anormal', 'abnormal', 'risk', 'risque'
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in morbidity_keywords):
                potential_targets.append(col)
        
        # Also look for binary columns that might represent morbidity
        for col in df.columns:
            if df[col].nunique() == 2:
                unique_vals = df[col].unique()
                if any(str(val).lower() in ['yes', 'no', 'oui', 'non', '1', '0', 'true', 'false'] 
                       for val in unique_vals):
                    potential_targets.append(col)
        
        return list(set(potential_targets))
    
    def clean_data(self, df, target_column=None):
        """Clean and preprocess the data"""
        df_cleaned = df.copy()
        
        # Remove completely empty rows and columns
        df_cleaned = df_cleaned.dropna(how='all')
        df_cleaned = df_cleaned.dropna(axis=1, how='all')
        
        # Handle missing values
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
        
        # Impute numeric columns with median
        if len(numeric_columns) > 0:
            self.imputers['numeric'] = SimpleImputer(strategy='median')
            df_cleaned[numeric_columns] = self.imputers['numeric'].fit_transform(df_cleaned[numeric_columns])
        
        # Impute categorical columns with mode
        if len(categorical_columns) > 0:
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            df_cleaned[categorical_columns] = self.imputers['categorical'].fit_transform(df_cleaned[categorical_columns])
        
        # Remove columns with too many missing values (>50%)
        missing_threshold = 0.5
        columns_to_drop = []
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() / len(df_cleaned) > missing_threshold:
                columns_to_drop.append(col)
        
        if columns_to_drop:
            df_cleaned = df_cleaned.drop(columns=columns_to_drop)
        
        return df_cleaned, columns_to_drop
    
    def encode_categorical_variables(self, df, target_column=None):
        """Encode categorical variables"""
        df_encoded = df.copy()
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != target_column:  # Don't encode target if it's categorical
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
        
        # Handle target column separately if it's categorical
        if target_column and target_column in categorical_columns:
            self.label_encoders[target_column] = LabelEncoder()
            df_encoded[target_column] = self.label_encoders[target_column].fit_transform(df_encoded[target_column].astype(str))
        
        return df_encoded
    
    def scale_features(self, df, target_column=None):
        """Scale numeric features"""
        df_scaled = df.copy()
        
        # Get feature columns (exclude target)
        feature_columns = [col for col in df_scaled.columns if col != target_column]
        numeric_features = df_scaled[feature_columns].select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) > 0:
            df_scaled[numeric_features] = self.scaler.fit_transform(df_scaled[numeric_features])
        
        return df_scaled
    
    def prepare_for_modeling(self, df, target_column):
        """Prepare data for machine learning modeling"""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle target variable
        if y.dtype == 'object':
            # Binary classification
            if y.nunique() == 2:
                y = (y == y.unique()[0]).astype(int)
            else:
                # Multi-class classification
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                self.label_encoders['target'] = le_target
        
        return X, y
    
    def get_feature_importance_data(self, df, target_column):
        """Get data for feature importance analysis"""
        if target_column not in df.columns:
            return None
        
        # Calculate correlation with target for numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for feature in numeric_features:
            if feature != target_column:
                try:
                    corr = df[feature].corr(df[target_column])
                    if not np.isnan(corr):
                        correlations[feature] = abs(corr)
                except:
                    pass
        
        return correlations
