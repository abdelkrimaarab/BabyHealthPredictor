import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.model_results = {}
        self.best_params = {}
        
    def get_available_models(self):
        """Get dictionary of available models"""
        return {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=5000, solver='liblinear'),
            'SVM': SVC(random_state=42, probability=True),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Neural Network': MLPClassifier(random_state=42, max_iter=2000)
        }
    
    def get_hyperparameter_grids(self):
        """Get hyperparameter grids for model tuning"""
        return {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    
    def train_model(self, model_name, X_train, X_test, y_train, y_test, tune_hyperparameters=False):
        """Train a single model"""
        try:
            models_dict = self.get_available_models()
            if model_name not in models_dict:
                raise ValueError(f"Model {model_name} not available")
            
            model = models_dict[model_name]
            
            # Hyperparameter tuning if requested
            if tune_hyperparameters:
                param_grids = self.get_hyperparameter_grids()
                if model_name in param_grids:
                    grid_search = GridSearchCV(
                        model, 
                        param_grids[model_name], 
                        cv=5, 
                        scoring='accuracy',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    self.best_params[model_name] = grid_search.best_params_
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred_proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Additional metrics for binary classification
            metrics = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
            }
            
            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2 and y_pred_proba_test is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba_test)
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                metrics['feature_importances'] = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                metrics['feature_importances'] = np.abs(model.coef_[0]).tolist()
            
            # Store model and results
            self.models[model_name] = model
            self.model_results[model_name] = metrics
            
            return model, metrics
            
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return None, None
    
    def train_multiple_models(self, model_names, X_train, X_test, y_train, y_test, tune_hyperparameters=False):
        """Train multiple models"""
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(model_names):
            status_text.text(f"Training {model_name}...")
            progress_bar.progress((i + 1) / len(model_names))
            
            model, metrics = self.train_model(
                model_name, X_train, X_test, y_train, y_test, tune_hyperparameters
            )
            
            if model is not None:
                results[model_name] = metrics
        
        status_text.text("Training completed!")
        return results
    
    def get_model_comparison(self):
        """Get comparison of all trained models"""
        if not self.model_results:
            return None
        
        comparison_data = []
        for model_name, metrics in self.model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Test Accuracy': metrics['test_accuracy'],
                'CV Mean': metrics['cv_mean'],
                'CV Std': metrics['cv_std'],
                'ROC AUC': metrics.get('roc_auc', 'N/A')
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, metric='test_accuracy'):
        """Get the best performing model based on specified metric"""
        if not self.model_results:
            return None, None
        
        best_model_name = max(
            self.model_results.keys(),
            key=lambda x: self.model_results[x].get(metric, 0)
        )
        
        return best_model_name, self.models[best_model_name]
    
    def predict_new_data(self, model_name, X_new):
        """Make predictions on new data"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        predictions = model.predict(X_new)
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_new)
        
        return predictions, probabilities
    
    def get_feature_importance(self, model_name, feature_names):
        """Get feature importance for a trained model"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
