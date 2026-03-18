"""
Modular components for the House Price Prediction Dashboard
Separated logic from UI for better maintainability
"""

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING & CACHING
# ═══════════════════════════════════════════════════════════════════

class ModelManager:
    """Manages model loading and predictions"""
    
    def __init__(self, model_path='gbr_model.joblib', features_path='feature_columns.joblib'):
        self.model_path = model_path
        self.features_path = features_path
        self.model = None
        self.feature_columns = None
    
    def load(self):
        """Load model and feature columns"""
        try:
            self.model = joblib.load(self.model_path)
            self.feature_columns = joblib.load(self.features_path)
            return True, "Model loaded successfully"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def predict(self, input_df):
        """Make prediction on input data"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Ensure all features are present
        input_df = input_df.reindex(columns=self.feature_columns, fill_value=0)
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        return prediction
    
    def is_loaded(self):
        """Check if model is ready"""
        return self.model is not None and self.feature_columns is not None


class DataLoader:
    """Handles data loading from CSV files"""
    
    @staticmethod
    def load_training_data(path='dataset/train.csv'):
        """Load training dataset"""
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    @staticmethod
    def load_test_data(path='dataset/test.csv'):
        """Load test dataset"""
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════
# DATA ANALYSIS & STATISTICS
# ═══════════════════════════════════════════════════════════════════

class DataAnalyzer:
    """Performs statistical analysis on datasets"""
    
    def __init__(self, data):
        self.data = data
    
    def get_summary_stats(self, target_column='SalePrice'):
        """Get summary statistics"""
        if target_column not in self.data.columns:
            return {}
        
        return {
            'mean': self.data[target_column].mean(),
            'median': self.data[target_column].median(),
            'std': self.data[target_column].std(),
            'min': self.data[target_column].min(),
            'max': self.data[target_column].max(),
            'q1': self.data[target_column].quantile(0.25),
            'q3': self.data[target_column].quantile(0.75),
            'skewness': self.data[target_column].skew(),
            'kurtosis': self.data[target_column].kurtosis()
        }
    
    def get_feature_types(self):
        """Get feature type breakdown"""
        dtypes = self.data.dtypes
        return {
            'numeric': len(dtypes[dtypes.isin(['int64', 'float64'])]),
            'categorical': len(dtypes[dtypes == 'object']),
            'datetime': len(dtypes[dtypes == 'datetime64[ns]']),
            'total': len(dtypes)
        }
    
    def get_correlation_with_target(self, target_column='SalePrice', top_n=10):
        """Get top features correlated with target"""
        numeric_data = self.data.select_dtypes(include=['number'])
        
        if target_column not in numeric_data.columns:
            return pd.Series()
        
        correlations = numeric_data.corr()[target_column].drop(target_column)
        return correlations.abs().sort_values(ascending=False).head(top_n)
    
    def get_missing_value_summary(self):
        """Get missing values info"""
        missing = self.data.isnull().sum()
        missing_percent = (missing / len(self.data)) * 100
        return pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percent': missing_percent
        }).sort_values('Missing_Count', ascending=False).head(20)
    
    def get_outlier_summary(self, numeric_only=True):
        """Detect potential outliers using IQR method"""
        if numeric_only:
            data = self.data.select_dtypes(include=['number'])
        else:
            data = self.data
        
        outlier_summary = {}
        for col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_summary[col] = len(outliers)
        
        return pd.Series(outlier_summary).sort_values(ascending=False).head(10)


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION GENERATORS
# ═══════════════════════════════════════════════════════════════════

class ChartGenerator:
    """Creates professional visualizations using Plotly"""
    
    @staticmethod
    def create_distribution_chart(data, column, title="Distribution", nbins=50):
        """Create distribution histogram"""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data[column],
            nbinsx=nbins,
            name=column,
            marker_color='rgba(102, 126, 234, 0.7)',
            marker_line_color='rgba(102, 126, 234, 1)',
            marker_line_width=1.5
        ))
        
        fig.update_layout(
            title=f'<b>{title}</b>',
            xaxis_title=column,
            yaxis_title='Frequency',
            hovermode='x unified',
            plot_bgcolor='rgba(245, 247, 250, 0.5)',
            paper_bgcolor='white',
            font=dict(size=12, color='#1f2937'),
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_correlation_chart(correlations, title="Feature Correlation"):
        """Create correlation bar chart"""
        fig = go.Figure(data=[
            go.Bar(
                y=correlations.index,
                x=correlations.values,
                orientation='h',
                marker=dict(
                    color=correlations.values,
                    colorscale='Viridis',
                    showscale=True
                )
            )
        ])
        
        fig.update_layout(
            title=f'<b>{title}</b>',
            xaxis_title='Correlation Coefficient',
            yaxis_title='Features',
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(245, 247, 250, 0.5)',
            paper_bgcolor='white',
            font=dict(size=12, color='#1f2937')
        )
        
        return fig
    
    @staticmethod
    def create_scatter_chart(data, x_col, y_col, title="Scatter Plot", color_col=None):
        """Create scatter plot"""
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col if color_col else None,
            title=f"<b>{title}</b>",
            labels={x_col: x_col.replace('_', ' '), y_col: y_col.replace('_', ' ')},
            opacity=0.7
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(245, 247, 250, 0.5)',
            paper_bgcolor='white',
            font=dict(size=12, color='#1f2937')
        )
        
        return fig
    
    @staticmethod
    def create_box_plot(data, columns, title="Distribution Comparison"):
        """Create box plot for multiple columns"""
        fig = px.box(
            data,
            y=columns,
            title=f"<b>{title}</b>",
            labels={col: col.replace('_', ' ').title() for col in columns}
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(245, 247, 250, 0.5)',
            paper_bgcolor='white',
            font=dict(size=12, color='#1f2937')
        )
        
        return fig
    
    @staticmethod
    def create_heatmap(data, numeric_only=True, title="Correlation Heatmap"):
        """Create correlation heatmap"""
        if numeric_only:
            data = data.select_dtypes(include=['number'])
        
        corr_matrix = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=f'<b>{title}</b>',
            height=600,
            paper_bgcolor='white',
            font=dict(size=10, color='#1f2937')
        )
        
        return fig


# ═══════════════════════════════════════════════════════════════════
# PREDICTION UTILITIES
# ═══════════════════════════════════════════════════════════════════

class PredictionHelper:
    """Helper functions for making predictions"""
    
    @staticmethod
    def create_input_dataframe(input_dict):
        """Convert input dictionary to DataFrame"""
        return pd.DataFrame([input_dict])
    
    @staticmethod
    def validate_input(input_df, feature_columns):
        """Validate input data"""
        if input_df is None or input_df.empty:
            return False, "Input data is empty"
        
        if len(input_df) == 0:
            return False, "No records in input"
        
        # Check for NaN values
        if input_df.isnull().any().any():
            return False, "Input contains NaN values"
        
        return True, "Input validation passed"
    
    @staticmethod
    def format_prediction(prediction, decimals=2):
        """Format prediction output"""
        return f"${prediction:,.{decimals}f}"
    
    @staticmethod
    def get_prediction_context(prediction, reference_data, target_column='SalePrice'):
        """Get contextual information about prediction"""
        if reference_data is None:
            return {}
        
        target_stats = reference_data[target_column].describe()
        
        return {
            'vs_mean': ((prediction - target_stats['mean']) / target_stats['mean']) * 100,
            'vs_median': ((prediction - target_stats['50%']) / target_stats['50%']) * 100,
            'percentile': (reference_data[target_column] <= prediction).sum() / len(reference_data) * 100,
            'is_high': prediction > target_stats['75%'],
            'is_low': prediction < target_stats['25%']
        }


# ═══════════════════════════════════════════════════════════════════
# EXPORT UTILITIES
# ═══════════════════════════════════════════════════════════════════

class ExportManager:
    """Manages data export functionality"""
    
    @staticmethod
    def dataframe_to_csv(df, filename):
        """Convert DataFrame to CSV bytes"""
        return df.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def create_prediction_report(input_data, prediction, context=None):
        """Create a report DataFrame from prediction"""
        report = pd.DataFrame({
            'Feature': list(input_data.keys()),
            'Value': list(input_data.values())
        })
        
        return report
