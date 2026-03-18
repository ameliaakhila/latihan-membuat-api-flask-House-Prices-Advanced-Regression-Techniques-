"""
Advanced Utilities for House Price Prediction Dashboard
Includes export, validation, and helper functions
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# INPUT VALIDATION & SANITIZATION
# ═══════════════════════════════════════════════════════════════════

class InputValidator:
    """Validates and sanitizes user inputs"""
    
    # Define valid ranges for key features
    FEATURE_RANGES = {
        'OverallQual': (1, 10),
        'YearBuilt': (1800, 2024),
        'LotArea': (500, 500000),
        'GrLivArea': (300, 10000),
        'GarageArea': (0, 5000),
        'TotalBsmtSF': (0, 10000),
        '1stFlrSF': (0, 10000),
        '2ndFlrSF': (0, 5000),
        'FullBath': (0, 5),
        'HalfBath': (0, 3),
        'BedroomAbvGr': (0, 10),
        'TotRmsAbvGrd': (1, 15)
    }
    
    @classmethod
    def validate_input_value(cls, feature: str, value: float) -> Tuple[bool, str]:
        """
        Validate if a feature value is within acceptable range
        
        Args:
            feature: Feature name
            value: Feature value
            
        Returns:
            Tuple of (is_valid, message)
        """
        if feature not in cls.FEATURE_RANGES:
            return True, "Feature not in validation list"
        
        min_val, max_val = cls.FEATURE_RANGES[feature]
        
        if value < min_val:
            return False, f"{feature} below minimum ({min_val})"
        elif value > max_val:
            return False, f"{feature} exceeds maximum ({max_val})"
        
        return True, "Valid"
    
    @classmethod
    def validate_all_inputs(cls, input_dict: Dict) -> Tuple[bool, List[str]]:
        """
        Validate all input values
        
        Returns:
            Tuple of (all_valid, error_messages)
        """
        errors = []
        
        for feature, value in input_dict.items():
            is_valid, message = cls.validate_input_value(feature, value)
            if not is_valid:
                errors.append(message)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def sanitize_input(value: any) -> float:
        """Convert and sanitize input value to float"""
        try:
            return float(value)
        except:
            return 0.0


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING & TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════

class FeatureTransformer:
    """Handles feature transformations and engineering"""
    
    @staticmethod
    def log_transform(value: float) -> float:
        """Apply log transformation to values"""
        return np.log1p(value)  # log1p handles 0 values
    
    @staticmethod
    def inverse_log_transform(value: float) -> float:
        """Reverse log transformation"""
        return np.expm1(value)
    
    @staticmethod
    def normalize(value: float, min_val: float, max_val: float) -> float:
        """Min-Max normalization"""
        return (value - min_val) / (max_val - min_val + 1e-8)
    
    @staticmethod
    def denormalize(value: float, min_val: float, max_val: float) -> float:
        """Reverse Min-Max normalization"""
        return value * (max_val - min_val) + min_val
    
    @staticmethod
    def create_derived_features(input_dict: Dict) -> Dict:
        """
        Create derived features from base features
        
        Example:
            - Price per sq ft = GrLivArea / SalePrice
            - Age = current_year - YearBuilt
        """
        derived = input_dict.copy()
        current_year = 2024
        
        # Age of property
        if 'YearBuilt' in input_dict:
            derived['Age'] = current_year - input_dict['YearBuilt']
        
        # Total area (approximate)
        if 'GrLivArea' in input_dict and 'TotalBsmtSF' in input_dict:
            derived['TotalArea'] = input_dict['GrLivArea'] + input_dict['TotalBsmtSF']
        
        # Bathroom count
        if 'FullBath' in input_dict and 'HalfBath' in input_dict:
            derived['TotalBath'] = input_dict['FullBath'] + 0.5 * input_dict['HalfBath']
        
        return derived


# ═══════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════

class StatisticalAnalyzer:
    """Statistical analysis and comparison functions"""
    
    @staticmethod
    def calculate_z_score(value: float, mean: float, std: float) -> float:
        """
        Calculate Z-score for a value
        
        Z = (value - mean) / std
        Indicates how many standard deviations from mean
        """
        if std == 0:
            return 0
        return (value - mean) / std
    
    @staticmethod
    def get_percentile_rank(value: float, data: pd.Series) -> float:
        """
        Calculate percentile rank of a value
        
        Returns:
            Percentile (0-100)
        """
        return (data <= value).sum() / len(data) * 100
    
    @staticmethod
    def calculate_confidence_interval(predictions: np.array, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for predictions
        
        Args:
            predictions: Array of predictions
            confidence: Confidence level (default 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        from scipy import stats
        
        mean = np.mean(predictions)
        std = np.std(predictions)
        margin = std * stats.t.ppf((1 + confidence) / 2, len(predictions) - 1)
        
        return mean - margin, mean + margin
    
    @staticmethod
    def detect_anomalies(data: pd.Series, threshold: float = 3.0) -> np.array:
        """
        Detect anomalies using Z-score method
        
        Args:
            data: Data series
            threshold: Z-score threshold (default 3.0)
            
        Returns:
            Boolean array of anomalies
        """
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / (std + 1e-8))
        return z_scores > threshold


# ═══════════════════════════════════════════════════════════════════
# PREDICTION AUGMENTATION
# ═══════════════════════════════════════════════════════════════════

class PredictionAugmenter:
    """Enhances predictions with context and explanations"""
    
    @staticmethod
    def classify_price_level(prediction: float, reference_data: pd.Series) -> str:
        """
        Classify price as budget/standard/premium
        
        Budget: < 25th percentile
        Standard: 25-75th percentile
        Premium: > 75th percentile
        """
        q1 = reference_data.quantile(0.25)
        q3 = reference_data.quantile(0.75)
        
        if prediction < q1:
            return "Budget"
        elif prediction > q3:
            return "Premium"
        else:
            return "Standard"
    
    @staticmethod
    def get_price_description(prediction: float, reference_data: pd.Series) -> str:
        """Get descriptive text about prediction"""
        mean = reference_data.mean()
        
        if prediction > mean * 1.5:
            return "Significantly above market average"
        elif prediction > mean * 1.2:
            return "Above market average"
        elif prediction < mean * 0.8:
            return "Below market average"
        elif prediction < mean * 0.5:
            return "Significantly below market average"
        else:
            return "At market average"
    
    @staticmethod
    def calculate_market_position(prediction: float, reference_data: pd.Series) -> Dict:
        """Calculate detailed market position"""
        mean = reference_data.mean()
        median = reference_data.median()
        std = reference_data.std()
        
        return {
            'vs_mean_percent': ((prediction - mean) / mean * 100),
            'vs_median_percent': ((prediction - median) / median * 100),
            'std_deviations': (prediction - mean) / std,
            'percentile': (reference_data <= prediction).sum() / len(reference_data) * 100
        }


# ═══════════════════════════════════════════════════════════════════
# COMPARISON & SIMILAR PROPERTIES
# ═══════════════════════════════════════════════════════════════════

class PropertyComparator:
    """Finds and compares similar properties"""
    
    @staticmethod
    def find_similar_properties(input_features: Dict, reference_data: pd.DataFrame, 
                               n_similar: int = 5) -> pd.DataFrame:
        """
        Find similar properties using Euclidean distance
        
        Args:
            input_features: Input property features
            reference_data: Reference dataset
            n_similar: Number of similar properties to find
            
        Returns:
            DataFrame of similar properties
        """
        # Select numeric columns
        numeric_cols = reference_data.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col in input_features.keys()]
        
        # Calculate distances
        distances = np.zeros(len(reference_data))
        
        for col in numeric_cols:
            if col in input_features:
                distances += (reference_data[col] - input_features[col]) ** 2
        
        distances = np.sqrt(distances)
        
        # Get top n similar
        similar_indices = np.argsort(distances)[:n_similar]
        return reference_data.iloc[similar_indices]
    
    @staticmethod
    def compare_property_metrics(input_features: Dict, reference_data: pd.DataFrame) -> Dict:
        """
        Compare input property with dataset statistics
        
        Returns metrics comparison
        """
        comparison = {}
        
        for feature, value in input_features.items():
            if feature in reference_data.columns:
                col_data = reference_data[feature]
                
                if pd.api.types.is_numeric_dtype(col_data):
                    comparison[feature] = {
                        'input_value': value,
                        'dataset_mean': col_data.mean(),
                        'dataset_median': col_data.median(),
                        'dataset_std': col_data.std(),
                        'percentile': (col_data <= value).sum() / len(col_data) * 100
                    }
        
        return comparison


# ═══════════════════════════════════════════════════════════════════
# DATA EXPORT & REPORTING
# ═══════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Generates prediction reports and exports"""
    
    @staticmethod
    def create_prediction_report(input_data: Dict, prediction: float, context: Dict) -> pd.DataFrame:
        """Create detailed prediction report"""
        report_data = {
            'Metric': [],
            'Value': []
        }
        
        # Add input features
        for feature, value in input_data.items():
            report_data['Metric'].append(f"Input: {feature}")
            report_data['Value'].append(str(value))
        
        # Add prediction
        report_data['Metric'].append("Predicted Price")
        report_data['Value'].append(f"${prediction:,.2f}")
        
        # Add context
        if context:
            for key, value in context.items():
                report_data['Metric'].append(f"Context: {key}")
                if isinstance(value, float):
                    report_data['Value'].append(f"{value:.2f}")
                else:
                    report_data['Value'].append(str(value))
        
        return pd.DataFrame(report_data)
    
    @staticmethod
    def create_batch_prediction_report(predictions_df: pd.DataFrame) -> str:
        """Create summary report for batch predictions"""
        summary = f"""
        BATCH PREDICTION REPORT
        {'='*50}
        
        Total Predictions: {len(predictions_df)}
        Average Predicted Price: ${predictions_df['Prediction'].mean():,.2f}
        Median Predicted Price: ${predictions_df['Prediction'].median():,.2f}
        Min Predicted Price: ${predictions_df['Prediction'].min():,.2f}
        Max Predicted Price: ${predictions_df['Prediction'].max():,.2f}
        Std Dev: ${predictions_df['Prediction'].std():,.2f}
        
        {'='*50}
        """
        return summary
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, filename: str) -> bytes:
        """Export DataFrame to CSV bytes"""
        return df.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def export_to_json(data: Dict, filename: str) -> str:
        """Export data to JSON string"""
        import json
        return json.dumps(data, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════
# PERFORMANCE MONITORING
# ═══════════════════════════════════════════════════════════════════

class PerformanceMonitor:
    """Monitor model and application performance"""
    
    def __init__(self):
        self.predictions = []
        self.inference_times = []
        self.errors = []
    
    def log_prediction(self, input_features: Dict, prediction: float, inference_time: float):
        """Log a prediction"""
        self.predictions.append({
            'features': input_features,
            'prediction': prediction,
            'timestamp': pd.Timestamp.now(),
            'inference_time': inference_time
        })
    
    def log_error(self, error: str, error_type: str):
        """Log an error"""
        self.errors.append({
            'error': error,
            'type': error_type,
            'timestamp': pd.Timestamp.now()
        })
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        if not self.predictions:
            return {}
        
        inference_times = [p['inference_time'] for p in self.predictions]
        
        return {
            'total_predictions': len(self.predictions),
            'avg_inference_time': np.mean(inference_times),
            'max_inference_time': np.max(inference_times),
            'min_inference_time': np.min(inference_times),
            'total_errors': len(self.errors)
        }
    
    def get_predictions_dataframe(self) -> pd.DataFrame:
        """Get all predictions as DataFrame"""
        if not self.predictions:
            return pd.DataFrame()
        
        data = []
        for pred in self.predictions:
            pred_dict = pred['features'].copy()
            pred_dict['prediction'] = pred['prediction']
            pred_dict['timestamp'] = pred['timestamp']
            pred_dict['inference_time'] = pred['inference_time']
            data.append(pred_dict)
        
        return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════

def example_usage():
    """Example of how to use the utilities"""
    
    # Validate input
    test_input = {
        'OverallQual': 8,
        'YearBuilt': 2005,
        'LotArea': 15000,
        'GrLivArea': 2500
    }
    
    is_valid, errors = InputValidator.validate_all_inputs(test_input)
    print(f"Input valid: {is_valid}")
    
    # Create derived features
    transformer = FeatureTransformer()
    derived = transformer.create_derived_features(test_input)
    print(f"Derived features: {derived}")
    
    # Example: Use other classes similarly
    # analyzer = StatisticalAnalyzer()
    # monitor = PerformanceMonitor()

if __name__ == "__main__":
    example_usage()
