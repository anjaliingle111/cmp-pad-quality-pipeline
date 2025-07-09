import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

class CMPDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = SelectKBest(f_classif, k=10)
        
    def process_cmp_features(self, df):
        """Process CMP manufacturing features"""
        # Encode categorical variables
        df_processed = df.copy()
        df_processed['material_type_encoded'] = self.label_encoder.fit_transform(df['material_type'])
        
        # Feature engineering
        df_processed['pressure_temp_ratio'] = df_processed['pressure'] / df_processed['temperature']
        df_processed['speed_force_product'] = df_processed['rotation_speed'] * df_processed['head_force']
        df_processed['efficiency_score'] = (df_processed['rotation_speed'] * df_processed['polish_time']) / df_processed['slurry_flow_rate']
        
        # Select features for model
        features = [
            'pressure', 'temperature', 'rotation_speed', 'polish_time',
            'slurry_flow_rate', 'pad_conditioning', 'head_force',
            'back_pressure', 'pad_age', 'material_type_encoded',
            'pressure_temp_ratio', 'speed_force_product', 'efficiency_score'
        ]
        
        X = df_processed[features]
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, features
    
    def load_and_process_data(self, file_path):
        """Load and process CMP data from CSV"""
        df = pd.read_csv(file_path)
        X, features = self.process_cmp_features(df)
        y = df['is_faulty'].values
        
        return X, y, features, df