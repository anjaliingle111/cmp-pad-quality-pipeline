import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
import warnings

# Suppress the specific sklearn warning
warnings.filterwarnings('ignore', message='X does not have valid feature names')

class PredictionService:
    def __init__(self):
        try:
            self.model = joblib.load("models/quality_predictor_v1.pkl")
            self.scaler = joblib.load("models/feature_scaler_v1.pkl")
            self.label_encoder = joblib.load("models/label_encoder_v1.pkl")
            
            # EXACT features used in training (12 features)
            self.feature_names = [
                'pressure', 'temperature', 'rotation_speed', 'polish_time', 
                'slurry_flow_rate', 'pad_conditioning', 'head_force', 
                'back_pressure', 'pad_age', 'material_type_encoded',
                'pressure_temp_ratio', 'speed_force_product'
            ]
            
            print(f"Model loaded successfully. Expected features: {len(self.feature_names)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
    def predict_quality(self, sensor_data):
        """Make real-time quality prediction"""
        try:
            # Validate required fields
            required_fields = [
                'pressure', 'temperature', 'rotation_speed', 'polish_time',
                'slurry_flow_rate', 'pad_conditioning', 'head_force',
                'back_pressure', 'pad_age', 'material_type'
            ]
            
            for field in required_fields:
                if field not in sensor_data:
                    return {'error': f'Missing required field: {field}'}
            
            # Encode material type
            try:
                material_encoded = self.label_encoder.transform([sensor_data['material_type']])[0]
            except ValueError as e:
                return {'error': f'Unknown material type: {sensor_data["material_type"]}. Valid types: Cu, W, Al'}
            
            # Feature engineering (same as training)
            pressure_temp_ratio = sensor_data['pressure'] / sensor_data['temperature']
            speed_force_product = sensor_data['rotation_speed'] * sensor_data['head_force']
            
            # Create DataFrame with proper feature names (FIXES THE WARNING)
            feature_data = {
                'pressure': [sensor_data['pressure']], 
                'temperature': [sensor_data['temperature']], 
                'rotation_speed': [sensor_data['rotation_speed']], 
                'polish_time': [sensor_data['polish_time']],
                'slurry_flow_rate': [sensor_data['slurry_flow_rate']], 
                'pad_conditioning': [sensor_data['pad_conditioning']],
                'head_force': [sensor_data['head_force']], 
                'back_pressure': [sensor_data['back_pressure']], 
                'pad_age': [sensor_data['pad_age']], 
                'material_type_encoded': [material_encoded],
                'pressure_temp_ratio': [pressure_temp_ratio], 
                'speed_force_product': [speed_force_product]
            }
            
            # Create DataFrame with proper column names
            features_df = pd.DataFrame(feature_data)
            
            print(f"Features shape: {features_df.shape}")
            print(f"Features: {features_df.iloc[0].values}")
            
            # Scale features (now with proper column names)
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            
            # CORRECT INTERPRETATION:
            # Model training: 0 = good, 1 = faulty
            # prediction_proba[0] = probability of good (class 0)
            # prediction_proba[1] = probability of faulty (class 1)
            
            is_faulty = bool(prediction == 1)  # 1 = faulty, 0 = good
            
            # Quality score based on probability of being good
            quality_score = prediction_proba[0] * 100  # Higher = better quality
            
            result = {
                'pad_id': sensor_data.get('pad_id', 'unknown'),
                'quality_score': round(quality_score, 2),
                'is_faulty': is_faulty,
                'confidence': round(max(prediction_proba) * 100, 2),
                'prediction_time': datetime.now().isoformat(),
                'model_version': 'v1.0',
                'raw_prediction': int(prediction),
                'probabilities': {
                    'good': round(prediction_proba[0] * 100, 2),
                    'faulty': round(prediction_proba[1] * 100, 2)
                }
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def batch_predict(self, sensor_data_batch):
        """Make batch predictions"""
        predictions = []
        for data in sensor_data_batch:
            pred = self.predict_quality(data)
            predictions.append(pred)
        return predictions

# Test the prediction service
if __name__ == "__main__":
    print("=== Testing CMP Quality Prediction Service (Warning-Free) ===")
    
    try:
        predictor = PredictionService()
        
        # Test with sample data
        test_data = {
            'pad_id': 'TEST001',
            'pressure': 4.8,
            'temperature': 24.0,
            'rotation_speed': 95.0,
            'polish_time': 58.0,
            'slurry_flow_rate': 195.0,
            'pad_conditioning': 1,
            'head_force': 9.8,
            'back_pressure': 2.1,
            'pad_age': 25,
            'material_type': 'Cu'
        }
        
        print("Testing with sample data (no warnings)...")
        result = predictor.predict_quality(test_data)
        print('Test Prediction Result:')
        print(json.dumps(result, indent=2))
        
        # Test with faulty data
        print("\nTesting with faulty conditions...")
        faulty_data = {
            'pad_id': 'FAULTY_TEST',
            'pressure': 3.5,  # Low pressure (causes faults)
            'temperature': 33.0,  # High temperature (causes faults)
            'rotation_speed': 70.0,  # Low speed (causes faults)
            'polish_time': 75.0,  # Long time (causes faults)
            'slurry_flow_rate': 170.0,
            'pad_conditioning': 0,
            'head_force': 8.0,
            'back_pressure': 2.5,
            'pad_age': 90,
            'material_type': 'Cu'
        }
        
        result_faulty = predictor.predict_quality(faulty_data)
        print('Faulty Prediction Result:')
        print(json.dumps(result_faulty, indent=2))
        
    except Exception as e:
        print(f"Error during testing: {e}")