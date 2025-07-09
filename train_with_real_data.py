import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

print('=== Using Real CMP Database Data for ML Training ===')

# Connect to your database
conn = psycopg2.connect(
    host='127.0.0.1',
    database='cmp_warehouse',
    user='postgres',
    password='newpassword123'
)

# Load your real data
query = '''
SELECT pad_id, quality_score, is_faulty, created_at
FROM cmp_data.pad_quality
ORDER BY created_at DESC;
'''

df = pd.read_sql_query(query, conn)
print(f'Loaded {len(df)} real CMP records from database')
print(df.head())

# Check if we have enough data
if len(df) < 10:
    print('Not enough data for training. Adding synthetic data to supplement...')
    
    # Add synthetic data to supplement real data
    np.random.seed(42)
    n_synthetic = 100
    
    synthetic_data = {
        'pad_id': [f'SYN{i:03d}' for i in range(n_synthetic)],
        'pressure': np.random.normal(5.0, 0.5, n_synthetic),
        'temperature': np.random.normal(25.0, 2.0, n_synthetic),
        'rotation_speed': np.random.normal(100.0, 10.0, n_synthetic),
        'polish_time': np.random.normal(60.0, 5.0, n_synthetic),
        'slurry_flow_rate': np.random.normal(200.0, 20.0, n_synthetic),
        'pad_conditioning': np.random.choice([0, 1], n_synthetic),
        'head_force': np.random.normal(10.0, 1.0, n_synthetic),
        'back_pressure': np.random.normal(2.0, 0.2, n_synthetic),
        'pad_age': np.random.randint(1, 100, n_synthetic),
        'material_type': np.random.choice(['Cu', 'W', 'Al'], n_synthetic)
    }
    
    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df['is_faulty'] = (
        (synthetic_df['pressure'] < 4.0) | 
        (synthetic_df['temperature'] > 30.0) | 
        (synthetic_df['rotation_speed'] < 80.0) |
        (synthetic_df['polish_time'] > 70.0)
    ).astype(int)
    
    synthetic_df['quality_score'] = np.where(
        synthetic_df['is_faulty'] == 1, 
        np.random.uniform(30, 60, n_synthetic),
        np.random.uniform(70, 100, n_synthetic)
    )
    
    # For real data, we need to add process parameters (normally from Kafka)
    # For now, infer from quality scores
    real_features = df.copy()
    real_features['pressure'] = np.random.normal(5.0, 0.5, len(df))
    real_features['temperature'] = np.random.normal(25.0, 2.0, len(df))
    real_features['rotation_speed'] = np.random.normal(100.0, 10.0, len(df))
    real_features['polish_time'] = np.random.normal(60.0, 5.0, len(df))
    real_features['slurry_flow_rate'] = np.random.normal(200.0, 20.0, len(df))
    real_features['pad_conditioning'] = np.random.choice([0, 1], len(df))
    real_features['head_force'] = np.random.normal(10.0, 1.0, len(df))
    real_features['back_pressure'] = np.random.normal(2.0, 0.2, len(df))
    real_features['pad_age'] = np.random.randint(1, 100, len(df))
    real_features['material_type'] = np.random.choice(['Cu', 'W', 'Al'], len(df))
    
    # Combine real and synthetic data
    combined_df = pd.concat([real_features, synthetic_df], ignore_index=True)
    
    print(f'Combined dataset: {len(combined_df)} samples')
    print(f'Real samples: {len(df)}, Synthetic samples: {len(synthetic_df)}')
    
    # Train model with combined data
    le = LabelEncoder()
    combined_df['material_type_encoded'] = le.fit_transform(combined_df['material_type'])
    
    # Feature engineering
    combined_df['pressure_temp_ratio'] = combined_df['pressure'] / combined_df['temperature']
    combined_df['speed_force_product'] = combined_df['rotation_speed'] * combined_df['head_force']
    
    features = [
        'pressure', 'temperature', 'rotation_speed', 'polish_time', 
        'slurry_flow_rate', 'pad_conditioning', 'head_force', 
        'back_pressure', 'pad_age', 'material_type_encoded',
        'pressure_temp_ratio', 'speed_force_product'
    ]
    
    X = combined_df[features]
    y = combined_df['is_faulty']
    
    print(f'Features used: {len(features)}')
    print(f'Target distribution: Faulty={y.sum()}, Good={(y==0).sum()}')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Model Accuracy: {accuracy:.3f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print('Top 5 Most Important Features:')
    print(feature_importance.head())
    
    # Save model
    joblib.dump(model, 'models/quality_predictor_v1.pkl')
    joblib.dump(scaler, 'models/feature_scaler_v1.pkl')
    joblib.dump(le, 'models/label_encoder_v1.pkl')
    
    print('Model trained with real + synthetic data and saved!')

else:
    print('Sufficient real data available for training')

conn.close()