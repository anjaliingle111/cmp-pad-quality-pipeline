import sys
sys.path.append('src/ml')
from prediction_service import PredictionService
import json

def test_ml_predictions():
    """Test ML predictions with real pad data"""
    print("=== Testing CMP Quality ML Predictions with Real Data ===\n")
    
    try:
        # Initialize prediction service
        predictor = PredictionService()
        
        # Test data based on your real database records
        test_cases = [
            {
                'name': 'PAD001 (Expected: Good Quality - 95.5 score)',
                'data': {
                    'pad_id': 'PAD001',
                    'pressure': 5.1,
                    'temperature': 24.8,
                    'rotation_speed': 102.0,
                    'polish_time': 59.0,
                    'slurry_flow_rate': 198.0,
                    'pad_conditioning': 1,
                    'head_force': 10.1,
                    'back_pressure': 2.0,
                    'pad_age': 15,
                    'material_type': 'Cu'
                },
                'expected_quality': 95.5,
                'expected_faulty': False
            },
            {
                'name': 'PAD002 (Expected: Good Quality - 87.2 score)',
                'data': {
                    'pad_id': 'PAD002',
                    'pressure': 4.9,
                    'temperature': 25.2,
                    'rotation_speed': 98.5,
                    'polish_time': 60.5,
                    'slurry_flow_rate': 195.0,
                    'pad_conditioning': 1,
                    'head_force': 9.8,
                    'back_pressure': 2.1,
                    'pad_age': 22,
                    'material_type': 'Al'
                },
                'expected_quality': 87.2,
                'expected_faulty': False
            },
            {
                'name': 'PAD003 (Expected: Faulty - 45.8 score)',
                'data': {
                    'pad_id': 'PAD003',
                    'pressure': 3.7,  # Low pressure (fault condition)
                    'temperature': 31.5,  # High temperature (fault condition)
                    'rotation_speed': 78.0,  # Low speed (fault condition)
                    'polish_time': 71.0,  # Long time (fault condition)
                    'slurry_flow_rate': 175.0,
                    'pad_conditioning': 0,
                    'head_force': 8.2,
                    'back_pressure': 2.4,
                    'pad_age': 89,
                    'material_type': 'Cu'
                },
                'expected_quality': 45.8,
                'expected_faulty': True
            },
            {
                'name': 'PAD004 (Expected: Good Quality - 92.1 score)',
                'data': {
                    'pad_id': 'PAD004',
                    'pressure': 5.0,
                    'temperature': 24.5,
                    'rotation_speed': 101.0,
                    'polish_time': 58.5,
                    'slurry_flow_rate': 200.0,
                    'pad_conditioning': 1,
                    'head_force': 10.0,
                    'back_pressure': 2.0,
                    'pad_age': 18,
                    'material_type': 'W'
                },
                'expected_quality': 92.1,
                'expected_faulty': False
            },
            {
                'name': 'PAD005 (Expected: Good Quality - 78.3 score)',
                'data': {
                    'pad_id': 'PAD005',
                    'pressure': 4.8,
                    'temperature': 25.8,
                    'rotation_speed': 95.0,
                    'polish_time': 61.0,
                    'slurry_flow_rate': 190.0,
                    'pad_conditioning': 1,
                    'head_force': 9.5,
                    'back_pressure': 2.2,
                    'pad_age': 35,
                    'material_type': 'Al'
                },
                'expected_quality': 78.3,
                'expected_faulty': False
            }
        ]
        
        # Test each pad
        results = []
        for test_case in test_cases:
            print(f"Testing: {test_case['name']}")
            print("-" * 60)
            
            # Make prediction
            result = predictor.predict_quality(test_case['data'])
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}\n")
                continue
            
            # Display results
            print(f"Pad ID: {result['pad_id']}")
            print(f"ML Quality Score: {result['quality_score']}%")
            print(f"Expected Quality: {test_case['expected_quality']}%")
            print(f"ML Prediction: {'Faulty' if result['is_faulty'] else 'Good'}")
            print(f"Expected: {'Faulty' if test_case['expected_faulty'] else 'Good'}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Probabilities: Good={result['probabilities']['good']}%, Faulty={result['probabilities']['faulty']}%")
            
            # Check if prediction matches expectation
            prediction_correct = result['is_faulty'] == test_case['expected_faulty']
            print(f"Prediction Accuracy: {'✅ Correct' if prediction_correct else '❌ Incorrect'}")
            
            results.append({
                'pad_id': result['pad_id'],
                'ml_quality_score': result['quality_score'],
                'expected_quality': test_case['expected_quality'],
                'ml_prediction': result['is_faulty'],
                'expected_prediction': test_case['expected_faulty'],
                'correct': prediction_correct,
                'confidence': result['confidence']
            })
            
            print("\n")
        
        # Summary
        print("="*60)
        print("SUMMARY")
        print("="*60)
        
        correct_predictions = sum(1 for r in results if r['correct'])
        total_predictions = len(results)
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        print(f"Total Tests: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        print("\nDetailed Results:")
        for result in results:
            status = "✅" if result['correct'] else "❌"
            print(f"{status} {result['pad_id']}: ML={result['ml_quality_score']}%, Expected={result['expected_quality']}%, Confidence={result['confidence']}%")
        
        # Test edge cases
        print("\n" + "="*60)
        print("EDGE CASE TESTING")
        print("="*60)
        
        edge_cases = [
            {
                'name': 'Perfect Conditions',
                'data': {
                    'pad_id': 'PERFECT',
                    'pressure': 5.0,
                    'temperature': 25.0,
                    'rotation_speed': 100.0,
                    'polish_time': 60.0,
                    'slurry_flow_rate': 200.0,
                    'pad_conditioning': 1,
                    'head_force': 10.0,
                    'back_pressure': 2.0,
                    'pad_age': 20,
                    'material_type': 'Cu'
                }
            },
            {
                'name': 'Extreme Fault Conditions',
                'data': {
                    'pad_id': 'EXTREME_FAULT',
                    'pressure': 3.0,  # Very low
                    'temperature': 35.0,  # Very high
                    'rotation_speed': 60.0,  # Very low
                    'polish_time': 80.0,  # Very long
                    'slurry_flow_rate': 150.0,
                    'pad_conditioning': 0,
                    'head_force': 7.0,
                    'back_pressure': 3.0,
                    'pad_age': 95,
                    'material_type': 'W'
                }
            }
        ]
        
        for edge_case in edge_cases:
            print(f"\n{edge_case['name']}:")
            result = predictor.predict_quality(edge_case['data'])
            if 'error' not in result:
                print(f"  Quality Score: {result['quality_score']}%")
                print(f"  Prediction: {'Faulty' if result['is_faulty'] else 'Good'}")
                print(f"  Confidence: {result['confidence']}%")
            else:
                print(f"  Error: {result['error']}")
    
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_batch_predictions():
    """Test batch predictions"""
    print("\n" + "="*60)
    print("BATCH PREDICTION TESTING")
    print("="*60)
    
    try:
        predictor = PredictionService()
        
        # Batch data
        batch_data = [
            {
                'pad_id': 'BATCH_001',
                'pressure': 5.2,
                'temperature': 24.0,
                'rotation_speed': 105.0,
                'polish_time': 57.0,
                'slurry_flow_rate': 205.0,
                'pad_conditioning': 1,
                'head_force': 10.5,
                'back_pressure': 1.9,
                'pad_age': 12,
                'material_type': 'Cu'
            },
            {
                'pad_id': 'BATCH_002',
                'pressure': 3.5,
                'temperature': 32.0,
                'rotation_speed': 70.0,
                'polish_time': 75.0,
                'slurry_flow_rate': 170.0,
                'pad_conditioning': 0,
                'head_force': 8.0,
                'back_pressure': 2.5,
                'pad_age': 90,
                'material_type': 'Al'
            }
        ]
        
        # Test batch prediction
        batch_results = predictor.batch_predict(batch_data)
        
        print("Batch Prediction Results:")
        for result in batch_results:
            if 'error' not in result:
                print(f"  {result['pad_id']}: {result['quality_score']}% ({'Faulty' if result['is_faulty'] else 'Good'})")
            else:
                print(f"  Error: {result['error']}")
    
    except Exception as e:
        print(f"Error during batch testing: {e}")

def test_error_handling():
    """Test error handling"""
    print("\n" + "="*60)
    print("ERROR HANDLING TESTING")
    print("="*60)
    
    try:
        predictor = PredictionService()
        
        # Test missing fields
        incomplete_data = {
            'pad_id': 'INCOMPLETE',
            'pressure': 5.0,
            'temperature': 25.0
            # Missing other required fields
        }
        
        result = predictor.predict_quality(incomplete_data)
        print(f"Missing fields test: {result.get('error', 'No error')}")
        
        # Test invalid material type
        invalid_material = {
            'pad_id': 'INVALID',
            'pressure': 5.0,
            'temperature': 25.0,
            'rotation_speed': 100.0,
            'polish_time': 60.0,
            'slurry_flow_rate': 200.0,
            'pad_conditioning': 1,
            'head_force': 10.0,
            'back_pressure': 2.0,
            'pad_age': 20,
            'material_type': 'InvalidMaterial'
        }
        
        result = predictor.predict_quality(invalid_material)
        print(f"Invalid material test: {result.get('error', 'No error')}")
        
    except Exception as e:
        print(f"Error during error handling testing: {e}")

if __name__ == "__main__":
    test_ml_predictions()
    test_batch_predictions()
    test_error_handling()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)