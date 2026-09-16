"""
Unit Testing Module
Validates data processing logic and ML engine output.
"""
import pytest
from data_processor import DataProcessor
from model_engine import DiagnosticEngine

def test_data_processor_validation():
    processor = DataProcessor()
    invalid_data = {'age': -5, 'glucose': 100, 'blood_pressure': 80, 'bmi': 22.0, 'insulin': 50}
    with pytest.raises(ValueError):
        processor.validate_inputs(invalid_data)

def test_prediction_output():
    processor = DataProcessor()
    engine = DiagnosticEngine()
    
    valid_data = {'age': 40, 'glucose': 120, 'blood_pressure': 70, 'bmi': 25.0, 'insulin': 90}
    processed = processor.preprocess(valid_data)
    prob, level = engine.predict_risk(processed)
    
    assert 0.0 <= prob <= 1.0
    assert level in ["Low Risk", "Moderate Risk", "High Risk"]
