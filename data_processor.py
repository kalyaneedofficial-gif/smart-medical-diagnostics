"""
Module 1: User Data Ingestion & Preprocessing
Handles clinical input validation, scaling, and transformation.
"""
import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self):
        self.feature_names = ['age', 'glucose', 'blood_pressure', 'bmi', 'insulin']

    def validate_inputs(self, input_dict):
        """Validates patient input ranges."""
        for feature, val in input_dict.items():
            if val < 0:
                raise ValueError(f"Invalid negative value for {feature}: {val}")
        return True

    def preprocess(self, input_dict):
        """Transforms input dictionary into scaled numpy array for ML models."""
        self.validate_inputs(input_dict)
        data = [input_dict[feat] for feat in self.feature_names]
        # Direct standardization scaling (mock standard scale: mean ~50, std ~15)
        scaled_data = (np.array(data) - 50.0) / 15.0
        return scaled_data.reshape(1, -1)
