"""
Module 2: Diagnostic Machine Learning Engine
Handles model evaluation, mock inference, and decision thresholding.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class DiagnosticEngine:
    def __init__(self):
        # Initializing synthetic model for demo evaluation
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(20, 5)
        y_dummy = np.random.randint(0, 2, size=20)
        self.model.fit(X_dummy, y_dummy)

    def predict_risk(self, preprocessed_features):
        """Returns risk probability and risk level label."""
        prob = self.model.predict_proba(preprocessed_features)[0][1]
        
        if prob > 0.7:
            risk_level = "High Risk"
        elif prob > 0.4:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"
            
        return prob, risk_level

    def get_feature_importance(self):
        """Extracts feature importance scores from trained model."""
        return self.model.feature_importances_
