import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
import json

class PowerPredictor:
    def __init__(self):
        self.model = load_model('lstm_power_model.h5')
        self.scaler_X = joblib.load('scaler_X.joblib')
        self.scaler_y = joblib.load('scaler_y.joblib')
        self.df = pd.read_csv('./merged_data_10.csv')
        self.df['Month_Year'] = pd.to_datetime(self.df['Month_Year'])
        
    def create_sequences(self, X, seq_length=10):
        X_seq = []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
        return np.array(X_seq)
    
    def predict_generation(self, data):
        X_scaled = self.scaler_X.transform(data)
        X_seq = self.create_sequences(X_scaled)
        predictions = self.model.predict(X_seq)
        return self.scaler_y.inverse_transform(predictions)
    
    def infer(self, start_date, end_date):
        try:
            # Convert and validate dates
            start = pd.to_datetime(start_date).replace(year=2019)
            end = pd.to_datetime(end_date).replace(year=2019)
            
            if (end - start).days < 10:
                return {"error": "Date range must be at least 10 days"}
            
            # Filter data
            df_filtered = self.df[
                (self.df['Month_Year'] >= start) & 
                (self.df['Month_Year'] <= end)
            ].copy()
            
            if df_filtered.empty:
                return {"error": "No data available for specified date range"}
            
            # Extract features
            time_features = {
                'year': df_filtered['Month_Year'].dt.year,
                'month': df_filtered['Month_Year'].dt.month,
                'day': df_filtered['Month_Year'].dt.day,
                'day_of_week': df_filtered['Month_Year'].dt.dayofweek,
                'day_of_year': df_filtered['Month_Year'].dt.dayofyear,
                'week_of_year': df_filtered['Month_Year'].dt.isocalendar().week,
                'quarter': df_filtered['Month_Year'].dt.quarter,
                'is_weekend': df_filtered['Month_Year'].dt.dayofweek >= 5,
                'is_month_start': df_filtered['Month_Year'].dt.is_month_start,
                'is_month_end': df_filtered['Month_Year'].dt.is_month_end
            }
            
            for feature, value in time_features.items():
                df_filtered[feature] = value
            
            # Prepare features
            features = [
                'temperature', 'humidity', 'precipitation', 'wind_speed', 
                'solar_radiation', 'max_temp', 'min_temp', 'dew_point', 
                'pressure', 'clouds', 'year', 'month', 'day', 'day_of_week',
                'day_of_year', 'week_of_year', 'quarter', 'is_weekend',
                'is_month_start', 'is_month_end'
            ]
            
            X = pd.get_dummies(df_filtered[features + ['Type']], columns=['Type'])
            
            # Generate predictions
            predictions = self.predict_generation(X)
            
            # Process results
            df_filtered['Predicted_Generation'] = np.nan
            df_filtered.loc[df_filtered.index[10:], 'Predicted_Generation'] = predictions.flatten()
            
            # Group predictions
            results = df_filtered.groupby('Type')['Predicted_Generation'].sum().to_dict()
            
            return {
                "success": True,
                "predictions": {
                    k: float(v) for k, v in results.items()  # Convert numpy types to native Python
                },
                "date_range": {
                    "start": start_date,
                    "end": end_date
                }
            }
            
        except Exception as e:
            return {"error": str(e)}