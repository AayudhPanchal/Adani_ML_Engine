import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

def predict_usage_for_date(target_date, historical_data, model_path='energy_prediction_model.pkl'):
    """
    Predict energy usage for a specific date using the trained model.
    
    Parameters:
    target_date: str or datetime - The date to predict usage for (format: 'YYYY-MM-DD')
    historical_data: pd.DataFrame - Historical usage data with datetime index and 'Usage' column
    model_path: str - Path to the saved model file
    
    Returns:
    float - Predicted usage value for the target date
    """
    # Load the saved model and preprocessing objects
    saved_objects = joblib.load(model_path)
    model = saved_objects['model']
    scaler = saved_objects['scaler']
    feature_names = saved_objects['feature_names']
    
    # Convert target_date to datetime if it's a string
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    # Ensure historical_data has 'Usage' column and is sorted
    historical_data = historical_data.copy()
    if 'Usage' not in historical_data.columns:
        raise ValueError("Historical data must contain 'Usage' column")
    historical_data = historical_data.sort_index()
    
    # Apply log transformation to historical data
    historical_data['Usage'] = np.log1p(historical_data['Usage'])
    
    # Create a DataFrame with sufficient historical context
    # Include dates before the target date to calculate all features
    date_range = pd.date_range(end=target_date, periods=31, freq='D')
    target_df = pd.DataFrame(index=date_range)
    target_df = target_df.join(historical_data[['Usage']], how='left')
    
    # Add time-based features
    target_df['hour'] = target_df.index.hour
    target_df['day_of_week'] = target_df.index.dayofweek
    target_df['month'] = target_df.index.month
    target_df['day_of_year'] = target_df.index.dayofyear
    target_df['week_of_year'] = target_df.index.isocalendar().week
    target_df['quarter'] = target_df.index.quarter
    target_df['is_weekend'] = target_df['day_of_week'].isin([5, 6]).astype(int)
    target_df['is_month_start'] = target_df.index.is_month_start.astype(int)
    target_df['is_month_end'] = target_df.index.is_month_end.astype(int)
    
    # Add cyclical features
    for col, max_val in [('hour', 24), ('day_of_week', 7), ('month', 12),
                        ('day_of_year', 365), ('week_of_year', 52)]:
        target_df[f'{col}_sin'] = np.sin(2 * np.pi * target_df[col]/max_val)
        target_df[f'{col}_cos'] = np.cos(2 * np.pi * target_df[col]/max_val)
    
    # Create lag features
    lags = [1, 2, 3, 7, 14, 21, 28]
    windows = [7, 14, 30]
    
    # Add lag features
    for lag in lags:
        target_df[f'Usage_lag_{lag}'] = target_df['Usage'].shift(lag)
        if lag <= 7:
            target_df[f'Usage_diff_{lag}'] = target_df['Usage'].diff(lag)
            target_df[f'Usage_pct_change_{lag}'] = target_df['Usage'].pct_change(lag)
    
    # Add rolling features
    for window in windows:
        target_df[f'Usage_rolling_mean_{window}'] = target_df['Usage'].rolling(window=window).mean()
        target_df[f'Usage_rolling_std_{window}'] = target_df['Usage'].rolling(window=window).std()
        target_df[f'Usage_rolling_min_{window}'] = target_df['Usage'].rolling(window=window).min()
        target_df[f'Usage_rolling_max_{window}'] = target_df['Usage'].rolling(window=window).max()
        target_df[f'Usage_rolling_skew_{window}'] = target_df['Usage'].rolling(window=window).skew()
        target_df[f'Usage_ewm_mean_{window}'] = target_df['Usage'].ewm(span=window).mean()
    
    # Select only the target date
    target_df = target_df.loc[[target_date]]
    
    # Fill any remaining NaN values with mean values from historical data
    for column in target_df.columns:
        if target_df[column].isna().any():
            if column in historical_data.columns:
                target_df[column] = historical_data[column].mean()
            else:
                target_df[column] = 0  # Default to 0 for new features
    
    # Ensure all required features are present and in correct order
    for feature in feature_names:
        if feature not in target_df.columns:
            target_df[feature] = 0
    
    target_df = target_df[feature_names]
    
    # Scale features
    target_df_scaled = pd.DataFrame(
        scaler.transform(target_df),
        columns=target_df.columns,
        index=target_df.index
    )
    
    # Make prediction
    prediction = model.predict(target_df_scaled)
    
    # Transform prediction back to original scale
    final_prediction = np.expm1(prediction[0])
    
    return final_prediction

# Example usage function
def predict_multiple_dates(start_date, end_date, historical_data, model_path='energy_prediction_model.pkl'):
    """
    Predict usage for a range of dates.
    
    Parameters:
    start_date: str - Start date in 'YYYY-MM-DD' format
    end_date: str - End date in 'YYYY-MM-DD' format
    historical_data: pd.DataFrame - Historical usage data
    model_path: str - Path to the saved model file
    
    Returns:
    pd.DataFrame - Predictions for the date range
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    predictions = []
    
    for date in dates:
        try:
            pred = predict_usage_for_date(date, historical_data, model_path)
            predictions.append({'Date': date, 'Predicted_Usage': pred})
        except Exception as e:
            print(f"Error predicting for {date}: {str(e)}")
            continue
    
    return pd.DataFrame(predictions).set_index('Date')

# Load your historical data
# historical_data = pd.read_csv('./train_preprocessed.csv', index_col='Dates', parse_dates=True)

# Predict for a single date
# try:
#     prediction = predict_usage_for_date('2019-12-31', historical_data)
#     print(f"Predicted usage for 2019-12-31: {prediction:.2f}")
# except Exception as e:
#     print(f"Error: {str(e)}")

# # Predict for a range of dates
# predictions_df = predict_multiple_dates('2020-12-01', '2020-12-31', historical_data)
# print("\nPredictions for December 2019:")
# print(predictions_df)