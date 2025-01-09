import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

def get_user_date_input():
    """Get date input from user with validation"""
    while True:
        try:
            date_str = input("Enter date (YYYY-MM-DD): ")
            return pd.to_datetime(date_str)
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD format.")

def validate_weather_data(weather_data, required_features):
    """Validate that all required weather features are present"""
    missing_features = [f for f in required_features if f not in weather_data.columns]
    if missing_features:
        raise ValueError(f"Missing required weather features: {missing_features}")

def predict_usage_for_date(target_date, historical_data, model_path='energy_prediction_model.pkl'):
    """
    Predict energy usage for a specific date using the trained model.
    """
    required_features = ['temperature', 'humidity', 'precipitation', 
                        'pressure', 'clouds', 'solar_radiation']
    
    # Validate required features
    missing_features = [f for f in required_features if f not in historical_data.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Load model and preprocessing objects
    saved_objects = joblib.load(model_path)
    model = saved_objects['model']
    scaler = saved_objects['scaler']
    feature_names = saved_objects['feature_names']
    
    # Create prediction dataframe using the last 30 days of historical data for rolling features
    end_date = target_date - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=30)
    target_df = historical_data[start_date:end_date].copy()
    target_df.loc[target_date] = historical_data.loc[target_date:target_date].iloc[0]
    
    # Add time features
    target_df = add_time_based_features(target_df)
    
    # Create weather interactions
    target_df = create_weather_interactions(target_df)
    
    # Create rolling features
    target_df = create_rolling_features(target_df)
    
    # Select only the target date
    target_df = target_df.loc[[target_date]]
    
    # Ensure all model features are present
    for feature in feature_names:
        if feature not in target_df.columns:
            target_df[feature] = 0
    
    # Select and order features
    target_df = target_df[feature_names]
    
    # Scale features
    target_df_scaled = pd.DataFrame(
        scaler.transform(target_df),
        columns=target_df.columns,
        index=target_df.index
    )
    
    # Make prediction
    prediction = model.predict(target_df_scaled)[0]
    
    return prediction

def main():
    """Main function to handle user interaction"""
    # Load historical data
    try:
        historical_data = pd.read_csv(r'E:\Programs\Adani_Thinkbiz\ML_Models\Datasets\Merged_Datasets\train_preprocessed.csv', 
                                    index_col='Dates', parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        return

    while True:
        print("\nEnergy Usage Prediction System")
        print("1. Predict for a specific date")
        print("2. Predict for a date range")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            target_date = get_user_date_input()
            try:
                prediction = predict_usage_for_date(target_date, historical_data)
                print(f"\nPredicted energy usage for {target_date.date()}: {prediction:.2f} units")
            except Exception as e:
                print(f"Error making prediction: {str(e)}")
        
        elif choice == '2':
            start_date = get_user_date_input()
            end_date = get_user_date_input()
            
            if start_date > end_date:
                print("Start date must be before end date!")
                continue
            
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            predictions = []
            
            for date in date_range:
                try:
                    pred = predict_usage_for_date(date, historical_data)
                    predictions.append({'Date': date, 'Predicted_Usage': pred})
                except Exception as e:
                    print(f"Error predicting for {date.date()}: {str(e)}")
                    continue
            
            results_df = pd.DataFrame(predictions).set_index('Date')
            print("\nPredictions:")
            print(results_df)
            
            # Option to save predictions
            save_option = input("\nWould you like to save the predictions to a CSV file? (y/n): ")
            if save_option.lower() == 'y':
                filename = f"predictions_{start_date.date()}_{end_date.date()}.csv"
                results_df.to_csv(filename)
                print(f"Predictions saved to {filename}")
        
        elif choice == '3':
            print("Exiting program...")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()