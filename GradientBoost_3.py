import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def create_weather_interactions(df):
    """Create interaction features between weather parameters"""
    # Temperature interactions
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['temp_pressure'] = df['temperature'] * df['pressure']
    df['temp_clouds'] = df['temperature'] * df['clouds']
    
    # Solar radiation interactions
    df['solar_temp'] = df['solar_radiation'] * df['temperature']
    df['solar_clouds'] = df['solar_radiation'] * df['clouds']
    df['solar_humidity'] = df['solar_radiation'] * df['humidity']
    
    # Precipitation interactions
    df['precip_humidity'] = df['precipitation'] * df['humidity']
    df['precip_pressure'] = df['precipitation'] * df['pressure']
    
    return df

def add_time_based_features(df):
    """Add time-based features using the index"""
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # One-hot encode day of week
    for day in range(7):
        df[f'day_{day}'] = (df['day_of_week'] == day).astype(int)
    
    # Add season indicators
    df['is_summer'] = df['month'].isin([5, 6, 7, 8]).astype(int)
    df['is_winter'] = df['month'].isin([11, 12, 1, 2]).astype(int)
    
    return df

def create_rolling_features(df, windows=[24, 168, 720]):  # 1 day, 1 week, 1 month
    """Create rolling statistics for weather and usage"""
    for window in windows:
        # Rolling features for usage
        df[f'usage_rolling_mean_{window}h'] = df['Usage'].rolling(window=window, min_periods=1).mean()
        df[f'usage_rolling_std_{window}h'] = df['Usage'].rolling(window=window, min_periods=1).std()
        
        # Rolling features for weather parameters
        for col in ['temperature', 'humidity', 'solar_radiation']:
            df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std()
    
    return df

# Load data
print("Loading data...")
train_df = pd.read_csv(r'E:\Programs\Adani_Thinkbiz\ML_Models\Datasets\Merged_Datasets\train_preprocessed.csv', 
                       parse_dates=['Dates'], index_col='Dates')

# Split into train and test
test_size = int(len(train_df) * 0.2)
test_df = train_df[-test_size:]
train_df = train_df[:-test_size]

# Feature engineering
print("Creating features...")
for df in [train_df, test_df]:
    df = create_weather_interactions(df)
    df = add_time_based_features(df)
    df = create_rolling_features(df)

# Handle missing values and outliers
def clean_dataset(df):
    df = df.copy()
    # Handle outliers using IQR method
    for col in ['Usage', 'temperature', 'humidity', 'solar_radiation']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    
    # Fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

train_df = clean_dataset(train_df)
test_df = clean_dataset(test_df)

# Prepare features and target
feature_cols = [col for col in train_df.columns if col != 'Usage']
X_train = train_df[feature_cols]
y_train = train_df['Usage']
X_test = test_df[feature_cols]
y_test = test_df['Usage']

# Scale features
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Train model
print("Training model...")
model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f'Test RMSE: {rmse:.2f}')
print(f'Test MAE: {mae:.2f}')
print(f'R² Score: {r2:.4f}')

# Plot feature importance
plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.show()

# Save the model
print("Saving model...")
joblib.dump({
    'model': model,
    'scaler': scaler,
    'feature_names': feature_cols
}, 'energy_prediction_model.pkl')

print("Process completed successfully!")