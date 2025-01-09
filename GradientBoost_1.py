import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import joblib

def create_advanced_lag_features(df, lags, target_col):
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        if lag <= 7:
            df[f'{target_col}_diff_{lag}'] = df[target_col].diff(lag)
            df[f'{target_col}_pct_change_{lag}'] = df[target_col].pct_change(lag)
    return df

def create_advanced_rolling_features(df, windows, target_col):
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        df[f'{target_col}_rolling_skew_{window}'] = df[target_col].rolling(window=window).skew()
        df[f'{target_col}_ewm_mean_{window}'] = df[target_col].ewm(span=window).mean()
    return df

def add_advanced_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    df['quarter'] = df.index.quarter
    
    # Binary flags
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    
    # Cyclical features
    for col, max_val in [('hour', 24), ('day_of_week', 7), ('month', 12), 
                        ('day_of_year', 365), ('week_of_year', 52)]:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df

# Load data
print("Loading data...")
train_df = pd.read_csv(r'E:\Programs\Adani_Thinkbiz\ML_Models\Datasets\Merged_Datasets\train_preprocessed.csv', 
                       index_col='Dates', parse_dates=True)
test_df = pd.read_csv(r'E:\Programs\Adani_Thinkbiz\ML_Models\Datasets\Merged_Datasets\test_preprocessed.csv', 
                      index_col='Dates', parse_dates=True)

# Handle outliers
def handle_outliers(df, col, n_sigmas=3):
    mean = df[col].mean()
    std = df[col].std()
    return df[col].clip(mean - n_sigmas * std, mean + n_sigmas * std)

print("Preprocessing data...")
train_df['Usage'] = handle_outliers(train_df, 'Usage')
test_df['Usage'] = handle_outliers(test_df, 'Usage')

# Log transform
train_df['Usage'] = np.log1p(train_df['Usage'])
test_df['Usage'] = np.log1p(test_df['Usage'])

# Create features
print("Creating features...")
lags = [1, 2, 3, 7, 14, 21, 28]  # Reduced lag features
windows = [7, 14, 30]  # Reduced window sizes

# Apply feature engineering
for df in [train_df, test_df]:
    df = create_advanced_lag_features(df, lags, 'Usage')
    df = create_advanced_rolling_features(df, windows, 'Usage')
    df = add_advanced_time_features(df)

# Drop rows with NaN values
print("Handling missing values...")
train_df = train_df.dropna()
test_df = test_df.dropna()

# Verify data is not empty
if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError("After preprocessing, one or both datasets are empty!")

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Separate features and target
X_train = train_df.drop(columns=['Usage'])
y_train = train_df['Usage']
X_test = test_df.drop(columns=['Usage'])
y_test = test_df['Usage']

# Scale features
print("Scaling features...")
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

# Define and train model
print("Training model...")
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test_scaled)

# Transform predictions back to original scale
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)

# Calculate metrics
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print("\nModel Performance:")
print(f'Test MSE: {mse:.2f}')
print(f'Test RMSE: {rmse:.2f}')
print(f'Test MAE: {mae:.2f}')
print(f'R² Score: {r2:.2f}')

# Plot feature importance
plt.figure(figsize=(12, 6))
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(y_test_original.index, y_test_original, label='Actual', alpha=0.7)
plt.plot(y_test_original.index, y_pred_original, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Usage')
plt.xlabel('Date')
plt.ylabel('Usage')
plt.legend()
plt.tight_layout()
plt.show()

# Save the model and preprocessing objects
print("Saving model...")
joblib.dump({
    'model': model,
    'scaler': scaler,
    'feature_names': X_train.columns.tolist()
}, 'energy_prediction_model.pkl')

print("Process completed successfully!")