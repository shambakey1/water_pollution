import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import re
from io import StringIO

# --- Data Loading and Preprocessing ---

# MANDATORY: Accessing the uploaded CSV file content using the environment placeholder.
# In this execution environment, we access the file content directly as a string, 
# rather than reading from a file path. DO NOT change this line.
data_content = "c_timeseries_3x_100k.csv"

try:
    # Read the content directly into a pandas DataFrame using StringIO
    # This treats the string variable 'data_content' as a file-like object.
    df = pd.read_csv(data_content)

except Exception as e:
    # Handle the error if the file cannot be read or is corrupted
    print(f"FATAL ERROR: Could not load the data from the uploaded file.")
    print(f"Error details: {e}")
    # Exit the script if the data cannot be loaded
    exit()

# 1. Reshape the data from Wide to Long format.
# The x-coordinates are embedded in the column names C(x=...).
print("Original data shape:", df.shape)

# Identify the concentration columns (e.g., C(x=0.5), C(x=1), C(x=2))
concentration_cols = [col for col in df.columns if col.startswith('C(x=')]

# Melt the DataFrame to create a single column for the 'x' feature and the 'C' target
df_long = pd.melt(df, id_vars=['t'],
                  value_vars=concentration_cols,
                  var_name='x_col',
                  value_name='C')

# 2. Extract the 'x' value (position) from the column name (e.g., 'C(x=0.5)' -> 0.5)
def extract_x(col_name):
    # Use regex to find the floating-point number inside C(x=...)
    match = re.search(r'C\(x=([\d.]+)\)', col_name)
    try:
        return float(match.group(1)) if match else np.nan
    except AttributeError:
        # Should not happen with valid input but keeps function robust
        return np.nan

df_long['x'] = df_long['x_col'].apply(extract_x)
df_long.drop('x_col', axis=1, inplace=True) # Drop the temporary column

# Remove any rows where 'x' extraction or concentration value failed
df_long.dropna(subset=['x', 'C', 't'], inplace=True)

print("Long format data shape:", df_long.shape)
print("\nFirst 5 rows of the prepared data:")
print(df_long.head())

# --- Model Training ---

# Define features (X) and target (y)
X = df_long[['t', 'x']]
y = df_long['C']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Initialize and train the XGBoost Regressor model
print("\nTraining XGBoost Regressor...")
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

xgb_model.fit(X_train, y_train)
print("Training complete.")

# --- Evaluation and Prediction ---

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R-squared (R²): {r2:.6f}")

# Sample prediction for new inputs
# Predict concentration at t=50s for x=1.5m
new_input = pd.DataFrame({'t': [50.0], 'x': [1.5]})
predicted_c = xgb_model.predict(new_input)[0]

print("\n--- Example Prediction ---")
print(f"Input: Time (t) = 50.0 s, Position (x) = 1.5 m")
print(f"Predicted Concentration (C): {predicted_c:.4f} kg/m³")

# Display a few actual vs. predicted values
comparison = pd.DataFrame({
    'Actual C': y_test.head(10),
    'Predicted C': y_pred[:10]
}).reset_index(drop=True)
print("\n--- Sample of Actual vs. Predicted Values (First 10) ---")
print(comparison)
