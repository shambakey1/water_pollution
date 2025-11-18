import pandas as pd
import numpy as np
import re
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Configuration ---
# The number of previous time steps to use to predict the next value
LOOKBACK_WINDOW = 10
# The specific file content placeholder
DATA_FILE_NAME = "c_timeseries_3x_100k.csv"

# --- Data Loading ---
try:
    df=pd.read_csv(DATA_FILE_NAME)
except Exception as e:
    print(f"FATAL ERROR: Could not load the data from the uploaded file: {e}")
    exit()

# 1. Reshape data from Wide to Long format (t, x, C)
print("Starting data preparation...")
concentration_cols = [col for col in df.columns if col.startswith('C(x=')]

df_long = pd.melt(df, id_vars=['t'],
                  value_vars=concentration_cols,
                  var_name='x_col',
                  value_name='C')

# Extract 'x' value (position) from the column name
def extract_x(col_name):
    match = re.search(r'C\(x=([\d.]+)\)', col_name)
    try:
        return float(match.group(1)) if match else np.nan
    except:
        return np.nan

df_long['x'] = df_long['x_col'].apply(extract_x)
df_long.drop('x_col', axis=1, inplace=True)
df_long.dropna(subset=['x', 'C', 't'], inplace=True)
df_long.sort_values(by=['x', 't'], inplace=True)

# 2. Scaling: Normalize all features for optimal Deep Learning performance
# We scale t, x, and C together to capture their relative magnitudes
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_long[['t', 'x', 'C']])
df_long_scaled = pd.DataFrame(scaled_data, columns=['t_scaled', 'x_scaled', 'C_scaled'])
df_long_scaled['x_original'] = df_long['x'].values # Keep original x for grouping

print(f"Total data points after preparation: {df_long_scaled.shape[0]}")

# 3. Sequence Creation for LSTM
def create_sequences(data_array, lookback):
    X, y = [], []
    # data_array contains [t_scaled, x_scaled, C_scaled]
    for i in range(len(data_array) - lookback):
        # Input sequence is the past 'lookback' steps of all 3 features
        feature_sequence = data_array[i:(i + lookback)]
        X.append(feature_sequence)
        
        # Target is the concentration at the next time step (index: 2 is C_scaled)
        target = data_array[i + lookback, 2]
        y.append(target)
    return np.array(X), np.array(y)

# Create sequences for each unique position (x) and concatenate them
X_sequences, y_targets = [], []
for x_val, group in df_long_scaled.groupby('x_original'):
    # We only need the scaled columns for sequence generation
    group_data = group[['t_scaled', 'x_scaled', 'C_scaled']].values
    
    X_group, y_group = create_sequences(group_data, LOOKBACK_WINDOW)
    X_sequences.append(X_group)
    y_targets.append(y_group)

X = np.concatenate(X_sequences)
y = np.concatenate(y_targets)

# 4. Split and Prepare Data for Keras
# X shape: (N_samples, LOOKBACK_WINDOW, N_features)
# y shape: (N_samples,)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False # Keep time sequence intact
)

print(f"\nTraining sequences shape: {X_train.shape}")
print(f"Testing targets shape: {y_test.shape}")

# --- Model Definition ---

# The input shape is (LOOKBACK_WINDOW, N_features), where N_features=3 (t, x, C)
n_features = X_train.shape[2] 

model = Sequential([
    # LSTM layer to capture sequential dependencies
    LSTM(units=50, activation='relu', input_shape=(LOOKBACK_WINDOW, n_features), return_sequences=False),
    # Dropout to prevent overfitting
    Dropout(0.2),
    # Dense output layer for regression (predicting one value: C_scaled)
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print("\nKeras Model Summary:")
model.summary()

# --- Model Training ---
print("\nTraining LSTM Network...")
# Training in a controlled way, setting verbose=0 to avoid excessive output during training
history = model.fit(X_train, y_train, epochs=10, batch_size=64, 
                    validation_split=0.1, verbose=1)
print("Training complete.")

# --- Evaluation and Inverse Transform ---

# Make predictions on the test set
y_pred_scaled = model.predict(X_test)

# To get meaningful results, we must inverse transform the scaled predictions and targets.
# The scaler was fitted on [t, x, C]. We need to reconstruct a temporary array 
# with the correct feature structure for the inverse transform: [0, 0, C_scaled]
# We only care about C, so we fill the t and x slots with arbitrary constants (e.g., 0)

# Inverse transform Y_test
y_test_temp = np.zeros((len(y_test), n_features))
y_test_temp[:, 2] = y_test.flatten()
y_test_original = scaler.inverse_transform(y_test_temp)[:, 2]

# Inverse transform Predictions
y_pred_temp = np.zeros((len(y_pred_scaled), n_features))
y_pred_temp[:, 2] = y_pred_scaled.flatten()
y_pred_original = scaler.inverse_transform(y_pred_temp)[:, 2]


# --- Evaluation Metrics ---

mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print("\n--- LSTM Model Evaluation Metrics (Original Units) ---")
print(f"Mean Absolute Error (MAE): {mae:.6f} kg/m³")
print(f"R-squared (R²): {r2:.6f}")

# Display a few actual vs. predicted values
comparison = pd.DataFrame({
    'Actual C': y_test_original[:10],
    'Predicted C': y_pred_original[:10]
}).reset_index(drop=True)
print("\n--- Sample of Actual vs. Predicted Concentrations (First 10) ---")
print(comparison)
