import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare the data
def load_and_prepare_data(file_path):
    """
    Load the CSV file and transform it into a format suitable for XGBoost
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Create a list to store the transformed data
    data_rows = []
    
    # Extract time column
    time_values = df['t'].values
    
    # Extract concentration columns for different x values
    x_columns = [col for col in df.columns if col.startswith('C(x=')]
    
    # Transform data into (t, x, c) format
    for i, t in enumerate(time_values):
        for col in x_columns:
            # Extract x value from column name
            x_val = float(col.split('='),[object Object],rstrip(')'))
            c_val = df.iloc[i][col]
            
            data_rows.append({
                't': t,
                'x': x_val,
                'c': c_val
            })
    
    return pd.DataFrame(data_rows)

# Feature engineering
def create_features(df):
    """
    Create additional features for better model performance
    """
    df_features = df.copy()
    
    # Time-based features
    df_features['t_squared'] = df_features['t'] ** 2
    df_features['t_log'] = np.log(df_features['t'] + 1)
    df_features['t_sqrt'] = np.sqrt(df_features['t'])
    
    # Spatial features
    df_features['x_squared'] = df_features['x'] ** 2
    df_features['x_log'] = np.log(df_features['x'] + 0.1)  # Add small value to avoid log(0)
    
    # Interaction features
    df_features['t_x_interaction'] = df_features['t'] * df_features['x']
    df_features['t_squared_x'] = df_features['t_squared'] * df_features['x']
    
    # Lag features (if temporal ordering exists)
    df_features = df_features.sort_values(['x', 't']).reset_index(drop=True)
    for x_val in df_features['x'].unique():
        mask = df_features['x'] == x_val
        df_features.loc[mask, 'c_lag_1'] = df_features.loc[mask, 'c'].shift(1)
        df_features.loc[mask, 'c_lag_2'] = df_features.loc[mask, 'c'].shift(2)
    
    # Fill NaN values for lag features
    df_features['c_lag_1'].fillna(df_features['c_lag_1'].mean(), inplace=True)
    df_features['c_lag_2'].fillna(df_features['c_lag_2'].mean(), inplace=True)
    
    return df_features

# XGBoost Model Class
class WaterPollutionXGBoost:
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, df):
        """Prepare features and target variables"""
        # Create features
        df_features = create_features(df)
        
        # Define feature columns (exclude target 'c')
        feature_cols = [col for col in df_features.columns if col != 'c']
        
        X = df_features[feature_cols].values
        y = df_features['c'].values.reshape(-1, 1)
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, df, test_size=0.2, random_state=42):
        """Train the XGBoost model"""
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Scale features and target
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y).ravel()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=random_state
        )
        
        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Initialize XGBoost regressor
        xgb_model = xgb.XGBRegressor(random_state=random_state, n_jobs=-1)
        
        # Grid search with cross-validation
        print("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Inverse transform predictions
        y_train_pred_orig = self.scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
        y_test_pred_orig = self.scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
        y_train_orig = self.scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_test_orig = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train_orig, y_train_pred_orig)
        test_mse = mean_squared_error(y_test_orig, y_test_pred_orig)
        train_r2 = r2_score(y_train_orig, y_train_pred_orig)
        test_r2 = r2_score(y_test_orig, y_test_pred_orig)
        train_mae = mean_absolute_error(y_train_orig, y_train_pred_orig)
        test_mae = mean_absolute_error(y_test_orig, y_test_pred_orig)
        
        print("\n=== Model Performance ===")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Testing MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Testing MAE: {test_mae:.4f}")
        
        return {
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_mae': train_mae, 'test_mae': test_mae,
            'y_test_true': y_test_orig, 'y_test_pred': y_test_pred_orig
        }
    
    def predict(self, t_values, x_values):
        """Make predictions for given t and x values"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create DataFrame for prediction
        pred_df = pd.DataFrame({
            't': t_values,
            'x': x_values,
            'c': np.zeros(len(t_values))  # Dummy values
        })
        
        # Create features
        pred_features = create_features(pred_df)
        
        # Extract features (exclude target 'c')
        X_pred = pred_features[self.feature_names].values
        
        # Scale features
        X_pred_scaled = self.scaler_X.transform(X_pred)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_pred_scaled)
        
        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        return y_pred
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importance (XGBoost)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df

# Usage example
def main():
    # Load and prepare data
    print("Loading data...")
    df = load_and_prepare_data('c_timeseries_3x_100k.csv')
    
    print(f"Data shape: {df.shape}")
    print(f"Data summary:\n{df.describe()}")
    
    # Initialize and train model
    model = WaterPollutionXGBoost()
    
    print("\nTraining XGBoost model...")
    results = model.train(df)
    
    # Plot feature importance
    print("\nFeature importance:")
    feature_importance = model.plot_feature_importance()
    print(feature_importance.head(10))
    
    # Make sample predictions
    print("\nMaking sample predictions...")
    sample_t = [5.0, 10.0, 15.0, 20.0]
    sample_x = [0.5, 1.0, 2.0, 1.5]
    
    predictions = model.predict(sample_t, sample_x)
    
    prediction_df = pd.DataFrame({
        't': sample_t,
        'x': sample_x,
        'predicted_c': predictions
    })
    
    print("Sample predictions:")
    print(prediction_df)
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(results['y_test_true'], results['y_test_pred'], alpha=0.6)
    plt.plot([results['y_test_true'].min(), results['y_test_true'].max()], 
             [results['y_test_true'].min(), results['y_test_true'].max()], 'r--', lw=2)
    plt.xlabel('Actual Concentration')
    plt.ylabel('Predicted Concentration')
    plt.title('Actual vs Predicted Concentration Values')
    plt.tight_layout()
    plt.show()
    
    return model, results

# Run the model
if __name__ == "__main__":
    model, results = main()
