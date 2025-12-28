"""
Python Native Parallel Hyperparameter Tuning
Supports CPU multiprocessing, GPU parallel execution, and hybrid CPU+GPU
Works for both LSTM and XGBoost models
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from itertools import product
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA PREPROCESSING MODULE
# ============================================================================

class DataPreprocessor:
    """Handles data preprocessing for both LSTM and XGBoost."""
    
    @staticmethod
    def prepare_xgboost_data(data_path):
        """Prepare data for XGBoost using only t and x as features."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.model_selection import train_test_split
        
        # Load data
        df = pd.read_csv(data_path)
        time = df['t'].values
        concentration_cols = [col for col in df.columns if col != 't']
        positions = [float(col.split('=')[1].rstrip(')')) for col in concentration_cols]
        
        print(f"Data shape: {df.shape}")
        print(f"Time steps: {len(time)}")
        print(f"Spatial positions: {positions}")
        
        # Create features: only t and x
        X_list, y_list = [], []
        
        for i, col in enumerate(concentration_cols):
            C_values = df[col].values
            
            for j in range(len(C_values)):
                # Features: [time, position]
                features = [time[j], positions[i]]
                
                X_list.append(features)
                y_list.append(C_values[j])
        
        X = np.array(X_list)
        y = np.array(y_list).reshape(-1, 1)
        
        print(f"Feature matrix: X={X.shape} (features: t, x), y={y.shape}")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=True
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
        )
        
        # Normalize
        scaler_features = StandardScaler()
        scaler_target = MinMaxScaler()
        
        X_train = scaler_features.fit_transform(X_train)
        X_val = scaler_features.transform(X_val)
        X_test = scaler_features.transform(X_test)
        
        y_train = scaler_target.fit_transform(y_train).ravel()
        y_val = scaler_target.transform(y_val).ravel()
        y_test = scaler_target.transform(y_test).ravel()
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler_features': scaler_features,
            'scaler_target': scaler_target
        }
    
    @staticmethod
    def prepare_lstm_data(data_path, sequence_length=10):
        """Prepare data for LSTM using only t and x as features."""
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        
        # Load data
        df = pd.read_csv(data_path)
        time = df['t'].values
        concentration_cols = [col for col in df.columns if col != 't']
        positions = [float(col.split('=')[1].rstrip(')')) for col in concentration_cols]
        
        print(f"Data shape: {df.shape}")
        print(f"Time steps: {len(time)}")
        print(f"Spatial positions: {positions}")
        
        # Create sequences with only t and x as features
        X_list, y_list = [], []
        
        for i, col in enumerate(concentration_cols):
            C_values = df[col].values
            
            for j in range(len(C_values) - sequence_length):
                seq_features = []
                for k in range(sequence_length):
                    # Features: [time, position]
                    seq_features.append([
                        time[j + k],
                        positions[i]
                    ])
                X_list.append(seq_features)
                y_list.append(C_values[j + sequence_length])
        
        X = np.array(X_list)
        y = np.array(y_list).reshape(-1, 1)
        
        print(f"Sequence data: X={X.shape} (features: t, x), y={y.shape}")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=True
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
        )
        
        # Normalize
        scaler_features = MinMaxScaler()
        scaler_target = MinMaxScaler()
        
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        X_train_scaled = scaler_features.fit_transform(X_train_reshaped)
        X_val_scaled = scaler_features.transform(X_val_reshaped)
        X_test_scaled = scaler_features.transform(X_test_reshaped)
        
        X_train = X_train_scaled.reshape(X_train.shape)
        X_val = X_val_scaled.reshape(X_val.shape)
        X_test = X_test_scaled.reshape(X_test.shape)
        
        y_train = scaler_target.fit_transform(y_train).ravel()
        y_val = scaler_target.transform(y_val).ravel()
        y_test = scaler_target.transform(y_test).ravel()
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler_features': scaler_features,
            'scaler_target': scaler_target
        }


# ============================================================================
# 2. WORKER FUNCTIONS FOR PARALLEL EXECUTION
# ============================================================================

def train_xgboost_worker(args):
    """
    Worker function for training XGBoost model.
    Designed to run in separate process.
    """
    job_id, params, data_dict = args
    
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Record start time
    start_time_ns = time.time_ns()
    
    print(f"[Job {job_id}] Training XGBoost: {params}")
    
    # Extract data
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']
    
    # Build and train model
    model = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=42,
        n_jobs=1,  # Use single thread per worker
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluate
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Record end time
    end_time_ns = time.time_ns()
    duration_ns = end_time_ns - start_time_ns
    duration_seconds = duration_ns / 1e9
    
    print(f"[Job {job_id}] Completed in {duration_seconds:.2f}s - Val MSE: {val_mse:.6f}")
    
    results = {
        'job_id': job_id,
        'params': params,
        'val_mse': float(val_mse),
        'val_mae': float(val_mae),
        'val_r2': float(val_r2),
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'best_iteration': int(model.best_iteration) if hasattr(model, 'best_iteration') else params['n_estimators'],
        'start_time_ns': int(start_time_ns),
        'end_time_ns': int(end_time_ns),
        'duration_ns': int(duration_ns),
        'duration_seconds': float(duration_seconds)
    }
    
    return results, model


def train_random_forest_worker(args):
    """
    Worker function for training Random Forest model.
    Designed to run in separate process.
    """
    job_id, params, data_dict = args
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Record start time
    start_time_ns = time.time_ns()
    
    print(f"[Job {job_id}] Training Random Forest: {params}")
    
    # Extract data
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']
    
    # Build and train model
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        bootstrap=params['bootstrap'],
        max_samples=params.get('max_samples', None),
        random_state=42,
        n_jobs=1,  # Use single thread per worker
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Record end time
    end_time_ns = time.time_ns()
    duration_ns = end_time_ns - start_time_ns
    duration_seconds = duration_ns / 1e9
    
    print(f"[Job {job_id}] Completed in {duration_seconds:.2f}s - Val MSE: {val_mse:.6f}")
    
    results = {
        'job_id': job_id,
        'params': params,
        'val_mse': float(val_mse),
        'val_mae': float(val_mae),
        'val_r2': float(val_r2),
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'n_trees': params['n_estimators'],
        'start_time_ns': int(start_time_ns),
        'end_time_ns': int(end_time_ns),
        'duration_ns': int(duration_ns),
        'duration_seconds': float(duration_seconds)
    }
    
    return results, model


def train_mlp_worker_cpu(args):
    """
    Worker function for training MLP model on CPU.
    Designed to run in separate process.
    """
    job_id, params, data_dict = args
    
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Record start time
    start_time_ns = time.time_ns()
    
    print(f"[Job {job_id}] Training MLP on CPU: {params}")
    
    # Extract data
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']
    
    # Build MLP model
    model = Sequential()
    
    # Input layer
    model.add(Dense(params['hidden_units'], 
                   activation=params['activation'],
                   input_shape=(X_train.shape[1],)))
    if params['use_batch_norm']:
        model.add(BatchNormalization())
    model.add(Dropout(params['dropout_rate']))
    
    # Hidden layers
    for _ in range(params['num_layers'] - 1):
        model.add(Dense(params['hidden_units'], 
                       activation=params['activation']))
        if params['use_batch_norm']:
            model.add(BatchNormalization())
        model.add(Dropout(params['dropout_rate']))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=20, 
                               restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                  patience=10, min_lr=1e-7, verbose=0)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params.get('epochs', 200),
        batch_size=params.get('batch_size', 64),
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Evaluate
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    y_val_pred = model.predict(X_val, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)
    
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Record end time
    end_time_ns = time.time_ns()
    duration_ns = end_time_ns - start_time_ns
    duration_seconds = duration_ns / 1e9
    
    print(f"[Job {job_id}] Completed in {duration_seconds:.2f}s - Val MSE: {val_loss:.6f}")
    
    results = {
        'job_id': job_id,
        'params': params,
        'val_mse': float(val_loss),
        'val_mae': float(val_mae),
        'val_r2': float(val_r2),
        'test_mse': float(test_loss),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'epochs_trained': len(history.history['loss']),
        'start_time_ns': int(start_time_ns),
        'end_time_ns': int(end_time_ns),
        'duration_ns': int(duration_ns),
        'duration_seconds': float(duration_seconds)
    }
    
    return results, model


def train_mlp_worker_gpu(job_id, params, data_dict, gpu_id):
    """
    Worker function for training MLP model on specific GPU.
    Runs in thread (not process) to share GPU memory efficiently.
    """
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Set memory growth to avoid OOM
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Record start time
    start_time_ns = time.time_ns()
    
    print(f"[Job {job_id}] Training MLP on GPU {gpu_id}: {params}")
    
    # Extract data
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']
    
    # Build MLP model
    with tf.device(f'/GPU:{gpu_id}'):
        model = Sequential()
        
        # Input layer
        model.add(Dense(params['hidden_units'], 
                       activation=params['activation'],
                       input_shape=(X_train.shape[1],)))
        if params['use_batch_norm']:
            model.add(BatchNormalization())
        model.add(Dropout(params['dropout_rate']))
        
        # Hidden layers
        for _ in range(params['num_layers'] - 1):
            model.add(Dense(params['hidden_units'], 
                           activation=params['activation']))
            if params['use_batch_norm']:
                model.add(BatchNormalization())
            model.add(Dropout(params['dropout_rate']))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=20, 
                                   restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                      patience=10, min_lr=1e-7, verbose=0)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params.get('epochs', 200),
            batch_size=params.get('batch_size', 64),
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        
        y_val_pred = model.predict(X_val, verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)
        
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)
    
    # Record end time
    end_time_ns = time.time_ns()
    duration_ns = end_time_ns - start_time_ns
    duration_seconds = duration_ns / 1e9
    
    print(f"[Job {job_id}] Completed in {duration_seconds:.2f}s - Val MSE: {val_loss:.6f}")
    
    results = {
        'job_id': job_id,
        'params': params,
        'val_mse': float(val_loss),
        'val_mae': float(val_mae),
        'val_r2': float(val_r2),
        'test_mse': float(test_loss),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'epochs_trained': len(history.history['loss']),
        'start_time_ns': int(start_time_ns),
        'end_time_ns': int(end_time_ns),
        'duration_ns': int(duration_ns),
        'duration_seconds': float(duration_seconds),
        'gpu_id': gpu_id
    }
    
    return results, model


def train_lstm_worker_cpu(args):
    """
    Worker function for training LSTM model on CPU.
    Designed to run in separate process.
    """
    job_id, params, data_dict = args
    
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Record start time
    start_time_ns = time.time_ns()
    
    print(f"[Job {job_id}] Training LSTM on CPU: {params}")
    
    # Extract data
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']
    
    # Build LSTM model
    model = Sequential()
    
    if params['num_layers'] == 1:
        model.add(LSTM(params['lstm_units'], 
                      input_shape=(X_train.shape[1], X_train.shape[2])))
    else:
        model.add(LSTM(params['lstm_units'], 
                      return_sequences=True,
                      input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(params['dropout_rate']))
    
    for i in range(1, params['num_layers']):
        if i == params['num_layers'] - 1:
            model.add(LSTM(params['lstm_units']))
        else:
            model.add(LSTM(params['lstm_units'], return_sequences=True))
        model.add(Dropout(params['dropout_rate']))
    
    model.add(Dense(1))
    
    optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, 
                               restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                  patience=5, min_lr=1e-7, verbose=0)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params.get('epochs', 100),
        batch_size=params.get('batch_size', 32),
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Evaluate
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    y_val_pred = model.predict(X_val, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)
    
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Record end time
    end_time_ns = time.time_ns()
    duration_ns = end_time_ns - start_time_ns
    duration_seconds = duration_ns / 1e9
    
    print(f"[Job {job_id}] Completed in {duration_seconds:.2f}s - Val MSE: {val_loss:.6f}")
    
    results = {
        'job_id': job_id,
        'params': params,
        'val_mse': float(val_loss),
        'val_mae': float(val_mae),
        'val_r2': float(val_r2),
        'test_mse': float(test_loss),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'epochs_trained': len(history.history['loss']),
        'start_time_ns': int(start_time_ns),
        'end_time_ns': int(end_time_ns),
        'duration_ns': int(duration_ns),
        'duration_seconds': float(duration_seconds)
    }
    
    return results, model


def train_lstm_worker_gpu(job_id, params, data_dict, gpu_id):
    """
    Worker function for training LSTM model on specific GPU.
    Runs in thread (not process) to share GPU memory efficiently.
    """
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Set memory growth to avoid OOM
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Record start time
    start_time_ns = time.time_ns()
    
    print(f"[Job {job_id}] Training LSTM on GPU {gpu_id}: {params}")
    
    # Extract data
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']
    
    # Build LSTM model
    with tf.device(f'/GPU:{gpu_id}'):
        model = Sequential()
        
        if params['num_layers'] == 1:
            model.add(LSTM(params['lstm_units'], 
                          input_shape=(X_train.shape[1], X_train.shape[2])))
        else:
            model.add(LSTM(params['lstm_units'], 
                          return_sequences=True,
                          input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(params['dropout_rate']))
        
        for i in range(1, params['num_layers']):
            if i == params['num_layers'] - 1:
                model.add(LSTM(params['lstm_units']))
            else:
                model.add(LSTM(params['lstm_units'], return_sequences=True))
            model.add(Dropout(params['dropout_rate']))
        
        model.add(Dense(1))
        
        optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=15, 
                                   restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                      patience=5, min_lr=1e-7, verbose=0)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params.get('epochs', 100),
            batch_size=params.get('batch_size', 32),
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        
        y_val_pred = model.predict(X_val, verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)
        
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)
    
    # Record end time
    end_time_ns = time.time_ns()
    duration_ns = end_time_ns - start_time_ns
    duration_seconds = duration_ns / 1e9
    
    print(f"[Job {job_id}] Completed in {duration_seconds:.2f}s - Val MSE: {val_loss:.6f}")
    
    results = {
        'job_id': job_id,
        'params': params,
        'val_mse': float(val_loss),
        'val_mae': float(val_mae),
        'val_r2': float(val_r2),
        'test_mse': float(test_loss),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'epochs_trained': len(history.history['loss']),
        'start_time_ns': int(start_time_ns),
        'end_time_ns': int(end_time_ns),
        'duration_ns': int(duration_ns),
        'duration_seconds': float(duration_seconds),
        'gpu_id': gpu_id
    }
    
    return results, model


# ============================================================================
# 3. PARALLEL HYPERPARAMETER TUNER
# ============================================================================

class ParallelHyperparameterTuner:
    """
    Python native parallel hyperparameter tuning.
    Supports CPU multiprocessing, GPU parallel execution, and hybrid.
    """
    
    def __init__(self, data_path, model_type='xgboost', output_dir='./parallel_tuning'):
        self.data_path = data_path
        self.model_type = model_type
        self.output_dir = output_dir
        
        # Create directory structure
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        
        self.data_dict = None
        self.results = []
        
    def prepare_data(self, **data_prep_kwargs):
        """Prepare data for training."""
        print("Preprocessing data...")
        
        if self.model_type in ['xgboost', 'random_forest', 'mlp']:
            self.data_dict = DataPreprocessor.prepare_xgboost_data(
                self.data_path, **data_prep_kwargs
            )
        else:  # lstm
            self.data_dict = DataPreprocessor.prepare_lstm_data(
                self.data_path, **data_prep_kwargs
            )
        
        print("Data preprocessing completed!")
        return self.data_dict
    
    def generate_param_combinations(self, param_grid):
        """Generate all parameter combinations."""
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(product(*values))
        
        params_list = []
        for i, combo in enumerate(combinations):
            params_list.append((i, dict(zip(keys, combo))))
        
        return params_list
    
    def tune_cpu(self, param_grid, n_workers=None, **data_prep_kwargs):
        """
        Hyperparameter tuning using CPU multiprocessing.
        
        Args:
            param_grid: Dictionary of hyperparameters
            n_workers: Number of parallel workers (default: CPU count)
            **data_prep_kwargs: Data preparation arguments
        """
        # Prepare data
        if self.data_dict is None:
            self.prepare_data(**data_prep_kwargs)
        
        # Generate parameter combinations
        params_list = self.generate_param_combinations(param_grid)
        
        # Determine number of workers
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        print(f"\n{'='*70}")
        print(f"CPU PARALLEL HYPERPARAMETER TUNING")
        print(f"{'='*70}")
        print(f"Model: {self.model_type}")
        print(f"Total configurations: {len(params_list)}")
        print(f"CPU workers: {n_workers}")
        print(f"{'='*70}\n")
        
        # Prepare worker arguments
        if self.model_type == 'xgboost':
            worker_func = train_xgboost_worker
        elif self.model_type == 'random_forest':
            worker_func = train_random_forest_worker
        elif self.model_type == 'mlp':
            worker_func = train_mlp_worker_cpu
        else:  # lstm
            worker_func = train_lstm_worker_cpu
        
        worker_args = [(job_id, params, self.data_dict) 
                      for job_id, params in params_list]
        
        # Execute in parallel
        start_time = time.time()
        results = []
        models = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(worker_func, args) for args in worker_args]
            
            for future in as_completed(futures):
                try:
                    result, model = future.result()
                    results.append(result)
                    models.append(model)
                except Exception as e:
                    print(f"Error in worker: {str(e)}")
        
        end_time = time.time()
        wall_clock_time = end_time - start_time
        
        self.results = results
        self.models = models
        
        # Print summary
        self._print_summary(results, wall_clock_time)
        
        return results
    
    def tune_gpu(self, param_grid, gpu_ids=None, **data_prep_kwargs):
        """
        Hyperparameter tuning using GPU(s).
        
        Args:
            param_grid: Dictionary of hyperparameters
            gpu_ids: List of GPU IDs to use (default: [0])
            **data_prep_kwargs: Data preparation arguments
        """
        if self.model_type not in ['lstm', 'mlp']:
            print(f"GPU tuning is only supported for LSTM and MLP models!")
            print("Use tune_cpu() for XGBoost and Random Forest.")
            return None
        
        # Check GPU availability
        import tensorflow as tf
        available_gpus = tf.config.list_physical_devices('GPU')
        
        if not available_gpus:
            print("No GPUs available! Falling back to CPU tuning.")
            return self.tune_cpu(param_grid, **data_prep_kwargs)
        
        # Prepare data
        if self.data_dict is None:
            self.prepare_data(**data_prep_kwargs)
        
        # Generate parameter combinations
        params_list = self.generate_param_combinations(param_grid)
        
        # Determine GPU IDs
        if gpu_ids is None:
            gpu_ids = [0]
        
        print(f"\n{'='*70}")
        print(f"GPU PARALLEL HYPERPARAMETER TUNING")
        print(f"{'='*70}")
        print(f"Model: {self.model_type}")
        print(f"Total configurations: {len(params_list)}")
        print(f"GPUs: {gpu_ids}")
        print(f"Parallel jobs per GPU: {len(params_list) // len(gpu_ids) + 1}")
        print(f"{'='*70}\n")
        
        # Select worker function
        if self.model_type == 'lstm':
            worker_func = train_lstm_worker_gpu
        else:  # mlp
            worker_func = train_mlp_worker_gpu
        
        # Execute in parallel using ThreadPoolExecutor (better for GPU)
        start_time = time.time()
        results = []
        models = []
        
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = []
            for i, (job_id, params) in enumerate(params_list):
                gpu_id = gpu_ids[i % len(gpu_ids)]
                future = executor.submit(worker_func, job_id, params, 
                                       self.data_dict, gpu_id)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result, model = future.result()
                    results.append(result)
                    models.append(model)
                except Exception as e:
                    print(f"Error in worker: {str(e)}")
        
        end_time = time.time()
        wall_clock_time = end_time - start_time
        
        self.results = results
        self.models = models
        
        # Print summary
        self._print_summary(results, wall_clock_time)
        
        return results
    
    def tune_hybrid(self, param_grid, gpu_ids=None, n_cpu_workers=None, 
                   **data_prep_kwargs):
        """
        Hyperparameter tuning using both CPU and GPU.
        
        Args:
            param_grid: Dictionary of hyperparameters
            gpu_ids: List of GPU IDs to use
            n_cpu_workers: Number of CPU workers
            **data_prep_kwargs: Data preparation arguments
        """
        if self.model_type != 'lstm':
            print("Hybrid tuning is only supported for LSTM models!")
            return None
        
        # Check GPU availability
        import tensorflow as tf
        available_gpus = tf.config.list_physical_devices('GPU')
        
        if not available_gpus:
            print("No GPUs available! Using CPU only.")
            return self.tune_cpu(param_grid, n_workers=n_cpu_workers, 
                               **data_prep_kwargs)
        
        # Prepare data
        if self.data_dict is None:
            self.prepare_data(**data_prep_kwargs)
        
        # Generate parameter combinations
        params_list = self.generate_param_combinations(param_grid)
        
        # Determine resources
        if gpu_ids is None:
            gpu_ids = [0]
        if n_cpu_workers is None:
            n_cpu_workers = mp.cpu_count() // 2
        
        # Split work between GPU and CPU
        n_gpu_jobs = len(params_list) * 2 // 3  # 2/3 on GPU
        gpu_params = params_list[:n_gpu_jobs]
        cpu_params = params_list[n_gpu_jobs:]
        
        print(f"\n{'='*70}")
        print(f"HYBRID CPU+GPU PARALLEL HYPERPARAMETER TUNING")
        print(f"{'='*70}")
        print(f"Model: {self.model_type}")
        print(f"Total configurations: {len(params_list)}")
        print(f"GPU jobs: {len(gpu_params)} (GPUs: {gpu_ids})")
        print(f"CPU jobs: {len(cpu_params)} (Workers: {n_cpu_workers})")
        print(f"{'='*70}\n")
        
        # Execute in parallel
        start_time = time.time()
        results = []
        models = []
        
        # GPU jobs (ThreadPoolExecutor)
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as gpu_executor:
            gpu_futures = []
            for i, (job_id, params) in enumerate(gpu_params):
                gpu_id = gpu_ids[i % len(gpu_ids)]
                future = gpu_executor.submit(train_lstm_worker_gpu, job_id, params,
                                           self.data_dict, gpu_id)
                gpu_futures.append(future)

def main():
    parser = argparse.ArgumentParser(
        description='Python Native Parallel Hyperparameter Tuning'
    )
    parser.add_argument('--data', required=True, help='Path to data CSV file')
    parser.add_argument('--model', choices=['xgboost', 'random_forest', 'mlp', 'lstm'], 
                       required=True, help='Model type to tune')
    parser.add_argument('--output-dir', default='./parallel_tuning',
                       help='Output directory')
    parser.add_argument('--mode', choices=['cpu', 'gpu', 'hybrid'], default='cpu',
                       help='Parallel execution mode')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of CPU workers (default: CPU count)')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=None,
                       help='GPU IDs to use (e.g., --gpu-ids 0 1)')
    
    args = parser.parse_args()
    
    # Initialize tuner
    tuner = ParallelHyperparameterTuner(
        data_path=args.data,
        model_type=args.model,
        output_dir=args.output_dir
    )
    
    # Define parameter grids based on model type
    if args.model == 'xgboost':
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 1.5]
        }
        data_prep_kwargs = {}
    elif args.model == 'random_forest':
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'max_samples': [0.8, 1.0]
        }
        data_prep_kwargs = {}
    elif args.model == 'mlp':
        param_grid = {
            'hidden_units': [64, 128, 256, 512],
            'num_layers': [2, 3, 4, 5],
            'activation': ['relu', 'tanh'],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.0001, 0.0005, 0.001],
            'use_batch_norm': [True, False],
            'epochs': [200],
            'batch_size': [64, 128]
        }
        data_prep_kwargs = {}
    else:  # lstm
        param_grid = {
            'lstm_units': [32, 64, 128],
            'num_layers': [1, 2, 3],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.0001, 0.001],
            'epochs': [100],
            'batch_size': [32]
        }
        data_prep_kwargs = {'sequence_length': 10}
    
    # Run tuning based on mode
    if args.mode == 'cpu':
        results = tuner.tune_cpu(
            param_grid=param_grid,
            n_workers=args.n_workers,
            **data_prep_kwargs
        )
    elif args.mode == 'gpu':
        if args.model in ['xgboost', 'random_forest']:
            print(f"{args.model.upper()} doesn't benefit from GPU for hyperparameter tuning.")
            print("Using CPU mode instead.")
            results = tuner.tune_cpu(
                param_grid=param_grid,
                n_workers=args.n_workers,
                **data_prep_kwargs
            )
        else:
            results = tuner.tune_gpu(
                param_grid=param_grid,
                gpu_ids=args.gpu_ids,
                **data_prep_kwargs
            )
    elif args.mode == 'hybrid':
        if args.model in ['xgboost', 'random_forest']:
            print(f"Hybrid mode only works for LSTM and MLP. Using CPU mode for {args.model.upper()}.")
            results = tuner.tune_cpu(
                param_grid=param_grid,
                n_workers=args.n_workers,
                **data_prep_kwargs
            )
        else:
            results = tuner.tune_hybrid(
                param_grid=param_grid,
                gpu_ids=args.gpu_ids,
                n_cpu_workers=args.n_workers,
                **data_prep_kwargs
            )
    
    print(f"\n{'='*70}")
    print("ALL DONE!")
    print(f"{'='*70}")
    print(f"Results saved in: {args.output_dir}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    main()
