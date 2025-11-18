import numpy as np
import sys
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from permetrics import RegressionMetric

# ==============================================================================
# --- 1. Correlation Coefficient (R) Metrics ---
# ==============================================================================

def calculate_correlation_metrics(y_true, y_pred):
    """Calculates Pearson's R (standard) and Spearman's Rho (variational)."""
    
    # 1. Standard Form: Pearson's R (Measures linear relationship)
    # R (Correlation Coefficient) 
    pearson_r, _ = pearsonr(y_true, y_pred)
    
    # 2. Variational Form: Spearman's Rho (Measures monotonic/rank relationship) 
    spearman_rho, _ = spearmanr(y_true, y_pred)
    
    return {
        "Pearson_R (Standard)": pearson_r,
        "Spearman_Rho (Variational)": spearman_rho
    }


# ==============================================================================
# --- 2. Coefficient of Determination (R^2) Metrics ---
# ==============================================================================

def calculate_r_squared_metrics(y_true, y_pred, N, K):
    """Calculates R^2 (standard) and Adjusted R^2 (variational)."""

    # 1. Standard Form: R^2 (Coefficient of Determination, COD) 
    r2_std = r2_score(y_true, y_pred)
    
    # 2. Variational Form: Adjusted R^2 (Penalizes model complexity K) 
    # Formula: 1 - (1 - R^2) * (N - 1) / (N - K - 1)
    if N - K - 1 <= 0:
        # Cannot compute if model is over-parameterized (K >= N-1)
        adj_r2 = np.nan
    else:
        adj_r2 = 1 - (1 - r2_std) * (N - 1) / (N - K - 1)
        
    return {
        "R2_COD (Standard)": r2_std,
        "Adjusted_R2 (Variational)": adj_r2
    }


# ==============================================================================
# --- 3. Scatter Index (SI) / NRMSE Metrics ---
# ==============================================================================

def calculate_scatter_index_metrics(y_true, y_pred):
    """
    Calculates SI normalized by range (standard) and mean (variational, requested).
    SI is synonymous with Normalized Root Mean Square Error (NRMSE). 
    """
    
    # Calculate RMSE first (required for all SI variants) 
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Denominator components
    y_mean = np.mean(y_true)
    y_range = np.max(y_true) - np.min(y_true)

    # 1. Standard Form: SI normalized by Range (Common NRMSE default) [1]
    if y_range == 0:
        si_range = np.nan
    else:
        si_range = rmse / y_range

    # 2. Variational Form (Requested): SI normalized by Mean (SI_Mean) 
    # This is often termed Relative RMSE (rRMSE). Formula: RMSE / Mean(Y_true)
    if y_mean == 0:
        si_mean = np.nan
    else:
        si_mean = rmse / y_mean

    return {
        "SI_Range (Standard)": si_range,
        "SI_Mean (Variational)": si_mean
    }


# ==============================================================================
# --- 4. Willmott's Index of Agreement (d) Metrics ---
# ==============================================================================

def modified_willmott_index(O, P):
    """
    Calculates the Modified Index of Agreement (d_mod).
    Uses absolute residuals, providing greater robustness against outliers. [2]
    """
    O_bar = np.mean(O)
    
    # Numerator: Sum of absolute differences |P_i - O_i|
    num = np.sum(np.abs(P - O))
    
    # Denominator: Sum of absolute potential error components |P_i - O_bar| + |O_i - O_bar|
    denom = np.sum(np.abs(P - O_bar) + np.abs(O - O_bar))
    
    # Formula: 1 - (Absolute Error Sum) / (Absolute Potential Error Sum) [2, 3]
    if denom == 0:
        return np.nan
    else:
        d_mod = 1 - (num / denom)
        return d_mod

def calculate_willmott_metrics(y_true, y_pred):
    """Calculates Original WI (standard) and Modified WI (variational)."""

    # 1. Standard Form: Original Willmott Index (d_orig or WI) [4, 5]
    # Uses squared differences; provided by permetrics.
    permetrics_evaluator = RegressionMetric(y_true, y_pred)
    wi_orig = permetrics_evaluator.willmott_index()
    
    # 2. Variational Form: Modified Willmott Index (d_mod) 
    wi_mod = modified_willmott_index(y_true, y_pred)
    
    return {
        "WI_Original (Standard)": wi_orig,
        "WI_Modified (Variational, d_mod)": wi_mod
    }

# ==============================================================================
# --- 5. User Input and Execution ---
# ==============================================================================

def get_user_data():
    """Prompts the user for data input and validates arrays."""
    
    print("\n==========================================================")
    print("        INPUT DATA FOR REGRESSION METRICS ANALYSIS        ")
    print("==========================================================")
    
    # --- Input Observed Data (y_true) ---
    while True:
        y_true_input = input(
            "Enter Observed Data (y_true) as a comma-separated list of numbers "
            "(e.g., 10.5, 12.0, 9.8, 11.5, 13.0,...): \n> "
        ).strip()
        try:
            y_true_list = [float(x.strip()) for x in y_true_input.split(',') if x.strip()]
            if not y_true_list: raise ValueError
            y_true = np.array(y_true_list)
            break
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

    # --- Input Predicted Data (y_pred) ---
    while True:
        y_pred_input = input(
            "\nEnter Predicted Data (y_pred) as a comma-separated list of numbers "
            "(must have the same length as Observed Data): \n> "
        ).strip()
        try:
            y_pred_list = [float(x.strip()) for x in y_pred_input.split(',') if x.strip()]
            y_pred = np.array(y_pred_list)
            
            if len(y_pred)!= len(y_true):
                print(f"Error: Predicted Data length ({len(y_pred)}) must match Observed Data length ({len(y_true)}).")
                continue
            if not y_pred_list: raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

    # --- Input K (Number of Predictors) ---
    N = len(y_true)
    while True:
        k_input = input(
            f"\nEnter the Number of Predictors (K) used in your model (integer, N={N} observations): \n> "
        ).strip()
        try:
            K = int(k_input)
            if K < 1:
                print("K must be a positive integer (number of independent variables).")
                continue
            if K >= N:
                print(f"Warning: K={K} is >= N={N}. Adjusted R^2 will be undefined or unreliable.")
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    return y_true, y_pred, N, K

if __name__ == "__main__":
    
    # Example data for quick execution (comment out get_user_data() to use this)
    # y_true_ex = np.array([10.5, 12.0, 9.8, 11.5, 13.0, 10.1, 12.5, 9.5, 11.8, 12.2])
    # y_pred_ex = np.array([10.3, 11.9, 10.0, 11.2, 12.8, 10.3, 12.4, 9.7, 11.5, 12.0])
    # K_ex = 3 
    # N_ex = len(y_true_ex)
    # y_true, y_pred, N, K = y_true_ex, y_pred_ex, N_ex, K_ex

    # Get user input interactively
    try:
        y_true, y_pred, N, K = get_user_data()
    except KeyboardInterrupt:
        print("\nAnalysis terminated by user.")
        sys.exit(0)

    # --- Calculate all metric sets ---
    correlation_results = calculate_correlation_metrics(y_true, y_pred)
    r2_results = calculate_r_squared_metrics(y_true, y_pred, N, K)
    si_results = calculate_scatter_index_metrics(y_true, y_pred)
    willmott_results = calculate_willmott_metrics(y_true, y_pred)
    
    # --- Print Results ---
    print("\n==========================================================")
    print("          MODEL PERFORMANCE METRICS (N=%d, K=%d)          " % (N, K))
    print("==========================================================")
    
    print("\n[A] Correlation Coefficient (R) Metrics:")
    print("    Used to measure the strength and direction of association.")
    for name, value in correlation_results.items():
        print(f"    - {name:<30}: {value:.4f}")

    print("\n Coefficient of Determination (R^2) Metrics:")
    print("    Used to measure the proportion of variance explained.")
    for name, value in r2_results.items():
        print(f"    - {name:<30}: {value:.4f}")
        
    print("\n[C] Scatter Index (SI) / Normalized RMSE (NRMSE) Metrics:")
    print("    Used to measure error magnitude standardized by data scale.")
    for name, value in si_results.items():
        print(f"    - {name:<30}: {value:.4f}")

    print("\n Willmott Index of Agreement (WI) Metrics:")
    print("    Used to quantify overall model agreement and bias sensitivity.")
    for name, value in willmott_results.items():
        print(f"    - {name:<30}: {value:.4f}")
    
    print("\n==========================================================")
