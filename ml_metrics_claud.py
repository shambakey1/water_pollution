import numpy as np
from typing import Dict, Union
from scipy.stats import spearmanr, kendalltau

class MLPerformanceMetrics:
    """
    Comprehensive ML performance metrics calculator with standard and variational forms.
    
    All metrics compare observed (ground truth) vs predicted (model output) values.
    """
    
    @staticmethod
    def correlation_coefficient(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Correlation Coefficient - measures linear relationship strength
        
        Variants:
        - pearson: Standard Pearson correlation coefficient (parametric, -1 to 1)
        - spearman: Spearman's rank correlation (non-parametric, handles monotonic relationships)
        - kendall: Kendall's tau (non-parametric, robust to outliers)
        - squared_pearson: r² (proportion of variance explained)
        - fisher_z: Fisher Z-transformation (normalizes distribution for inference)
        """
        # Remove any NaN values
        mask = ~(np.isnan(observed) | np.isnan(predicted))
        obs = observed[mask]
        pred = predicted[mask]
        
        # Pearson correlation (standard)
        pearson_r = np.corrcoef(obs, pred)[0, 1]
        
        # Spearman's rank correlation
        spearman_rho, _ = spearmanr(obs, pred)
        
        # Kendall's tau
        kendall_tau, _ = kendalltau(obs, pred)
        
        # Squared Pearson (coefficient of determination variant)
        r_squared = pearson_r ** 2
        
        # Fisher Z-transformation
        fisher_z = 0.5 * np.log((1 + pearson_r) / (1 - pearson_r)) if abs(pearson_r) < 1 else np.inf
        
        return {
            'pearson': pearson_r,
            'spearman': spearman_rho,
            'kendall': kendall_tau,
            'squared_pearson': r_squared,
            'fisher_z': fisher_z
        }
    
    @staticmethod
    def coefficient_of_determination(observed: np.ndarray, predicted: np.ndarray, 
                                     n_predictors: int = 1) -> Dict[str, float]:
        """
        Coefficient of Determination (R²) - proportion of variance explained
        
        Variants:
        - standard: Classic R² (can be negative if model worse than mean)
        - adjusted: Adjusted R² (penalizes additional predictors)
        - predicted: Predicted R² (using PRESS statistic)
        - unbiased: Unbiased estimator of population R²
        - efron: Efron's pseudo-R² (alternative formulation)
        
        Args:
            n_predictors: Number of predictors in model (for adjusted R²)
        """
        n = len(observed)
        mean_obs = np.mean(observed)
        
        # Residual sum of squares
        ss_res = np.sum((observed - predicted) ** 2)
        
        # Total sum of squares
        ss_tot = np.sum((observed - mean_obs) ** 2)
        
        # Standard R²
        r2_standard = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Adjusted R² (penalizes model complexity)
        p = n_predictors
        r2_adjusted = 1 - ((1 - r2_standard) * (n - 1) / (n - p - 1)) if (n - p - 1) > 0 else r2_standard
        
        # Predicted R² (using PRESS residuals - approximation)
        # For exact PRESS, would need leave-one-out predictions
        press_residuals = (observed - predicted) / (1 - 1/n)  # Approximation
        press = np.sum(press_residuals ** 2)
        r2_predicted = 1 - (press / ss_tot) if ss_tot != 0 else 0
        
        # Unbiased R² (population estimate)
        r2_unbiased = 1 - ((n - 1) / (n - p - 1)) * (ss_res / ss_tot) if (n - p - 1) > 0 and ss_tot != 0 else 0
        
        # Efron's pseudo-R²
        numerator = np.sum((observed - predicted) ** 2)
        denominator = np.sum((observed - mean_obs) ** 2)
        r2_efron = 1 - (numerator / denominator) if denominator != 0 else 0
        
        return {
            'standard': r2_standard,
            'adjusted': r2_adjusted,
            'predicted': r2_predicted,
            'unbiased': r2_unbiased,
            'efron': r2_efron
        }
    
    @staticmethod
    def scatter_index(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Scatter Index (SI) - normalized measure of scatter/dispersion
        
        Variants:
        - standard: SI = RMSE / mean(observed)
        - unbiased: Removes systematic bias before calculation
        - cv_rmse: Coefficient of variation of RMSE (normalized by std)
        - nrmse_range: RMSE normalized by data range
        - nrmse_iqr: RMSE normalized by interquartile range (robust to outliers)
        """
        # Calculate basic statistics
        rmse = np.sqrt(np.mean((observed - predicted) ** 2))
        mean_obs = np.mean(observed)
        std_obs = np.std(observed, ddof=1)
        
        # Standard Scatter Index
        si_standard = (rmse / mean_obs) if mean_obs != 0 else np.inf
        
        # Unbiased Scatter Index (removes systematic bias)
        bias = np.mean(predicted - observed)
        unbiased_errors = (observed - predicted) - bias
        rmse_unbiased = np.sqrt(np.mean(unbiased_errors ** 2))
        si_unbiased = (rmse_unbiased / mean_obs) if mean_obs != 0 else np.inf
        
        # Coefficient of Variation of RMSE
        cv_rmse = (rmse / std_obs) if std_obs != 0 else np.inf
        
        # NRMSE normalized by range
        data_range = np.max(observed) - np.min(observed)
        nrmse_range = (rmse / data_range) if data_range != 0 else np.inf
        
        # NRMSE normalized by IQR (robust to outliers)
        q75, q25 = np.percentile(observed, [75, 25])
        iqr = q75 - q25
        nrmse_iqr = (rmse / iqr) if iqr != 0 else np.inf
        
        return {
            'standard': si_standard,
            'unbiased': si_unbiased,
            'cv_rmse': cv_rmse,
            'nrmse_range': nrmse_range,
            'nrmse_iqr': nrmse_iqr
        }
    
    @staticmethod
    def willmott_index(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Willmott Index of Agreement - measures agreement between observations and predictions
        
        Variants:
        - standard: Original Willmott's d (1982) - range [0, 1]
        - modified: Modified index d1 (1985) - uses absolute differences, range [0, 1]
        - refined: Refined index dr (2012) - can be negative, range [-1, 1]
        - second_order: Considers second-order effects
        - relative: Relative index of agreement (dimensionless)
        """
        mean_obs = np.mean(observed)
        n = len(observed)
        
        # Standard Willmott Index (d) - Original 1982
        numerator = np.sum((observed - predicted) ** 2)
        denominator = np.sum((np.abs(predicted - mean_obs) + np.abs(observed - mean_obs)) ** 2)
        d_standard = 1 - (numerator / denominator) if denominator != 0 else 0
        
        # Modified Willmott Index (d1) - 1985
        numerator_mod = np.sum(np.abs(observed - predicted))
        denominator_mod = np.sum(np.abs(predicted - mean_obs) + np.abs(observed - mean_obs))
        d1_modified = 1 - (numerator_mod / denominator_mod) if denominator_mod != 0 else 0
        
        # Refined Willmott Index (dr) - 2012
        sum_abs_diff = np.sum(np.abs(observed - predicted))
        sum_potential_error = 2 * np.sum(np.abs(observed - mean_obs))
        
        if sum_potential_error != 0:
            if sum_abs_diff <= sum_potential_error:
                dr_refined = 1 - (sum_abs_diff / sum_potential_error)
            else:
                dr_refined = (sum_potential_error / sum_abs_diff) - 1
        else:
            dr_refined = 0
        
        # Second-order Willmott Index (considers variance)
        var_obs = np.var(observed)
        var_pred = np.var(predicted)
        covar = np.mean((observed - mean_obs) * (predicted - mean_obs))
        
        if var_obs != 0 and var_pred != 0:
            d_second_order = 2 * covar / (var_obs + var_pred + (mean_obs - np.mean(predicted)) ** 2)
        else:
            d_second_order = 0
        
        # Relative Index of Agreement
        mse = np.mean((observed - predicted) ** 2)
        potential_error = np.mean((np.abs(predicted - mean_obs) + np.abs(observed - mean_obs)) ** 2)
        d_relative = 1 - (mse / potential_error) if potential_error != 0 else 0
        
        return {
            'standard': d_standard,
            'modified': d1_modified,
            'refined': dr_refined,
            'second_order': d_second_order,
            'relative': d_relative
        }
    
    @classmethod
    def compute_all_metrics(cls, observed: np.ndarray, predicted: np.ndarray,
                           n_predictors: int = 1) -> Dict[str, Dict[str, float]]:
        """
        Compute all metrics and their variants in one call.
        
        Args:
            observed: Ground truth values (1D array)
            predicted: Model predictions (1D array)
            n_predictors: Number of predictors used in model (for adjusted R²)
            
        Returns:
            Nested dictionary with all metrics and their variants
        """
        observed = np.asarray(observed).flatten()
        predicted = np.asarray(predicted).flatten()
        
        if observed.shape != predicted.shape:
            raise ValueError(f"Shape mismatch: observed {observed.shape} vs predicted {predicted.shape}")
        
        if len(observed) < 2:
            raise ValueError("Need at least 2 samples for metric calculation")
        
        return {
            'correlation_coefficient': cls.correlation_coefficient(observed, predicted),
            'coefficient_of_determination': cls.coefficient_of_determination(observed, predicted, n_predictors),
            'scatter_index': cls.scatter_index(observed, predicted),
            'willmott_index': cls.willmott_index(observed, predicted)
        }
    
    @staticmethod
    def print_results(results: Dict[str, Dict[str, float]], title: str = "ML Performance Metrics"):
        """Pretty print all metrics results."""
        print("\n" + "=" * 80)
        print(f"{title:^80}")
        print("=" * 80)
        
        for metric_name, variants in results.items():
            print(f"\n{metric_name.replace('_', ' ').upper()}")
            print("-" * 80)
            for variant_name, value in variants.items():
                if np.isfinite(value):
                    print(f"  {variant_name:25s}: {value:12.6f}")
                else:
                    print(f"  {variant_name:25s}: {str(value):>12s}")
        
        print("\n" + "=" * 80 + "\n")


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data with known relationship
    n_samples = 150
    observed = np.random.randn(n_samples) * 15 + 100  # Mean=100, Std=15
    
    # Create predictions with some error
    noise = np.random.randn(n_samples) * 5  # Error term
    bias = 2  # Systematic bias
    predicted = observed + noise + bias
    
    # Initialize metrics calculator
    metrics_calc = MLPerformanceMetrics()
    
    # Method 1: Compute all metrics at once
    print("\n" + "█" * 80)
    print("METHOD 1: Compute All Metrics")
    print("█" * 80)
    
    all_results = metrics_calc.compute_all_metrics(
        observed=observed,
        predicted=predicted,
        n_predictors=3  # Example: model with 3 predictors
    )
    
    metrics_calc.print_results(all_results)
    
    # Method 2: Compute individual metrics
    print("\n" + "█" * 80)
    print("METHOD 2: Individual Metric Computation")
    print("█" * 80)
    
    print("CORRELATION COEFFICIENTS:")
    corr_results = metrics_calc.correlation_coefficient(observed, predicted)
    for name, value in corr_results.items():
        print(f"  • {name:20s}: {value:8.4f}")
    
    print("COEFFICIENT OF DETERMINATION:")
    r2_results = metrics_calc.coefficient_of_determination(observed, predicted, n_predictors=3)
    for name, value in r2_results.items():
        print(f"  • {name:20s}: {value:8.4f}")
    
    print("SCATTER INDEX:")
    si_results = metrics_calc.scatter_index(observed, predicted)
    for name, value in si_results.items():
        print(f"  • {name:20s}: {value:8.4f}")
    
    print("WILLMOTT INDEX OF AGREEMENT:")
    willmott_results = metrics_calc.willmott_index(observed, predicted)
    for name, value in willmott_results.items():
        print(f"  • {name:20s}: {value:8.4f}")
    
    # Example with perfect predictions
    print("\n\n" + "█" * 80)
    print("EXAMPLE: Perfect Predictions (predicted = observed)")
    print("█" * 80)
    
    perfect_results = metrics_calc.compute_all_metrics(
        observed=observed,
        predicted=observed,  # Perfect predictions
        n_predictors=1
    )
    
    metrics_calc.print_results(perfect_results, "Perfect Prediction Scenario")
    
    print("\n✓ Code execution complete!\n")
