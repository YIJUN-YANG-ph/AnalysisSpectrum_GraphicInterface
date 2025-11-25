import numpy as np
from scipy import stats

def evaluate_fit_quality(xdata: np.ndarray, 
                         ydata: np.ndarray, 
                         model_func: callable, 
                         popt: np.ndarray, 
                         pcov: np.ndarray = None,
                         model_name: str = 'Lorentzian') -> dict:
    """
    Evaluate the goodness of fit for a model to the data.
    
    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data (e.g., frequency).
    ydata : np.ndarray
        Dependent variable data (e.g., transmission).
    model_func : callable
        The fitting function (e.g., f_func_Lorentzian).
    popt : np.ndarray
        Optimized parameters from curve_fit.
    pcov : np.ndarray, optional
        Covariance matrix from curve_fit.
    model_name : str, optional
        Name of the model for reporting purposes.
    
    Returns
    -------
    dict
        Dictionary containing goodness-of-fit metrics:
        - 'r_squared': R² coefficient of determination (0 to 1, higher is better)
        - 'adjusted_r_squared': Adjusted R² accounting for number of parameters
        - 'rmse': Root Mean Square Error (lower is better)
        - 'nrmse': Normalized RMSE (percentage, lower is better)
        - 'chi_squared': Chi-squared statistic
        - 'reduced_chi_squared': Reduced chi-squared (should be close to 1)
        - 'residuals': Array of residuals (ydata - y_fit)
        - 'residuals_std': Standard deviation of residuals
        - 'max_residual': Maximum absolute residual
        - 'parameter_uncertainties': Standard errors of parameters (if pcov provided)
        - 'is_good_fit': Boolean indicating if fit meets quality thresholds
        - 'fit_quality': String rating ('Excellent', 'Good', 'Fair', 'Poor')
        - 'warnings': List of warning messages about fit quality
    """
    
    # Calculate fitted values
    y_fit = model_func(xdata, *popt)
    
    # Calculate residuals
    residuals = ydata - y_fit
    
    # Number of data points and parameters
    n = len(ydata)
    p = len(popt)
    
    # Calculate Sum of Squares, ss: sum of squares
    ss_res = np.sum(residuals**2)  # Residual sum of squares
    ss_tot = np.sum((ydata - np.mean(ydata))**2)  # Total sum of squares
    
    # R-squared (coefficient of determination)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Adjusted R-squared
    if n > p + 1:
        adjusted_r_squared = 1 - (ss_res / (n - p)) / (ss_tot / (n - 1))
    else:
        adjusted_r_squared = r_squared
    
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(ss_res / n)
    
    # Normalized RMSE (as percentage)
    y_range = np.max(ydata) - np.min(ydata)
    nrmse = (rmse / y_range * 100) if y_range > 0 else np.inf
    
    # Chi-squared statistics
    # Assuming equal weights (uncertainty = 1) if not provided
    chi_squared = ss_res
    
    # Reduced chi-squared
    if n > p:
        reduced_chi_squared = chi_squared / (n - p)
    else:
        reduced_chi_squared = np.inf
    
    # Residual statistics
    residuals_std = np.std(residuals)
    max_residual = np.max(np.abs(residuals))
    
    # Parameter uncertainties (standard errors)
    parameter_uncertainties = None
    if pcov is not None:
        try:
            parameter_uncertainties = np.sqrt(np.diag(pcov))
        except:
            parameter_uncertainties = None
    
    # Determine fit quality based on multiple criteria
    warnings = []
    is_good_fit = True
    
    # Criterion 1: R-squared should be high (> 0.95 for excellent, > 0.90 for good)
    if r_squared < 0.85:
        warnings.append(f"Low R² = {r_squared:.4f} (should be > 0.90)")
        is_good_fit = False
    
    # Criterion 2: Reduced chi-squared should be close to 1
    if reduced_chi_squared > 5 or reduced_chi_squared < 0.2:
        warnings.append(f"Reduced χ² = {reduced_chi_squared:.4f} is far from 1")
        is_good_fit = False
    
    # Criterion 3: NRMSE should be small (< 5% for good fit)
    if nrmse > 10:
        warnings.append(f"High normalized RMSE = {nrmse:.2f}% (should be < 5%)")
        is_good_fit = False
    
    # Criterion 4: Check for systematic residuals (runs test)
    # Count the number of sign changes in residuals
    if n > 10:
        sign_changes = np.sum(np.diff(np.sign(residuals)) != 0)
        expected_changes = (n - 1) / 2
        # If too few sign changes, residuals may be systematic
        if sign_changes < expected_changes * 0.5:
            warnings.append(f"Residuals show systematic pattern (sign changes: {sign_changes}, expected: {expected_changes:.1f})")
            is_good_fit = False
    
    # Criterion 5: Check if parameters are reasonable (e.g., uncertainties)
    if parameter_uncertainties is not None:
        relative_uncertainties = parameter_uncertainties / np.abs(popt)
        if np.any(relative_uncertainties > 1.0):
            warnings.append("Some parameter uncertainties exceed parameter values")
            is_good_fit = False
    
    # Assign fit quality rating
    if r_squared >= 0.98 and nrmse < 3 and 0.5 < reduced_chi_squared < 2.0:
        fit_quality = 'Excellent'
    elif r_squared >= 0.95 and nrmse < 5 and 0.3 < reduced_chi_squared < 3.0:
        fit_quality = 'Good'
    elif r_squared >= 0.90 and nrmse < 10:
        fit_quality = 'Fair'
    else:
        fit_quality = 'Poor'
    
    # Compile results
    results = {
        'r_squared': r_squared,
        'adjusted_r_squared': adjusted_r_squared,
        'rmse': rmse,
        'nrmse': nrmse,
        'chi_squared': chi_squared,
        'reduced_chi_squared': reduced_chi_squared,
        'residuals': residuals,
        'residuals_std': residuals_std,
        'max_residual': max_residual,
        'parameter_uncertainties': parameter_uncertainties,
        'is_good_fit': is_good_fit,
        'fit_quality': fit_quality,
        'warnings': warnings,
        'n_data_points': n,
        'n_parameters': p,
    }
    
    return results


def print_fit_report(fit_results: dict, popt: np.ndarray = None, 
                    param_names: list = None, model_name: str = 'Model'):
    """
    Print a formatted report of fit quality metrics.
    
    Parameters
    ----------
    fit_results : dict
        Output from evaluate_fit_quality function.
    popt : np.ndarray, optional
        Optimized parameters to display.
    param_names : list, optional
        Names of parameters for display.
    model_name : str, optional
        Name of the model.
    """
    print(f"\n{'='*60}")
    print(f"FIT QUALITY REPORT - {model_name}")
    print(f"{'='*60}")
    
    print(f"\nOverall Fit Quality: {fit_results['fit_quality']}")
    print(f"Suitable for Analysis: {'YES' if fit_results['is_good_fit'] else 'NO'}")
    
    print(f"\n{'-'*60}")
    print("Goodness-of-Fit Metrics:")
    print(f"{'-'*60}")
    print(f"  R² (coefficient of determination): {fit_results['r_squared']:.6f}")
    print(f"  Adjusted R²:                        {fit_results['adjusted_r_squared']:.6f}")
    print(f"  RMSE (Root Mean Square Error):      {fit_results['rmse']:.6e}")
    print(f"  Normalized RMSE:                    {fit_results['nrmse']:.3f}%")
    print(f"  χ² (Chi-squared):                   {fit_results['chi_squared']:.6e}")
    print(f"  Reduced χ²:                         {fit_results['reduced_chi_squared']:.6f}")
    print(f"  Residual Std Dev:                   {fit_results['residuals_std']:.6e}")
    print(f"  Max Absolute Residual:              {fit_results['max_residual']:.6e}")
    
    if popt is not None:
        print(f"\n{'-'*60}")
        print("Fitted Parameters:")
        print(f"{'-'*60}")
        if param_names is None:
            param_names = [f"p{i}" for i in range(len(popt))]
        
        for i, (name, val) in enumerate(zip(param_names, popt)):
            if fit_results['parameter_uncertainties'] is not None:
                unc = fit_results['parameter_uncertainties'][i]
                rel_unc = unc / np.abs(val) * 100 if val != 0 else np.inf
                print(f"  {name:15s} = {val:.6e} ± {unc:.6e} ({rel_unc:.2f}%)")
            else:
                print(f"  {name:15s} = {val:.6e}")
    
    if fit_results['warnings']:
        print(f"\n{'-'*60}")
        print("WARNINGS:")
        print(f"{'-'*60}")
        for warning in fit_results['warnings']:
            print(f"  ⚠ {warning}")
    
    print(f"\n{'='*60}\n")


def plot_residuals(xdata: np.ndarray, 
                   ydata: np.ndarray, 
                   model_func: callable, 
                   popt: np.ndarray,
                   fit_results: dict = None,
                   ax: 'matplotlib.axes._axes.Axes' = None) -> 'matplotlib.axes._axes.Axes':
    """
    Plot residuals to visualize fit quality.
    
    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    model_func : callable
        The fitting function.
    popt : np.ndarray
        Optimized parameters.
    fit_results : dict, optional
        Results from evaluate_fit_quality for additional info.
    ax : matplotlib.axes._axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    
    Returns
    -------
    ax : matplotlib.axes._axes.Axes
        The axes with the residual plot.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Calculate fitted values and residuals
    y_fit = model_func(xdata, *popt)
    residuals = ydata - y_fit
    
    # Plot residuals
    ax.scatter(xdata, residuals, alpha=0.6, s=20, label='Residuals')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Zero line')
    
    # Add ±σ bands if fit_results provided
    if fit_results is not None:
        std = fit_results['residuals_std']
        ax.axhline(y=std, color='orange', linestyle=':', linewidth=1, label=f'±1σ ({std:.3e})')
        ax.axhline(y=-std, color='orange', linestyle=':', linewidth=1)
        ax.axhline(y=2*std, color='yellow', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axhline(y=-2*std, color='yellow', linestyle=':', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('X data')
    ax.set_ylabel('Residuals (Data - Fit)')
    ax.set_title('Residual Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def quick_fit_check(xdata: np.ndarray, 
                   ydata: np.ndarray, 
                   model_func: callable, 
                   popt: np.ndarray, 
                   pcov: np.ndarray = None,
                   threshold_r2: float = 0.90,
                   threshold_nrmse: float = 10.0) -> bool:
    """
    Quick boolean check if a fit meets minimum quality thresholds.
    
    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    model_func : callable
        The fitting function.
    popt : np.ndarray
        Optimized parameters.
    pcov : np.ndarray, optional
        Covariance matrix.
    threshold_r2 : float, optional
        Minimum acceptable R² value (default: 0.90).
    threshold_nrmse : float, optional
        Maximum acceptable normalized RMSE in % (default: 10.0).
    
    Returns
    -------
    bool
        True if fit meets quality thresholds, False otherwise.
    
    Example
    -------
    >>> is_good = quick_fit_check(xdata, ydata, f_func_Lorentzian, popt, pcov)
    >>> if not is_good:
    >>>     print("Warning: Fit quality is poor, consider different model")
    """
    results = evaluate_fit_quality(xdata, ydata, model_func, popt, pcov)
    
    meets_r2 = results['r_squared'] >= threshold_r2
    meets_nrmse = results['nrmse'] <= threshold_nrmse
    
    return meets_r2 and meets_nrmse

