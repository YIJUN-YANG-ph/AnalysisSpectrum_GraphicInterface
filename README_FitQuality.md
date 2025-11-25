# Goodness-of-Fit Evaluation for Lorentzian Fitting

## Overview

This module (`F_GoodnessOfFit.py`) provides comprehensive tools to evaluate whether a Lorentzian model is suitable for fitting your resonance data. It calculates multiple statistical metrics and provides clear feedback on fit quality.

## Features

The fit quality evaluation includes:

1. **R² (Coefficient of Determination)**: Measures how well the model explains the variance in the data (0 to 1, higher is better)
2. **Adjusted R²**: R² adjusted for the number of parameters
3. **RMSE (Root Mean Square Error)**: Average deviation between data and fit
4. **Normalized RMSE**: RMSE as a percentage of data range
5. **Chi-squared Statistics**: Statistical goodness-of-fit measure
6. **Reduced Chi-squared**: Should be close to 1 for a good fit
7. **Residual Analysis**: Checks for systematic patterns in residuals
8. **Parameter Uncertainties**: Standard errors of fitted parameters

## Quality Ratings

The module automatically assigns a quality rating:
- **Excellent**: R² ≥ 0.98, NRMSE < 3%, Reduced χ² between 0.5-2.0
- **Good**: R² ≥ 0.95, NRMSE < 5%, Reduced χ² between 0.3-3.0
- **Fair**: R² ≥ 0.90, NRMSE < 10%
- **Poor**: Below the above thresholds

## Usage

### 1. Automatic Evaluation in Pipeline

When using the `PL_FitResonance` method in `C_SingleHighQ.py`, fit quality is automatically evaluated and displayed:

```python
from C_SingleHighQ import SingleHighQ

Q = SingleHighQ()
T_corrected = Q.load_data_singleQ(file_name='your_data.txt', model='FineScan')

T_normalised = Q.PL_FitResonance(
    T_corrected=T_corrected,
    center_wl_nm=1550,
    ax_T_normalised=ax,
    ax_T=ax_T,
    param_rel={'Rel_FWHM': 0.2, 'Rel_A': 0.2, 'Rel_Peak': 0.2},
    distance=10, prominence=0.001, width=5, rel_height=1
)

# Access fit results
fit_results = Q.fit_results
if fit_results['is_good_fit']:
    print("Fit is suitable for analysis")
else:
    print("Warning: Fit quality issues detected")
    print(fit_results['warnings'])
```

### 2. Standalone Evaluation

You can also evaluate any fit independently:

```python
from F_GoodnessOfFit import evaluate_fit_quality, print_fit_report

# After curve fitting
from F_LorentzianModel import f_locatized
opt, pcov, f_func = f_locatized(xdata, ydata, bounds, model='SingleLorentzian')

# Evaluate fit quality
fit_results = evaluate_fit_quality(
    xdata=xdata,
    ydata=ydata,
    model_func=f_func,
    popt=opt,
    pcov=pcov,
    model_name='Single Lorentzian'
)

# Print detailed report
param_names = ['Amplitude', 'FWHM (Hz)', 'Resonance (Hz)']
print_fit_report(fit_results, popt=opt, param_names=param_names)

# Check result
if fit_results['is_good_fit']:
    print("✓ Model is suitable")
else:
    print("✗ Model may not be appropriate")
```

### 3. Quick Boolean Check

For a simple yes/no answer:

```python
from F_GoodnessOfFit import quick_fit_check

is_good = quick_fit_check(
    xdata=xdata,
    ydata=ydata,
    model_func=f_func,
    popt=opt,
    pcov=pcov,
    threshold_r2=0.90,      # Minimum R²
    threshold_nrmse=10.0    # Maximum normalized RMSE (%)
)

if not is_good:
    print("Warning: Consider using a different model")
```

### 4. Visualize Residuals

To see if there are systematic patterns in the fit errors:

```python
from F_GoodnessOfFit import plot_residuals
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plot_residuals(xdata, ydata, f_func, opt, fit_results, ax=ax)
plt.show()
```

Or use the built-in diagnostic plots in `C_SingleHighQ`:

```python
Q.plot_fit_diagnostics(
    T_normalised=T_normalised,
    model_func=f_func,
    popt=opt,
    fit_results=fit_results
)
```

## Interpretation Guide

### Good Fit Indicators
- **R² > 0.95**: The model explains >95% of data variance
- **NRMSE < 5%**: Fit errors are less than 5% of data range
- **Reduced χ² ≈ 1**: Statistical consistency (0.5 < χ² < 2.0)
- **Random residuals**: No systematic patterns in residual plot
- **Low parameter uncertainties**: Uncertainties < 50% of parameter values

### Poor Fit Indicators
- **R² < 0.85**: Model doesn't capture data well
- **NRMSE > 10%**: Large fitting errors
- **Reduced χ² ≫ 1 or ≪ 1**: Statistical inconsistency
- **Systematic residuals**: Clear patterns (curves, waves) in residual plot
- **High parameter uncertainties**: Uncertainties > 100% of values

### Common Issues and Solutions

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Low R² | Wrong model type | Try DoubleLorentzian for doublet peaks |
| High NRMSE | Background not removed | Adjust Savgol filter parameters |
| Systematic residuals | Asymmetric line shape | Consider Fano resonance model |
| High χ² | Fitting bounds too tight | Increase `Rel_FWHM`, `Rel_A`, `Rel_Peak` |
| Parameter uncertainty > 100% | Noisy data or poor initialization | Improve peak finding parameters |

## Example Output

```
============================================================
FIT QUALITY REPORT - Single Lorentzian
============================================================

Overall Fit Quality: Good
Suitable for Analysis: YES

------------------------------------------------------------
Goodness-of-Fit Metrics:
------------------------------------------------------------
  R² (coefficient of determination): 0.967234
  Adjusted R²:                        0.965891
  RMSE (Root Mean Square Error):      1.234e-03
  Normalized RMSE:                    4.567%
  χ² (Chi-squared):                   5.678e-02
  Reduced χ²:                         1.234567
  Residual Std Dev:                   1.123e-03
  Max Absolute Residual:              3.456e-03

------------------------------------------------------------
Fitted Parameters:
------------------------------------------------------------
  Amplitude       = 8.765432e-01 ± 2.345e-02 (2.68%)
  FWHM (Hz)       = 1.234567e+08 ± 3.456e+06 (2.80%)
  Resonance (Hz)  = 1.935000e+14 ± 1.234e+06 (0.00%)

============================================================
```

## Integration with Your Workflow

1. **Data Loading**: Use `load_data_singleQ()` to load your measurement
2. **Background Removal**: Use `Remove_Offset()` or `Q_RemoveOffset_Savgol()`
3. **Fitting**: Use `PL_FitResonance()` which now includes automatic fit evaluation
4. **Review Results**: Check `Q.fit_results` for detailed metrics
5. **Visual Inspection**: Use `plot_fit_diagnostics()` to see residuals
6. **Decision**: Use fit quality to decide if results are trustworthy

## Files Created/Modified

1. **F_GoodnessOfFit.py** (NEW): Core evaluation functions
2. **C_SingleHighQ.py** (MODIFIED): Integrated fit evaluation in `PL_FitResonance()`
3. **Example_FitQuality.py** (NEW): Example usage script

## Tips for Best Results

1. **Start with good peak finding**: Accurate initial guesses improve fitting
2. **Remove background properly**: Use Savgol filter with appropriate parameters
3. **Set reasonable bounds**: Use `Rel_FWHM=0.2`, `Rel_A=0.2`, `Rel_Peak=0.2` as starting points
4. **Check residuals visually**: Always inspect the residual plot for patterns
5. **Consider doublet peaks**: If fit is poor, check if you have overlapping resonances
6. **Iterate if needed**: Adjust parameters based on warnings and try again

## Questions?

If the fit quality is consistently poor, consider:
- Is your resonance actually Lorentzian-shaped?
- Do you have a doublet/multiplet instead of single peak?
- Is there remaining background that should be removed?
- Are there measurement artifacts or noise issues?
