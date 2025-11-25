# QUICK START: Goodness-of-Fit Evaluation

## 🚀 Immediate Usage (3 Steps)

### Step 1: Your existing code works as before
```python
from C_SingleHighQ import SingleHighQ
import matplotlib.pyplot as plt

Q = SingleHighQ()
T_corrected = Q.load_data_singleQ(file_name='your_data.txt', model='FineScan')
```

### Step 2: Run the fitting pipeline (same as before)
```python
fig, ax = plt.subplots()
T_normalised = Q.PL_FitResonance(
    T_corrected=T_corrected,
    center_wl_nm=1550,
    ax_T_normalised=ax
)
```

### Step 3: Check the quality (NEW - automatic!)
```python
# The fit quality is now automatically evaluated and printed!
# Check if fit is trustworthy:
if Q.fit_results['is_good_fit']:
    print("✓ Results are reliable")
else:
    print("⚠ Fit quality issues - review warnings")
```

That's it! No changes needed to your existing workflow. The quality check is automatic.

---

## 📊 What You'll See

### Console Output (NEW)
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
  Normalized RMSE:                    4.567%
  Reduced χ²:                         1.234567

------------------------------------------------------------
Fitted Parameters:
------------------------------------------------------------
  Amplitude       = 8.765e-01 ± 2.345e-02 (2.68%)
  FWHM (Hz)       = 1.235e+08 ± 3.456e+06 (2.80%)
  Resonance (Hz)  = 1.935e+14 ± 1.234e+06 (0.00%)

============================================================
```

### Plot Label (UPDATED)
Before: `"Lorentzian Fit Q=5.23M@1550nm"`
After:  `"Lorentzian Fit Q=5.23M@1550nm [Good] ✓"`

---

## 🎯 What the Metrics Mean (Simple)

| Metric | What it tells you | Good Value |
|--------|-------------------|------------|
| **R²** | How much variance is explained | > 0.95 (95%) |
| **NRMSE** | Fit error as % of data range | < 5% |
| **Reduced χ²** | Statistical consistency | 0.5 - 2.0 |

**Rule of thumb**: If all three are in the "Good Value" range, your fit is reliable!

---

## ⚠️ If You See Warnings

### "Low R²"
**Problem**: Model doesn't fit data well  
**Try**: Adjust fitting bounds or try DoubleLorentzian for doublet peaks

### "High normalized RMSE"
**Problem**: Large fitting errors  
**Try**: Check background removal (Savgol filter parameters)

### "Reduced χ² is far from 1"
**Problem**: Statistical inconsistency  
**Try**: Increase `Rel_FWHM`, `Rel_A`, `Rel_Peak` in param_rel

### "Residuals show systematic pattern"
**Problem**: Model shape is wrong  
**Try**: Check if you have doublet peaks or non-Lorentzian line shape

---

## 🔍 Advanced: Access Detailed Results

```python
# All metrics are stored in Q.fit_results
results = Q.fit_results

print(f"R² value: {results['r_squared']:.4f}")
print(f"RMSE: {results['rmse']:.6e}")
print(f"Quality: {results['fit_quality']}")
print(f"Good fit? {results['is_good_fit']}")

# Check warnings
if results['warnings']:
    print("\nWarnings:")
    for warning in results['warnings']:
        print(f"  - {warning}")

# Get parameter uncertainties
if results['parameter_uncertainties'] is not None:
    uncertainties = results['parameter_uncertainties']
    print(f"\nAmplitude uncertainty: {uncertainties[0]:.6e}")
    print(f"FWHM uncertainty: {uncertainties[1]:.6e}")
    print(f"Resonance uncertainty: {uncertainties[2]:.6e}")
```

---

## 📈 Visualize Fit Diagnostics

```python
# Create diagnostic plots (data vs fit + residuals)
Q.plot_fit_diagnostics(
    T_normalised=T_normalised,
    model_func=lambda x, *p: -p[0]/(1+((x-p[2])/(0.5*p[1]))**2)+1,
    popt=[Q.fitting_options['A'], Q.fitting_options['FWHM'], Q.fitting_options['res']],
    fit_results=Q.fit_results
)
plt.show()
```

This creates a 2-panel figure:
- **Top**: Your data vs the fitted curve
- **Bottom**: Residuals (should look random, not patterned)

---

## 🧪 Test the Implementation

Run the test script to verify everything works:

```bash
python Test_GoodnessOfFit.py
```

Expected output:
```
✓ Test 1 PASSED (Perfect fit)
✓ Test 2 PASSED (Good fit with noise)
✓ Test 3 PASSED (Poor fit detection)
✓ Test 4 PASSED (Quick check function)
✓ Test 5 PASSED (Visualization)

ALL TESTS PASSED!
```

---

## 📝 Common Use Cases

### Case 1: Quick check if fit is OK
```python
if Q.fit_results['is_good_fit']:
    # Use the Q-factor
    Q_factor = Q.fitting_options['FWHM']  # ... calculate Q
    print(f"Q-factor: {Q_factor:.2e} (reliable)")
else:
    print("Don't trust this Q-factor - fit is poor")
```

### Case 2: Compare different fits
```python
# Fit with tight bounds
param_rel_tight = {'Rel_FWHM': 0.1, 'Rel_A': 0.1, 'Rel_Peak': 0.1}
T1 = Q.PL_FitResonance(T_corrected, param_rel=param_rel_tight)
r2_tight = Q.fit_results['r_squared']

# Fit with loose bounds
param_rel_loose = {'Rel_FWHM': 0.5, 'Rel_A': 0.5, 'Rel_Peak': 0.5}
T2 = Q.PL_FitResonance(T_corrected, param_rel=param_rel_loose)
r2_loose = Q.fit_results['r_squared']

print(f"Tight bounds: R² = {r2_tight:.4f}")
print(f"Loose bounds: R² = {r2_loose:.4f}")
```

### Case 3: Decide between Single vs Double Lorentzian
```python
# Try single Lorentzian
T_single = Q.PL_FitResonance(T_corrected, ...)
r2_single = Q.fit_results['r_squared']

# Try double Lorentzian (your existing code for doublet)
# ... (fit with DoubleLorentzian)
# r2_double = ...

if r2_double - r2_single > 0.05:
    print("Use DoubleLorentzian - significantly better fit")
else:
    print("SingleLorentzian is sufficient")
```

---

## 🎓 Key Concepts

**R² (R-squared)**
- Fraction of variance explained by the model
- R² = 0.95 means model explains 95% of data variation
- Higher is better (max = 1.0)

**RMSE (Root Mean Square Error)**
- Average magnitude of fitting errors
- In same units as your y-data
- Lower is better

**Normalized RMSE**
- RMSE expressed as % of data range
- Makes it easier to judge: < 5% is good
- Independent of your data scale

**Reduced Chi-squared**
- χ²/(n-p) where n=data points, p=parameters
- Should be ≈ 1 for good fit
- >> 1 means poor fit, << 1 means overfitting or underestimated errors

**Residuals**
- Difference between data and fit at each point
- Should be randomly scattered
- Patterns indicate model doesn't capture physics

---

## 💡 Pro Tips

1. **Always check the residual plot** - it reveals issues R² might miss
2. **Parameter uncertainties matter** - if uncertainty > 50% of value, be cautious
3. **R² alone is not enough** - also check NRMSE and reduced χ²
4. **Trust the overall rating** - it combines all metrics intelligently
5. **Warnings are specific** - they tell you exactly what to fix

---

## 📚 Need More Info?

- **Full documentation**: `README_FitQuality.md`
- **Usage examples**: `Example_FitQuality.py`
- **Workflow diagram**: `WORKFLOW_DIAGRAM.md`
- **Summary**: `SUMMARY_FitQuality.md`

---

## ✅ Checklist for Your Next Analysis

- [ ] Run your existing fitting code (works as before)
- [ ] Check console output for fit quality report
- [ ] Verify `is_good_fit` is True
- [ ] If False, read warnings and adjust accordingly
- [ ] (Optional) Generate diagnostic plots with `plot_fit_diagnostics()`
- [ ] Use Q-factor with confidence if fit is good!

**That's it! Your fits now come with automatic quality assessment.** 🎉
