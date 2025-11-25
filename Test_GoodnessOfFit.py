"""
Test script for goodness-of-fit evaluation functions.
This script tests the fit quality evaluation with synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from F_GoodnessOfFit import evaluate_fit_quality, print_fit_report, plot_residuals, quick_fit_check

# Create synthetic Lorentzian data
def lorentzian(x, A, FWHM, x0):
    """Lorentzian function"""
    return -A / (1 + ((x - x0) / (0.5 * FWHM))**2) + 1

# Test parameters
A_true = 0.8
FWHM_true = 50e6  # 50 MHz
x0_true = 193.5e12  # 193.5 THz

# Generate data
x = np.linspace(x0_true - 200e6, x0_true + 200e6, 500)
y_true = lorentzian(x, A_true, FWHM_true, x0_true)

print("\n" + "="*70)
print("TESTING GOODNESS-OF-FIT EVALUATION")
print("="*70)

# Test 1: Perfect fit (no noise)
print("\n" + "-"*70)
print("Test 1: Perfect Fit (No Noise)")
print("-"*70)

y_perfect = y_true.copy()
popt_perfect = np.array([A_true, FWHM_true, x0_true])

fit_results_perfect = evaluate_fit_quality(
    xdata=x,
    ydata=y_perfect,
    model_func=lorentzian,
    popt=popt_perfect,
    model_name='Perfect Lorentzian'
)

print_fit_report(fit_results_perfect, popt=popt_perfect,
                param_names=['Amplitude', 'FWHM (Hz)', 'Resonance (Hz)'])

assert fit_results_perfect['r_squared'] > 0.9999, "Perfect fit should have R² ≈ 1"
assert fit_results_perfect['fit_quality'] == 'Excellent', "Perfect fit should be Excellent"
print("✓ Test 1 PASSED")

# Test 2: Good fit with small noise
print("\n" + "-"*70)
print("Test 2: Good Fit (Small Noise)")
print("-"*70)

np.random.seed(42)
noise_small = np.random.normal(0, 0.01, len(x))
y_good = y_true + noise_small

# Simulate fitted parameters (close to true values)
popt_good = np.array([A_true * 1.01, FWHM_true * 0.99, x0_true + 1e6])

fit_results_good = evaluate_fit_quality(
    xdata=x,
    ydata=y_good,
    model_func=lorentzian,
    popt=popt_good,
    model_name='Noisy Lorentzian'
)

print_fit_report(fit_results_good, popt=popt_good,
                param_names=['Amplitude', 'FWHM (Hz)', 'Resonance (Hz)'])

assert fit_results_good['r_squared'] > 0.90, "Good fit should have R² > 0.90"
print("✓ Test 2 PASSED")

# Test 3: Poor fit (wrong model)
print("\n" + "-"*70)
print("Test 3: Poor Fit (Wrong Parameters)")
print("-"*70)

# Use very wrong parameters
popt_poor = np.array([A_true * 2.0, FWHM_true * 0.3, x0_true + 50e6])

fit_results_poor = evaluate_fit_quality(
    xdata=x,
    ydata=y_good,
    model_func=lorentzian,
    popt=popt_poor,
    model_name='Poorly Fitted Lorentzian'
)

print_fit_report(fit_results_poor, popt=popt_poor,
                param_names=['Amplitude', 'FWHM (Hz)', 'Resonance (Hz)'])

assert fit_results_poor['r_squared'] < 0.90, "Poor fit should have R² < 0.90"
assert not fit_results_poor['is_good_fit'], "Poor fit should be flagged"
assert len(fit_results_poor['warnings']) > 0, "Poor fit should have warnings"
print("✓ Test 3 PASSED")

# Test 4: Quick fit check
print("\n" + "-"*70)
print("Test 4: Quick Fit Check Function")
print("-"*70)

is_good_perfect = quick_fit_check(x, y_perfect, lorentzian, popt_perfect)
is_good_good = quick_fit_check(x, y_good, lorentzian, popt_good)
is_good_poor = quick_fit_check(x, y_good, lorentzian, popt_poor, threshold_r2=0.90)

print(f"Perfect fit passes quick check: {is_good_perfect}")
print(f"Good fit passes quick check: {is_good_good}")
print(f"Poor fit passes quick check: {is_good_poor}")

assert is_good_perfect, "Perfect fit should pass quick check"
assert is_good_good, "Good fit should pass quick check"
assert not is_good_poor, "Poor fit should fail quick check"
print("✓ Test 4 PASSED")

# Test 5: Visualize results
print("\n" + "-"*70)
print("Test 5: Creating Visualization")
print("-"*70)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Row 1: Perfect fit
axes[0, 0].plot(x*1e-6, y_perfect, 'o', alpha=0.5, markersize=3, label='Data')
axes[0, 0].plot(x*1e-6, lorentzian(x, *popt_perfect), 'r-', linewidth=2, label='Fit')
axes[0, 0].set_title('Perfect Fit: Data vs Model')
axes[0, 0].set_xlabel('Frequency (MHz)')
axes[0, 0].set_ylabel('Transmission')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

plot_residuals(x, y_perfect, lorentzian, popt_perfect, fit_results_perfect, ax=axes[0, 1])
axes[0, 1].set_title(f'Perfect Fit: Residuals (R²={fit_results_perfect["r_squared"]:.6f})')

# Row 2: Good fit
axes[1, 0].plot(x*1e-6, y_good, 'o', alpha=0.5, markersize=3, label='Data')
axes[1, 0].plot(x*1e-6, lorentzian(x, *popt_good), 'r-', linewidth=2, label='Fit')
axes[1, 0].set_title('Good Fit: Data vs Model (with noise)')
axes[1, 0].set_xlabel('Frequency (MHz)')
axes[1, 0].set_ylabel('Transmission')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

plot_residuals(x, y_good, lorentzian, popt_good, fit_results_good, ax=axes[1, 1])
axes[1, 1].set_title(f'Good Fit: Residuals (R²={fit_results_good["r_squared"]:.6f})')

# Row 3: Poor fit
axes[2, 0].plot(x*1e-6, y_good, 'o', alpha=0.5, markersize=3, label='Data')
axes[2, 0].plot(x*1e-6, lorentzian(x, *popt_poor), 'r-', linewidth=2, label='Fit')
axes[2, 0].set_title('Poor Fit: Data vs Model (wrong parameters)')
axes[2, 0].set_xlabel('Frequency (MHz)')
axes[2, 0].set_ylabel('Transmission')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

plot_residuals(x, y_good, lorentzian, popt_poor, fit_results_poor, ax=axes[2, 1])
axes[2, 1].set_title(f'Poor Fit: Residuals (R²={fit_results_poor["r_squared"]:.6f})')

plt.tight_layout()
plt.savefig('test_fit_quality.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved as 'test_fit_quality.png'")
print("✓ Test 5 PASSED")

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nThe goodness-of-fit evaluation functions are working correctly.")
print("You can now use them with confidence in your resonance fitting workflow.")

plt.show()
