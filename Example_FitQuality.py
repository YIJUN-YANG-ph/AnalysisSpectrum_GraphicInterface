"""
Example script demonstrating how to evaluate Lorentzian fit quality.

This script shows how to:
1. Load and process resonance data
2. Fit with Lorentzian model
3. Evaluate if the fit is good
4. Visualize fit diagnostics
"""

import matplotlib.pyplot as plt
from C_SingleHighQ import SingleHighQ
from os.path import join

# Initialize the SingleHighQ class
Q = SingleHighQ()

# Setup plotting
Fig_T = plt.figure(figsize=(12, 5))
ax_T = Fig_T.add_subplot(111)

# Load data (change these paths as needed)
FolderName = r'C:\Users\yijun.yang\OneDrive\1A_PostDoc\SiN\202511SiN700A_4P_HighQ-PC\FineScan Measurement\D75'
FileName = r'RRW1.1G0.5L1560.438F10mHzA5V(2).txt'
file_name = join(FolderName, FileName)

# Load the data
T_corrected = Q.load_data_singleQ(
    file_name=file_name,
    model='FineScan'
)
ax_T.plot(T_corrected['nu_Hz'], T_corrected['T_linear'], 'o', 
         label='resonance', alpha=0.3)

# Setup figure for normalized transmission and fit
Fig_T_normalised = plt.figure(figsize=(12, 5))
ax_T_normalised = Fig_T_normalised.add_subplot(111)

# Define parameters for peak finding
param_find_peaks = {
    'distance': 30/3,
    'prominence': 0.001*1,
    'width': 5,
    'rel_height': 1,
}

# Define relative bounds for fitting
param_rel = {
    'Rel_FWHM': 0.2,
    'Rel_A': 0.2,
    'Rel_Peak': 0.2,
}

# Perform the fitting pipeline
print("\n" + "="*70)
print("PERFORMING LORENTZIAN FIT")
print("="*70)

T_normalised = Q.PL_FitResonance(
    T_corrected=T_corrected,
    center_wl_nm=1560,
    ax_T_normalised=ax_T_normalised,
    ax_T=ax_T,
    param_rel=param_rel,
    **param_find_peaks
)

# Access fit quality results
if hasattr(Q, 'fit_results'):
    fit_results = Q.fit_results
    
    print("\n" + "="*70)
    print("FIT QUALITY SUMMARY")
    print("="*70)
    print(f"Overall Quality: {fit_results['fit_quality']}")
    print(f"Is Good Fit: {fit_results['is_good_fit']}")
    print(f"R²: {fit_results['r_squared']:.6f}")
    print(f"Normalized RMSE: {fit_results['nrmse']:.3f}%")
    print(f"Reduced χ²: {fit_results['reduced_chi_squared']:.6f}")
    
    if fit_results['warnings']:
        print("\n⚠ WARNINGS:")
        for warning in fit_results['warnings']:
            print(f"  - {warning}")
    
    # Create diagnostic plots
    print("\nGenerating diagnostic plots...")
    Q.plot_fit_diagnostics(
        T_normalised=T_normalised,
        model_func=Q.fitting_options,  # This would need the actual function
        popt=[Q.fitting_options['A'], Q.fitting_options['FWHM'], Q.fitting_options['res']],
        fit_results=fit_results
    )
    
    # Decision making based on fit quality
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    if fit_results['is_good_fit']:
        print("✓ The Lorentzian model fits well with your data.")
        print("  You can proceed with confidence to use the fitted parameters.")
        print(f"  Q factor: {Q_factor:.2e}")
    else:
        print("✗ The Lorentzian model may NOT be suitable for your data.")
        print("\n  Possible reasons:")
        print("  1. The resonance has a non-Lorentzian shape (e.g., Fano resonance)")
        print("  2. There's remaining background that wasn't properly removed")
        print("  3. The data contains noise or artifacts")
        print("  4. The fitting bounds are too restrictive")
        print("  5. You may have a doublet resonance instead of single peak")
        print("\n  Suggestions:")
        print("  - Check the residual plot for systematic patterns")
        print("  - Try adjusting the fitting bounds (Rel_FWHM, Rel_A, Rel_Peak)")
        print("  - Review the background removal process")
        print("  - Consider if you need a different model (e.g., DoubleLorentzian)")

plt.show()
