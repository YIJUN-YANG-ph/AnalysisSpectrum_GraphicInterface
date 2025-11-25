# Goodness-of-Fit Workflow Diagram

## Overall Process Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Load Resonance Data                         │
│              (load_data_singleQ or load_data_Offset)           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Background Removal                              │
│           (Remove_Offset / Q_RemoveOffset_Savgol)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Peak Finding                                  │
│                    (Q_FindPeak)                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                Lorentzian Fitting                                │
│              (f_locatized with bounds)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            *** NEW: Goodness-of-Fit Evaluation ***              │
│              (evaluate_fit_quality)                              │
│                                                                  │
│  Calculates:                                                     │
│  • R² (coefficient of determination)                            │
│  • RMSE (Root Mean Square Error)                                │
│  • Chi-squared statistics                                        │
│  • Residual analysis                                             │
│  • Parameter uncertainties                                       │
│                                                                  │
│  Checks:                                                         │
│  • Is R² high enough? (> 0.90)                                  │
│  • Is NRMSE low enough? (< 10%)                                 │
│  • Is Reduced χ² close to 1?                                    │
│  • Are residuals random or systematic?                           │
│  • Are parameter uncertainties reasonable?                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────┴────────┐
                    │                 │
              YES   │  Is Fit Good?   │   NO
                    │                 │
         ┌──────────┤                 ├──────────┐
         │          └─────────────────┘          │
         ▼                                        ▼
┌──────────────────┐                  ┌────────────────────┐
│   USE RESULTS    │                  │  REVIEW WARNINGS   │
│                  │                  │                    │
│ • Q-factor       │                  │ Possible issues:   │
│ • FWHM           │                  │ • Wrong model      │
│ • Resonance freq │                  │ • Poor background  │
│ • Uncertainties  │                  │ • Tight bounds     │
│                  │                  │ • Noisy data       │
│ ✓ High confidence│                  │ • Doublet peak     │
└──────────────────┘                  └─────────┬──────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────┐
                                      │  Adjust & Re-fit    │
                                      │                     │
                                      │ • Change bounds     │
                                      │ • Try DoubleLorentz │
                                      │ • Fix background    │
                                      └─────────────────────┘
```

## Detailed Fit Quality Assessment

```
evaluate_fit_quality()
         │
         ├─► Calculate Fitted Values: y_fit = model(x, *params)
         │
         ├─► Calculate Residuals: residuals = y_data - y_fit
         │
         ├─► Statistical Metrics
         │   │
         │   ├─► R² = 1 - (SS_residual / SS_total)
         │   │   Target: > 0.95 for good fit
         │   │
         │   ├─► RMSE = sqrt(mean(residuals²))
         │   │   Target: < 5% of data range
         │   │
         │   └─► Reduced χ² = χ² / (n - p)
         │       Target: 0.5 < χ² < 2.0
         │
         ├─► Residual Pattern Analysis
         │   │
         │   ├─► Count sign changes in residuals
         │   │   (detect systematic patterns)
         │   │
         │   └─► Calculate std dev of residuals
         │
         ├─► Parameter Uncertainty Check
         │   │
         │   └─► σ_param / |param| < 1.0
         │       (uncertainty should be < 100% of value)
         │
         └─► Generate Report
             │
             ├─► Quality Rating: Excellent/Good/Fair/Poor
             ├─► Boolean: is_good_fit (YES/NO)
             └─► Warnings: List of specific issues
```

## Integration Points in Your Code

```
C_SingleHighQ.PL_FitResonance()
         │
         ├─► [Existing] Load & process data
         ├─► [Existing] Find peaks
         ├─► [Existing] Remove background
         ├─► [Existing] Fit with Lorentzian
         │
         └─► [NEW] Evaluate fit quality ◄── AUTOMATIC!
             │
             ├─► Store in self.fit_results
             ├─► Print detailed report
             └─► Update plot with quality indicator
```

## What You See in the Plot

```
Before (old):
  "Lorentzian Fit Q=5.23M@1550nm"

After (new):
  "Lorentzian Fit Q=5.23M@1550nm [Good] ✓"
   └─ Added: quality rating and checkmark
```

## Diagnostic Plot Structure

```
┌───────────────────────────────────────────────┐
│  Top Panel: Data vs Fit                       │
│  ┌─────────────────────────────────────┐      │
│  │ ○○○○ Data points                    │      │
│  │ ──── Fitted model                   │      │
│  │                                      │      │
│  │      Shows how well model matches    │      │
│  └─────────────────────────────────────┘      │
│                                                │
│  Bottom Panel: Residuals                       │
│  ┌─────────────────────────────────────┐      │
│  │ ........ ±2σ bands (light)          │      │
│  │ - - - - ±1σ bands (darker)          │      │
│  │ ────────  Zero line (red)           │      │
│  │ ○○○○○○  Residuals (should be random)│      │
│  │                                      │      │
│  │   If residuals show pattern → Bad!  │      │
│  │   If residuals are random → Good!   │      │
│  └─────────────────────────────────────┘      │
└───────────────────────────────────────────────┘
```

## Decision Tree

```
                    Fit Complete
                         │
                         ▼
              ┌──────────────────┐
              │  R² > 0.95?      │
              └────┬─────────┬───┘
                YES│         │NO
                   ▼         ▼
            ┌─────────┐  ┌────────────┐
            │NRMSE<5%?│  │ Poor Fit   │
            └──┬───┬──┘  │ Check:     │
          YES│   │NO     │ • Data     │
             ▼   ▼       │ • Model    │
        ┌────┐ ┌────┐    │ • Bounds   │
        │Good│ │Fair│    └────────────┘
        │Fit │ │Fit │
        └────┘ └────┘
           │      │
           └──┬───┘
              ▼
        Use Results
```

## File Organization

```
Your Project Directory/
│
├── F_GoodnessOfFit.py          ◄── NEW: Core evaluation functions
│   ├── evaluate_fit_quality()
│   ├── print_fit_report()
│   ├── plot_residuals()
│   └── quick_fit_check()
│
├── C_SingleHighQ.py             ◄── MODIFIED: Integrated evaluation
│   ├── PL_FitResonance()       ←── Now includes fit quality check
│   └── plot_fit_diagnostics()  ←── NEW method for visualization
│
├── F_LorentzianModel.py         ◄── UNCHANGED: Your fitting functions
│
├── README_FitQuality.md         ◄── NEW: Complete documentation
├── SUMMARY_FitQuality.md        ◄── NEW: Quick reference
├── Example_FitQuality.py        ◄── NEW: Usage example
└── Test_GoodnessOfFit.py        ◄── NEW: Validation tests
```

## Quick Reference: When to Worry

```
✓ GOOD SIGNS
  • R² > 0.95
  • NRMSE < 5%
  • Reduced χ² ≈ 1 (between 0.5 and 2)
  • Random residuals (scattered evenly around zero)
  • Small parameter uncertainties (< 20% of value)
  • "is_good_fit" = True
  • fit_quality = "Excellent" or "Good"

✗ WARNING SIGNS
  • R² < 0.85
  • NRMSE > 10%
  • Reduced χ² >> 1 or << 1
  • Systematic residuals (curves, waves, patterns)
  • Large parameter uncertainties (> 50% of value)
  • Multiple warnings in report
  • fit_quality = "Fair" or "Poor"
```
