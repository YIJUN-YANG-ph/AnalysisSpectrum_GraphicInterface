# a function for the bound to fit Lorentzian model
import numpy as np
from scipy.optimize import curve_fit
def F_Bounds_SingleLorentzian(A_lower: float = 0, A_upper: float = 1,
                              FWHM_lower: float = 0, FWHM_upper: float = 1,
                              Peak_lower: float = 0, Peak_upper: float = 1,
                              A0: float = None, Rel_A: float = 0.1,
                              FWHM0: float = None, Rel_FWHM: float = 0.1,
                              Peak0: float = None, Rel_Peak: float = 0.1
                              )->tuple[np.ndarray, np.ndarray]:
    """ A function to define the bounds for Lorentzian fitting for single resonance
        if given A_lower, A_upper, FWHM_lower, FWHM_upper, Peak_lower, Peak_upper,
        the bounds will be directly defined by these values.
        Otherwise, the bounds will be defined around the initial guess values A0, FWHM0, Peak0,
        and the relative fluctuations Rel_A, Rel_FWHM, Rel_Peak.
        Comments by Yijun
        Args:
           * **transmission_linear** (np.ndarray): the linear transmission data.

    """
    if A0 is not None:
        A_lower = A0*(1-Rel_A)
        A_upper = A0*(1+Rel_A)
    if FWHM0 is not None:
        FWHM_lower = FWHM0*(1-Rel_FWHM)
        FWHM_upper = FWHM0*(1+Rel_FWHM)
    if Peak0 is not None:
        Peak_lower = Peak0*(1-Rel_Peak)
        Peak_upper = Peak0*(1+Rel_Peak)

    lower_bounds = np.array([A_lower, FWHM_lower, Peak_lower])
    upper_bounds = np.array([A_upper, FWHM_upper, Peak_upper])
    bounds = (lower_bounds, upper_bounds)
    return bounds

if __name__ == "__main__":
    bounds = F_Bounds_SingleLorentzian(A0=0.5, Rel_A=0.2,
                                       FWHM0=0.01e12, Rel_FWHM=0.5,
                                       Peak0=193.4e12, Rel_Peak=0.001)
    print(bounds)