import numpy as np
# from uncertainties import ufloat
def F_QFactor(nu_peak, FWHM, fitting_opt=None, fitting_pcov=None)->float:
    """
    Calculate the Q factor given the peak frequency and FWHM.

    Parameters:
    nu_peak (float): Peak frequency in Hz.
    FWHM (float): Full Width at Half Maximum in Hz.
    fitting_opt (dict, optional): Additional fitting options.

    Returns:
    float: Q factor.
    """
    Q_factor = nu_peak / FWHM
    # incertaint based on fitting covariance if provided
    if fitting_pcov is not None:
        
        # nu_peak_ufloat = ufloat(nu_peak, np.sqrt(fitting_pcov[0,0]))
        # FWHM_ufloat = ufloat(FWHM, np.sqrt(fitting_pcov[1,1]))
        # Q_factor_ufloat = nu_peak_ufloat / FWHM_ufloat
        # Q_factor = Q_factor_ufloat.n  # nominal value
        pass


    return Q_factor