def F_QFactor(nu_peak, FWHM, fitting_opt=None)->float:
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
    return Q_factor