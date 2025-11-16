import numpy as np
from scipy.optimize import curve_fit
##############  try to define a localized fitting function ###############
def f_locatized(nu_res:np.ndarray,
                signal_res:np.ndarray,
                bounds:tuple[np.ndarray, np.ndarray],
                model:str = 'SingleLorentzian'
                )->tuple[np.ndarray, np.ndarray, callable]:
    """ A localized fitting fucntion. The localized grating response is fitted by polynomial function
        The resonance responce is fitted by Lorentzian fucntion
        Comments by Yijun
        Args:
           * **nu_res,signal_res** (array): the mesurements around a given resonance nu
           * **bounds** (array): bounds for Lorentzian fitting
           * **model** (str): currently only 'SingleLorentzian' is implemented
        Ref:
            
    """
    #f_coupler_poly = np.polyfit(nu_res,signal_res,1)
    #f_coupler = np.poly1d(f_coupler_poly)
    
    def f_func_Lorentzian(lbd, A, FWHM, lbd_res):
        """ Lorentzian fitting function for degenerate resonance
            Comments by Yijun wl or freq. don't matter
            Args:
               * **lbd** (float): wavelength in nm or frequency in Hz
               * **A** (float): parameter to find extinction ratio
               * **HWHM** (float): half width at half maximum
               * **lbd_res** (float): resonance frequency
            Ref:
                https://doi.org/10.1364/OE.381224
        """
        return -A / (1+((lbd-lbd_res)/(0.5*FWHM))**2) + 1 #f_coupler(lbd)
    

    if model == 'SingleLorentzian':
        popt, pcov = curve_fit(f_func_Lorentzian,nu_res,signal_res, bounds=bounds)
    else:
        raise NotImplementedError(f'Model {model} not implemented yet.')
    
    return popt, pcov, f_func_Lorentzian #,f_coupler