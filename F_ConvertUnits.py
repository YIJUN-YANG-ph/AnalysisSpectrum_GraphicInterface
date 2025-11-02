import numpy as np
import scipy
import scipy.constants as consts
def wl2nu(wl):
    """ from wavelength scale nm to frequency scale hz
        Comments by Yijun
        Args:
           * **wl** (float): in nm
           
        Ref:
            
    """
    return consts.c/(wl*1e-9)# in hz
def nu2wl(nu):
    """ from freq scale hz to wavelegnth scale nm
        Comments by Yijun
        Args:
           * **nu** (float): in hz
        Ref:
            
    """
    return consts.c/(nu)*1e9# in nm
def dB2linear(signal_dB):
    """ from dB to linear scale
        Comments by Yijun
        Args:
           * **signal_dB** (float): in dB
        Ref:
    """
    signal_linear =10**(signal_dB/10)# the signal_dB is transfered to linear signal
    return signal_linear

def linear2dB(signal_linear):
    """ from linear to dB scale
        Comments by Yijun
        Args:
           * **signal_linear** (float): in linear unit
        Ref:
    """
    signal_dB =10*np.log10(signal_linear)# the signal_dB is transfered to linear signal
    return signal_dB