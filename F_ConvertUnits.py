# from click import Tuple
import numpy as np
import scipy
import scipy.constants as consts
import pandas as pd
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
        Created by Yijun
        Args:
           * **signal_dB** (float): in dB
        Ref:
    """
    signal_linear =10**(signal_dB/10)# the signal_dB is transfered to linear signal
    return signal_linear

def linear2dB(signal_linear):
    """ from linear to dB scale
        Created by Yijun
        Args:
           * **signal_linear** (float): in linear unit
        Ref:
    """
    signal_dB =10*np.log10(signal_linear)# the signal_dB is transfered to linear signal
    return signal_dB

def time2nu(t_s:np.ndarray,T_data_dB:np.ndarray,
                 Params:dict = {'Vpp_V':4,'Freq_Hz':30*1e-3,'Tunning_Hz_V':300*1e6})->pd.DataFrame:
    """ convert the temporal transmission data to optical frequency domain transmission data. Based on the laser tuning speed and the frequency modulation.
        Created by Yijun
        Args:
           * **t_s** (np.ndarray): time data in seconds
           * **T_data_dB** (np.ndarray): transmission data in dB or dBm
           * **Params** (dict): parameters. Function generator: Vpp_V, Freq_Hz; Laser tuning speed: Tunning_Hz_V
                                Vpp_V: peak-to-peak voltage in volts, 
                                Freq_Hz: frequency in hertz, the frequency of the modulation signal
                                Tunning_Hz_V: optical frequency tuning rate in hertz per volt, based on the laser specification.

    """
    Period = 1/Params['Freq_Hz']# in s
    nu_vs_t = Params['Vpp_V']*Params['Tunning_Hz_V']/Period# in hz/s
    nu_Hz = nu_vs_t*t_s# in hz

    T = pd.DataFrame()
    T['nu_Hz'] = nu_Hz
    T['T_dB'] = T_data_dB
    T['T_linear'] = dB2linear(T['T_dB'].values)

    return T

def Heterodyne(nu_Hz:np.ndarray,
               Pe_dBm:np.ndarray,)->pd.DataFrame:
    """ convert the heterodyne measurement data to optical frequency domain transmission data.
        Created by Yijun
        Args:
           * **nu_Hz** (np.ndarray): frequency data in hertz
           * **Pe_dBm** (np.ndarray): electrical power data in dBm

    """
    T = pd.DataFrame()
    T['nu_Hz'] = nu_Hz
    T['Pe_dBm'] = Pe_dBm
    T['T_dB'] = Pe_dBm # optical power in dB is equal to electrical power in dB, see Explanation in notebook
    T['T_linear'] = dB2linear(T['T_dB'].values)
    return T