import pandas as pd
import numpy as np
def FindPeaks(transmission_linear,**param_find_peaks)->tuple[np.ndarray,dict]:
    """ find peaks for FSR and pepareing for resonances fitting 
        Keyword Args:
           * **frequency_Hz** : the frequency in Hz (converted from wavelength).
           * **transmission_linear** : the linear transmission data.
           * **???** : ???
        Returns:
           * **peaks** : pandas dataframe. idx_peaks: the index the peaks in input frequency_Hz.
    """
    from scipy.signal import find_peaks
    if param_find_peaks:
        param_find_peaks = param_find_peaks
    else:
        param_find_peaks = {'distance':2000*0.8,
                            'prominence':0.0001}
    idx_peaks, properties = find_peaks(-transmission_linear, **param_find_peaks)# when in linear unit
    # peaks = pd.DataFrame(data = {'idx_peaks':idx_peaks,})
    
    return idx_peaks, properties

def F_SelectPeaks(
                 transmission_linear:np.ndarray,
                 param_find_peaks:dict = None,
                 num_peaks:int = 1)->dict:
    """ given the transmission data, find the top N peaks. By default N=1. Return the parameters to find these peaks.
        Keyword Args:
           * **transmission_linear** : the linear transmission data.
           * **param_find_peaks** : initial parameters for peak finding. ex. {'distance':400,# in number of points
                                                                      'prominence':0.001*0.1,# need to be adjusted according to the actual data
                                                                      'width':5,# in number of points
                                                                      'rel_height':1,# don't change this one
                                                                     }
           * **num_peaks** : number of peaks to select to find. Default is 1.
        Returns:
           * **params_find_peaks_selected** : dict, containing the parameters to find the top N peaks.
        """
    if param_find_peaks is None:
        param_find_peaks = {'distance':100,# in number of points
                            'prominence':1,# need to be adjusted according to the actual data
                            'width':5,# in number of points
                            'rel_height':1,# don't change this one
                           }
    count_peak = 0
    count_loop = 0
    params_find_peaks_selected = param_find_peaks.copy()
    # don't try more than 10 times to find the required number of peaks
    while count_peak < num_peaks and count_loop < 10:
        idx_peaks, properties = FindPeaks(transmission_linear,**params_find_peaks_selected)
        prominences = properties['prominences']
        if len(prominences) < num_peaks:
            # reduce the prominence requirement to find more peaks
            params_find_peaks_selected['prominence'] *= 0.2
            # print(f"Reducing prominence to {params_find_peaks_selected['prominence']} to find more peaks.")
        else:
            count_peak = len(prominences)
    return params_find_peaks_selected