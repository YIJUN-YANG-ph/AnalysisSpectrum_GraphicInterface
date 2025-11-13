import pandas as pd
import numpy as np
def FindPeaks(transmission_linear,**param_find_peaks)->np.ndarray:
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
    idx_peaks, _ = find_peaks(-transmission_linear, **param_find_peaks)# when in linear unit
    # peaks = pd.DataFrame(data = {'idx_peaks':idx_peaks,})
    
    return idx_peaks