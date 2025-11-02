# this a script to check different functionilty of file path handling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

import os

from components_AnalysisSpectrum import analysis_main,OffsetRemover
from F_LoadData import load_data
from F_ConvertUnits import wl2nu, dB2linear
if __name__ == "__main__":
    Param_legend = {'bbox_to_anchor':(1.2,1),
                    'loc':'upper left',
                    'borderaxespad':0.0, 
                    'shadow':True}
    Param_legend_twinx = {'bbox_to_anchor':(1.2, 0.5),
                          'loc':'upper left',
                          'borderaxespad':0.0, 
                          'shadow':True}
    # =============================================================================
    # diamter:diameter of the ring in um.
    # width: the waveguide width of the ring. in nm
    # =============================================================================
    Param_RingResonator = {'diameter':200,
                           'width':1100,
                           'range_wl':[1500,1640]}
    # Param_RingResonator = {'diameter':200,
    #                        'width':1100,
    #                        'range_wl':[1510,1570]}
    
    # =============================================================================
    # distance: distancenumber, optional
    # Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. 
    # Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
    # FSR_lbd = lbd^2/(ng*pi*Diamter)
    # pts_distance = FSR_lbd/1pm, 1 pm is the resolution
    #
    # prominence:
    # The prominence of a peak measures how much a peak stands out from the surrounding 
    # baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.
    # empiric value:
    # if transmission is pre-normalised, 'prominence':0.3,
    # if transmission is not pre-normalised, prominence':0.001 
    # =============================================================================
    # don't add new terms
    # Param_find_peaks = {'distance':1600,
    #                     'prominence':0.001,}
    Param_find_peaks = {'distance':1600,
                        'prominence':0.01,}
    # =============================================================================
    # Explanation:
    # window_length: an odd number used for Savgol fitting. This number need to adapted to Fabry perot oscillation at the backgroup of the transmission
    # points_to_remove: number of points around one resonance, which present the whole peak.
    # polyorder: for Savgol filter. Generally don't need to change.
    # =============================================================================
    Param_Savgol_fitting = {'window_length':101,
                            'points_to_remove':151,
                            'polyorder':2,}
    # =============================================================================
    # Explanation:
    # peak_range_nm: wavelength range around one resonace that can define the peak
    # A_margin: for the peak height, a little margin for better fitting. it should be << 1
    # =============================================================================
    Param_peaks_fitting = {'peak_range_nm':1,
                           'A_margin': 0.02,}
    # =============================================================================
    # Explanation:
    #     fitting_order:2 or 3. the order of polynomial to fit for group delay (1/FSR)
    #     nb_sigma: 3. number of sigma for robust polynomial fitting for FSR
    # =============================================================================
    Param_FSR_fitting = {'fitting_order':2,
                         'nb_sigma':3,}
    # =============================================================================
    # Explanation
    # to distinguish a and t
    # 'wl_critical': in nm
    # =============================================================================
    Param_loss_calcul = {'wl_critical':1535,}
    
    """
    
    FileName = 'test_offsetremoved.csv'
    FileName = 'test_W800_R100um_G600.csv'
    # FileName = 'W1100_R100um_G500nm.txt'
    # FileName = 'W1100_R100um_G600nm.txt'
    # FileName = '/Users/yangyijun/Library/CloudStorage/OneDrive-Personal/1A_thesis/Analysis_Transmission_GraphicInterface/Transmission mesurements/test_W1100_R100um_G400_BeforeAnnealing.csv'
    FileName = '/test_W1100_R100um_G600_AfterAnnealing.csv'
    FileName = 'ChirpPer0.46Len600Square_R.txt'


    FileName = 'MZI_P0.46_W1.25_Wend_3_L300_BP1_R.txt'
    # T = load_data(FileName,range_wl=[1500,1574])
    # FileName = '/Users/yangyijun/Downloads/Measures_Yijun/Ring50GC_A1.csv'
    # FileName = 'Ring200_W1100_R100um_G600nm_AfterAnnealing.csv'
    try:
        T = load_data(FileName,range_wl=Param_RingResonator['range_wl'],wl_name='wavelength',data_name='transmission')
    except ValueError:
        # T = load_data(FileName,range_wl=[1500,1609],wl_name='L',data_name='1')
        print('Loading data fails!\n')

    '''Add the transmission and reflection of pulse compressor'''
    FileName_Transmission = 'PC_P0.46_W1.25_Wend_3_L300_BP1_T.txt'
    FileName_Reflection = 'PC_P0.46_W1.25_Wend_3_L300_BP1_R.txt'
    T_T = load_data(FileName_Transmission,range_wl=Param_RingResonator['range_wl'],wl_name='wavelength',data_name='transmission')
    T_R = load_data(FileName_Reflection,range_wl=Param_RingResonator['range_wl'],wl_name='wavelength',data_name='transmission')    

    '''Add offset'''
    FileName_Offset = 'ref_1150.txt'
    T_Offset = load_data(FileName_Offset,range_wl=Param_RingResonator['range_wl'],wl_name='wavelength',data_name='transmission')
    
    '''remove offset from T, T_T and T_R'''
    T_T = OffsetRemover(T_T,T_Offset)
    T_R = OffsetRemover(T_R,T_Offset)
    '''Addition dict'''
    Addition_dict = {'T_T':T_T,
                    'T_R':T_R}

    """

    FileName = 'test_W1100_R100um_G400_AfterAnnealing.csv'
    T = load_data(FileName,range_wl=Param_RingResonator['range_wl'],wl_name='wavelength',data_name='transmission')


    Fig_FSR,Fig_peak_fitting,Fig_resonance,fig_loss = analysis_main(T,
                  Param_RingResonator,
                  Param_find_peaks,
                  Param_Savgol_fitting,
                  Param_peaks_fitting,
                  Param_FSR_fitting,
                  Param_loss_calcul,
                  Param_legend = {'bbox_to_anchor':(1.2,1),
                                  'loc':'upper left',
                                  'borderaxespad':0.0, 
                                  'shadow':True},
                  Title = FileName,
                #   Addition_dict = Addition_dict
                  )  