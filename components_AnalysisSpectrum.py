#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 22:59:50 2024

@author: yangyijun
"""

'''
This script define necessary component functions for spectrum with resonances.
'''

import pandas as pd
import numpy as np
import sys
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from F_ConvertUnits import wl2nu, nu2wl, dB2linear, linear2dB
from F_LoadData import load_data
import scipy.constants as const

C = const.c  # Speed of light in vacuum (m/s)
'''
def wl2nu(wl):
    """ from wavelength scale nm to frequency scale hz
        Comments by Yijun
        Args:
           * **wl** (float): in nm
           
        Ref:
            
    """
    return C/(wl*1e-9)# in hz
def nu2wl(nu):
    """ from freq scale hz to wavelegnth scale nm
        Comments by Yijun
        Args:
           * **nu** (float): in hz
        Ref:
            
    """
    return C/(nu)*1e9# in nm
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

def load_data(file_name, wl_name='wavelength', data_name='transmission', range_wl=None)->pd.DataFrame:
    """ 
    Load data from file with flexible column name handling.
    
    Keyword Args:
       * **file_name** : the file name of the measurement.
       * **wl_name** : the column name that indicates wavelength in nm.
       * **data_name** : the column name that indicates the transmission in dB.
       * **range_wl** : a list or tuple specifying the wavelength range to filter [min_wavelength, max_wavelength].
    
    Returns:
       * **T** : pandas dataframe: wavelength_nm, nu_Hz, T_dB, T_linear.
    """
    
    # Load the data
    try:
        # Attempt to load the data with auto-detection of the separator
        data = pd.read_csv(file_name,engine='python',sep=None)
    # Handle exceptions
    except Exception as e:
        print(f"Error reading file {file_name}: {e}")


    # Try with the primary column names
    try:
        if wl_name in data.columns and data_name in data.columns:
            wavelength = data[wl_name].values
            transmission = data[data_name].values
        else:
            raise ValueError

    # Fallback to alternative column names if not found
    except ValueError:
        alt_wl_name, alt_data_name = 'L', '1'
        if alt_wl_name in data.columns and alt_data_name in data.columns:
            wavelength = data[alt_wl_name].values
            transmission = data[alt_data_name].values
        else:
            raise ValueError("Column names not found in data file.")

    # Apply the range filter if specified
    if range_wl is not None:
        mask = (wavelength >= range_wl[0]) & (wavelength <= range_wl[1])
        wavelength = wavelength[mask]
        transmission = transmission[mask]

    # Calculate derived data
    nu = wl2nu(wavelength)
    transmission_linear = dB2linear(transmission)

    # Create and return DataFrame
    T = pd.DataFrame({
        'wavelength_nm': wavelength,
        'T_dB': transmission,
        'T_linear': transmission_linear,
        'nu_Hz': nu
    })

    return T
'''

def find_peaks_for_FSR(frequency_Hz,transmission_linear,**param_find_peaks):
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
    peaks = pd.DataFrame(data = {'idx_peaks':idx_peaks,})
    return peaks
#%%
def calcul_FSR(frequency_Hz,transmission_linear,idx_peaks):
    """ Calculate FSR: free spectral range.
        Keyword Args:
           * **frequency_Hz** : pandas.series. the frequency in Hz (converted from wavelength).
           * **transmission_linear** : pandas.series. the linear transmission data.
           * **idx_peaks** : pandas.series. index of frequency_Hz where peaks are.
        Returns:
           * **FSR** : pandas dataframe. 
                       frequency_Hz_FSR: frequency of each FSR.
                       wavelength_nm_FSR: wavelength in nm of each FSR.
                       FSR_Hz: FSR in Hz.
    """
    frequency_Hz_peaks = frequency_Hz[idx_peaks]
    # transmission_linear_peaks = transmission_linear[idx_peaks]
    FSR_Hz = np.abs(np.diff(frequency_Hz_peaks))
    frequency_Hz_peaks = frequency_Hz_peaks[:-1]

    FSR = pd.DataFrame(data = {'frequency_Hz_FSR':frequency_Hz_peaks,
                               'wavelength_nm_FSR':nu2wl(frequency_Hz_peaks),
                               'FSR_Hz':FSR_Hz,})
    return FSR

#%%
def calcul_ng(FSR_Hz,length_m, cavity_type = 'ring'):
    """ Calculate ng from FSR: FSR = C/(ng*L). ref: https://www.rp-photonics.com/free_spectral_range.html
        Keyword Args:
           * **FSR_Hz** : numpy.array. the FSR in Hz.
           * **length_m** : float. the length in m of the cavity.
           * **cavity_type** : by default it is 'ring'.
        Returns:
           * **ng** : group index.
    """
    ng = C/(FSR_Hz*(length_m))
    #f_ng = C/(f_FSR(x_f)*L)
    # fig_Disp = plt.figure(figsize=(8,3.5))# plot for peaks and resonances
    # ax_ng = fig_Disp.add_subplot(121)
    # wl_peaks = nu2wl(nu_peaks)
    # ax_ng.plot(nu2wl(nu_peaks),ng,'o',label = guide_names)
    return ng


def calcul_dispersion(wavelength_nm,FSR_Hz,length_m,order=2):
    """ Calculate dispersion from FSR: 
        D = d(1/vg)/(dlambda) = d(ng/C)/(dlambda)
        ref: https://www.rp-photonics.com/group_velocity_dispersion.html
        Group delay = 1/FSR_Hz. ref: Yijun's thesis
        GVD parameter D = 1/length_m * (d(Group delay)/d(lambda_nm)), gives ps/nm/m
        
        Keyword Args:
           * **wavelength_nm** : wavelength in nm.
           * **FSR_Hz** : free spectrum range in Hz.
           * **length_m** : length of the cavity.
           * **order**: fitting order, by default is 2
        Returns:
           * **GVD** : pandas dataframe. 
                       wavelength_nm_GVD: wavelength in nm.
                       GVD_D: D in ps/nm/m.
                       GVD_beta_2: beta_2 in ps^2/km.
    """
    
    from scipy.optimize import curve_fit

    group_delay_ps = 1/FSR_Hz * 1e12 # in ps
    # # do a fitting for group delay vs. wavelength
    # f_groupdelay_wl_poly = np.polyfit(wavelength_nm,group_delay_ps,order)# first order (GVD) or second order (GVD+third order dispersion) fitting
    # d_groupdelay_over_d_wavelength = 2*f_groupdelay_wl_poly[0]*wavelength_nm + f_groupdelay_wl_poly[1]
    
    # Polynomial fitting for group delay vs. wavelength
    f_groupdelay_wl_poly = np.polyfit(wavelength_nm, group_delay_ps, int(order))# first order (GVD) or second order (GVD+third order dispersion) fitting
    f_groupdelay_wl = np.poly1d(f_groupdelay_wl_poly)
    
    # First derivative of the group delay polynomial
    f_groupdelay_wl_deriv = np.polyder(f_groupdelay_wl)
    d_groupdelay_over_d_wavelength = f_groupdelay_wl_deriv(wavelength_nm)
    
    
    
    
    GVD_D = 1/length_m * (d_groupdelay_over_d_wavelength)#ps/nm/m
    # beta_2 = -lbd^2/(2*pi*C)*D
    GVD_beta_2 = - (wavelength_nm**2/(2*np.pi*C)) * GVD_D
    GVD_beta_2 = GVD_beta_2 * 1e6#ps^2/km
    GVD = pd.DataFrame(data = {'wavelength_nm_GVD':wavelength_nm,
                               'GVD_D':GVD_D,
                               'GVD_beta_2':GVD_beta_2})
    return GVD

##############  try to define a localized fitting function ###############
def f_locatized(nu_res,signal_res,bounds):
    """ A localized fitting fucntion. The localized grating response is fitted by polynomial function
        The resonance responce is fitted by Lorentzian fucntion
        Comments by Yijun
        Args:
           * **nu_res,signal_res** (array): the mesurements around a given resonance nu
           * **bounds** (array): bounds for Lorentzian fitting
        Ref:
            
    """
    f_coupler_poly = np.polyfit(nu_res,signal_res,1)
    f_coupler = np.poly1d(f_coupler_poly)
    
    def f_func_Lorentzian(lbd, A, HWHM, lbd_res):
        """ Lorentzian fitting function for degenerate resonance
            Comments by Yijun wl or freq. don't matter
            Args:
               * **lbd** (float): wavelength in nm
               * **A** (float): parameter to find extinction ratio
               * **HWHM** (float): half width at half maximum
               * **lbd_res** (float): resonance frequency
            Ref:
                https://doi.org/10.1364/OE.381224
        """
        return -A / (1+((lbd-lbd_res)/HWHM)**2) + f_coupler(lbd)
    popt, pcov = curve_fit(f_func_Lorentzian,nu_res,signal_res, bounds=bounds)
    return popt, pcov, f_func_Lorentzian,f_coupler

def resonances_fitting(wavelength_nm,transmission_linear,idx_res,peak_range_nm = 1,A_margin = 0.02):
    """ To fit the resonance with Lorentzian model.
        Comments by Yijun
        Args:
           * **wavelength_nm** (np.array or pd.Series): wavelength in nm
           * **transmission** (np.array or pd.Series): transmission in linear scale.
           * **idx_res** (pd.Series): the index that indicates where the resonances are.
           * **peak_range_nm** (float): the wavelength range around one resonance that can define a peak.
           * **A_margin** (float): a margin for A fitting, the amplitude of the peak

           
        Outputs:
           * **peaks_f_param** (pd.dataframe): parameters for the Lorentizan fitting
                                             'frequency_0_f':nu_peak,central frequency
                                             'FWHM_f':FWHM,
                                             'A_f':A,
                                             'Q_f':Q,
           * **peaks_fitted** (pd.dataframe): sampled fitted peaks.
                                               'nu_samples': nu_samples_list,
                                               'T_f_Lorentzian': T_f_Lorentzian_list
    """
    ##########################################################################
    nu_peak_array = np.arange(np.size(idx_res),dtype=float)
    Q_array = np.arange(np.size(idx_res),dtype=float)
    FWHM_array = np.arange(np.size(idx_res),dtype=float)
    A_array = np.arange(np.size(idx_res),dtype=float)
    R_ext_array = np.arange(np.size(idx_res),dtype=float)
    # Lists to store nu_res and T_f_Lorentzian from all resonances
    nu_samples_list = []
    T_f_Lorentzian_list = []
    
    i = 0
    for idx in idx_res:
        """ Do fitting using Lorentian distribution for each resonance
            Comments by Yijun:
            1. select the resonance range nu_res
            2. find transmission data in the resonance range transmission_res
            
        """
        # peak_range_nm = 1 #OHYP13, nm around each resonance
        nu = wl2nu(wavelength_nm)
        nu_res_1 = wl2nu(wavelength_nm[idx]+peak_range_nm/2)
        nu_res_2 = wl2nu(wavelength_nm[idx]-peak_range_nm/2)
        #nu_res = np.linspace(nu_res_1, nu_res_2,128)
        idx_res = np.where((nu>nu_res_1) & (nu<nu_res_2))# idx_res is a tuple
        idx_res = idx_res[0]# idx_res is a np array
        # idx_res.tolist()
        nu_res = nu[idx_res]
        transmission_res = transmission_linear[idx_res]

        
        # try to refind the best bounds for Lorentian fitting
        A_upper = 1-transmission_linear[idx] + A_margin
        A_lower = 0.01
        # bounds set for (A, HWHM, lbd_res)
        bounds = ([A_lower,0,wl2nu(wavelength_nm[idx]+0.02)],[A_upper,0.1*1e12,wl2nu(wavelength_nm[idx]-0.02)])

        # f_locatized: fitting function using Lorentizian 
        popt, pcov, f_func_Lorentzian,f_coupler = f_locatized(nu_res,transmission_res,bounds)
        
        
        # print('A; half width at half maximum (HWHM)(nm); resonance peak(nm)',*popt,sep='\n',end='\n')
        print(f'A: {popt[0]}\nHalf width at half maximum (HWHM) (Hz): {popt[1]}\nResonance peak (Hz): {popt[2]}\n')


        ''' Calculate Q factor'''
        nu_peak = popt[2]
        FWHM = 2*popt[1]
        A = popt[0]
        Q = nu_peak/FWHM
        nu_peak_array[i] = nu_peak.copy()
        Q_array[i] = Q.copy()# Q factor
        FWHM_array[i] = FWHM # FWHM
        A_array[i] = A# to find out the extinction ratio
        R_ext_array[i] = f_coupler(nu_peak)/(f_coupler(nu_peak)-A)
        print('Q = ',Q,end='\n\n')
        
        # Append current nu_res and corresponding T_f_Lorentzian values
        nu_samples_list.extend(nu_res)
        T_f_Lorentzian_list.extend(f_func_Lorentzian(nu_res, A, FWHM/2, nu_peak))
        
        i = i+1
    
    # Create pandas DataFrame from the collected data: fitted resonances
    peaks_fitted = pd.DataFrame({
        'nu_samples': nu_samples_list,
        'T_f_Lorentzian': T_f_Lorentzian_list
    })
    # prepare a pd.dataframe to record all the parameters
    peaks_f_params = pd.DataFrame({'frequency_0_f':nu_peak_array,
                                'FWHM_f':FWHM_array,
                                'A_f':A_array,
                                'Q_f':Q_array,
                                'R_ext':R_ext_array,
                                })
    
    return peaks_f_params,peaks_fitted

def RemoveOffset_Savgol(wavelength, transmission, idx, window_length=101, polyorder=2, points_to_remove=201):
    """ A Savgol filter is used to distinguish background offset and resonances.
        Comments by Yijun
        Args:
           * **wavelength** (np.array or pd.Series): wavelength in nm
           * **transmission** (np.array or pd.Series): transmission in linear scale.
           * **idx** (pd.Series): the index that indicates where the resonances are.
           * **window_length** (int): window length for Savgol filter, need to be odd number.
           * **polyorder** (int): polynomial order for Savgol filter
           * **points_to_remove** (int): the number of points that will be removed around each resonances when doing Savgol filtering.
        Outputs:
           * **transmission_corrected** (np.array): the transmission where the background offset is removed. The resonance shapes are kept.
           * **offset** (np.array): the back ground offset.
    """
    from scipy.signal import savgol_filter
    # Step 1: Create a mask that excludes 50 points before and after each resonance
    mask = np.ones_like(transmission, dtype=bool)  # Start with all True (include all)
    
    half_window = points_to_remove // 2
    for i in idx:
        # Set the region around each resonance to False (exclude it)
        mask[int(max(0, i - half_window)): int(min(len(transmission), i + half_window + 1))] = False

    # Step 2: Apply the Savitzky-Golay filter to the data without the resonances
    # Filter only the non-resonance regions
    transmission_filtered = transmission.copy()  # Start with a copy to store the results
    polyorder = int(polyorder)
    window_length = int(window_length)
    transmission_filtered[mask] = savgol_filter(transmission[mask], window_length=window_length, polyorder=polyorder)

    # Step 3: Use linear interpolation to fill in the regions around resonances
    transmission_filtered = pd.Series(transmission_filtered)
    transmission_filtered[~mask] = np.nan  # Mark the resonance regions as NaN
    transmission_filtered = transmission_filtered.interpolate(method='cubic').to_numpy() # the interpolate method may be changed to have a better interpolation.

    # Step 4: Subtract the background from the original transmission data
    # transmission_corrected = transmission - transmission_filtered + 1  # Add 1 to maintain baseline around 1
    transmission_corrected = transmission /transmission_filtered 

    offset = np.copy(transmission_filtered)
    return transmission_corrected, offset
    # return transmission_filtered

def fit_fsr_polynomial(nu_res,FSR,order=2,nb_sigma=1):
    """ Fit a polynomial to the FSR data within several-sigma for a more accurate FSR calculation.
        This allow to filter out multiple modes perturbation.
        Comments by Yijun
        Args:
            * **nu_res** (): the resonance frequencies.
            * **FSR** (): free spectrum range.
            * **order** (): order of FSR fitting, 2 default, 3 maximum.
            * **nb_sigma** (float): number of sigma. the FSR inside the mean around this number of sigma will be included for fitting
           
        Outputs:
            * **f_FSR** (?): the function of FSR fitting.
    """
    sigma = np.std(FSR)
    idx_3sigma = np.where((FSR < FSR.mean() + nb_sigma * sigma) & (FSR > FSR.mean() - nb_sigma * sigma))# idx_3sigma is a tuple
    idx_3sigma = idx_3sigma[0]

    f_FSR_poly = np.polyfit(nu_res[idx_3sigma], FSR[idx_3sigma],order)
    f_FSR = np.poly1d(f_FSR_poly)
    return f_FSR
def fit_fsr_polynomial_robust(nu_res, FSR, order=2, nb_sigma=1, max_iter=5):
    """ 
    Fit a polynomial to the FSR data while filtering out outliers.
    This allows ignoring multiple modes perturbations (bossing).
    
    Args:
       * nu_res (array-like): The resonance frequencies.
       * FSR (array-like): Free spectral range values.
       * order (int): Order of the polynomial for FSR fitting (default 2).
       * nb_sigma (float): Sigma threshold for outlier removal (default 1).
       * max_iter (int): Maximum number of iterations for robust fitting (default 5).
       
    Returns:
       * f_FSR (np.poly1d): The robust polynomial fit function for FSR.
       * inliers (array-like): Boolean array indicating inliers used in the final fit.
    """
    inliers = np.ones(len(FSR), dtype=bool)  # Start with all points as inliers
    
    for _ in range(max_iter):
        # Perform polynomial fit on the inliers
        f_FSR_poly = np.polyfit(nu_res[inliers], FSR[inliers], order)
        f_FSR = np.poly1d(f_FSR_poly)
        
        # Calculate residuals and standard deviation of residuals
        residuals = FSR - f_FSR(nu_res)
        sigma_residuals = np.std(residuals[inliers])
        
        # Update inliers: keep points within nb_sigma * sigma_residuals
        new_inliers = np.abs(residuals) < nb_sigma * sigma_residuals
        
        # Stop if inliers don't change
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers

    # Final polynomial fit using refined inliers
    f_FSR_poly = np.polyfit(nu_res[inliers], FSR[inliers], order)
    f_FSR = np.poly1d(f_FSR_poly)
    
    return f_FSR, inliers

def calculate_dispersion(wl_fitting, ng_fitting):
    """ 
    Calculate dispersion from group index variation over wavelength.
    
    Args:
       * **wl_fitting** (array like): wavelength in nm.
       * **ng_fitting** (array like): group index.
       
    """
    
    D = np.diff(ng_fitting) / np.diff(wl_fitting) / C
    return D * 1e12  # ps/nm/m

def calculate_loss_coupling(nu_peak_array, Finesse_array, R_ext_array, Diameter, isDistinguishable=1, wl_critical=1535):
    """
    Calculate loss, coupling, and other values based on input arrays.
    ref: https://doi.org/10.1364/OE.17.018971
    Extracting coupling and loss coefficients from a ring resonator.
    Args:
       * **nu_peak_array**: array of frequency resonance positions
       * **Finesse_array**: array of finesse values
       * **R_ext_array**: array of extinction ratios values
       * **Diameter**: the resonator diameter
       * **isDistinguishable**: flag to distinguish loss and coupling
       * **wl_critical**: critical wavelength to distinguish between loss and coupling
    
    Returns:
       * **root_1**: array of coupling values
       * **root_2**: array of loss values
       * **loss_dBcm**: array of loss in dB/cm
       * **loss_dBcm_mean**: mean loss in dB/cm
    """
    
    # Compute A, B, root_1, and root_2
    A = np.cos(np.pi/Finesse_array) / (1 + np.sin(np.pi/Finesse_array))
    B = 1 - (1 - np.cos(np.pi/Finesse_array)) / (1 + np.cos(np.pi/Finesse_array)) * 1/R_ext_array[:-1]
    root_2 = np.sqrt(A/B) + np.sqrt(A/B - A)
    root_1 = np.sqrt(A/B) - np.sqrt(A/B - A)

    # Manually distinguish the loss and coupling if necessary
    nu_peak_array = nu_peak_array[:-1]
    if isDistinguishable == 1:
        nu_critical = wl2nu(wl_critical)
        idx_nu_inf = np.where((nu_peak_array < nu_critical))[0]
        idx_nu_sup = np.where((nu_peak_array >= nu_critical))[0]
        root_1_distinguished = np.concatenate((root_2.copy()[idx_nu_sup], root_1.copy()[idx_nu_inf]))
        root_2_distinguished = np.concatenate((root_1.copy()[idx_nu_sup], root_2.copy()[idx_nu_inf]))
        root_1 = root_1_distinguished
        root_2 = root_2_distinguished

    # Calculate loss in dB/cm
    loss_dBcm = np.log10(root_2**2) * 10 / (np.pi * Diameter / 10000)
    loss_dBcm_mean = np.log10((root_2.mean())**2) * 10 / (np.pi * Diameter / 10000)
    
    return root_1, root_2, loss_dBcm, loss_dBcm_mean

def calculate_extinction_ratio(nu_res_array, A_array, f_coupler):
    """ Calculate extinction ratio R_ext for coupling/loss coefficient calculation.
        Comments by Yijun
        Args:
           * **nu_res_array** (): 
           * **A_array** (): 
           * **f_coupler** (): 
           
        Outputs:
           * **?** (?): ?
    """
    R_ext_array = f_coupler(nu_res_array) / (f_coupler(nu_res_array) - A_array)
    return R_ext_array

def calculate_finesse(FSR_peak_array, FWHM_array):
    """Calculate finesse as FSR/FWHM."""
    return FSR_peak_array / FWHM_array



def analysis_main(T,
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
                  Param_legend_twinx = {'bbox_to_anchor':(1.2, 0.5),
                                        'loc':'upper left',
                                        'borderaxespad':0.0, 
                                        'shadow':True},
                  Title = 'Not defined',
                  Addition_dict:dict = {},
                  **kwargs):
    """ Do analysis.
        Comments by Yijun
        Args:
           * **T** (pd.dataframe): 
           * **?** (): 
           * **?** (): 
           * **kwargs** ():other 
           
        Outputs:
           * **?** (?): ?
    """
    # FileName = 'test_offsetremoved.csv'
    # FileName = 'test_W800_R100um_G600.csv'
    # # FileName = 'W1100_R100um_G500nm.txt'
    # # FileName = 'W1100_R100um_G600nm.txt'
    # # FileName = '/Users/yangyijun/Library/CloudStorage/OneDrive-Personal/1A_thesis/Analysis_Transmission_GraphicInterface/Transmission mesurements/test_W1100_R100um_G400_BeforeAnnealing.csv'
    # FileName = '/Users/yangyijun/Library/CloudStorage/OneDrive-Personal/1A_thesis/Analysis_Transmission_GraphicInterface/Transmission mesurements/test_W1100_R100um_G400_AfterAnnealing.csv'
    # # T = load_data(FileName,range_wl=[1500,1574])
    # try:
    #     T = load_data(FileName,range_wl=Param_RingResonator['range_wl'],wl_name='wavelength',data_name='transmission')
    # except ValueError:
    #     # T = load_data(FileName,range_wl=[1500,1609],wl_name='L',data_name='1')
    #     print('Loading data fails!\n')
            

    Diamter = Param_RingResonator['diameter']
    RoundTrip = np.pi*Diamter*1e-6
    Fig_FSR,(ax_T,ax_FSR,ax_dispersion) = plt.subplots(nrows =3,ncols=1,sharex=True,sharey = False,figsize = [10,8.5])
    manager = plt.get_current_fig_manager()
    try:
        manager.window.setGeometry(350, 50, 800, 800)  # x, y, width, height
    except AttributeError:
        print("Setting the window position is not supported in this backend.")
    
    
    ax_T.plot(T['wavelength_nm'],T['T_linear'],)
    ax_T.set_title(Title)
    ax_T.set_ylabel('T')
    '''
    find peaks
    '''
    peaks = find_peaks_for_FSR(T['nu_Hz'],T['T_linear'],**Param_find_peaks)
    ax_T.plot(nu2wl(T['nu_Hz'][peaks['idx_peaks']]),T['T_linear'][peaks['idx_peaks']],"x",label = 'resonance')
    ax_T.legend()
    # ax_T.legend(**Param_legend)

    
    '''
    calculate free spectrum range in Hz, FSR
    '''
    FSR = calcul_FSR(T['nu_Hz'],T['T_linear'],peaks['idx_peaks'])
    ax_FSR.plot(FSR['wavelength_nm_FSR'],FSR['FSR_Hz']*1e-12,"o",label = 'FSR')
    ax_FSR.set_ylabel('FSR (THz)')
    ax_FSR_ng = ax_FSR.twinx()
    ng = calcul_ng(FSR['FSR_Hz'],RoundTrip,cavity_type='ring')
    ax_FSR_ng.plot(FSR['wavelength_nm_FSR'],ng,'s',color='tab:orange',label='$n_g$')
    ax_FSR_ng.set_ylabel(r'$n_g$')
    # ax_FSR_ng.legend(**{'bbox_to_anchor':(1.05,1),
    #                 'loc':'lower left',
    #                 'borderaxespad':0.0, 
    #                 'shadow':True})
    ax_FSR_ng.legend()
    ax_FSR.legend()

    '''
    calculate the dispersion
    '''
    GVD = calcul_dispersion(FSR['wavelength_nm_FSR'],FSR['FSR_Hz'],RoundTrip)
    ax_dispersion.plot(GVD['wavelength_nm_GVD'],GVD['GVD_D'],"o",label = 'GVD_D')
    ax_dispersion.set_xlabel('wavelength (nm)')
    ax_dispersion.set_ylabel('D (ps/nm/m)')
    ax_dispersion_beta_2 = ax_dispersion.twinx()
    ax_dispersion_beta_2.plot(GVD['wavelength_nm_GVD'],GVD['GVD_beta_2'],"s",color='tab:orange',label = r'$\beta_2$')
    ax_dispersion_beta_2.set_ylabel(r'$\beta_2$ ($ps^2/km$)')
    ax_dispersion_beta_2.legend()
    # ax_dispersion.legend()
    Fig_FSR.tight_layout()
    
    '''
    a new figure to show the group delay from FSR
    '''
    Fig_GD, (ax_GD) = plt.subplots(nrows = 1,ncols=1,figsize = [7,3])
    ax_TR = ax_GD.twinx() # a twin x axis for corresponded T_T and T_R
    GD = 1/FSR['FSR_Hz']# second
    ax_GD.plot(FSR['frequency_Hz_FSR'],GD * 1e12,".",label = 'group delay')
    ax_GD.set_ylabel('group delay (ps)')
    ax_GD.set_xlabel('frequency (Hz)')
    secax_GD = ax_GD.secondary_xaxis('top', functions=(nu2wl, wl2nu))
    secax_GD.set_xlabel('wavelength nm')
    # ax_GD.plot(FSR['wavelength_nm_FSR'],GD * 1e12,"o",label = 'group delay')
    ax_GD.set_ylim([5,15])
    # show T_T and T_R
    if 'T_T' in Addition_dict.keys() and 'T_R' in Addition_dict.keys():
        T_T = Addition_dict['T_T']
        T_R = Addition_dict['T_R']
        ax_TR.plot(T_T['nu_Hz'],T_T['T_linear'],'--',label = 'T_T',color = 'grey',alpha = 0.5)
        ax_TR.plot(T_R['nu_Hz'],T_R['T_linear'],'--',label = 'T_R',color = 'orange',alpha = 0.5)
        # the total collected energy
        # choose one point every 10 points
        downsampled_nu_Hz = T_T['nu_Hz'][::70]
        downsampled_total_T_linear = (T_T['T_linear'] + T_R['T_linear'])[::70]
        ax_TR.plot(downsampled_nu_Hz,downsampled_total_T_linear,'-',label = 'total',color = 'black',alpha = 0.5)
        # ax_TR.plot(T_T['nu_Hz'],T_T['T_linear']+T_R['T_linear'],'-',label = 'total',color = 'green',alpha = 0.5)    
        ax_TR.set_ylim([-0.05,1.05])
    # add a fitting to the group delay
    isFitting = True
    if isFitting:
        # if Addition_dict has the key Params_GD
        if 'Params_GD' in Addition_dict.keys():
            Params_GD = Addition_dict['Params_GD']
            mask = (FSR['frequency_Hz_FSR'] <= wl2nu(Params_GD['wl_fit_start'])) & (FSR['frequency_Hz_FSR'] >= wl2nu(Params_GD['wl_fit_end']))
        else:
            mask = (FSR['frequency_Hz_FSR'] <= wl2nu(1515)) & (FSR['frequency_Hz_FSR'] >= wl2nu(1565))
        filtered_nu = FSR['frequency_Hz_FSR'][mask]
        filtered_GD_r = GD[mask]

        # Perform linear fit
        fit_params = np.polyfit(filtered_nu, filtered_GD_r, 1)  # Linear fit (degree=1)
        fit_line = np.polyval(fit_params, filtered_nu)
        fit_endpoints = np.polyval(fit_params, [filtered_nu.iloc[0], filtered_nu.iloc[-1]]) # in seconds
        slope_ps_nm =(fit_endpoints[1]-fit_endpoints[0]) * 1e12 / (nu2wl(filtered_nu.iloc[-1]) - nu2wl(filtered_nu.iloc[0])) # in ps/nm

        # Calculate residuals
        residuals = (filtered_GD_r - fit_line)*1e12

        # Calculate standard error of the estimate
        n = len(filtered_GD_r)
        standard_error = np.sqrt(np.sum(residuals**2) / (n - 2))
        ax_GD.plot(filtered_nu, fit_line*1e12, 'r--',
                   label=f'Linear Fit:\n{np.real(fit_params[0]*1e12*1e12):.3f} $ps^2$ \n STD: {np.real(standard_error):.3f} ps \n slope: {slope_ps_nm:.3f} ps/nm')
        # print(slope_ps_nm)
        # add legend
        # ax2.legend()
    ax_GD.legend(**Param_legend)
    # title
    ax_GD.set_title(Title)
    Fig_GD.tight_layout()
    '''
    a new figure to shown fitting for each resonances
    '''
    Fig_resonance,(ax_r1,ax_r2) = plt.subplots(nrows = 2,ncols=1,sharex=True,sharey = False,figsize = [10,6])
    manager = plt.get_current_fig_manager()
    try:
        manager.window.setGeometry(1150, 50, 700, 600)  # x, y, width, height
    except AttributeError:
        print("Setting the window position is not supported in this backend.")
    transmission_corrected, offset = RemoveOffset_Savgol(T['wavelength_nm'],T['T_linear'],peaks['idx_peaks'],
                                                         **Param_Savgol_fitting)
    ax_r1.plot(T['wavelength_nm'],T['T_linear'],'o',label=r'original $T_{linear}$',alpha = 0.5,color='gray')
    ax_r1.plot(T['wavelength_nm'],offset,'--',label = 'background offset')
    # Format the dictionary into a string
    param_str = ',\n'.join(f'{k}={v}' for k, v in Param_Savgol_fitting.items())
    ax_r2.plot(T['wavelength_nm'],transmission_corrected, '--',label = f'corrected $T_{{linear}}$ \n{param_str}')
    ax_r2.set_xlabel('wavelength (nm)')
    ax_r2.set_ylabel('T')
    # ax_r2.legend(**Param_legend)
    ax_r2.legend()

    ax_r2.set_title('corrected')
    ax_r1.set_xlabel('wavelength (nm)')
    ax_r1.set_ylabel('T')
    # ax_r1.legend(**Param_legend)
    ax_r1.legend()

    ax_r1.set_title('Transimission correction')
    Fig_resonance.tight_layout()
    
    
    
    '''
    a figure to shown fitting quality for resonances
    '''
    peaks_f_params,peaks_fitted_points = resonances_fitting(T['wavelength_nm'],transmission_corrected,peaks['idx_peaks'],**Param_peaks_fitting)
    
    Fig_peak_fitting,(ax_pf1,ax_pf2) = plt.subplots(nrows = 2,ncols=1,sharex=True,sharey = False,figsize = [10,6])
    manager = plt.get_current_fig_manager()
    try:
        manager.window.setGeometry(550, 700, 600, 400)  # x, y, width, height
    except AttributeError:
        print("Setting the window position is not supported in this backend.")
    ax_pf1.plot(T['wavelength_nm'],transmission_corrected,label='sampled points')
    ax_pf1.plot(nu2wl(peaks_fitted_points['nu_samples']),peaks_fitted_points['T_f_Lorentzian'],'o',
                alpha=0.5,label='fitted points')
    ax_pf1.set_ylim([0,1.2])
    ax_pf1.set_ylabel('T')
    ax_pf1.set_title('Lorentzian fitting for peaks')
    # ax_pf1.legend(**Param_legend)
    ax_pf1.legend()

    
    ax_pf2.plot(T['wavelength_nm'][peaks['idx_peaks']],peaks_f_params['Q_f'],'s',label='Q factor')
    ax_pf2.set_ylabel('Q factor')
    ax_pf2.set_xlabel('wavelength (nm)')
    # ax_pf2.legend(**Param_legend)
    ax_pf2.legend()

    
    Fig_peak_fitting.tight_layout()
    
    
    '''
    Calculate loss from Lorentzian fitting of each peaks.
    '''
# =============================================================================
#     # parameters retrived from Lorentzian fitting
#     # all the names with '_array' means it come from direct or indirectly from Laurentzian fitting.
# =============================================================================
    nu_peak_array=peaks_f_params['frequency_0_f'];# center frequency (resonance) of each peaks
    Q_array=peaks_f_params['Q_f'];
    FWHM_array = peaks_f_params['FWHM_f']
    A_array = peaks_f_params['A_f']
    FSR_array = np.abs(np.diff(nu_peak_array))
    R_ext_array = peaks_f_params['R_ext']
# =============================================================================
#     # use a 'robust' polynomial method to fit FSR in Hz with wavelength in nm. This method is resistant to bossing.
#     # inliers (array-like): Boolean array indicating inliers used in the final fit.
# =============================================================================
    f_FSR,inliers = fit_fsr_polynomial_robust(nu2wl(nu_peak_array[:-1]), FSR_array,
                                order = Param_FSR_fitting['fitting_order'],
                                nb_sigma = Param_FSR_fitting['nb_sigma'],)
    
    ax_FSR.plot(nu2wl(nu_peak_array),f_FSR(nu2wl(nu_peak_array))*1e-12,"-",color='blue',label = 'FSR fitting based on peak fitting')
    # FSR from pink fitting nu_res_array whichi is not considered for robust polynomial fitting
    ax_FSR.plot(nu2wl(nu_peak_array[:-1][np.invert(inliers)]), FSR_array[np.invert(inliers)]*1e-12,
                'x',
                color = 'purple',
                label='FSR not considered')
    # FSR from pink fitting nu_res_array whichi is considered for robust polynomial fitting
    ax_FSR.plot(nu2wl(nu_peak_array[:-1][inliers]), FSR_array[inliers]*1e-12,
                'v',
                color = 'blue',alpha=0.5,
                label = 'FSR considered')
    ax_FSR.legend(**Param_legend)
    # ax_FSR.legend()
# =============================================================================
#     # calculate group index from FSR fitting based on Lorentzian fitting on each peaks.
# =============================================================================
    ng_array = calcul_ng(f_FSR(nu2wl(nu_peak_array)),RoundTrip,cavity_type='ring')
    ax_FSR_ng.plot(nu2wl(nu_peak_array),ng_array,'-',color='red',label='$n_g$ fitting based on peak fitting')
    ax_FSR_ng.legend(**Param_legend_twinx)
    # ax_FSR_ng.legend()

# =============================================================================
#     # Recalculate GVD parameters base on FSR fitting, which is based on Larentzian fitting on each peaks. 
#     # This should be more precise than previous GVD parameters calculation based on raw measurement.
# =============================================================================
    GVD_array = calcul_dispersion(nu2wl(nu_peak_array[:-1][inliers]),FSR_array[inliers],RoundTrip,order = Param_FSR_fitting['fitting_order'])
    ax_dispersion.plot(GVD_array['wavelength_nm_GVD'],GVD_array['GVD_D'],
                       color='blue',label='GVD D from fitting')
    ax_dispersion_beta_2.plot(GVD_array['wavelength_nm_GVD'],GVD_array['GVD_beta_2'],
                              color='red',label=r'$\beta_2$ from fitting')
    ax_dispersion.legend(**Param_legend)
    ax_dispersion_beta_2.legend(**Param_legend_twinx)
    # ax_dispersion.legend()
    # ax_dispersion_beta_2.legend()
    Fig_FSR.tight_layout()

    
    
# =============================================================================
#     # Calculate finesse
# =============================================================================
    Finesse_array = calculate_finesse(FSR_array, FWHM_array[:-1])

    # In the main, call the calculation function
    root_1, root_2, loss_dBcm, loss_dBcm_mean = calculate_loss_coupling(nu_peak_array, Finesse_array, R_ext_array, Param_RingResonator['diameter'],wl_critical=Param_loss_calcul['wl_critical'])
    
    # Now, perform the plotting
    fig_loss = plt.figure(figsize=(8, 3.5))  # plot for peaks and resonances
    
    
    # Subplot for coupling and loss coefficients
    ax_loss = fig_loss.add_subplot(121)
    ax_loss.set_xlabel('frequency (Hz)')
    ax_loss.set_ylabel('Coupling or Loss')
    ax_loss.plot(nu_peak_array[:-1], root_2, 'o', label=r'single-path ampli. trans. $a$')
    ax_loss.plot(nu_peak_array[:-1], root_1, 'o', label=r'self-coupling coef. $t$')
    ax_loss.legend()
    try:
        # ax_loss.set_ylim((0.978, 1.002))
        ax_loss.set_ylim(np.min([np.min(root_2),np.min(root_1)])-0.01, 1.002)
    except (ValueError, SyntaxError):
        print('Yijun says: range ERROR!! May have Nan in loss_1 and loss_2. \n')
        # ax_loss.set_ylim(np.min([np.min(root_2),np.min(root_1)])-0.01, 1.002)

    ax_loss.set_title('coupling and loss coefficients')
    
    # Add secondary x-axis for wavelength
    secax_loss = ax_loss.secondary_xaxis('top', functions=(nu2wl, wl2nu))
    secax_loss.set_xlabel('wavelength nm')
    
    # Subplot for loss alpha
    ax_alpha = fig_loss.add_subplot(122)
    ax_alpha.plot(nu_peak_array[:-1], loss_dBcm, 'o', label='loss')
    ax_alpha.legend()
    try:
        # ax_alpha.set_ylim((-3, 0.1))
        ax_alpha.set_ylim((np.min(loss_dBcm)-1, 0.1))
    except (ValueError, SyntaxError):
        print('Yijun says: range ERROR!! Message from loss_dBcm \n')

    ax_alpha.set_xlabel('frequency (Hz)')
    ax_alpha.set_ylabel('loss dB/cm')
    ax_alpha.grid()
    
    # Add secondary x-axis for wavelength
    secax_alpha = ax_alpha.secondary_xaxis('top', functions=(nu2wl, wl2nu))
    secax_alpha.set_xlabel('wavelength nm')
    
    # Set layout to adjust subplots
    fig_loss.tight_layout()
    
    # Get the screen dimensions and set figure location to the bottom-right
    manager = plt.get_current_fig_manager()
    screen_width, screen_height = manager.window.screen().availableGeometry().width(), manager.window.screen().availableGeometry().height()
    fig_width, fig_height = fig_loss.get_size_inches() * fig_loss.dpi
    
    # Calculate bottom-right coordinates
    x_position = screen_width - fig_width
    y_position = screen_height - fig_height
    
    # Move the figure to the bottom-right corner
    manager.window.setGeometry(int(x_position)+50, int(y_position)-50, int(fig_width), int(fig_height)+50)
    Fig_FSR.tight_layout()
    # Fig_FSR.close()
    plt.show()
    
    
    return Fig_FSR,Fig_peak_fitting,Fig_resonance,fig_loss

def OffsetRemover(T,T_offset)->pd.DataFrame:
    """
    Remove the offset from the transmission data.
    Args:
       * **T** (pd.dataframe): transmission data.
       * **T_offset** (pd.dataframe): offset data.
       
    Returns:
       * **T_corrected** (pd.dataframe): corrected transmission data.
    """
    # Interpolate the offset data to match the measurement wavelength if needed
    f = interp1d(T_offset['wavelength_nm'], 
                 T_offset['T_linear'], bounds_error=False, fill_value="extrapolate")
    interpolated_offset = f(T['wavelength_nm'])
    T_corrected_linear = T['T_linear'] / interpolated_offset
    T_corrected_dB = 10 * np.log10(T_corrected_linear)

    # interpolated_offset = f(wavelength_meas)
    # T = pd.DataFrame({
    #     'wavelength_nm': wavelength,
    #     'T_dB': transmission,
    #     'T_linear': transmission_linear,
    #     'nu_Hz': nu
    # })

    T_corrected = pd.DataFrame({
        'wavelength_nm': T['wavelength_nm'],
        'T_dB': T_corrected_dB,
        'T_linear': T_corrected_linear,
        'nu_Hz': T['nu_Hz'],
    })
    return T_corrected
'''
the main for the test of each function.
'''
# isTest = 1

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
                           'range_wl':[1500,1600]}
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
    Param_find_peaks = {'distance':60,
                        'prominence':0.0001,}
    # =============================================================================
    # Explanation:
    # window_length: an odd number used for Savgol fitting. This number need to adapted to Fabry perot oscillation at the backgroup of the transmission
    # points_to_remove: number of points around one resonance, which present the whole peak.
    # polyorder: for Savgol filter. Generally don't need to change.
    # =============================================================================
    Param_Savgol_fitting = {'window_length':101,
                            'points_to_remove':101,
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
                  Addition_dict = Addition_dict)   


    

    