import numpy as np
import pandas as pd
from F_LoadData import load_data
from F_ConvertUnits import Heterodyne, nu2wl, time2nu, wl2nu
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths
from F_QFactor import F_QFactor
class SingleHighQ():
    def __init__(self):
        self.x_data = np.array([])
        self.y_data = np.array([])
        # self.T_Data = self.load_data_singleQ()
        # self.T_Offset = self.load_data_Offset()
        # self.T_corrected = self.Remove_Offset()
    def load_data_Offset(self,ax = None):
        """
        Opens a file dialog to load a .txt or .
        """
        file_name = r'231025-ReferenceBackgroud.txt'
        file_name = r'241025-ReferenceBackgroud.txt'

        if file_name:
                try:                
                    T = load_data(file_name, wl_name='frequency', data_name='power1')
                    #change the key names according to the actual file
                    T = T.rename(columns={'wavelength_nm':'nu_Hz', 'nu_Hz':'wavelength_nm',})
                    # convert nu_Hz from GHz to Hz
                    T['nu_Hz'] = T['nu_Hz'] * 1e9
                    T['wavelength_nm'] = nu2wl(T['nu_Hz']) # convert to nm
                    

                    # rearrange columns
                    T['Pe_dBm'] = T['T_dB'].copy() # electrical power in dBm
                    T['T_dB'] = T['Pe_dBm'] * 1 # optical power = (electrical power)*0.5 in dB
                    T['T_linear'] = 10**(T['T_dB']/10) # convert dB to linear scale
                    if ax:
                        import matplotlib.pyplot as plt
                        # Fig = plt.figure()
                        # plot
                        # ax = Fig.add_subplot(111)
                        ax.plot(T['nu_Hz'], T['Pe_dBm'],label='offset')
                        # plt.plot(T['nu_Hz'], T['T_dB'])
                        ax.set_xlabel('Frequency (Hz)')
                        ax.set_ylabel('Electrical Power (dBm)')
                        ax.set_title('Measurement')
                        
                        # print(T.head())

                    return T
                except Exception as e:
                    print(f"Error loading file: {e}")
    def load_data_singleQ(self,file_name = None, ax = None,
                          model = 'Heterodyne',ScanParams:dict = None)-> pd.DataFrame:
            """
            Opens a file dialog to load a .txt or .csv file.
            Assumes 2-column data (x, y) separated by comma, space, or tab.
            Inputs:
               * **file_name** (str): Path to the data file. If None, a default file will be used.
               * **ax** (matplotlib.axes._axes.Axes): Axes object for plotting. If None, no plot will be made.
               * **model** (str): Model type for data conversion. Options are 'Heterodyne', 'FineScan', or default.
               * **ScanParams** (dict): Dictionary containing scan parameters.
            returns:    **T**: DataFrame containing the loaded data.
            **T['nu_Hz']**: Hz
            **T['wavelength_nm']**: nm
            **T['Pe_dBm']**: electrical power in dBm
            **T['T_dB']**: optical power in dB
            **T['T_linear']**: optical power in linear scale

            """
            # file_name, _ = QFileDialog.getOpenFileName(
            #     self, "Open Data File", "", "Text Files (*.txt *.csv)"
            # )
            if file_name is None:
                file_name = r'D80-G400-W1598.679-0-1.5(241025).txt'
                file_name = r'D75-G400-W1549.524-2.5-3.9.txt'
            # file_name = r'D75-G400-W1600.014-1.7-2.2.txt'
            # file_name = r'D75-G400-W1600.014-0-3.txt'
            """
            if file_name:
                try:                
                    # file_name = r'D80-G400-W1598.679-0-1.5(241025).txt'
                    T = load_data(file_name, wl_name='frequency', data_name='power1')
                    #change the key names according to the actual file
                    T = T.rename(columns={'wavelength_nm':'nu_Hz', 'nu_Hz':'wavelength_nm',})
                    # convert nu_Hz from GHz to Hz
                    T['nu_Hz'] = T['nu_Hz'] * 1e9
                    T['wavelength_nm'] = nu2wl(T['nu_Hz']) # convert to nm
                    

                    # rearrange columns
                    T['Pe_dBm'] = T['T_dB'].copy() # electrical power in dBm
                    T['T_dB'] = T['Pe_dBm'] * 1 # optical power = (electrical power)*0.5 in dB
                    T['T_linear'] = 10**(T['T_dB']/10) # convert dB to linear scale

                    self.x_data = T['nu_Hz'].values
                    self.y_data = T['T_dB'].values
                    if ax:
                        import matplotlib.pyplot as plt
                        ax.plot(T['nu_Hz'], T['Pe_dBm'],label='resonance')
                        # plt.plot(T['nu_Hz'], T['T_dB'])
                        ax.set_xlabel('Frequency (Hz)')
                        ax.set_ylabel('Electrical Power (dBm)')
                    
                    # print(T.head())

                    
                    return T
                except Exception as e:
                    print(f"Error loading file: {e}")
            """
            # switch
            if model == 'Heterodyne':
                T = load_data(file_name, wl_name='frequency', data_name='power1',
                    f_convert=lambda nu, Pe: Heterodyne(nu*1e9, Pe))
            elif model =='FineScan':
                # params = {'Vpp_V':10,'Freq_Hz':10*1e-3,'Tunning_Hz_V':300*1e6}
                if ScanParams is not None:
                    params = ScanParams
                else:
                    params = {'Vpp_V':4,'Freq_Hz':5*1e-3,'Tunning_Hz_V':300*1e6}
                T = load_data(file_name, wl_name='Time(s)', data_name='ct400.detectors.P_detector1(dBm)',
                f_convert=lambda t, d: time2nu(t, d, Params=params))
            else:# default
                T = load_data(file_name, wl_name='L', data_name='1')   
                 
            if ax:
                import matplotlib.pyplot as plt
                ax.plot(T['nu_Hz'], T['T_dB'],label='resonance')
                # plt.plot(T['nu_Hz'], T['T_dB'])
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power (dB)')
            return T
    def Remove_Offset(self,
                      ax:matplotlib.axes._axes.Axes = None)-> pd.DataFrame:
         """Remove the offset from the single Q data using the offset data.
         Steps:
            1. Interpolate the offset data to match the x_data of the single Q data.
            2. Subtract the interpolated offset from the single Q y_data.
         """
         if self.T_Offset is not None and self.T_Data is not None:
            from scipy.interpolate import interp1d
            # Create interpolation function for the offset data
            interp_func = interp1d(self.T_Offset['nu_Hz'], self.T_Offset['Pe_dBm'], kind='linear', fill_value="extrapolate")
            # Interpolate offset values at the x_data points
            # offset_values = interp_func(self.x_data)
            offset_values = interp_func(self.T_Data['nu_Hz'])
            # Subtract offset from y_data
            # corrected_y_data = self.y_data - offset_values
            corrected_y_data = self.T_Data['Pe_dBm'] - offset_values

            # Create a new DataFrame to hold the corrected data
            T_corrected = self.T_Data.copy()
            T_corrected['Pe_dBm'] = corrected_y_data
            T_corrected['T_dB'] = T_corrected['Pe_dBm'] * 1 # optical power = (electrical power)*0.5 in dB
            T_corrected['T_linear'] = 10**(T_corrected['T_dB']/10) # update linear scale as well
            if ax:
                # plot corrected data
                import matplotlib.pyplot as plt
                # Fig = plt.figure()
                # plot
                # ax = Fig.add_subplot(111)
                ax.plot(T_corrected['nu_Hz'], T_corrected['T_linear'],label='corrected T',alpha=0.7)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('T linear scale')
                ax.set_title('Corrected optical power after offset removal')
            return T_corrected
    
    def Q_RemoveOffset_Savgol(self,T_corrected:pd.DataFrame,
                              idx_peaks:np.ndarray,
                              points_to_remove:int = 50,
                              window_length:int=101,
                              polyorder:int=2,
                              ax:matplotlib.axes._axes.Axes = None)-> tuple[pd.DataFrame, np.ndarray]:
        """ Remove the background offset using Savgol filter.
        Args:
           * **T_corrected** (pd.DataFrame): DataFrame containing the corrected transmission data. A direct usable T that contains no huge background offset.
           * **idx_peaks** (np.ndarray): the index that indicates where the resonances are.
           * **points_to_remove** (np.int): the number of points that will be removed around each resonances when doing Savgol filtering.
           * **window_length** (int): window length for Savgol filter, need to be odd number.
           * **polyorder** (int): polynomial order for Savgol filter
        Outputs:
           * **transmission_corrected** (np.array): the transmission where the background offset is removed. The resonance shapes are kept.
           * **offset** (np.array): the back ground offset. The same as dimension as transmission_corrected.
        """
        from F_RemoveOffsetSavgol import RemoveOffset_Savgol
        # T_corrected = self.T_corrected
        T_normalised = T_corrected.copy()
        T_normalised['T_linear'], offset = RemoveOffset_Savgol(
            T_corrected['T_linear'].to_numpy(),
            idx_peaks,
            points_to_remove=points_to_remove,
            window_length=window_length,
            polyorder=polyorder
        )
        T_normalised['T_dB'] = 10 * np.log10(T_normalised['T_linear'])
        # delete T_normalised['Pe_dBm']
        T_normalised = T_normalised.drop(columns=['Pe_dBm'], errors='ignore')
        if ax:
            # ax.plot(T_corrected['nu_Hz'], T_normalised,label='Savgol corrected T',alpha=0.7)

            # ax = Fig_Savgol.add_subplot(111)
            # ax.plot(transmission, label='Original Transmission', alpha=0.5, linestyle='--')
            ax.plot(T_corrected['nu_Hz'], offset, label='Savgol Fitted Offset/filtered', alpha=0.7, linestyle='--')
            # ax.plot(T_corrected['nu_Hz'],T_normalised['T_linear'], label='Corrected Transmission', alpha=1,linestyle='dotted')
            # ax.set_title('Savgol Filter Background Removal')
            ax.legend()
        return T_normalised, offset
    
    
    def Q_FindPeak(self,T_corrected:pd.DataFrame,
                   ax:matplotlib.axes._axes.Axes,
                   **param_find_peaks)->tuple[np.ndarray,dict,np.ndarray,np.ndarray,np.ndarray,dict]:
        """
        Find peaks from the corrected optical transmission T_corrected.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes, optional, if provided, the peaks will be plotted on this axis.
        param_find_peaks : dict, optional, {'distance':2000*0.8,'prominence':0.1,'width':5,'rel_height':0.5,}
        
        Returns
        -------
        idx_peaks : np.ndarray
            Indices of the detected peaks in T_corrected['T_linear'].
        properties_peaks : dict
            Properties of the detected peaks as returned by scipy.signal.find_peaks.
        right_ips : np.ndarray
            Right intersection points of the peak widths.
        left_ips : np.ndarray
            Left intersection points of the peak widths.
        width_peaks : np.ndarray
            Widths of the peaks in the same units as T_corrected['nu_Hz'].
        """
        from F_FindPeaks import FindPeaks
        if param_find_peaks:
            param_find_peaks = param_find_peaks
        else:
            param_find_peaks = {'distance':2000*0.8,
                                'prominence':0.1,
                                'width':5,
                                'rel_height':1,}
        # T_corrected = self.T_corrected
        idx_peaks, properties_peaks = FindPeaks(T_corrected['T_linear'],**param_find_peaks)
        
        promences = properties_peaks['prominences']
        # Calculate the height of each peak’s contour line and plot the results
        contour_heights = T_corrected['T_linear'][idx_peaks] + promences
        # Find all peaks and calculate their widths at the relative height of 0.5 (contour line at half the prominence height) 
        # and 1 (at the lowest contour line at full prominence height).
        width_heights = -properties_peaks['width_heights']
        left_ips = properties_peaks['left_ips'].astype(int)
        right_ips = properties_peaks['right_ips'].astype(int)
        width_peaks = T_corrected['nu_Hz'].iloc[right_ips].to_numpy() - T_corrected['nu_Hz'].iloc[left_ips].to_numpy()
        
        
        
        # find width at half prominence, use rel_height=0.5, to get FWHM
        param_find_peaks['rel_height'] = 0.5
        idx_peaks_2,properties_peaks_half = FindPeaks(T_corrected['T_linear'],**param_find_peaks)
        FWHM_peaks = properties_peaks_half['widths'] * (T_corrected['nu_Hz'].iloc[10]-T_corrected['nu_Hz'].iloc[9])
        properties_peaks['FWHM'] = FWHM_peaks


        
        
        
        if ax:
            ax.plot(T_corrected['nu_Hz'][idx_peaks],
                    T_corrected['T_linear'][idx_peaks],
                    'o')
            
            ax.vlines(x=T_corrected['nu_Hz'][idx_peaks], ymin=contour_heights, ymax=T_corrected['T_linear'][idx_peaks],
                        linestyles="dashdot", color="C1",label='prominence')
            
            ax.hlines(width_heights, T_corrected['nu_Hz'].iloc[left_ips].values, T_corrected['nu_Hz'].iloc[right_ips].values, 
                        linestyles='--',color="C2",label=f'width {(width_peaks*1e-6).astype(int)} MHz')
            ax.hlines(-properties_peaks_half['width_heights'], T_corrected['nu_Hz'].iloc[properties_peaks_half['left_ips']].values, 
                        T_corrected['nu_Hz'].iloc[properties_peaks_half['right_ips']].values, 
                        linestyles=':',color="C3",label=f'FWHM{(FWHM_peaks*1e-6).astype(int)} MHz')
            ax.set_title('Peak Finding Results')
            ax.legend()

        return idx_peaks, properties_peaks, right_ips, left_ips, width_peaks, properties_peaks_half
    


    def Q_FindDoubletResonance(self,T_corrected:pd.DataFrame,
                               ax:matplotlib.axes._axes.Axes,
                               **param_find_DoubletResonance)->tuple[np.ndarray,dict,np.ndarray,np.ndarray,np.ndarray]:
        """
        Find peaks of doublet resonance from the corrected optical transmission T_corrected(or even normalised T).

        Parameters
        ----------
        T_corrected : pd.DataFrame. Should better be the normalised T that contains no huge background offset.
        ax : matplotlib.axes._axes.Axes, optional, if provided, the peaks will be plotted on this axis.
        param_find_DoubletResonance : dict, optional, {'distance':2000*0.8,'prominence':0.1,'width':5,'rel_height':0.5,}
        
        Returns
        -------
        idx_peaks : np.ndarray
            Indices of the detected peaks in T_corrected['T_linear'].
        properties_peaks : dict
            Properties of the detected peaks as returned by scipy.signal.find_peaks.
        right_ips : np.ndarray
            Right intersection points of the peak widths.
        left_ips : np.ndarray
            Left intersection points of the peak widths.
        width_peaks : np.ndarray
            Widths of the peaks in the same units as T_corrected['nu_Hz'].
        """
        from F_FindPeaks import FindPeaks
        if param_find_DoubletResonance:
            param_find_DoubletResonance = param_find_DoubletResonance
        else:
            param_find_DoubletResonance = {'distance':10,
                                'prominence':0.1,
                                'width':5,
                                'rel_height':1,}
        # T_corrected = self.T_corrected
        idx_peaks, properties_peaks = FindPeaks(T_corrected['T_linear'],**param_find_DoubletResonance)
        
        promences = properties_peaks['prominences']
        # Calculate the height of each peak’s contour line and plot the results
        contour_heights = T_corrected['T_linear'][idx_peaks] + promences
        # Find all peaks and calculate their widths at the relative height of 0.5 (contour line at half the prominence height) 
        # and 1 (at the lowest contour line at full prominence height).
        width_heights = -properties_peaks['width_heights']
        left_ips = properties_peaks['left_ips'].astype(int)
        right_ips = properties_peaks['right_ips'].astype(int)
        width_peaks = T_corrected['nu_Hz'].iloc[right_ips].to_numpy() - T_corrected['nu_Hz'].iloc[left_ips].to_numpy()
        
        
        
        # find width at half prominence, use rel_height=0.5, to get FWHM
        # param_find_peaks['rel_height'] = 0.5
        # idx_peaks_2,properties_peaks_half = FindPeaks(T_corrected['T_linear'],**param_find_peaks)
        # FWHM_peaks = properties_peaks_half['widths'] * (T_corrected['nu_Hz'].iloc[1]-T_corrected['nu_Hz'].iloc[0])
        # properties_peaks['FWHM'] = FWHM_peaks


        
        
        
        if ax:
            ax.plot(T_corrected['nu_Hz'], T_corrected['T_linear'],'o',alpha=0.5)
            ax.plot(T_corrected['nu_Hz'][idx_peaks],
                    T_corrected['T_linear'][idx_peaks],
                    'o')
            
            ax.vlines(x=T_corrected['nu_Hz'][idx_peaks], ymin=contour_heights, ymax=T_corrected['T_linear'][idx_peaks],
                        linestyles="dashdot", color="C1",label='prominence')
            
            ax.hlines(width_heights, T_corrected['nu_Hz'].iloc[left_ips].values, T_corrected['nu_Hz'].iloc[right_ips].values, 
                        linestyles='--',color="C2",label=f'width {(width_peaks*1e-6).astype(int)} MHz')
            # ax.hlines(-properties_peaks_half['width_heights'], T_corrected['nu_Hz'].iloc[properties_peaks_half['left_ips']].values, 
            #             T_corrected['nu_Hz'].iloc[properties_peaks_half['right_ips']].values, 
            #             linestyles=':',color="C3",label=f'FWHM{(FWHM_peaks*1e-6).astype(int)} MHz')
            ax.set_title('Doublet Peak Finding Results')
            ax.legend()
        return idx_peaks, properties_peaks, right_ips, left_ips, width_peaks#, properties_peaks_half
    
    def Q_FitDoubletResonance(self, T_normalised: pd.DataFrame,
                              center_wl_nm:float = 1550,
                              param_find_DoubletResonance: dict = None,
                              param_rel_Doublet: dict = None,
                              ax_Doublet: matplotlib.axes._axes.Axes = None
                              )-> tuple[np.ndarray, np.ndarray, callable]:
        """
        Fit doublet resonance from the normalised optical transmission T_normalised.
        Parameters:
              * **T_normalised** (pd.DataFrame): DataFrame containing the normalised optical transmission data.
              * **center_wl_nm** (float): Center wavelength in nanometers.
              * **param_find_DoubletResonance** (dict, optional): Parameters for doublet peak finding.
              * **param_rel_Doublet** (dict, optional): Relative parameters for doublet fitting.
              * **ax_Doublet** (matplotlib.axes._axes.Axes, optional): Axes object for plotting.
        Returns:
            * **opt_Doublet** (np.ndarray): Optimized parameters for the doublet fit.
            * **pcov_Doublet** (np.ndarray): Covariance of the optimized parameters.
            * **f_func_DoubleLorentzian** (callable): Fitted double Lorentzian function.
        """
        # 1st: peak finding
        if param_find_DoubletResonance is not None:
            param_find_DoubletResonance = param_find_DoubletResonance
        else:
            param_find_DoubletResonance = {'distance':10,
                                    'prominence':0.1,
                                    'width':5,
                                    'rel_height':0.5,}
        idx_peaks, properties_peaks, right_ips, left_ips, width_peaks = Q.Q_FindDoubletResonance(
            T_normalised,
            ax=ax_Doublet,
            **param_find_DoubletResonance)
        

        # if there is param_rel, update the Rel_A, Rel_FWHM, Rel_Peak
        if param_rel_Doublet is not None:
            param_rel_Doublet = param_rel_Doublet   
        else:
            param_rel_Doublet = {'Rel_A1':0.5,
                                'Rel_A2':0.5,
                                'Rel_FWHM1':1,
                                'Rel_FWHM2':1,
                                'Rel_Peak1':0.5,
                                'Rel_Peak2':0.5,}
        from F_Bounds import F_Bounds_DoubleLorentzian
        # bound for A1, A2, FWHM1, FWHM2, lbd_res1, lbd_res2
        # A1
        A1_0 = properties_peaks['prominences'].max() # initial guess from peak finding
        # A2_0 = properties_peaks['prominences'][1] # initial guess from peak finding
        A2_0 = A1_0 # the second peak amplitude initial guess set to be the same as the first one
        FWHM1_0 = properties_peaks['widths'].max() * (T_normalised['nu_Hz'].iloc[10]-T_normalised['nu_Hz'].iloc[9]) # initial guess from peak finding
        # FWHM2_0 = properties_peaks['widths'].max() * (T_normalised['nu_Hz'].iloc[1]-T_normalised['nu_Hz'].iloc[0]) # initial guess from peak finding
        FWHM2_0 = FWHM1_0 # set the second FWHM initial guess to be the same as the first one
        FWMH_max = width_peaks.max() # set the max FWHM to be the max width found from peak finding, knowing that re_height=0.5
        FWHM_min = width_peaks.min() # set the min FWHM to be the min width found from peak finding, knowing that re_height=0.5
        Peak1_0 = T_normalised['nu_Hz'].iloc[idx_peaks[0]] # initial guess from peak finding
        Peak2_0 = T_normalised['nu_Hz'].iloc[idx_peaks[1]] # initial guess from peak finding
        bounds_Doublet = F_Bounds_DoubleLorentzian(A1_0=A1_0, Rel_A1=param_rel_Doublet['Rel_A1'],
                                                A2_0=A2_0, Rel_A2=param_rel_Doublet['Rel_A2'],
                                                FWHM1_lower=FWHM_min, 
                                                FWHM1_upper=FWMH_max,
                                                FWHM2_lower=FWHM_min,
                                                FWHM2_upper=FWMH_max,
                                                # FWHM1_0=FWHM1_0, Rel_FWHM1=param_rel_Doublet['Rel_FWHM1'],
                                                # FWHM2_0=FWHM2_0, Rel_FWHM2=param_rel_Doublet['Rel_FWHM2'],
                                                Peak1_0=Peak1_0, Rel_Peak1=param_rel_Doublet['Rel_Peak1'],
                                                Peak2_0=Peak2_0, Rel_Peak2=param_rel_Doublet['Rel_Peak2'],
                                                )
        # step2: fitting using Lorentzian doublet model with bounds
        from F_LorentzianModel import f_locatized
        opt_Doublet, pcov_Doublet, f_func_DoubleLorentzian = f_locatized(
            nu_res=T_normalised['nu_Hz'].to_numpy(),
            signal_res=T_normalised['T_linear'].to_numpy(),
            bounds=bounds_Doublet, model='DoubleLorentzian')
        # Q factor for the first resonance
        Q_factor1 = F_QFactor(
            nu_peak=wl2nu(center_wl_nm),
            FWHM=opt_Doublet[1]
        )
        Q_factor2 = F_QFactor(
            nu_peak=wl2nu(center_wl_nm),
            FWHM=opt_Doublet[4]
        )

        # Evaluate fit quality
        from F_GoodnessOfFit import evaluate_fit_quality, print_fit_report
        fit_results = evaluate_fit_quality(
            xdata=T_normalised['nu_Hz'].to_numpy(),
            ydata=T_normalised['T_linear'].to_numpy(),
            model_func=f_func_DoubleLorentzian,
            popt=opt_Doublet,
            pcov=pcov_Doublet,
            model_name='Double Lorentzian'
        )
        
        # Store fit results for later access
        self.fit_results = fit_results
        
        # Print fit quality report
        param_names = ['Amplitude', 'FWHM (Hz)', 'Resonance (Hz)']
        print_fit_report(fit_results, popt=opt_Doublet, param_names=param_names, 
                        model_name='Double Lorentzian')
        
        if ax_Doublet is not None:
            fit_quality_marker = '✓' if fit_results['is_good_fit'] else '✗'
            label_text = (f'Doublet Fit Q1={Q_factor1*1e-6:.2f}M,\n '
                          f'Q2={Q_factor2*1e-6:.2f}M @ {center_wl_nm}nm\n'
                          f'[{fit_results["fit_quality"]}] {fit_quality_marker}')
            ax_Doublet.plot(T_normalised['nu_Hz'],
                            f_func_DoubleLorentzian(T_normalised['nu_Hz'], *opt_Doublet),
                            label=label_text,
                            linestyle='--',
                            color = 'C4')
            ax_Doublet.set_xlabel('RelFreq (Hz)')
            ax_Doublet.set_ylabel('T linear scale')
            ax_Doublet.legend()

        

        return opt_Doublet, pcov_Doublet, f_func_DoubleLorentzian
    def plot_fit_diagnostics(self, T_normalised: pd.DataFrame, 
                            model_func, 
                            popt: np.ndarray,
                            fit_results: dict = None):
        """
        Create diagnostic plots for fit quality assessment.
        
        Parameters
        ----------
        T_normalised : pd.DataFrame
            Normalized transmission data.
        model_func : callable
            The fitted model function.
        popt : np.ndarray
            Optimized parameters.
        fit_results : dict, optional
            Results from evaluate_fit_quality.
        """
        import matplotlib.pyplot as plt
        from F_GoodnessOfFit import plot_residuals
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Top plot: Data and fit
        ax1 = axes[0]
        ax1.plot(T_normalised['nu_Hz']*1e-6, T_normalised['T_linear'], 
                'o', alpha=0.5, label='Data', markersize=4)
        ax1.plot(T_normalised['nu_Hz']*1e-6, 
                model_func(T_normalised['nu_Hz'], *popt), 
                'r-', linewidth=2, label='Fit')
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylabel('Normalized Transmission')
        ax1.set_title('Fit Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Residuals
        ax2 = axes[1]
        plot_residuals(T_normalised['nu_Hz'].to_numpy(), 
                      T_normalised['T_linear'].to_numpy(),
                      model_func, popt, fit_results, ax=ax2)
        ax2.set_xlabel('Frequency (Hz)')
        
        plt.tight_layout()
        return fig
    
    def PL_FitResonance(self,
                        T_corrected:pd.DataFrame,
                        center_wl_nm:float = 1550,
                        ax_T:matplotlib.axes._axes.Axes = None,
                        ax_T_normalised:matplotlib.axes._axes.Axes = None,
                        param_rel:dict=None,
                        **param_find_peaks,
                        )-> pd.DataFrame:
        """
        A pipe line (PL) to fit each resonance.
        Parameters
        ----------
        T_corrected : pd.DataFrame
            DataFrame containing the corrected transmission data. A direct usable T that contains no huge background offset.
        ax_T : matplotlib.axes._axes.Axes, optional
            Axis to plot the peak finding results.
        ax_T_normalised : matplotlib.axes._axes.Axes, optional
            Axis to plot the normalised transmission and fitting results.
        param_rel : dict, optional
            Relative parameters for fitting bounds. If None, default values will be used.
            * **Rel_FWHM** (float): relative range for FWHM around the initial guess.
            * **Rel_A** (float): relative range for Amplitude around the initial guess.
            * **Rel_Peak** (float): relative range for Peak position around the initial guess.
        """
        ### find the peak
        idx_peaks, properties_peaks, right_ips, left_ips, width_peaks,properties_peaks_half = self.Q_FindPeak(T_corrected, 
                                                                                                              ax = ax_T,
                                                                                                              **param_find_peaks)
        # if didn't find the peak, give a warning and return
        if len(idx_peaks) == 0:
            print("No peaks found in the provided transmission data. Please check the data or adjust peak finding parameters.")
            return
        if len(properties_peaks_half['widths']) == 0:
            print("No FWHM found in the provided transmission data. Please check the data or adjust peak finding parameters.")
            return
        
        ### determine the points to remove around each resonance for Savgol filtering
        points_to_remove = ((right_ips - left_ips)*param_rel['factor_remove_points']).astype(int)

        ### fit the back ground offset using Savgol filter
        from F_RemoveOffsetSavgol import RemoveOffset_Savgol

        T_normalised, offset = self.Q_RemoveOffset_Savgol(
            T_corrected,
            idx_peaks,
            window_length=9,
            polyorder=2,
            points_to_remove=points_to_remove[0],
            ax=ax_T
        )

        # Fig_T_normalised = plt.figure()
        # ax_T_normalised = Fig_T_normalised.add_subplot(111)
        if ax_T_normalised is not None:
            ax_T_normalised.plot(T_normalised['nu_Hz']*1e-6, T_normalised['T_linear'],label='Savgol corrected normalised T',
                                alpha=0.7)
            ax_T_normalised.set_xlabel('rel Freq (MHz)')
            ax_T_normalised.set_ylabel('T linear scale')
            ax_T_normalised.set_title('Normalised Transmission')
            ax_T_normalised.legend()

        ### fitting the resonance to get Q factor and other parameters
        # step 1: choose the bounds for fitting
        # try to refind the best bounds for Lorentian fitting
        from F_Bounds import F_Bounds_SingleLorentzian
        
        # if there is param_rel, update the Rel_A, Rel_FWHM, Rel_Peak
        if param_rel is not None:
            param_rel = param_rel
        else:
            param_rel = {'Rel_FWHM':0.2,
                         'Rel_A':0.2,
                         'Rel_Peak':0.2,
                         'factor_remove_points':1,}
            
        # bound for A, HWHM, lbd_res
        # A
        A0 = properties_peaks['prominences'][0] # initial guess from peak finding
        A0 = A0/offset[idx_peaks[0]] # normalise the A0
        Rel_A = param_rel['Rel_A']
        # FWHM
        FWHM0 = properties_peaks['FWHM'][0] # initial guess from peak finding
        Rel_FWHM = param_rel['Rel_FWHM']     
        # Peak position
        Peak0 = T_normalised['nu_Hz'].iloc[idx_peaks[0]] # initial guess from peak finding
        Rel_Peak = param_rel['Rel_Peak']

        bounds = F_Bounds_SingleLorentzian(
            A0=A0, Rel_A=Rel_A,
            FWHM0=FWHM0, Rel_FWHM=Rel_FWHM,
            Peak0=Peak0, Rel_Peak=Rel_Peak
        )
        # step2: fitting using Lorentzian model with bounds
        from F_LorentzianModel import f_locatized
        opt, pcov, f_func_Lorentzian = f_locatized(
            nu_res=T_normalised['nu_Hz'].to_numpy(),
            signal_res=T_normalised['T_linear'].to_numpy(),
            bounds=bounds, model='SingleLorentzian')
        fitting_options = {
            'A': opt[0],
            'FWHM': opt[1],
            'res': opt[2],
        }
        fitting_pcov = pcov
        self.fitting_options = fitting_options
        self.fitting_pcov = fitting_pcov

        from F_QFactor import F_QFactor
        Q_factor = F_QFactor(
            nu_peak=wl2nu(center_wl_nm),
            FWHM=fitting_options['FWHM']
        )
        
        # Evaluate fit quality
        from F_GoodnessOfFit import evaluate_fit_quality, print_fit_report
        fit_results = evaluate_fit_quality(
            xdata=T_normalised['nu_Hz'].to_numpy(),
            ydata=T_normalised['T_linear'].to_numpy(),
            model_func=f_func_Lorentzian,
            popt=opt,
            pcov=pcov,
            model_name='Single Lorentzian'
        )
        
        # Store fit results for later access
        self.fit_results = fit_results
        
        # Print fit quality report
        param_names = ['Amplitude', 'FWHM (Hz)', 'Resonance (Hz)']
        print_fit_report(fit_results, popt=opt, param_names=param_names, 
                        model_name='Single Lorentzian')
        
        if ax_T_normalised is not None:
            # Update label with fit quality indicator
            fit_quality_marker = '✓' if fit_results['is_good_fit'] else '✗'
            ax_T_normalised.plot(T_normalised['nu_Hz']*1e-6,
                                f_func_Lorentzian(T_normalised['nu_Hz'], *opt),
                                label=f'Lorentzian Fit Q={Q_factor*1e-6:.2f}M@{center_wl_nm}nm [{fit_results["fit_quality"]}] {fit_quality_marker}', 
                                linestyle='--')  
            ax_T_normalised.legend()
        
        return T_normalised
    
if __name__ == "__main__":
     
    import matplotlib.pyplot as plt
    ### an example usage for Heterodyne measurement
    # Fig_Pe = plt.figure()
    # ax_Pe = Fig_Pe.add_subplot(111)
    # Q = SingleHighQ()
    # Q.load_data_Offset(ax = ax_Pe)
    # Q.load_data_singleQ(ax = ax_Pe)
    # ax_Pe.legend()

    # Fig_T = plt.figure()
    # ax_T = Fig_T.add_subplot(111)
    # T_corrected = Q.Remove_Offset(ax = ax_T)
    # ax_T.legend()




    # Fig_T_normalised = plt.figure()
    # ax_T_normalised = Fig_T_normalised.add_subplot(111)

    # Q.PL_FitResonance(
    #     T_corrected=T_corrected,
    #     ax_T_normalised=ax_T_normalised,
    #     ax_T=ax_T)
    

    ### another example usage for Fine Scan measurement
    Q = SingleHighQ()
    Fig_T = plt.figure()
    ax_T = Fig_T.add_subplot(111)
    # T_corrected = Q.load_data_singleQ(file_name=r'C:\Users\yijun.yang\OneDrive\1A_PostDoc\SiN\202509_2509SiN700A_Multipassage\mesurement\Laser fine tunning\D80\RR0.4.txt',
    #                                   model='FineScan',
    #                                   )
    FolderName = r'C:\Users\yijun.yang\OneDrive\1A_PostDoc\SiN\202511SiN700A_4P_HighQ-PC\FineScan Measurement\D75'
    # FolderName = r'/Users/yangyijun/Library/CloudStorage/OneDrive-Personal/1A_PostDoc/SiN/202511SiN700A_4P_HighQ-PC/FineScan Measurement/D75'
    # FolderName = r'C:\Users\yijun.yang\OneDrive\1A_PostDoc\SiN\202511SiN700A_4P_HighQ-PC\FineScan Measurement\D80'
    FileName = r'RRW1.1G0.5L1520.664F10mHzA2V.txt'
    # FileName = r'RRW1.1G0.5L1569.368F10mHzA5V.txt'
    # FileName = r'RRW1.1G0.5L1560.438F10mHzA5V(2).txt'
    FileName = r'RRW2.8G0.5L1628.594F5mHzA2V.txt'
    FileName = r'RRW1.1G0.5L1534.283F10mHzA2V.txt'
    FileName = r'RRW1.1G0.5L1530.243F10mHzA2V.txt'
    FileName = r'RRW1.1G0.5L1609.707F10mHzA2V.txt'
    # RRW1.1G0.5L1570.515F510mHzA2V
    from F_GetConfig import F_GetConfig
    config = F_GetConfig(FileName = FileName)
    ScanConfig = {'Vpp_V':config['amplitude_V']*2,'Freq_Hz':config['freq_mHz']*1e-3,'Tunning_Hz_V':300*1e6}
    # ScanConfig = {'Vpp_V':config['amplitude_V']*2,'Freq_Hz':10*1e-3,'Tunning_Hz_V':300*1e6}
    from os.path import join
    file_name=join(FolderName,FileName)
    T_corrected = Q.load_data_singleQ(file_name=file_name,
                                      model='FineScan',ScanParams=ScanConfig)
    ax_T.plot(T_corrected['nu_Hz'], T_corrected['T_linear'],'o',label='resonance',alpha=0.3)
    
    Fig_T_normalised = plt.figure()
    ax_T_normalised = Fig_T_normalised.add_subplot(111)

    param_find_peaks = {'distance':400,# in number of points
                        'prominence':0.0003,# need to be adjusted according to the actual data
                        'width':5,# in number of points
                        'rel_height':1,# don't change this one
                        }
    param_rel = {'Rel_FWHM':0.2,
                 'Rel_A':0.2,
                 'Rel_Peak':0.2,
                 'factor_remove_points':0.3,# factor to multiply the width to get the points to remove for Savgol filtering
        }
    from F_FindPeaks import F_SelectPeaks
    # param_find_peaks = F_SelectPeaks(T_corrected['T_linear'].values, param_find_peaks, num_peaks=5)
    T_normalised = Q.PL_FitResonance(
        T_corrected=T_corrected,
        center_wl_nm=config['wavelength_nm'],
        ax_T_normalised=ax_T_normalised,
        ax_T=ax_T,
        param_rel=param_rel,
        **param_find_peaks)
    # plt.show()
    

    Fig_Doublet = plt.figure()
    ax_Doublet = Fig_Doublet.add_subplot(111)
    param_find_DoubletResonance = {'distance':11,
                                'prominence':0.1,
                                'width':2.5,
                                'rel_height':0.5,}
    
    param_rel_Doublet = {'Rel_A1':0.99,
                         'Rel_A2':0.99,
                        #  'Rel_FWHM1':0.8,
                        #  'Rel_FWHM2':0.8,
                         'Rel_Peak1': 0.8,
                         'Rel_Peak2':0.8,}
    
    opt_Doublet, pcov_Doublet, f_func_DoubleLorentzian = Q.Q_FitDoubletResonance(T_normalised,
                                                                                center_wl_nm=config['wavelength_nm'],
                                                                                param_find_DoubletResonance=param_find_DoubletResonance,
                                                                                param_rel_Doublet=param_rel_Doublet,
                                                                                ax_Doublet=ax_Doublet)
    
    '''
    idx_peaks, properties_peaks, right_ips, left_ips, width_peaks = Q.Q_FindDoubletResonance(
        T_normalised,
        ax=ax_Doublet,
        **param_find_DoubletResonance)
    ax_Doublet.set_xlabel('RelFreq (Hz)')
    ax_Doublet.set_ylabel('T linear scale')

    # if there is param_rel, update the Rel_A, Rel_FWHM, Rel_Peak
    from F_Bounds import F_Bounds_DoubleLorentzian
    # bound for A1, A2, FWHM1, FWHM2, lbd_res1, lbd_res2
    # A1
    A1_0 = properties_peaks['prominences'][0] # initial guess from peak finding
    A2_0 = properties_peaks['prominences'][1] # initial guess from peak finding
    FWHM1_0 = properties_peaks['widths'].max() * (T_normalised['nu_Hz'].iloc[1]-T_normalised['nu_Hz'].iloc[0]) # initial guess from peak finding
    FWHM2_0 = properties_peaks['widths'].max() * (T_normalised['nu_Hz'].iloc[1]-T_normalised['nu_Hz'].iloc[0]) # initial guess from peak finding
    Peak1_0 = T_normalised['nu_Hz'].iloc[idx_peaks[0]] # initial guess from peak finding
    Peak2_0 = T_normalised['nu_Hz'].iloc[idx_peaks[1]] # initial guess from peak finding
    bounds_Doublet = F_Bounds_DoubleLorentzian(A1_0=A1_0, Rel_A1=param_rel_Doublet['Rel_A1'],
                                               A2_0=A2_0, Rel_A2=param_rel_Doublet['Rel_A2'],
                                               FWHM1_0=FWHM1_0, Rel_FWHM1=param_rel_Doublet['Rel_FWHM1'],
                                               FWHM2_0=FWHM2_0, Rel_FWHM2=param_rel_Doublet['Rel_FWHM2'],
                                               Peak1_0=Peak1_0, Rel_Peak1=param_rel_Doublet['Rel_Peak1'],
                                               Peak2_0=Peak2_0, Rel_Peak2=param_rel_Doublet['Rel_Peak2'],
                                               )
    # step2: fitting using Lorentzian doublet model with bounds
    from F_LorentzianModel import f_locatized
    opt_Doublet, pcov_Doublet, f_func_DoubleLorentzian = f_locatized(
        nu_res=T_normalised['nu_Hz'].to_numpy(),
        signal_res=T_normalised['T_linear'].to_numpy(),
        bounds=bounds_Doublet, model='DoubleLorentzian')
    # Q factor for the first resonance
    Q_factor1 = F_QFactor(
        nu_peak=wl2nu(1550),
        FWHM=opt_Doublet[1]
    )
    ax_Doublet.plot(T_normalised['nu_Hz'],
                    f_func_DoubleLorentzian(T_normalised['nu_Hz'], *opt_Doublet),
                    label=f'Doublet Lorentzian Fit Q1={Q_factor1*1e-6:.2f}M', linestyle='--',
                    color = 'C4')
    ax_Doublet.legend()
    '''

    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    ### find the peak
    idx_peaks, properties_peaks, right_ips, left_ips, width_peaks,properties_peaks_half = Q.Q_FindPeak(T_corrected, ax = ax_T)
    points_to_remove = (right_ips - left_ips).astype(int)
    ### fit the back ground offset using Savgol filter
    from F_RemoveOffsetSavgol import RemoveOffset_Savgol

    T_normalised, offset = Q.Q_RemoveOffset_Savgol(
        T_corrected,
        idx_peaks,
        window_length=9,
        polyorder=2,
        points_to_remove=points_to_remove[0]+10,
        ax=ax_T
    )

    Fig_T_normalised = plt.figure()
    ax_T_normalised = Fig_T_normalised.add_subplot(111)
    ax_T_normalised.plot(T_normalised['nu_Hz']*1e-6, T_normalised['T_linear'],label='Savgol corrected normalised T',
                         alpha=0.7)
    ax_T_normalised.set_xlabel('Frequency (MHz)')
    ax_T_normalised.set_ylabel('T linear scale')
    ax_T_normalised.legend()

    ### fitting the resonance to get Q factor and other parameters
    # step 1: choose the bounds for fitting
    # try to refind the best bounds for Lorentian fitting
    from F_Bounds import F_Bounds_SingleLorentzian
    # bound for A, HWHM, lbd_res
    # A
    A0 = properties_peaks['prominences'][0] # initial guess from peak finding
    Rel_A = 0.2
    # FWHM
    FWHM0 = properties_peaks['FWHM'][0] # initial guess from peak finding
    Rel_FWHM = 0.2
    # Peak position
    Peak0 = T_normalised['nu_Hz'].iloc[idx_peaks[0]] # initial guess from peak finding
    Rel_Peak = 0.2
    bounds = F_Bounds_SingleLorentzian(
        A0=A0, Rel_A=Rel_A,
        FWHM0=FWHM0, Rel_FWHM=Rel_FWHM,
        Peak0=Peak0, Rel_Peak=Rel_Peak
    )
    # step2: fitting using Lorentzian model with bounds
    from F_LorentzianModel import f_locatized
    opt, pcov, f_func_Lorentzian = f_locatized(
        nu_res=T_normalised['nu_Hz'].to_numpy(),
        signal_res=T_normalised['T_linear'].to_numpy(),
        bounds=bounds, model='SingleLorentzian')
    fitting_options = {
        'A': opt[0],
        'FWHM': opt[1],
        'res': opt[2],
    }
    from F_QFactor import F_QFactor
    Q_factor = F_QFactor(
        nu_peak=wl2nu(1550),
        FWHM=fitting_options['FWHM']
    )

    ax_T_normalised.plot(T_normalised['nu_Hz']*1e-6,
                          f_func_Lorentzian(T_normalised['nu_Hz'], *opt),
                          label=f'Lorentzian Fit Q={Q_factor*1e-6:.2f}M', linestyle='--')  
    ax_T_normalised.legend()
    '''
