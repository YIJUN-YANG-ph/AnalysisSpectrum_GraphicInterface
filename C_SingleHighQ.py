import numpy as np
import pandas as pd
from F_LoadData import load_data
from F_ConvertUnits import nu2wl, wl2nu
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths
class SingleHighQ():
    def __init__(self):
        self.x_data = np.array([])
        self.y_data = np.array([])
        self.T_Data = self.load_data_singleQ()
        self.T_Offset = self.load_data_Offset()
        self.T_corrected = self.Remove_Offset()
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
                    T['T_dB'] = T['Pe_dBm'] * 0.5 # optical power = (electrical power)*0.5 in dB
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
    def load_data_singleQ(self,ax = None):
            """
            Opens a file dialog to load a .txt or .csv file.
            Assumes 2-column data (x, y) separated by comma, space, or tab.
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
            # file_name = r'D80-G400-W1598.679-0-1.5(241025).txt'
            file_name = r'D75-G400-W1549.524-2.5-3.9.txt'
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
                    T['T_dB'] = T['Pe_dBm'] * 0.5 # optical power = (electrical power)*0.5 in dB
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
            T_corrected['T_dB'] = T_corrected['Pe_dBm'] * 0.5 # optical power = (electrical power)*0.5 in dB
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
        _,properties_peaks_half = FindPeaks(T_corrected['T_linear'],**param_find_peaks)
        FWHM_peaks = properties_peaks_half['widths'] * (T_corrected['nu_Hz'].iloc[1]-T_corrected['nu_Hz'].iloc[0])
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
            ax.legend()
         
        


        return idx_peaks, properties_peaks, right_ips, left_ips, width_peaks, properties_peaks_half
    def Q_RemoveOffset_Savgol(self,
                              idx_peaks:np.ndarray,
                              points_to_remove:int = 50,
                              window_length:int=101,
                              polyorder:int=2,
                              ax:matplotlib.axes._axes.Axes = None)-> pd.DataFrame:
        """ Remove the background offset using Savgol filter.
        Args:
           * **idx_peaks** (np.ndarray): the index that indicates where the resonances are.
           * **points_to_remove** (np.int): the number of points that will be removed around each resonances when doing Savgol filtering.
           * **window_length** (int): window length for Savgol filter, need to be odd number.
           * **polyorder** (int): polynomial order for Savgol filter
        Outputs:
           * **transmission_corrected** (np.array): the transmission where the background offset is removed. The resonance shapes are kept.
        """
        from F_RemoveOffsetSavgol import RemoveOffset_Savgol
        T_corrected = self.T_corrected
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
        return T_normalised
    def PL_FitResonance(self,
                        T_corrected:pd.DataFrame,
                        ax_T:matplotlib.axes._axes.Axes = None):
        """
        A pipe line (PL) to fit each resonance.
        Parameters
        ----------
        T_corrected : pd.DataFrame
            DataFrame containing the corrected transmission data. A direct usable T that contains no huge background offset.
        """
        ### find the peak
        idx_peaks, properties_peaks, right_ips, left_ips, width_peaks,properties_peaks_half = self.Q_FindPeak(T_corrected, ax = ax_T)
        points_to_remove = ((right_ips - left_ips)*1.1).astype(int)

        ### fit the back ground offset using Savgol filter
        from F_RemoveOffsetSavgol import RemoveOffset_Savgol

        T_normalised = self.Q_RemoveOffset_Savgol(
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
        
        pass
    
if __name__ == "__main__":
     
    import matplotlib.pyplot as plt
    Fig_Pe = plt.figure()
    ax_Pe = Fig_Pe.add_subplot(111)
    Q = SingleHighQ()
    Q.load_data_Offset(ax = ax_Pe)
    Q.load_data_singleQ(ax = ax_Pe)
    ax_Pe.legend()

    Fig_T = plt.figure()
    ax_T = Fig_T.add_subplot(111)
    T_corrected = Q.Remove_Offset(ax = ax_T)
    ax_T.legend()

    ### find the peak
    idx_peaks, properties_peaks, right_ips, left_ips, width_peaks,properties_peaks_half = Q.Q_FindPeak(T_corrected, ax = ax_T)
    points_to_remove = (right_ips - left_ips).astype(int)
    ### fit the back ground offset using Savgol filter
    from F_RemoveOffsetSavgol import RemoveOffset_Savgol

    T_normalised = Q.Q_RemoveOffset_Savgol(
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
    
