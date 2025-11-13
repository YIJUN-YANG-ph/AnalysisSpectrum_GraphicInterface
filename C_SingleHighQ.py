import numpy as np
import pandas as pd
from F_LoadData import load_data
from F_ConvertUnits import nu2wl, wl2nu
import matplotlib
import matplotlib.pyplot as plt
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
                        
                        print(T.head())

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
            file_name = r'D80-G400-W1598.679-0-1.5(241025).txt'
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
                    
                    print(T.head())

                    
                    # Plot the raw data
                    # self.plot_data.setData(self.x_data, self.y_data)
                    # Also plot on the linear scale plot
                    # y_data_linear = T['T_linear'].values
                    # self.plot_data_linear.setData(self.x_data, y_data_linear)
                    '''
                    # Make intelligent guesses for initial parameters
                    self.spin_offset.setValue(np.min(self.y_data))
                    self.spin_amplitude.setValue(np.max(self.y_data) - np.min(self.y_data))
                    self.spin_center.setValue(self.x_data[np.argmax(self.y_data)])
                    self.spin_fwhm.setValue((self.x_data.max() - self.x_data.min()) * 0.1) # Guess 10% of range
                    
                    # Clear old fit plot
                    self.plot_fit.setData([], [])
                    '''
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
                ax.plot(T_corrected['nu_Hz'], T_corrected['T_linear'],label='corrected',alpha=0.7)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('T linear scale')
                ax.set_title('Corrected optical power after offset removal')
            return T_corrected
    def Q_FindPeak(self):
         from F_FindPeaks import FindPeaks
         return
         
         
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
    from F_FindPeaks import FindPeaks
    idx_peaks = FindPeaks(T_corrected['T_linear'])
    ax_T.plot(T_corrected['nu_Hz'][idx_peaks],
              T_corrected['T_linear'][idx_peaks],
              'o')
    # ax_T.plot(T_corrected['nu_'])

    