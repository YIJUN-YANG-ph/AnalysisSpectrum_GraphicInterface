import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from F_ConvertUnits import time2nu, wl2nu, nu2wl, dB2linear
def F_load_FuncGenOsci(file_name_fg = None, 
                       file_name_res = None,
                       ax = None,
                       model = 'FuncGenOsci',
                       ScanParams:dict = {'Tunning_Hz_V':300*1e6} # Tunics plus tuning rate in Hz/V
                       )-> pd.DataFrame:
        """
        To have T vs. freq
        First, load function generation signal and resonance signal from oscilloscope, then convert the function generation voltage to frequency using the tuning rate, and finally plot the resonance signal vs. frequency.
        Assumes 2-column data (x, y) separated by comma, space, or tab.

            Inputs:
                * **file_name_fg** (str): Path to the function generator data file. If None, a default file will be used.
                * **file_name_res** (str): Path to the resonance data file. If None, a default file will be used.
                * **ax** (matplotlib.axes._axes.Axes): Axes object for plotting. If None, no plot will be made.
                * **model** (str): Model type for data conversion. Options are 'Heterodyne', 'FineScan', or default.
                * **ScanParams** (dict): Dictionary containing scan parameters.
        returns:    
            **T**: DataFrame containing the loaded data.
            **T['nu_Hz']**: Hz
            **T['wavelength_nm']**: nm
            **T['T_dB']**: voltage(optical power) in dB
            **T['T_linear']**: voltage(optical power) in linear scale in V.

        """

        from F_LoadData import load_data
        import os
        from os.path import join
        import pandas as pd
        import numpy as np

        if file_name_fg is None or file_name_res is None:
            FolderName = r'C:\Users\yijun.yang\autolab\drivers\local\tektronix_TDS3014C\2511SiN700A_D75\RR_1549nm'
            file_name_fg = r'c1_functiongenerator_TDS5104BCH1_volts.txt'
            file_name_res = r'c2_resonance_TDS5104BCH2_volts.txt'
            file_fg = join(FolderName, file_name_fg)
            file_res = join(FolderName, file_name_res)
        # file_name = r'D75-G400-W1600.014-1.7-2.2.txt'
        # file_name = r'D75-G400-W1600.014-0-3.txt'

        # switch
        if model == 'FuncGenOsci':
            data_fg = pd.read_csv(file_fg,engine='python',sep=None)
            data_res = pd.read_csv(file_res,engine='python',sep=None)
            fg_volts = data_fg.iloc[:,1].values * 1000 # the channel decreased the voltage by 1000 times, so we need to multiply it back
            fg_T = data_fg.iloc[:,0].values
            res_volts = data_res.iloc[:,1].values
            # do a linear regression of fg_volts vs fg_T due to small variation of fg_T
            coeffs = np.polyfit(fg_T,fg_volts, 1)
            fg_volts_fit = np.polyval(coeffs, fg_T)
            
            # freq_Hz = ScanParams['Tunning_Hz_V'] * fg_volts # linear conversion from voltage to frequency
            freq_Hz = ScanParams['Tunning_Hz_V'] * fg_volts_fit # linear conversion from voltage to frequency

            
            # plot res_volts vs freq_Hz
            T = pd.DataFrame({
                'nu_Hz': freq_Hz,
                'T_linear': res_volts,
                'T_dB': 10 * np.log10(res_volts)
            })
            from matplotlib import pyplot as plt
            if ax:
                ax.plot(T['nu_Hz']*1e-6, T['T_linear'],)
                ax.set_xlabel('Frequency (MHz)')
                ax.set_ylabel('Transmission (V)')

        return T

if __name__ == "__main__":
    fig, ax = plt.subplots()
    T = F_load_FuncGenOsci(ax=ax)
    plt.show()
    # print(T.head())