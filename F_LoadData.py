import pandas as pd
import numpy as np
import os

from yaml import warnings
from F_ConvertUnits import wl2nu, nu2wl, dB2linear

def load_data(file_name, wl_name='wavelength', data_name='transmission', range_wl=None,
              f_convert:callable = None)->pd.DataFrame:
    """ 
    Load data from file with flexible column name handling.
    
    Keyword Args:
       * **file_name** : the file name of the measurement.
       * **wl_name** : the column name that indicates wavelength in nm.
       * **data_name** : the column name that indicates the transmission in dB.
       * **range_wl** : a list or tuple specifying the wavelength range to filter [min_wavelength, max_wavelength].
       * **f_convert** : a callable function to convert x-axis data if needed.
    
    Returns:
       * **T** : pandas dataframe: (wavelength_nm), nu_Hz, T_dB, T_linear.
    """
    
    # Load the data
    try:
        # Attempt to load the data with auto-detection of the separator
        data = pd.read_csv(file_name,engine='python',sep=None)
    # Handle exceptions
    except Exception as e:
        #print a warning and re-raise the exception

        warnings(f"Error reading file {file_name}: {e}")
        # print(f"Error reading file {file_name}: {e}")
        raise

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



    # Apply x-axis conversion if provided
    if f_convert is not None:
        T = f_convert(wavelength, transmission)
    else:
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

if __name__ == "__main__":
    # Example usage
    # file_name = r'test_W1100_R100um_G400_AfterAnnealing.csv'
    # T = load_data(file_name, wl_name='wavelength', data_name='transmission', range_wl=[1500,1600])
    # print(T.head())

    # Another example usage, with different column names,
    # From frequency sweeping measurement
    file_name = r'D80-G400-W1598.679-0-1.5(241025).txt'
    T = load_data(file_name, wl_name='frequency', data_name='power1')
    #change the key names according to the actual file
    T = T.rename(columns={'wavelength_nm':'nu_Hz', 'nu_Hz':'wavelength_nm',})
    # convert nu_Hz from THz to Hz
    T['nu_Hz'] = T['nu_Hz'] * 1e12
    T['wavelength_nm'] = nu2wl(T['nu_Hz']) # convert to nm
    import matplotlib.pyplot as plt
    plt.plot(T['nu_Hz'], T['T_dB'])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Electrical Power (dB)')
    print(T.head())


    # Another example usage, with time sweeping measurement
    from F_ConvertUnits import time2nu
    file_name = r'RR0.4-F30mHz-V2pp.txt'
    file_name = r'C:\Users\yijun.yang\OneDrive\1A_PostDoc\SiN\202509_2509SiN700A_Multipassage\mesurement\Laser fine tunning\D80\RR0.4.txt'
    params = {'Vpp_V':4,'Freq_Hz':30*1e-3,'Tunning_Hz_V':300*1e6}
    T = load_data(file_name, wl_name='Time(s)', data_name='ct400.detectors.P_detector1(dBm)',
                  f_convert=lambda t, d: time2nu(t, d, Params=params))
    # each 1s give 36*1e6 Hz
    # T['nu_Hz'] = T['wavelength_nm'] * 36e6
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(T['nu_Hz']*1e-6-11000-750, T['T_linear'])
    plt.xlabel('Frequency (MHz)')    
    plt.ylabel('Transmission (dB)')

    # Another example usage, with heterodyne measurement
    from F_ConvertUnits import Heterodyne
    file_name = r'D75-G400-W1549.524-2.5-3.9.txt'
    T = load_data(file_name, wl_name='frequency', data_name='power1',
                  f_convert=lambda nu, Pe: Heterodyne(nu*1e9, Pe))


