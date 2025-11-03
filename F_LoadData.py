import pandas as pd
import numpy as np
import os

from yaml import warnings
from F_ConvertUnits import wl2nu, nu2wl, dB2linear

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