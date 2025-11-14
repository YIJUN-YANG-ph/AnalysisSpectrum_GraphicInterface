import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def RemoveOffset_Savgol(transmission, idx, window_length=101, polyorder=2, points_to_remove=201):
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
    transmission_filtered = transmission_filtered.interpolate(method='polynomial', order=1).to_numpy() # the interpolate method may be changed to have a better interpolation.

    # Step 4: Subtract the background from the original transmission data
    # transmission_corrected = transmission - transmission_filtered + 1  # Add 1 to maintain baseline around 1
    transmission_corrected = transmission /transmission_filtered 

    offset = np.copy(transmission_filtered)

    
    # Fig_Savgol = plt.figure()
    # ax = Fig_Savgol.add_subplot(111)
    # ax.plot(transmission, label='Original Transmission', alpha=0.5, linestyle='--')
    # ax.plot(offset, label='Savgol Fitted Offset/filtered', alpha=0.7, linestyle='--')
    # ax.plot(transmission_corrected, label='Corrected Transmission', alpha=1)
    # ax.set_title('Savgol Filter Background Removal')
    # ax.legend()
    # plt.show()
        
    return transmission_corrected, offset