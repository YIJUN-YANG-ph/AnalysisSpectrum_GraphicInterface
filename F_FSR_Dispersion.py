# Give freq vs. FSR, plot the dispersion curve, and fit to a polynomial.

import numpy as np
import matplotlib.pyplot as plt

from F_ConvertUnits import wl2nu,nu2wl

def F_FSR_Dispersion(
    freq,
    FSR_array,
    order=3,
    pump_wavelength_nm=1550,
    R=100e-6,
    isPlot=True,
):
    """Run FSR fit, integrated dispersion, and extract D1/D2/D3/beta2/beta3.

    Args:
        freq (np.ndarray): Resonance frequencies (Hz).
        FSR_array (np.ndarray): FSR values (Hz).
        order (int): Polynomial order for FSR fit.
        pump_wavelength_nm (float): Pump wavelength in nm.
        R (float): Resonator radius (m) for beta2/beta3 calculation.
        isPlot (bool): If True, generate plots.
    """
    f_FSR, mu, Dint, plot_handles = F_FSR_intDisp(
        freq,
        FSR_array,
        order=order,
        pump_wavelength_nm=pump_wavelength_nm,
        isPlot=isPlot,
    )

    D1, D2, D3, beta2, beta3 = F_intDisp_HOD(
        mu,
        Dint,
        f_FSR,
        freq,
        R=R,
        plot_handles=plot_handles,
    )

    print(
        f"Extracted dispersion coefficients: D2 = {D2/(2*np.pi)*1e-6:.2f} MHz, "
        f"D3 = {D3/(2*np.pi)*1e-6:.2f} MHz"
    )
    print(
        f"Disp. coef.: beta_2 = {beta2*1e+27:.0f} ps^2/km, "
        f"beta_3 = {beta3*1e+39:.0f} ps^3/km"
    )

    return D1, D2, D3, beta2, beta3

def F_intDisp_HOD(mu, Dint, f_FSR, freq, R=100e-6, plot_handles=None):
    """Extract higher-order dispersion coefficients from integrated dispersion.

    Fits Dint vs. mode number mu with a fixed 3rd-order polynomial and
    returns D1, D2, D3 and corresponding beta2, beta3.

    Args:
        mu (np.ndarray): Mode number array (dimensionless), centered at mu=0.
        Dint (np.ndarray): Integrated dispersion values (rad/s).
        f_FSR (callable): FSR fit function; used to evaluate D1 at mu=0.
        freq (np.ndarray): Resonance frequencies (Hz), aligned with mu.
        R (float): Resonator radius (m). Used for beta2/beta3 calculation.
        plot_handles (dict | None): Plot handles from F_FSR_intDisp; if provided,
            the Dint fit curve is drawn on the dispersion axis.
    reference:
        thesis: Advanced characterization techniques of photonic devices with frequency combs, Chalmers
    """
    # HOD model: Dint = D2 * mu^2 / 2 + D3 * mu^3 / 6 + ...
    from F_fit_fsr_polynomial_robust import fit_fsr_polynomial_robust
    '''step1: fit'''
    # order 3 is engough to extract D2 and D3, and the D1 is from the FSR fit at mu=0
    f_Dint, inliers_Dint = fit_fsr_polynomial_robust(
        mu,
        Dint,
        order=3,
        nb_sigma=3,
        max_iter=5,
    )

    Dint_fit = f_Dint(mu)
    if plot_handles is not None:
        ax_disp = plot_handles["ax_disp"]
        ax_disp.plot(mu, Dint_fit / (2 * np.pi) * 1e-9, label=r'$D_{int}$ Fit', color='red')
        ax_disp.legend()
    '''step2: extract D2, D3, and calculate beta2, beta3'''
    coeffs = f_Dint.coefficients
    idx_mu0 = np.argmin(np.abs(mu))
    D1 = 2 * np.pi * f_FSR(freq[idx_mu0])
    D2 = 2 * coeffs[-3] # polynomial coeffs are in descending powers, so coeffs[-3] corresponds to mu^2 term, and the factor of 2 comes from the definition of Dint
    D3 = 6 * coeffs[-4]

    L = 2 * np.pi * R
    beta2 = -2 * np.pi / L * D2 / D1**3
    beta3 = -2 * np.pi / L * (D3 / D1**4)

    # print the D2, D3, beta2, beta3 under the graph with a box
    if plot_handles is not None:
        ax_disp = plot_handles["ax_disp"]
        textstr = "\n".join([
            rf'$D_2/2\pi = {D2/(2*np.pi)*1e-6:.2f}\,\mathrm{{MHz}}$',
            rf'$D_3/2\pi = {D3/(2*np.pi)*1e-6:.2f}\,\mathrm{{MHz}}$',
            rf'$\beta_2 = {beta2*1e+27:.0f}\,\mathrm{{ps^2/km}}$',
            rf'$\beta_3 = {beta3*1e+39:.0f}\,\mathrm{{ps^3/km}}$',
        ])
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax_disp.text(0.05, 0.95, textstr, transform=ax_disp.transAxes, fontsize=10,
                     verticalalignment='top', bbox=props)
        #tight layout
        plt.tight_layout()
        plt.show()

    return D1, D2, D3, beta2, beta3
        

def F_FSR_intDisp(
    freq: np.ndarray,
    FSR: np.ndarray,
    order: int = 3,
    nb_sigma: int = 3,
    max_iter: int = 5,
    pump_wavelength_nm: float = 1550,
    fig_size_fsr: tuple[float, float] = (8, 4),
    fig_size_disp: tuple[float, float] = (8, 4),
    freq_tick_rotation: float = 30,
    isPlot: bool = True,
):
    """Fit FSR vs. frequency and plot integrated dispersion.

    Args:
        freq (np.ndarray): Resonance frequencies of each mode (Hz).
        FSR (np.ndarray): Free spectral range values (Hz).
        order (int): Polynomial order for robust fit for FSR, not for Dint, order for Dint is later fixed at 3.
        nb_sigma (int): Sigma threshold for inlier detection.
        max_iter (int): Maximum iterations for robust fitting.
        pump_wavelength_nm (float): Pump wavelength in nm used to define mu=0.
        fig_size_fsr (tuple[float, float]): Figure size for FSR plot.
        fig_size_disp (tuple[float, float]): Figure size for dispersion plot.
        freq_tick_rotation (float): Rotation angle for frequency tick labels.
        isPlot (bool): If True, render plots; if False, still computes outputs.

    Returns:
        f_FSR (callable): Polynomial fit function for FSR vs. frequency.
        mu (np.ndarray): Mode numbers relative to the pump mode.
        Dint (np.ndarray): Integrated dispersion in rad/s.
        plot_handles (dict, optional): Only when isPlot=True. Contains figures and axes:
            {
                "fig_fsr": Figure,
                "ax_fsr": Axes,
                "fig_disp": Figure,
                "ax_disp": Axes,
                "ax_freq": SecondaryAxis,
                "ax_wl": SecondaryAxis,
            }
    """
    from F_fit_fsr_polynomial_robust import fit_fsr_polynomial_robust
    f_FSR, inliers = fit_fsr_polynomial_robust(freq, FSR, 
                                               order=order,
                                               nb_sigma=nb_sigma,
                                               max_iter=max_iter)
    
    # Plot the data and the fit
    plot_handles = None
    if isPlot:
        fig_fsr, ax_fsr = plt.subplots(figsize=fig_size_fsr)
    # plt.scatter(freq, FSR, label='Data', color='blue') 
    # FSR from pink fitting nu_res_array whichi is considered for robust polynomial fitting
        ax_fsr.scatter(freq[inliers], FSR[inliers], label='Inliers', color='blue', marker='v')
    # FSR from pink fitting nu_res_array whichi is not considered for robust polynomial fitting
        ax_fsr.scatter(freq[~inliers], FSR[~inliers], label='Outliers', color='blue', marker='x')
        ax_fsr.plot(freq[inliers], f_FSR((freq[inliers])), label='Fit', color='red') 
        ax_fsr.set_xlabel('Frequency (Hz)') 
        ax_fsr.set_ylabel('FSR (Hz)') 
        ax_fsr.set_title('FSR vs. Frequency with Polynomial Fit') 
    


    # step 2: resampling the FSR from the polynomial fit
    FSR_fit = f_FSR(freq)
    if isPlot:
        ax_fsr.scatter(freq, FSR_fit, label='FSR Fit sampled', color='green', alpha=0.3)
        ax_fsr.legend() 
        ax_fsr.grid() 

    # step 3: D1 = FSR_fit * 2pi
    D1 = FSR_fit * 2 * np.pi
    ## plot the D1 vs. frequency
    # fig, ax = plt.subplots(figsize=(5, 4))
    # ax.scatter(freq, D1, label='D1', color='purple',alpha=0.5)
    # ax.set_xlabel('Frequency (Hz)')
    # ax.set_ylabel('D1 (Hz)')
    # ax.set_title('D1 vs. Frequency')
    # ax.legend()
    # ax.grid()
    
    # step 4: integrated dispersion
    pump_wavelength = pump_wavelength_nm
    omega = freq * 2 * np.pi
    mu = np.round((freq - wl2nu(pump_wavelength)) / FSR_fit) # calculate mode numbers relative to the pump wavelength
    idx_mu0 = np.argmin(np.abs(mu)) # find the index of the mode closest to the pump wavelength, which is considered as mu = 0
    Dint = omega - omega[idx_mu0] - D1[idx_mu0] * mu # only D1 is sampled from FSR fit, the omega is from the original freq data, and the omega0 is from the original freq data as well, which is the freq closest to the pump wavelength
    wl_0 = nu2wl(freq[idx_mu0])
    if isPlot:
        fig_disp, ax_disp = plt.subplots(figsize=fig_size_disp)
        #using latex to show Dint/2pi in the y-axis label
        ax_disp.scatter(mu, Dint / (2 * np.pi) * 1e-9, label=r'$D_{int}/2\pi$', color='blue', alpha=0.5)
        label_pump = (
            f'Pump wavelength: {wl_0:.2f} nm \n'
            rf'$D_1/2\pi = {D1[idx_mu0]/(2*np.pi)*1e-9:.2f}$ GHz'
        )
        ax_disp.scatter(0, 0, label=label_pump, color='red', marker='x')

        ax_disp.set_xlabel(r'Mode Number ($\mu$)') 
        ax_disp.set_ylabel(r'$D_{int}/2\pi$ (GHz)') 
        ax_disp.set_title('Integrated Dispersion') 
        ax_disp.legend() 
        ax_disp.grid()
    # show the scale in frequency as well on the bottom x-axis
    # build a monotonic mapping between mu and freq for the secondary axis
    if isPlot:
        mu_sorted_idx = np.argsort(mu)
        mu_sorted = mu[mu_sorted_idx]
        freq_sorted = freq[mu_sorted_idx]
        wl_sorted = nu2wl(freq_sorted)
        wl_sorted_idx = np.argsort(wl_sorted)
        wl_sorted_asc = wl_sorted[wl_sorted_idx]
        mu_sorted_for_wl = mu_sorted[wl_sorted_idx]

        def mu_to_freq(x):
            return np.interp(x, mu_sorted, freq_sorted)

        def freq_to_mu(x):
            return np.interp(x, freq_sorted, mu_sorted)

        def mu_to_wl(x):
            return np.interp(x, mu_sorted, wl_sorted)

        def wl_to_mu(x):
            return np.interp(x, wl_sorted_asc, mu_sorted_for_wl)

        ax_freq = ax_disp.secondary_xaxis('bottom', functions=(mu_to_freq, freq_to_mu))
        ax_freq.set_xlabel('Frequency (Hz)')
        ax_freq.spines['bottom'].set_position(('outward', 40))
        ax_freq.tick_params(axis='x', direction='out')
        for label in ax_freq.get_xticklabels():
            label.set_rotation(freq_tick_rotation)
            label.set_ha('right')

        ax_wl = ax_disp.secondary_xaxis('top', functions=(mu_to_wl, wl_to_mu))
        ax_wl.set_xlabel('Wavelength (nm)')

        plot_handles = {
            "fig_fsr": fig_fsr,
            "ax_fsr": ax_fsr,
            "fig_disp": fig_disp,
            "ax_disp": ax_disp,
            "ax_freq": ax_freq,
            "ax_wl": ax_wl,
        }
    
    


     
    # plt.show() 
    if isPlot:
        return f_FSR, mu, Dint, plot_handles
    return f_FSR, mu, Dint
    



if __name__ == "__main__":
    FSR_array = np.array([
       2.21476368e+11, 2.21426217e+11, 2.21391221e+11, 2.21301952e+11,
       2.21135995e+11, 2.22219114e+11, 2.21250225e+11, 2.21297181e+11,
       2.21326200e+11, 2.21307649e+11, 2.21302220e+11, 2.21295426e+11,
       2.21270169e+11, 2.21280345e+11, 2.21237106e+11, 2.21243182e+11,
       2.21225092e+11, 2.21196130e+11, 2.21197339e+11, 2.21162669e+11,
       2.21171385e+11, 2.21134263e+11, 2.21131273e+11, 2.21115898e+11,
       2.21078103e+11, 2.21098560e+11, 2.21051274e+11, 2.21061340e+11,
       2.21020256e+11, 2.21020594e+11, 2.21019778e+11, 2.20972804e+11,
       2.20999145e+11, 2.20961163e+11, 2.20928446e+11, 2.20954638e+11,
       2.20910619e+11, 2.20886175e+11, 2.20883669e+11, 2.20869051e+11,
       2.20853478e+11, 2.20859975e+11, 2.20814034e+11, 2.20833873e+11,
       2.20785913e+11, 2.20786836e+11, 2.20769839e+11, 2.20727588e+11,
       2.20748216e+11, 2.20711752e+11, 2.20726016e+11, 2.20695758e+11,
       2.20690302e+11, 2.20659988e+11, 2.20658833e+11])
    
    nu_peak_array = np.array([
        1.996590e+14, 1.994375e+14, 1.992161e+14, 1.989947e+14,
        1.987734e+14, 1.985522e+14, 1.983300e+14, 1.981088e+14,
        1.978875e+14, 1.976661e+14, 1.974448e+14, 1.972235e+14,
        1.970022e+14, 1.967810e+14, 1.965597e+14, 1.963384e+14,
        1.961172e+14, 1.958960e+14, 1.956748e+14, 1.954536e+14,
        1.952324e+14, 1.950112e+14, 1.947901e+14, 1.945690e+14,
        1.943479e+14, 1.941268e+14, 1.939057e+14, 1.936846e+14,
        1.934636e+14, 1.932426e+14, 1.930215e+14, 1.928005e+14,
        1.925795e+14, 1.923585e+14, 1.921376e+14, 1.919167e+14,
        1.916957e+14, 1.914748e+14, 1.912539e+14, 1.910330e+14,
        1.908122e+14, 1.905913e+14, 1.903704e+14, 1.901496e+14,
        1.899288e+14, 1.897080e+14, 1.894872e+14, 1.892664e+14,
        1.890457e+14, 1.888250e+14, 1.886043e+14, 1.883835e+14,
        1.881628e+14, 1.879421e+14, 1.877215e+14, 1.875008e+14, ]) 
    
    
    freq = nu_peak_array[:-1]
    F_FSR_Dispersion(
        freq,
        FSR_array,
        order=3,
        pump_wavelength_nm=1550,
        R=100e-6,
        isPlot=True,
    )

    
