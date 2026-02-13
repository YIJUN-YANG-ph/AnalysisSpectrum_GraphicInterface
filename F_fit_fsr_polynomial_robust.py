import numpy as np

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