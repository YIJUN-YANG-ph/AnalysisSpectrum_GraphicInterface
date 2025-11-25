# analysis the uncertainty of the model, given the function model and fitted parameters
def f_Uncertainty(model, popt, pcov, xdata, ydata, alpha=0.05):
    """
    the model is a function
    """
    import numpy as np
    from scipy import stats
