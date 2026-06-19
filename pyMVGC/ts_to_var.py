import numpy as np
from .helpers import _validate_time_series


def ts_to_var(data,p): ## ts to autocov (A1)
    """Fit a VAR(p) model to time-series data using ordinary least squares.

    Parameters
    ----------
    data : numpy.ndarray
        Time-series samples arranged row-wise with shape ``(n_samples, n_variables)``.
    p : int
        VAR model order. Must be a positive integer smaller than the number of samples in data.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``Phi`` with shape ``(n_variables, n_variables, p)`` and residual covariance matrix with shape ``(n_variables, n_variables)``.
    """
    data = _validate_time_series(data, p)
    N,n = data.shape

    #normalise
    data -= np.mean(data,axis = 0) #demean

    #construct data
    #laged predictors
    X = np.hstack([data[p-lag:-lag] for lag in range(1, p+1)])
    Y = data[p:] #trim target


    # estimate parameters (Phi) using OLS
    Phi,_,_,_ = np.linalg.lstsq(X, Y, rcond=None) 

    #note residual cov mat should be produced by above but seems incorrect.
    # # calculated below instead
    resid = Y - X @ Phi # calculate residuals 
    resCovMat = resid.T @ resid / (resid.shape[0]-1) # residual covariance matrix
    
    # reshape stacked coefficients into (n,n,q)
    Phi = Phi.reshape(n, n, p, order = "F")
   
    return Phi, resCovMat