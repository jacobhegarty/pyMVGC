from .helpers import _validate_time_series
from .ts_to_var import ts_to_var
import numpy as np
import warnings

def estimate_model_order(data, maxLag, criterion="AIC", verbose=True):
    """Compute information criteria for increasing VAR model orders..

    Parameters
    ----------
    data : array-like
        Multivariate time-series data with shape (n_samples, n_variables).
    maxLag : int
        Maximum lag to evaluate. Must be less than number of samples in data.
    criterion : str, optional
        Information criterion to compute. One of ``"AIC"``
        and ``"BIC"``.
    verbose : bool, optional
        If ``True``, print criterion values for each lag.

    Returns
    -------
    numpy.ndarray
        Criterion values for lags ``1`` to ``maxLag``.
    """
    import time

    startId = time.time()
    print(f"ENTER estimate_model_order id={startId}")
    if not isinstance(maxLag, int) or maxLag <= 0:
        raise ValueError("maxLag must be a positive integer")
    if criterion not in {"AIC", "BIC"}:
        raise ValueError('criterion must be either "AIC" or "BIC"')
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean")


    _validate_time_series(data, maxLag)
    n, N = data.shape[1], data.shape[0]
    crits = np.zeros(maxLag)
    for i in range(maxLag):
        p=i+1
        k = p * n * n
        M = N - p

        _, resCovMat = ts_to_var(data, p)

        det = np.linalg.det(resCovMat)
        if det <= 0:
            warnings.warn(f"Residuals covariance not positive definite at lag {p}")
            crits[i] = np.nan
            continue

        LL = np.log(det) * (-M / 2)  # log-likelihood

        if criterion == "AIC":
            if M - k - 1 <= 0:
                warnings.warn(f"AIC undefined at lag {p} (M-k-1 <= 0), skipping")
                crits[i] = np.nan
                continue
            pen = 2 * k * (M / (M - k - 1))
        elif criterion == "BIC":
            pen = k * np.log(M)

        crits[i] = -2 * LL + pen

        if verbose:
            print(f"{criterion} at lag {p} = {crits[i]}")

    if verbose:
        print(f"Minimal {criterion} at lag {np.nanargmin(crits)+1}")

    return crits
    






            



