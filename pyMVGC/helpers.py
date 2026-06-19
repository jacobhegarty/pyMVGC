import numpy as np

def _validate_time_series(data, p):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array shaped (n_samples, n_variables)")
    if data.size == 0:
        raise ValueError("Data must be a 2D array shaped (n_samples, n_variables)")
    if not np.all(np.isfinite(data)):
        raise ValueError("Time-series data must contain only finite values")
    if not isinstance(p, (int, np.integer)):
        raise TypeError("Model order p must be a integer")
    if p <= 0:
        raise ValueError("Model order p must be positive")
    if data.shape[0] <= p:
        raise ValueError("Model order p must be smaller than the number of samples")
    return data



def _validate_autocovariance(G):
    G = np.asarray(G, dtype=float)
    if G.ndim != 3:
        raise ValueError("Autocovariance sequence must be a 3D array shaped (n, n, q+1)")
    if G.shape[0] != G.shape[1]:
        raise ValueError("Autocovariance sequence must have matching first two dimensions")
    if G.shape[2] < 2:
        raise ValueError("Autocovariance sequence must contain at least lag-0 and lag-1 slices")
    if not np.all(np.isfinite(G)):
        raise ValueError("Autocovariance sequence must contain only finite values")
    return G


def _validate_var_inputs(Phi, resCovMat):
    Phi = np.asarray(Phi, dtype=float)
    resCovMat = np.asarray(resCovMat, dtype=float)
    if Phi.ndim != 3:
        raise ValueError("Phi must be a 3D array shaped (n, n, p)")
    if Phi.shape[0] != Phi.shape[1]:
        raise ValueError("Phi must have matching first two dimensions")
    if resCovMat.ndim != 2 or resCovMat.shape[0] != resCovMat.shape[1]:
        raise ValueError("Residual covariance matrix must be square")
    if Phi.shape[0] != resCovMat.shape[0]:
        raise ValueError("Phi and residual covariance matrix must have matching variable dimensions")
    if not np.all(np.isfinite(Phi)) or not np.all(np.isfinite(resCovMat)):
        raise ValueError("Phi and residual covariance matrix must contain only finite values")
    return Phi, resCovMat


def _validate_indices(indices, n, name):

    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")

    indices = np.asarray(indices)
    if indices.ndim == 0:
        indices = indices.reshape(1)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be a 1D index array")
    if indices.size == 0:
        raise ValueError(f"{name} must not be empty")

    if np.issubdtype(indices.dtype, np.integer):
        int_idx = indices.astype(int, copy=False)
    else:
        if np.issubdtype(indices.dtype, np.floating):
            if not np.all(np.isfinite(indices)):
                raise ValueError(f"{name} must contain only finite values")
            if not np.all(np.equal(indices, np.floor(indices))):
                raise TypeError(f"{name} must contain integer indices")
            int_idx = indices.astype(int)
        else:
            raise TypeError(f"{name} must contain integer indices")

    if np.any(int_idx < 0) or np.any(int_idx >= n):
        raise IndexError(f"{name} contains indices outside the valid range [0, {n})")
    if np.unique(int_idx).size != int_idx.size:
        raise ValueError(f"{name} must not contain duplicate indices")

    return int_idx


def _reconstruct(missingInds,resCov):
    """Reinsert a reduced covariance matrix into its original square shape.

    Parameters
    ----------
    missingInds : array-like
        Indices that were removed from the original matrix.
    resCov : numpy.ndarray
        Reduced covariance matrix.

    Returns
    -------
    numpy.ndarray
        A zero-padded covariance matrix with the reduced matrix restored to
        the non-missing indices.
    """
    n = resCov.shape[0] + len(missingInds) #number of vars including missing
    resCovReconst = np.zeros((n,n)) # initialise reconstructed matrix

    notMissing = np.flatnonzero(~np.isin(np.arange(n), missingInds)) #get present vars index
    resCovReconst[np.ix_(notMissing,notMissing)] = resCov # fill in vars from residual covariance matrix to correct cosl/rows.
    return resCovReconst


def _isPosDef(sig,tol = 1e-10):
    sig = (sig + sig.T) / 2  # force exact symmetry
    eigvals = np.linalg.eigvalsh(sig)
    return np.all(eigvals > -tol)