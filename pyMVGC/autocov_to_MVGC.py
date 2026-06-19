import numpy as np
from .helpers import _reconstruct, _validate_autocovariance, _validate_indices
from .autocov_to_var import autocov_to_var

def autocov_to_MVGC(autocov,to_idx,from_idx,):
    """Compute multivariate Granger causality from an autocovariance sequence.

    Parameters
    ----------
    autocov : numpy.ndarray
        Autocovariance sequence with shape ``(n_variables, n_variables, q + 1)`` and lag 0 in the first slice.
    to_idx : array-like
        Variable(s) to calculate MVGC to (causee) ``[0, n_variables)``.
    from_idx : array-like
        Variable(s) to calculate MVGC from (causes)  ``[0, n_variables)``.

    Returns
    -------
    float
        Granger causlality from ``from_idx`` to ``to_idx`` variables.
    """
    
    autocov = _validate_autocovariance(autocov)
    to_idx = _validate_indices(to_idx, autocov.shape[0], "to_idx")
    from_idx = _validate_indices(from_idx, autocov.shape[0], "from_idx")
    if np.intersect1d(to_idx, from_idx).size:
        raise ValueError("to_idx and from_idx must not overlap")

    _,full = autocov_to_var(autocov) # calculate full VAR residual covariance from autocov

    LSIG = np.log(np.linalg.det(full[np.ix_(to_idx, to_idx)])) #calculate log linear determinant of residual covariance of child (to) variable(s) in full VAR

    redG =  np.delete(np.delete(autocov, from_idx, axis=0), from_idx, axis=1) # delete potential parent (from) variables from autocovariance

    _,red = autocov_to_var(redG) # get reduced residual covariance from reduced autocovariance sequence
    redReconst = _reconstruct(from_idx,red) # add back in parent variables (with zeros) to ensure indexing consistency
    
    LSIGj = np.log(np.linalg.det(redReconst[np.ix_(to_idx, to_idx)])) #log linear determinant of residual covarinace of child (to) variable(s) in reduced VAR

    GC = LSIGj - LSIG ##calculates GC
    
    return float(GC)
