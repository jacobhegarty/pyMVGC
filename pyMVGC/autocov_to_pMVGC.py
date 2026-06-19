import numpy as np
from .helpers import _validate_autocovariance
from .autocov_to_var import autocov_to_var
def autocov_to_pMVGC(autocov):
    """Compute pairwise multivariate Granger causality from autocovariance sequence.

    Parameters
    ----------
    autocov : numpy.ndarray
        Autocovariance sequence with shape ``(n_variables, n_variables, q + 1)`` and lag 0 in the first slice.

    Returns
    -------
    numpy.ndarray
        Square matrix of pairwise GC values with 'from' variables in columns, 'to' variables in rows.
    """
    autocov = _validate_autocovariance(autocov)
    n = autocov.shape[0] # number of variables

    _,full = autocov_to_var(autocov) # get full VAR from autocov

    LSIG = np.log(np.diag(full)) #get logged diagonal (residual variance for each variable)

    adjMat = np.zeros((n,n)) # initialise pairwise GC matrix

    #loop parent nodes
    for i in range(n): 

        redG =  np.delete(np.delete(autocov, i, axis=0), i, axis=1) # remove potential parent from autocov

        _, red = autocov_to_var(redG) # calculate reduced VAR from reduced autocovariance

        LSIGj = np.log(np.diag(red)) # get residual variance associated with each variable

        GCs = LSIGj - np.delete(LSIG,i) ##calculate GC
        
        GCs = np.insert(GCs, i, np.nan) # insert nan on diagonal
     
        adjMat[:,i] = GCs # store GC values 

    return(adjMat)