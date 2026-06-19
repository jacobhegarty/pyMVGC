import numpy as np
from .helpers import _validate_autocovariance


def autocov_to_var(autocov): 
    """Calculates VAR residual covariance matrix and coefficients from autocovariance sequence via Yule-Walker equations.

    Parameters
    ----------
    autocov : numpy.ndarray
        Autocovariance sequence with shape ``(n_variables, n_variables, q + 1)`` and lag 0 in slice 0.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Residual covariance matrix with shape ``(n_variables, n_variables)`` and stacked VAR coefficients with shape ``(n_variables, n_variables, q)``.
    """
    autocov = _validate_autocovariance(autocov)
    q = autocov.shape[2] - 1
    n = autocov.shape[1]
    qn= q*n

    G0 = autocov[:,:,0] # covariance (lag 0) 

    GF = autocov[:, :, 1:].reshape(n, qn, order = "F").T # forwards autocovariance sequence
    GB =np.reshape(np.transpose( np.flip(autocov[:, :, 1:], axis=2) , (0, 2, 1)), (qn, n), order = "F") # backwards autocovariance sequence
    
    #initialse for storage
    AF = np.zeros((n, qn))   # forward VAR coefficients
    AB = np.zeros((n, qn))   # backward coefficients 

    # initialise recursion 
    k = 1 # model order
    r = q - k 

    kf = slice(0, k*n) # forward index block
    kb = slice(r*n, qn)  # backward index block

    # Solve first Yule–Walker step
    AF[:, kf] = GB[kb, :] @ np.linalg.inv(G0)
    AB[:, kb] = GF[kf, :] @ np.linalg.inv(G0)

    # recursion
    for k in range(2, q+1):

        # 
        numF = GB[(r-1)*n:r*n, :] - AF[:, kf] @ GB[kb, :]
        denF = G0 - AB[:, kb] @ GB[kb, :]
        AAF = numF @ np.linalg.inv(denF)

        numB = GF[(k-1)*n:k*n, :] - AB[:, kb] @ GF[kf, :]
        denB = G0 - AF[:, kf] @ GF[kf, :]
        AAB = numB @ np.linalg.inv(denB)

        AFPREV = AF[:, kf].copy()
        ABPREV = AB[:, kb].copy()

        # update index windows
        r = q - k
        kf = slice(0, k*n)
        kb = slice(r*n, qn)

        # update forward and backward coefficient blocks
        AF[:, kf] = np.hstack([AFPREV - AAF @ ABPREV, AAF])
        AB[:, kb] = np.hstack([AAB, ABPREV - AAB @ AFPREV])

    # residual covariance
    SIG = G0 - AF @ GF

    # reshape stacked coefficients into (n,n,q)
    AF = AF.reshape(n, n, q, order = "F")

    return AF,SIG