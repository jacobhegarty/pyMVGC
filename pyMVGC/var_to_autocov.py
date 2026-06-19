import numpy as np
import warnings
from scipy.linalg import solve_discrete_lyapunov
from .helpers import _isPosDef, _validate_var_inputs
def var_to_autocov(Phi,resCovMat): 
    """Generate an autocovariance sequence from VAR coefficients and residual covariance matrix.

    Parameters
    ----------
    Phi : numpy.ndarray
        VAR coefficient matrix in stacked form with shape ``(n_variables, n_variables, p)``.
    resCovMat : numpy.ndarray
        Residual covariance matrix with shape ``(n_variables, n_variables)``.

    Returns
    -------
    numpy.ndarray
        Autocovariance sequence with shape ``(n_variables, n_variables, q + 1)`` and lag 0 in the first slice.
    """

    #####checks
    Phi, resCovMat = _validate_var_inputs(Phi, resCovMat)

    if  (~_isPosDef(resCovMat)):
        raise ValueError("Residual covariance matrix not positive-deinite")



    n = Phi.shape[0] #number of variables

    p = Phi.shape[2] # model order


    ### VAR coefficents for 1-lag
    Phi = Phi.reshape(n * p,n, order="F")
    PhiStack = Phi.T
    bottom = np.hstack([np.eye((p-1)*n), np.zeros(((p-1)*n, n))])
    A1 = np.vstack([PhiStack,bottom])

 
    ##calculate spectral radius - check for VAR stationarity
    specRad = np.max(np.abs(np.linalg.eigvals(A1)))
    if (specRad >= 1.0): 
        raise ValueError(f"Non-stationary VAR! Residual covariance matrix not positive-definative. \n Spectral radius: {specRad}")
   
   #construct residual covariance for 1-lag problem
    SIG1 = np.block([
        [resCovMat, np.zeros((n, (p-1)*n))],
        [np.zeros(((p-1)*n, n)), np.zeros(((p-1)*n, (p-1)*n))]
    ])

    #solve Lyapunov question for 1-lag covarinace
    covZero = solve_discrete_lyapunov(A1,SIG1)

    acRelErr = np.linalg.norm( A1 @ covZero @ A1.T - covZero + SIG1, ord='fro') / np.linalg.norm(SIG1, ord='fro')
    if (acRelErr>1e-8):
        warnings.warn(f"Large relative error in discrete Lyapunov for 1-lag problem. Relative error: {acRelErr}.",
                      RuntimeWarning,
        stacklevel=2
    )


    if (~ _isPosDef(covZero)):
        raise ValueError("1-lag covariance matrix is not positive-definite")

    ## Estimate minimum lags for autocovariance seq
    minACLags =np.ceil( np.log(1e-08) / np.log(specRad)) 
    q = int(minACLags)
 
    #autocovariance forward sequence
    G_init = covZero[:n, :]                  
    G_init = G_init.reshape(n, n, p,order = "F")
    G = np.concatenate(
        [G_init, np.zeros((n, n, (q+1) - p))],
        axis=2
    )

    #autocovariance backwards sequence
    B = np.vstack([
        np.zeros((((q+1) - p)*n, n)),
        covZero[:, -n:]                        
    ])  

    #recurssion
    for k in range(p,q+1):
        r = q+1 - k
        rows = slice(r*n, r*n + p*n) 
        G[:,:,k] = PhiStack @ B[rows,:]
        brow = slice((r-1)*n, r*n)    
        B[brow, :] = G[:, :, k]

    return G
