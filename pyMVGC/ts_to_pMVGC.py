from .ts_to_var import ts_to_var
from .var_to_autocov import var_to_autocov
from .autocov_to_pMVGC import autocov_to_pMVGC
from .helpers import _validate_time_series

def ts_to_pMVGC(data, p):
   """Estimate pairwise multivariate Granger causality from time series data.

   Parameters
   ----------
   data : numpy.ndarray
      Time-series samples arranged row-wise with shape ``(n_samples, n_variables)``.
   p : int
      VAR model order. Must be a positive integer smaller than the number of samples in data.

    Returns
    -------
    numpy.ndarray
        Square matrix of pairwise GC values with 'from' variables in columns, 'to' variables in rows.
    """
   _validate_time_series(data,p)
   Phi, resCov = ts_to_var(data = data,p=p)
   autocov = var_to_autocov(Phi = Phi,resCovMat=resCov)
   adjMat = autocov_to_pMVGC(autocov)
   return adjMat    