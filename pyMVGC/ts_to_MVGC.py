from .ts_to_var import ts_to_var
from .var_to_autocov import var_to_autocov
from .autocov_to_MVGC import autocov_to_MVGC
def ts_to_MVGC(data ,to_idx,from_idx,p ):
    """Estimate multivariate Granger causality directly from time series data.

    Parameters
    ----------
    data : numpy.ndarray
        Time-series samples arranged row-wise with shape ``(n_samples, n_variables)``.
    to_idx : array-like
        Variable(s) to calculate MVGC to (causee) ``[0, n_variables)``.
    from_idx : array-like
        Variable(s) to calculate MVGC from (causes)  ``[0, n_variables)``.
    p : int
        VAR model order. Must be a positive integer smaller than ``n_samples``.

    Returns
    -------
    float
        Granger causality value for the requested variable set.
    """
    Phi, resCov = ts_to_var(data = data,p=p)
    autocov= var_to_autocov(Phi = Phi,resCovMat=resCov)
    GC = autocov_to_MVGC(autocov, to_idx,from_idx)
    return GC

