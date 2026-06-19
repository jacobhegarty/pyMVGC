# pyMVGC: Python Multivariate Granger causality

A Python implementation of multivariate Granger causality (MVGC) for time-series data,  based on algorithms from Barnett & Seth's MVGC toolbox [1]. This method avoids explicit estimation of the reduced VAR via moving between autocovariance-sequence and regression-parameter representations of the VAR model. This reduces estimation error and imporoves statistical power and speed of estimation compared to the traditional estimation of separate full and reduced models.


## Usage
pyMVGC can be downloaded from PyPI via pip:
```
$ pip install pyMVGC
```
Multivariate granger causality may be estimated in one line of code:

```python
import pyMVGC as gc

# Multivariate Granger causality from one set of variables to another
ts_to_MVGC(data, to_idx = [0], from_idx = [1], p = 2)

#Pairwise granger causality between all variables in the system
ts_to_pMVGC(data, p = 2)
```

Optimal model order can be estimated from "AIC" or "BIC" scores:

```python
estimate_model_order(data, maxLag = 30, criterion = "AIC")
```

Functions allowing movement between autocovariance and VAR representations, used in MVGC calculation, are also available:

```python
Phi, resCov = gc.ts_to_var(data = data, p = p) # learn VAR from time series data.
autocov = gc.var_to_autocov(Phi = Phi, resCovMat = resCov) # learn autocovariance sequence from VAR.
Phi, resCov = gc.autocov_to_var(autocov = autocov)

GC = gc.autocov_to_MVGC(autocov, to_idx,from_idx) # get multivariate granger causality from from autocovarance sequence.
pGC = gc.autocov_to_pMVGC(autocov) # get pairwise granger causality from autocovariance sequnce.
    
```

See [notebooks/tutorial.ipynb](https://github.com/jacobhegarty/pyMVGC/blob/main/notebooks/tutorial.ipynb) for a more detailed tutorial.


## References
[1] L. Barnett and A. K. Seth, "The MVGC Multivariate Granger Causality Toolbox: A new
approach to Granger-causal inference", J. Neurosci. Methods 223, pp 50-68, 2014.