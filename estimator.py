import numpy as np
from scipy.optimize import minimize
from typing import Tuple

class WeibullEstimator:
    def fit_mle(self, failures: np.ndarray, censored: np.ndarray = None) -> Tuple[float, float]:
        if censored is None: censored = np.zeros_like(failures, dtype=bool)
        times = failures[failures > 0]
        cens = censored[failures > 0]
        
        def neg_ll(params):
            b, e = params
            if b <= 0 or e <= 0: return np.inf
            ll = 0
            for t, c in zip(times, cens):
                if c: ll -= (t/e)**b
                else: ll += np.log(b) - np.log(e) + (b-1)*np.log(t/e) - (t/e)**b
            return -ll
            
        res = minimize(neg_ll, [2.0, np.mean(times)], bounds=[(0.1, 10), (1, None)])
        return res.x[0], res.x[1]