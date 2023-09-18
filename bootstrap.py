import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from teaspoon.parameter_selection.FNN_n import FNN_n

from dimensions import basic, BESTIMATORS

def bootstrap(df,
              n_bootstrap,
              tau,
              predictors=None,
              estimators=BESTIMATORS,
              n_jobs=-1):
    
    predictors = df.columns if predictors is None else predictors
    predictors = [pred for pred in predictors if pred != 'year']

    ## Calculate point estimates
    print("Point estimates")
    point_estimates = pd.DataFrame(index=estimators, columns=predictors)
    for predictor in predictors:
        lagged, dic = transform_series(df[predictor], tau=tau)
        for estimator in estimators:
            point_estimates.loc[estimator, predictor] = basic(-1, lagged=lagged, estimator=estimator, dic={})['stat']
    
    ## Generator to feed the Pool
    it = phase_bootstrap(df=df,
                         n_bootstrap=n_bootstrap,
                         estimators=estimators,
                         predictors=predictors,
                         tau=tau,
                         point_estimates=point_estimates)

    res = [basic(*p) for p in it] if n_jobs == 1 else Parallel(n_jobs=n_jobs, backend='loky', verbose=1)(delayed(basic)(*p) for p in it)
    res = pd.DataFrame(res)
    return res


def phase_bootstrap(df,
                    n_bootstrap,
                    estimators,
                    predictors,
                    tau,
                    point_estimates):
    """Sample from reconstructed state-space"""
    df = df[predictors]
    for predictor in predictors:
        lagged, dic = transform_series(df[predictor], tau=tau)
        dic['predictor'] = predictor
        # dic['drop_dups'] = drop_dups
        for i in range(n_bootstrap):
            bs = lagged.sample(frac=1, replace=True)
            # bs = bs.drop_duplicates(ignore_index=True) if drop_dups else bs
            for estimator in estimators:
                theta_hat = point_estimates.loc[estimator, predictor]
                yield i, bs, estimator, {**dic, 'theta_hat': theta_hat}


def lag(df,
        lags):
    """Lags df according to lags. Lags are denoted with _underscore_, e.g. X_4.
    Observations, OTOH come with a sign: X4 or Y+3.
    """
    if type(df) == pd.Series:
        df = pd.DataFrame(index=df.index, columns=[df.name], data=df.values)
    ## If lags is int, we lag all variables 0,...,lags1. Here we
    ## create the lagging dictionary.
    if isinstance(lags, int):
        assert lags > 0, 'If lags is an integer it has to be > 0'
        dic = {}
        for col in df.columns:
            dic[col] = range(lags)
        lags = dic

    lagged = pd.DataFrame(index=df.index)
    for var, lag_list in lags.items():
        for lag in lag_list:
            assert lag >= 0
            lagged[f"{var}_{lag}"] = df[var].shift(lag)

    assert lagged.shape[0] > 0
    assert lagged.shape[1] > 0
    return lagged


def transform_series(series: pd.Series,
                     tau: int):

    if series.size < 100: ## 100 is arbitrary
        raise ValueError(f"Time series {series.name} length {series.size} < 100 is too short")
    
    if series.std() == 0:
        return 1
           
    assert tau > 0 
    maxDim = (series.size - 1) / tau + 1
    maxDim = min(10, int(maxDim))
    embedding_dimension = FNN_n(series.values, tau=tau, maxDim=maxDim)[1] 
    embedding_dimension = max(1, embedding_dimension)

    lags = tau * np.arange(embedding_dimension, dtype=int)
    lagged = lag(series, {series.name: lags}).dropna()

    return lagged, dict(embedding_dimension=embedding_dimension, tau=tau)
