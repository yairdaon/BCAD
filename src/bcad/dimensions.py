import time
import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import skdim

ESTIMATORS = {
    'MADA': skdim.id.MADA,
    'MLE': skdim.id.MLE,
    'CorrInt': skdim.id.CorrInt,
    'MOM': skdim.id.MOM,
    'TwoNN': skdim.id.TwoNN,
    'TLE': skdim.id.TLE,
    'FisherS': skdim.id.FisherS,
    'DANCo': skdim.id.DANCo,
    'MiND_ML': skdim.id.MiND_ML,
    'KNN': skdim.id.KNN,
    'lPCA': skdim.id.lPCA,
    'ESS': skdim.id.ESS
}

BESTIMATORS = ['CorrInt', 'MLE', 'MOM', 'FisherS']  # Best ESTimators


def res2subd(res):     
    grouper = ['cycle', 'estimator']
    theta_hat = subtract_dims(res, grouper=grouper, stat_col='theta_hat')
    bs = subtract_dims(res, grouper=grouper, stat_col='stat')
    assert (theta_hat.index == bs.index).all()
    subd = bs.assign(theta_hat=theta_hat.stat)
        
    subd.reset_index(drop=True, inplace=True)
    return subd


def gg(df, grp=None):
    """ID is ID(outcome) - ID(cause). For a true relation we expect
    ID(outcome) >= ID(cause) <===> ID >= 0"""
    x = df.iloc[:, 1].values
    pred = df.predictor.values
    data = np.subtract.outer(x, x).T.ravel()
    tt = dict(outcome=np.tile(pred, df.shape[0]),
              cause=np.repeat(pred, df.shape[0]),
              stat=data)
    tt = pd.DataFrame(tt)
    return tt if grp is None else (grp, tt)


def subtract_dims(df,
                  grouper=['instance', 'cycle', 'estimator'],
                  stat_col='stat',
                  n_jobs=1):
    """Calculate dim(outcome) - dim(cause). For a true relation we expect
    
    dim(outcome) >= dim(cause) <===> dim(outcome) - dim(cause) >= 0

    """
    def it(df, grouper, stat_col):
        for grp, data in df.groupby(grouper):
            yield data[['predictor', stat_col]], grp

    n_jobs = min(n_jobs, 12)
    gen = it(df, grouper=grouper, stat_col=stat_col)
    res = Parallel(n_jobs=n_jobs, backend='loky')(delayed(gg)(*pp) for pp in gen)
    res = pd.concat(dd.assign(**dict(zip(grouper, grp))) for grp, dd in res)
    return res.query("cause != outcome").reset_index(drop=True)


def basic(cycle: int,
          lagged: pd.DataFrame,
          estimator: str,
          dic: dict):
    """ Calculate intrinsic dimension with a single estimator"""
    ret = {'estimator': estimator, 'cycle': cycle, **dic}
    start = time.time()
    try:
        stat = ESTIMATORS[estimator]().fit_transform(X=lagged)
        error = ''
    except Exception as err:
        error = "".join(traceback.format_exception_only(type(err), err)).strip()
        if lagged.values.size == lagged.shape[0]:
            if lagged.nunique()[0] == 1:
                stat = 0
            else:
                stat = 1
        else:
            stat = np.nan
    end = time.time()
    if dic.pop('verbose', False):
        print(lagged.columns[0].split('_')[0], cycle, estimator, end - start, 'error' if len(error) > 0 else '',
              flush=True)
    return {**ret, 'stat': stat, 'error': error, 'time': end - start}
