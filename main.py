from pprint import pprint
import time
import pdb

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plac


from bootstrap import bootstrap
#from misc import OUT, DATA, STATES
from dimensions import BESTIMATORS, subtract_dims,  res2subd
from ssa import GSSA
from rypdal_sugihara import transform
from helpers import decisions
  
def denoise(df, M, predictors=None):
    if M == 0:
        return df
    
    predictors = df.columns if predictors is None else predictors
    for pred in predictors:
        df[pred] = GSSA.reconstruct(data=df[pred], M=M)  # SSA with Gavish's threshold
    return df


def infer(df,
          tau,
          predictors,
          n_bootstrap=200,
          estimators=BESTIMATORS):
    
    res = bootstrap(df=df,
                    n_bootstrap=n_bootstrap,
                    predictors=predictors,
                    tau=tau,
                    estimators=estimators).reset_index(drop=True)
    
    subd = res2subd(res)
    subd = subd.assign(tau=tau)  
    
    return subd
    
def main(n_bootstrap: ("number of bootstrap cycles", 'option', 'n') = 200,
         M_years: ("number of years in SSA window", 'option', 'M') = 0):
    

    shift = -1
    window = 12
    tau = 12

    df = pd.read_csv("data/Ohio.csv", index_col='time', infer_datetime_format=True, parse_dates=True)
    M = int(365 * M_years / 7)
    dd = denoise(df, M=M)
    dd = dd.assign(R=transform(dd.pi_inc, window=window, shift=shift)).dropna()
    
    res = infer(dd,
                n_bootstrap=n_bootstrap,
                tau=tau,
                estimators=BESTIMATORS,
                predictors=['R', 'AH'])

    aggd = decisions(res, estimators=BESTIMATORS)
    aggd['BCAD_refutes_causal'] = aggd[0.95] < 0
    pdb.set_trace()
    
if __name__ == '__main__':
    try:
        plac.call(main)
    except:
        import traceback, sys
        traceback.print_exc()
        tb = sys.exc_info()[2]        
        pdb.post_mortem(tb)
