import pdb

import pandas as pd
import plac
from bcad.dimensions import BESTIMATORS
from bcad.helpers import infer, decisions
from bcad.rypdal_sugihara import transform
from bcad.ssa import GSSA


def denoise(df, M, predictors=None):
    if M == 0:
        return df

    predictors = df.columns if predictors is None else predictors
    for pred in predictors:
        df[pred] = GSSA.reconstruct(data=df[pred], M=M)  # SSA with Gavish's threshold
    return df


def main(n_bootstrap: ("number of bootstrap cycles", 'option', 'n') = 200,
         M_years: ("number of years in SSA window", 'option', 'M') = 0,
         state: ("State to run on", 'option', 's') = 'Ohio'):
    shift = -1
    window = 12
    tau = 12

    ## AH absolute humidity, VP vapor pressure, 
    df = pd.read_csv(f"data/{state}.csv", index_col='time', infer_datetime_format=True, parse_dates=True)
    M = int(365 * M_years / 7)
    dd = denoise(df, M=M)

    ## R is the Reproduction number
    dd = dd.assign(R=transform(dd.pi_inc, window=window, shift=shift)).dropna()

    res = infer(dd,
                n_bootstrap=n_bootstrap,
                tau=tau,
                estimators=BESTIMATORS,
                predictors=['R', 'AH'])

    ## AH = absolute humidity, R = reproduction number calculated from
    ## weekly incidence.
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
