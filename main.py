import pdb
import pandas as pd
import plac
from matplotlib import pyplot as plt

from bcad.dimensions import BESTIMATORS
from bcad.helpers import infer, decisions
from bcad.rypdal_sugihara import transform
from bcad.ssa import denoise


def main(n_bootstrap: ("number of bootstrap cycles", 'option', 'n') = 200,
         M_years: ("number of years in SSA window", 'option', 'M') = 3, ## Three years window
         state: ("State to run on", 'option', 's') = 'Ohio'):
    shift = -1
    window = 12
    tau = 12

    ## AH absolute humidity, VP vapor pressure, 
    df = pd.read_csv(f"data/{state}.csv", index_col='time', infer_datetime_format=True, parse_dates=True)

    ## SSA window
    M = int(365 * M_years / 7) ## 365 days in a year, sampled weekly

    ## Denoise (smooth) with SSA
    dd = denoise(df, M=M)   
    
    ## R is the Reproduction number calculated from pi_inc (pneumonia and infulenza weekly incidence).
    dd = dd.assign(R=transform(dd.pi_inc, window=window, shift=shift)).dropna()

    ## Run BCAD, basically 
    res = infer(dd,
                n_bootstrap=n_bootstrap,
                tau=tau,
                estimators=BESTIMATORS,
                predictors=['R', 'AH'])    
    aggd = decisions(res, estimators=BESTIMATORS)

    ## BCAD refutes the causal realtion X -> Y if dim(X) > dim(Y) significantly.  
    aggd["Does BCAD refute this causal direction?"] = aggd[0.95] < 0
    print("\n\nAH: Absolute humidity.\nR: Instantaneous reproduction number of pneumonia and influenza incidence.")
    print("Significant p-values << 1 indicate refuting a causal relation.")
    print("\n\n")
    print(aggd[['pval', "Does BCAD refute this causal direction?"]])
    print("\n\n")
    pdb.set_trace()


if __name__ == '__main__':
    try:
        plac.call(main)
    except:
        import traceback, sys
        traceback.print_exc()
        tb = sys.exc_info()[2]
        pdb.post_mortem(tb)
