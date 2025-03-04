import pandas as pd
from scipy.stats import norm

from .bootstrap import bootstrap
from .dimensions import res2subd, BESTIMATORS

ALPHAS = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]


def infer(df,
          tau,
          predictors,
          n_bootstrap=200,
          estimators=BESTIMATORS):
    """
    Runs bcad basically
    """

    res = bootstrap(df=df,
                    n_bootstrap=n_bootstrap,
                    predictors=predictors,
                    tau=tau,
                    estimators=estimators).reset_index(drop=True)

    subd = res2subd(res)
    subd = subd.assign(tau=tau)

    return subd


def fon(x):
    """ x = {theta_i}_i=1^B and theta_i is ith BS sample."""
    mu, sig = x.mean(), x.std()
    ret = {alpha: mu + norm.ppf(alpha) * sig for alpha in ALPHAS}
    ret['pval'] = norm.cdf(mu / sig)
    ret = pd.Series(ret)
    return ret


def decisions(df, estimators):
    """
    Aggregates the output of infer to make decisions. 
    """
    
    df = df.query("estimator in @estimators")
    grouper = ['cause', 'outcome']
    theta_hat = df.query("cycle == 0").groupby(grouper).theta_hat.mean().sort_index()

    bs = df.groupby(grouper + ['cycle']).stat.mean().groupby(grouper).apply(fon).reset_index()
    bs = bs.pivot_table(index=grouper, columns='level_2', values='stat').sort_index()

    estimators = estimators if type(estimators) == str else ','.join(estimators)
    res = pd.concat([bs, theta_hat], axis=1).assign(estimator=estimators)
    return res


# def get_state_date(state):
#     from pathlib import Path
#     path = Path(__file__).absolute()
#     print(path)

    
