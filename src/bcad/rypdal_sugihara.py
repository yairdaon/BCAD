import numpy as np
import pandas as pd

def jacobian(X):
    x = X.fix
    y = X.shifted
    x_norm = np.linalg.norm(x)
    return np.dot(x/x_norm, y/x_norm)
    
    
def transform(x, window, shift):
    """
    The feature of Rypdal and Sugihara. 
    Returns numpy array L such that

    L[t] = argmin_l || l x[t:t-window] - x[t-shift:t-window-shift]  ||

    So that taking window =12 and shift = -1 gives a smoothed estimate
    for lambda such that x[t]lambda ~ x[t+1] at time t

    """
    ## x.shift(-1)[t] == x[t+1]: if x = [1,4,5,9] then x.shift(-1) = [4,5,9,nan]  
    dd = pd.DataFrame({'fix': x.values, 'shifted':x.shift(shift)}, index=x.index)
    
    L = np.full(dd.shape[0], np.nan)
    for i in range(window, dd.shape[0]):
        X = dd.iloc[i-window:i]
        L[i] = jacobian(X)

    return L
