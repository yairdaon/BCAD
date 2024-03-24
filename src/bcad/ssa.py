#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:08:37 2018

@author: Vivien Sainte Fare Garnot

This module is modified from mcssa: 
https://github.com/VSainteuf/mcssa
"""
import numpy as np
import pandas as pd
from optht import optht


def embedded(x, M):
    """Computes the M embedding of a time series

    Args:
        x (numpy array): data time series
        M (int): dimension of the desired embedding (window length)

    Returns:
        the embedded trajectory matrix
    """
    N2 = x.shape[0] - M + 1
    X = np.zeros((N2, M))
    for i in range(N2):
        X[i, :] = x[i:i + M]
    return np.matrix(X)


class GSSA:
    """Generic instance of a SSA analysis
    Args:
        data: array like, input time series, must be one dimensional

    Attributes:
        data (array): input time series
        index (index): index of the time series
        M (int): Window length
        N2 (int): Reduced length
        X (numpy matrix): Trajectory matrix
    """

    def __init__(self, M=None):
        self.M = M

    def fit(self, X):
        """Completes the Analysis on a SSA object

        Args:
            M (int): window length

        """
        X = np.array(X)
        X = np.squeeze(X)
        assert len(X.shape) == 1, "Data must be 1D"
        self.N = X.size
        self.N2 = self.N - self.M + 1

        if (self.N2 < self.M):
            raise ValueError('Window length is too big')
        else:
            self.X = embedded(X, self.M)

        self.U, self.s, self.Vt = np.linalg.svd(self.X)
        try:
            self.cutoff = optht(self.M / self.N2, sv=self.s)
        except Exception as e:
            print(str(e))
            self.cutoff = self.s.size
        return self

    def predict(self):
        S = np.zeros(self.X.shape)
        np.fill_diagonal(S, self.s)

        S[self.cutoff:] = 0
        H = self.U @ S @ self.Vt
        H = np.flipud(H)

        rec = np.empty(self.N)
        for k in range(self.N):
            rec[k] = np.diagonal(H, offset=-(self.N2 - 1 - k)).mean()
        return rec

    @staticmethod
    def reconstruct(data, M):
        if np.unique(data).size == 1:
            return data
        ssa = GSSA(M)
        if type(data) == pd.Series:
            return ssa.fit(data).predict()
        elif type(data) == pd.DataFrame:
            res = pd.DataFrame(index=data.index, columns=data.columns)
            for col in data.columns:
                res[col] = ssa.fit(data[col].values).predict()
            return res
        else:
            raise NotImplemented("Data must be pd.Series or pd.DataFrame")

