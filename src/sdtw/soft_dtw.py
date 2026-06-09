# Author: Mathieu Blondel
# License: Simplified BSD

import numpy as np

from .soft_dtw_fast import _soft_dtw
from .soft_dtw_fast import _soft_dtw_grad


def sdtw_divergence(D_xy, D_xx, D_yy, gamma):
    """
    Soft-DTW divergence (Blondel et al. 2021).

    D_γ(X, Y) = SDTW_γ(X,Y) − ½(SDTW_γ(X,X) + SDTW_γ(Y,Y))

    This is a true divergence: D_γ(X,X) = 0 and D_γ(X,Y) ≥ 0 when the local
    cost matrix derives from a W₂² cost (proved in RESSOURCES/.ressources/proof.tex).

    The three pre-computed distance matrices are passed in to let the caller
    share cached matrices with the barycenter optimiser.

    Parameters
    ----------
    D_xy : array, shape = [m, n]   cross distance matrix
    D_xx : array, shape = [m, m]   self distance matrix for X
    D_yy : array, shape = [n, n]   self distance matrix for Y
    gamma : float                  Soft-DTW regularisation parameter

    Returns
    -------
    divergence : float
    """
    sdtw_xy = SoftDTW(D_xy, gamma=gamma).compute()
    sdtw_xx = SoftDTW(D_xx, gamma=gamma).compute()
    sdtw_yy = SoftDTW(D_yy, gamma=gamma).compute()
    return sdtw_xy - 0.5 * (sdtw_xx + sdtw_yy)


class SoftDTW(object):

    def __init__(self, D, gamma=1.0):
        """
        Parameters
        ----------
        D: array, shape = [m, n] or distance object
            Distance matrix between elements of two time series.

        gamma: float
            Regularization parameter.
            Lower is less smoothed (closer to true DTW).

        Attributes
        ----------
        self.R_: array, shape = [m + 2, n + 2]
            Accumulated cost matrix (stored after calling `compute`).
        """
        if hasattr(D, "compute"):
            self.D = D.compute()
        else:
            self.D = D

        self.D = self.D.astype(np.float64)

        self.gamma = gamma

    def compute(self):
        """
        Compute soft-DTW by dynamic programming.

        Returns
        -------
        sdtw: float
            soft-DTW discrepancy.
        """
        m, n = self.D.shape

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the backward recursion.
        self.R_ = np.zeros((m+2, n+2), dtype=np.float64)

        _soft_dtw(self.D, self.R_, gamma=self.gamma)

        return self.R_[m, n]

    def grad(self):
        """
        Compute gradient of soft-DTW w.r.t. D by dynamic programming.

        Returns
        -------
        grad: array, shape = [m, n]
            Gradient w.r.t. D.
        """
        if not hasattr(self, "R_"):
            raise ValueError("Needs to call compute() first.")

        m, n = self.D.shape

        # Add an extra row and an extra column to D.
        # Needed to deal with edge cases in the recursion.
        D = np.vstack((self.D, np.zeros(n)))
        D = np.hstack((D, np.zeros((m+1, 1))))

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the recursion.
        E = np.zeros((m+2, n+2))

        _soft_dtw_grad(D, self.R_, E, gamma=self.gamma)

        return E[1:-1, 1:-1]
