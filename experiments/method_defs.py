"""Method table and factory functions shared across experiment scripts."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))

from costs import SqEuclidean, WaSPS
from softdtw import SoftDTW
from baselines.sta_wrapper import make_cost_fn as sta_cost_fn


# repr:   'params' → estimate distribution parameters first
#         'raw'    → use raw sample arrays directly
# *_nodiv: same representation as the base method, but is_divergence=False
#          (and, for wasps, log_correction=False) everywhere — for evaluating
#          non-divergence performance. Covers all 3 non-STA base methods (wasps,
#          eucl_params, eucl_raw) — STA has no divergence concept, no _nodiv variant.
#          See method_defs.py factory functions below and src/classification/nn.py's
#          is_divergence flag (KNN previously had no divergence concept at all).
_METHODS = {
    'wasps':             {'repr': 'params'},
    'wasps_nodiv':       {'repr': 'params'},
    'eucl_params':       {'repr': 'params'},
    'eucl_params_nodiv': {'repr': 'params'},
    'eucl_raw':          {'repr': 'raw'},
    'eucl_raw_nodiv':    {'repr': 'raw'},
    'sta':               {'repr': 'raw'},
}


def make_cost_fn(method: str, family: str, sta_epsilon: float):
    """Cost function for KNN and predict (data in positive-param space, no θ bijector)."""
    if method == 'wasps':
        return WaSPS(family, log_correction=True)
    if method == 'wasps_nodiv':
        return WaSPS(family, log_correction=False)
    if method in ('eucl_params', 'eucl_params_nodiv', 'eucl_raw', 'eucl_raw_nodiv'):
        return SqEuclidean()
    if method == 'sta':
        return sta_cost_fn(sta_epsilon)
    raise ValueError(f"unknown method '{method}'")


def make_softdtw_bary(method: str, family: str, sta_epsilon: float, gamma: float) -> SoftDTW:
    """SoftDTW instance for barycenter fitting."""
    if method == 'wasps':
        cost_fn = WaSPS(family, use_positivity_constraint=True)
        return SoftDTW(cost_fn, gamma, is_divergence=True, manual_grad=True)
    if method == 'wasps_nodiv':
        cost_fn = WaSPS(family, use_positivity_constraint=True, log_correction=False)
        return SoftDTW(cost_fn, gamma, is_divergence=False, manual_grad=True)
    if method in ('eucl_params', 'eucl_raw'):
        # use_positivity_constraint=True: optimize in θ-space (softplus reparametrization)
        # so the optimizer never drifts negative during gradient descent.
        cost = SqEuclidean(use_positivity_constraint=True)
        return SoftDTW(cost, gamma, is_divergence=True, manual_grad=False)
    if method in ('eucl_params_nodiv', 'eucl_raw_nodiv'):
        cost = SqEuclidean(use_positivity_constraint=True)
        return SoftDTW(cost, gamma, is_divergence=False, manual_grad=False)
    if method == 'sta':
        return SoftDTW(sta_cost_fn(sta_epsilon), gamma, is_divergence=True, manual_grad=False)
    raise ValueError(f"unknown method '{method}'")
