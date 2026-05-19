"""
Learning Shapelets for time series classification, extended with a
SoftDTW + Wasserstein distance measure for stochastic time series.

Original code by Bäumle et al.:
  https://github.com/benibaeumle/Learning-Shapelets

Extensions for WaSPS-DTW:
- ``SoftDTWWassersteinDistFunction``: custom ``torch.autograd.Function`` that
  wraps the Cython SoftDTW-Wasserstein forward pass and implements an analytic
  backward pass through ``WassersteinDistance.jacobian_product_Y``.
- ``MinSoftDTWWassersteinDistBlock``: shapelet block using the above distance.
  Shapelets are stored as log-λ parameters (log-parameterisation guarantees
  λ > 0 at all times).  The forward pass applies sliding-window SoftDTW and
  returns the minimum distance across all positions.
- ``ShapeletsDistBlocks``, ``LearningShapeletsModel``, and
  ``LearningShapelets`` are extended to accept
  ``dist_measure='soft_dtw_wasserstein'``.

Usage example::

    from src.optimizer.learning_shapelets import LearningShapelets
    import torch

    clf = LearningShapelets(
        shapelets_size_and_len={10: 4, 20: 4},
        loss_func=torch.nn.CrossEntropyLoss(),
        in_channels=1,
        num_classes=5,
        dist_measure='soft_dtw_wasserstein',
        gamma=0.5,
        to_cuda=False,
    )
    clf.set_optimizer(torch.optim.Adam(clf.model.parameters(), lr=1e-3))
    clf.fit(X_train, y_train, epochs=50, batch_size=32)
    preds = clf.predict(X_test)
"""

from collections import OrderedDict
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# WaSPS-DTW imports (sys.path trick used throughout this project)
# ---------------------------------------------------------------------------
_src_dir = Path(__file__).parent.parent          # src/
sys.path.insert(0, str(_src_dir))

from sdtw.soft_dtw import SoftDTW               # noqa: E402
from sdtw.distance import WassersteinDistance   # noqa: E402


# =============================================================================
# Original benibaeumle/Learning-Shapelets blocks (verbatim)
# =============================================================================

class MinEuclideanDistBlock(nn.Module):
    """
    Calculates the euclidean distances of a bunch of shapelets to a data set and performs global min-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MinEuclideanDistBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels

        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        self.shapelets.retain_grad()

    def forward(self, x):
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        x = torch.cdist(x, self.shapelets, p=2)
        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3)
        x, _ = torch.min(x, 3)
        return x

    def get_shapelets(self):
        return self.shapelets.transpose(1, 0)

    def set_shapelet_weights(self, weights):
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        weights = weights.transpose(1, 0)
        if not list(weights.shape) == list(self.shapelets.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets.shape)}"
                             f"compared to {list(weights.shape)}")
        self.shapelets = nn.Parameter(weights)
        self.shapelets.retain_grad()

    def set_weights_of_single_shapelet(self, j, weights):
        if not list(weights.shape) == list(self.shapelets[:, j].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets[:, j].shape)}"
                             f"compared to {list(weights[j].shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.Tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets[:, j] = weights
        self.shapelets = nn.Parameter(self.shapelets).contiguous()
        self.shapelets.retain_grad()


class MaxCosineSimilarityBlock(nn.Module):
    """
    Calculates the cosine similarity of a bunch of shapelets to a data set and performs global max-pooling.
    Parameters
    ----------
    shapelets_size : int
    num_shapelets : int
    in_channels : int
    cuda : bool
    """
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MaxCosineSimilarityBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        self.relu = nn.ReLU()

        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                                dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        self.shapelets.retain_grad()

    def forward(self, x):
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        x = x / x.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        shapelets_norm = self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)
        x = torch.matmul(x, shapelets_norm.transpose(1, 2))
        n_dims = x.shape[1]
        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3) / n_dims
        x = self.relu(x)
        x, _ = torch.max(x, 3)
        return x

    def get_shapelets(self):
        return self.shapelets.transpose(1, 0)

    def set_shapelet_weights(self, weights):
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        weights = weights.transpose(1, 0)
        if not list(weights.shape) == list(self.shapelets.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets.shape)} "
                             f"compared to {list(weights.shape)}")
        self.shapelets = nn.Parameter(weights)

    def set_weights_of_single_shapelet(self, j, weights):
        if not list(weights.shape) == list(self.shapelets[:, j].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets[:, j].shape)} "
                             f"compared to {list(weights[j].shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.Tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets[:, j] = weights
        self.shapelets = nn.Parameter(self.shapelets).contiguous()


class MaxCrossCorrelationBlock(nn.Module):
    """
    Calculates the cross-correlation of a bunch of shapelets to a data set via convolution
    and performs global max-pooling.
    """
    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MaxCrossCorrelationBlock, self).__init__()
        self.shapelets = nn.Conv1d(in_channels, num_shapelets, kernel_size=shapelets_size)
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.cuda()

    def forward(self, x):
        x = self.shapelets(x)
        x, _ = torch.max(x, 2, keepdim=True)
        return x.transpose(2, 1)

    def get_shapelets(self):
        return self.shapelets.weight.data

    def set_shapelet_weights(self, weights):
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        if not list(weights.shape) == list(self.shapelets.weight.data.shape):
            raise ValueError(f"Shapes do not match.")
        self.shapelets.weight.data = weights

    def set_weights_of_single_shapelet(self, j, weights):
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        if self.to_cuda:
            weights = weights.cuda()
        self.shapelets.weight.data[j, :] = weights


class ShapeletsDistanceLoss(nn.Module):
    """Shapelet distance regularization loss (top-k distance to data)."""
    def __init__(self, dist_measure='euclidean', k=6):
        super(ShapeletsDistanceLoss, self).__init__()
        # treat soft_dtw_wasserstein the same as euclidean for this regularizer
        if dist_measure == 'soft_dtw_wasserstein':
            dist_measure = 'euclidean'
        if dist_measure not in ('euclidean', 'cosine'):
            raise ValueError("Parameter 'dist_measure' must be either of 'euclidean' or 'cosine'.")
        if not isinstance(k, int):
            raise ValueError("Parameter 'k' must be an integer.")
        self.dist_measure = dist_measure
        self.k = k

    def forward(self, x):
        y_top, _ = torch.topk(x.clamp(1e-8), self.k,
                               largest=self.dist_measure != 'euclidean',
                               sorted=False, dim=0)
        if self.dist_measure == 'euclidean':
            return torch.mean(y_top)
        return torch.mean(1 - y_top)


class ShapeletsSimilarityLoss(nn.Module):
    """Cosine-similarity-based diversity regularization loss between shapelets."""
    def __init__(self):
        super(ShapeletsSimilarityLoss, self).__init__()

    def cosine_distance(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        x1 = x1.unfold(2, x2.shape[2], 1).contiguous()
        x1 = x1.transpose(0, 1)
        x1 = x1 / x1.norm(p=2, dim=3, keepdim=True).clamp(min=eps)
        x2 = x2 / x2.norm(p=2, dim=2, keepdim=True).clamp(min=eps)
        x1 = torch.matmul(x1, x2.transpose(1, 2))
        n_dims = x1.shape[1]
        x1 = torch.sum(x1, dim=1) / n_dims
        return x1

    def forward(self, shapelet_blocks):
        losses = 0.
        for block in shapelet_blocks:
            shapelets = block[1]
            shapelets.retain_grad()
            sim = self.cosine_distance(shapelets, shapelets)
            losses += torch.mean(sim)
        return losses


# =============================================================================
# WaSPS-DTW extension: SoftDTW-Wasserstein distance for shapelets
# =============================================================================

class SoftDTWWassersteinDistFunction(torch.autograd.Function):
    """
    Custom autograd function for the SoftDTW-Wasserstein distance between two
    scalar time series of exponential rate parameters λ.

    Forward pass
    ------------
    Calls the Cython SoftDTW implementation with a ``WassersteinDistance``
    cost matrix.

    Backward pass
    -------------
    Uses the analytic gradient of W2²(λ_x, λ_y) = (1/λ_x − 1/λ_y)² w.r.t.
    the shapelet parameters λ_y, computed by
    ``WassersteinDistance.jacobian_product_Y``.

    Inputs
    ------
    x_params : tensor(float) of shape (T,)
        Rate parameters (λ > 0) of the time series.  No gradient is computed
        for this input.
    shapelet_params : tensor(float) of shape (S,)
        Rate parameters (λ > 0) of the shapelet.  Gradient is computed here.
    gamma : float
        SoftDTW regularisation parameter.

    Returns
    -------
    distance : scalar tensor
        SoftDTW-Wasserstein distance (differentiable w.r.t. ``shapelet_params``).
    """

    @staticmethod
    def forward(ctx, x_params, shapelet_params, gamma):
        x_np = x_params.detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
        s_np = shapelet_params.detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
        g = float(gamma)

        dist_obj = WassersteinDistance(
            x_np, s_np,
            distribution='exponential',
            precompute_params=True,
            X_is_params=True,
            Y_is_params=True,
        )
        sdtw = SoftDTW(dist_obj, gamma=g)
        distance = float(sdtw.compute())
        E = sdtw.grad()                           # (T, S) — gradient w.r.t. D
        G_Y = dist_obj.jacobian_product_Y(E)      # (S, 1) — dL/dλ_y

        ctx.save_for_backward(
            torch.tensor(G_Y.flatten(), dtype=torch.float32)
        )
        return torch.tensor(distance, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        G_Y, = ctx.saved_tensors
        # Propagate gradient to shapelet params; move to same device as grad_output
        return None, grad_output * G_Y.to(grad_output.device), None


class MinSoftDTWWassersteinDistBlock(nn.Module):
    """
    Shapelet distance block using SoftDTW-Wasserstein as the distance measure.

    Each shapelet is a sequence of positive exponential rate parameters λ,
    stored in log space to enforce λ > 0 unconditionally.

    The forward pass computes, for each (sample, shapelet) pair, the minimum
    SoftDTW-Wasserstein distance over all sliding-window positions of the time
    series.  This mirrors the sliding-window min-pooling strategy of
    ``MinEuclideanDistBlock``.

    Parameters
    ----------
    shapelets_size : int
        Length S of the shapelets (in time steps).
    num_shapelets : int
        Number of shapelets in this block.
    gamma : float
        SoftDTW regularisation parameter γ (lower → closer to hard DTW).
    to_cuda : bool
        If True, move tensors to GPU.

    Notes
    -----
    The forward pass is **sequential** (loops over batch × shapelets ×
    windows) because the Cython SoftDTW kernel does not run on GPU and cannot
    be trivially vectorised.  Use small batch sizes and moderate numbers of
    shapelets for reasonable training times.
    """

    def __init__(self, shapelets_size, num_shapelets, gamma=1.0, to_cuda=False):
        super(MinSoftDTWWassersteinDistBlock, self).__init__()
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.gamma = gamma
        self.to_cuda = to_cuda

        # Log-parameterisation: log_shapelets ∈ ℝ, λ = exp(log_shapelets) > 0
        log_init = torch.randn(num_shapelets, shapelets_size)
        self.log_shapelets = nn.Parameter(log_init)
        self.log_shapelets.retain_grad()

    @property
    def shapelets(self):
        """Return the actual λ values (always positive via exp)."""
        return torch.exp(self.log_shapelets)     # (num_shapelets, S)

    def forward(self, x):
        """
        Compute the minimum SoftDTW-Wasserstein distance between each time
        series in the batch and each shapelet.

        Parameters
        ----------
        x : tensor(float) of shape (batch, 1, T)
            Exponential rate parameters λ of the input time series.  Values
            are clamped to (1e-6, ∞) to ensure positivity.

        Returns
        -------
        out : tensor(float) of shape (batch, 1, num_shapelets)
        """
        batch_size, _, T = x.shape
        S = self.shapelets_size
        n_positions = max(1, T - S + 1)

        lambda_shapelets = self.shapelets     # (num_shapelets, S)
        out_batch = []

        for b in range(batch_size):
            xi = x[b, 0, :].clamp(min=1e-6)  # (T,) positive λ values
            dists_per_shapelet = []

            for k in range(self.num_shapelets):
                sk = lambda_shapelets[k, :]   # (S,) — depends on log_shapelets

                # Sliding-window min-pooling
                window_dists = []
                for p in range(n_positions):
                    x_sub = xi[p:p + S].clamp(min=1e-6)
                    dist = SoftDTWWassersteinDistFunction.apply(
                        x_sub, sk, self.gamma
                    )
                    window_dists.append(dist)

                min_dist, _ = torch.stack(window_dists).min(dim=0)
                dists_per_shapelet.append(min_dist)

            out_batch.append(torch.stack(dists_per_shapelet))

        out = torch.stack(out_batch)           # (batch, num_shapelets)
        return out.unsqueeze(1)                # (batch, 1, num_shapelets)

    def get_shapelets(self):
        """Return shapelets as λ values, shape (num_shapelets, shapelets_size)."""
        return self.shapelets.detach()

    def set_shapelet_weights(self, weights):
        """Set shapelet weights from λ values (will be log-transformed internally)."""
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        self.log_shapelets = nn.Parameter(torch.log(weights.clamp(min=1e-6)))
        self.log_shapelets.retain_grad()

    def set_weights_of_single_shapelet(self, j, weights):
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        self.log_shapelets.data[j, :] = torch.log(weights.clamp(min=1e-6))


# =============================================================================
# ShapeletsDistBlocks — extended to support 'soft_dtw_wasserstein'
# =============================================================================

class ShapeletsDistBlocks(nn.Module):
    """
    Defines shapelet blocks (one per distinct length), concatenates their
    outputs.

    Parameters
    ----------
    shapelets_size_and_len : dict(int → int)
        ``{shapelet_length: num_shapelets, ...}``
    in_channels : int
    dist_measure : str
        ``'euclidean'``, ``'cross-correlation'``, ``'cosine'``, or
        ``'soft_dtw_wasserstein'``.
    to_cuda : bool
    gamma : float
        SoftDTW γ — only used when ``dist_measure='soft_dtw_wasserstein'``.
    """

    def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='euclidean',
                 to_cuda=True, gamma=1.0):
        super(ShapeletsDistBlocks, self).__init__()
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = OrderedDict(
            sorted(shapelets_size_and_len.items(), key=lambda x: x[0])
        )
        self.in_channels = in_channels
        self.dist_measure = dist_measure
        self.gamma = gamma

        if dist_measure == 'euclidean':
            self.blocks = nn.ModuleList([
                MinEuclideanDistBlock(
                    shapelets_size=sz, num_shapelets=n,
                    in_channels=in_channels, to_cuda=to_cuda)
                for sz, n in self.shapelets_size_and_len.items()
            ])
        elif dist_measure == 'cross-correlation':
            self.blocks = nn.ModuleList([
                MaxCrossCorrelationBlock(
                    shapelets_size=sz, num_shapelets=n,
                    in_channels=in_channels, to_cuda=to_cuda)
                for sz, n in self.shapelets_size_and_len.items()
            ])
        elif dist_measure == 'cosine':
            self.blocks = nn.ModuleList([
                MaxCosineSimilarityBlock(
                    shapelets_size=sz, num_shapelets=n,
                    in_channels=in_channels, to_cuda=to_cuda)
                for sz, n in self.shapelets_size_and_len.items()
            ])
        elif dist_measure == 'soft_dtw_wasserstein':
            self.blocks = nn.ModuleList([
                MinSoftDTWWassersteinDistBlock(
                    shapelets_size=sz, num_shapelets=n,
                    gamma=gamma, to_cuda=to_cuda)
                for sz, n in self.shapelets_size_and_len.items()
            ])
        else:
            raise ValueError(
                "dist_measure must be one of 'euclidean', 'cross-correlation', "
                "'cosine', or 'soft_dtw_wasserstein'"
            )

    def forward(self, x):
        out = (torch.tensor([], dtype=torch.float).cuda()
               if self.to_cuda else torch.tensor([], dtype=torch.float))
        for block in self.blocks:
            out = torch.cat((out, block(x)), dim=2)
        return out

    def get_blocks(self):
        return self.blocks

    def get_block(self, i):
        return self.blocks[i]

    def set_shapelet_weights_of_block(self, i, weights):
        self.blocks[i].set_shapelet_weights(weights)

    def get_shapelets_of_block(self, i):
        return self.blocks[i].get_shapelets()

    def get_shapelet(self, i, j):
        return self.blocks[i].get_shapelets()[j, :]

    def set_shapelet_weights_of_single_shapelet(self, i, j, weights):
        self.blocks[i].set_weights_of_single_shapelet(j, weights)

    def get_shapelets(self):
        max_shapelet_len = max(self.shapelets_size_and_len.keys())
        num_total = sum(self.shapelets_size_and_len.values())
        shapelets = torch.Tensor(num_total, self.in_channels, max_shapelet_len)
        shapelets[:] = np.nan
        start = 0
        for block in self.blocks:
            end = start + block.num_shapelets
            shapelets[start:end, :, :block.shapelets_size] = block.get_shapelets().unsqueeze(1)
            start = end
        return shapelets


# =============================================================================
# LearningShapeletsModel
# =============================================================================

class LearningShapeletsModel(nn.Module):
    """
    Puts ``ShapeletsDistBlocks`` together with a linear classifier.

    Parameters
    ----------
    shapelets_size_and_len : dict(int → int)
    in_channels : int
    num_classes : int
    dist_measure : str
    to_cuda : bool
    gamma : float
        SoftDTW γ (only for ``'soft_dtw_wasserstein'``).
    """

    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2,
                 dist_measure='euclidean', to_cuda=True, gamma=1.0):
        super(LearningShapeletsModel, self).__init__()
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocks(
            in_channels=in_channels,
            shapelets_size_and_len=shapelets_size_and_len,
            dist_measure=dist_measure,
            to_cuda=to_cuda,
            gamma=gamma,
        )
        self.linear = nn.Linear(self.num_shapelets, num_classes)
        if self.to_cuda:
            self.cuda()

    def forward(self, x, optimize='acc'):
        x = self.shapelets_blocks(x)
        if optimize == 'acc':
            x = self.linear(x)
        x = torch.squeeze(x, 1)
        return x

    def transform(self, X):
        return self.shapelets_blocks(X)

    def get_shapelets(self):
        return self.shapelets_blocks.get_shapelets()

    def set_shapelet_weights(self, weights):
        start = 0
        for i, (sz, n) in enumerate(self.shapelets_size_and_len.items()):
            end = start + n
            self.set_shapelet_weights_of_block(i, weights[start:end, :, :sz])
            start = end

    def set_shapelet_weights_of_block(self, i, weights):
        self.shapelets_blocks.set_shapelet_weights_of_block(i, weights)

    def set_weights_of_shapelet(self, i, j, weights):
        self.shapelets_blocks.set_shapelet_weights_of_single_shapelet(i, j, weights)


# =============================================================================
# LearningShapelets (sklearn-style wrapper)
# =============================================================================

class LearningShapelets:
    """
    Sklearn-style wrapper for Learning Shapelets time series classification.

    Extended to support ``dist_measure='soft_dtw_wasserstein'`` for
    stochastic time series represented as sequences of exponential rate
    parameters λ.

    Parameters
    ----------
    shapelets_size_and_len : dict(int → int)
        ``{shapelet_length: num_shapelets, ...}``
    loss_func : callable
        PyTorch loss function (e.g. ``torch.nn.CrossEntropyLoss()``).
    in_channels : int
    num_classes : int
    dist_measure : str
        ``'euclidean'``, ``'cross-correlation'``, ``'cosine'``, or
        ``'soft_dtw_wasserstein'``.
    verbose : int
    to_cuda : bool
    k : int
        Top-k for distance regularizer (set 0 to disable).
    l1 : float
        Weight for distance regularizer.
    l2 : float
        Weight for similarity regularizer.
    gamma : float
        SoftDTW γ — only used when ``dist_measure='soft_dtw_wasserstein'``.
    """

    def __init__(self, shapelets_size_and_len, loss_func, in_channels=1, num_classes=2,
                 dist_measure='euclidean', verbose=0, to_cuda=True,
                 k=0, l1=0.0, l2=0.0, gamma=1.0):

        self.model = LearningShapeletsModel(
            shapelets_size_and_len=shapelets_size_and_len,
            in_channels=in_channels,
            num_classes=num_classes,
            dist_measure=dist_measure,
            to_cuda=to_cuda,
            gamma=gamma,
        )
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.model.cuda()

        self.shapelets_size_and_len = shapelets_size_and_len
        self.loss_func = loss_func
        self.verbose = verbose
        self.optimizer = None

        if not all([k == 0, l1 == 0.0, l2 == 0.0]) and not all([k > 0, l1 > 0.0]):
            raise ValueError(
                "For the regularizer, 'k' and 'l1' must be greater than zero, "
                "or all three ('k', 'l1', 'l2') must be zero."
            )
        self.k = k
        self.l1 = l1
        self.l2 = l2
        self.loss_dist = ShapeletsDistanceLoss(dist_measure=dist_measure, k=k) if k > 0 else None
        self.loss_sim_block = ShapeletsSimilarityLoss()
        self.use_regularizer = k > 0 and l1 > 0.0

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_shapelet_weights(self, weights):
        self.model.set_shapelet_weights(weights)
        if self.optimizer is not None:
            warnings.warn(
                "Updating model parameters requires re-initialising the optimizer. "
                "Please call set_optimizer(optim) again."
            )

    def set_shapelet_weights_of_block(self, i, weights):
        self.model.set_shapelet_weights_of_block(i, weights)
        if self.optimizer is not None:
            warnings.warn(
                "Updating model parameters requires re-initialising the optimizer. "
                "Please call set_optimizer(optim) again."
            )

    def update(self, x, y):
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def loss_sim(self):
        blocks = [p for p in self.model.named_parameters() if 'shapelets_blocks' in p[0]]
        return self.loss_sim_block(blocks)

    def update_regularized(self, x, y):
        y_hat = self.model(x)
        loss_ce = self.loss_func(y_hat, y)
        loss_ce.backward(retain_graph=True)

        dists_mat = self.model(x, 'dists')
        loss_dist = self.loss_dist(dists_mat) * self.l1
        loss_dist.backward(retain_graph=True)

        loss_sim_val = 0.0
        if self.l2 > 0.0:
            loss_sim_val = self.loss_sim() * self.l2
            loss_sim_val.backward(retain_graph=True)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.l2 > 0.0:
            return loss_ce.item(), loss_dist.item(), loss_sim_val.item()
        return loss_ce.item(), loss_dist.item()

    def fit(self, X, Y, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        """
        Train the model.

        Parameters
        ----------
        X : array-like(float) of shape (n_samples, in_channels, len_ts)
        Y : array-like(long) of shape (n_samples,)
        """
        if self.optimizer is None:
            raise ValueError("No optimizer set. Call set_optimizer(optim) first.")

        if not isinstance(X, torch.Tensor):
            X = tensor(X, dtype=torch.float).contiguous()
        if not isinstance(Y, torch.Tensor):
            Y = tensor(Y, dtype=torch.long).contiguous()
        if self.to_cuda:
            X, Y = X.cuda(), Y.cuda()

        train_dl = DataLoader(TensorDataset(X, Y), batch_size=batch_size,
                              shuffle=shuffle, drop_last=drop_last)
        self.model.train()

        losses_ce, losses_dist, losses_sim = [], [], []
        pbar = tqdm(range(epochs), disable=self.verbose <= 0)
        curr_ce = curr_dist = curr_sim = 0.0

        for _ in pbar:
            for x, y in train_dl:
                if not self.use_regularizer:
                    curr_ce = self.update(x, y)
                    losses_ce.append(curr_ce)
                else:
                    result = self.update_regularized(x, y)
                    curr_ce, curr_dist = result[0], result[1]
                    curr_sim = result[2] if len(result) == 3 else 0.0
                    losses_ce.append(curr_ce)
                    losses_dist.append(curr_dist)
                    if self.l2 > 0.0:
                        losses_sim.append(curr_sim)

            desc = f"Loss: {curr_ce:.4f}"
            if self.use_regularizer:
                desc += f"  dist: {curr_dist:.4f}"
                if self.l2 > 0.0:
                    desc += f"  sim: {curr_sim:.4f}"
            pbar.set_description(desc)

        if not self.use_regularizer:
            return losses_ce
        if self.l2 > 0.0:
            return losses_ce, losses_dist, losses_sim
        return losses_ce, losses_dist

    def transform(self, X):
        """Shapelet transform (no gradient, returns numpy array)."""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        if self.to_cuda:
            X = X.cuda()
        with torch.no_grad():
            out = self.model.transform(X)
        return out.squeeze().cpu().detach().numpy()

    def fit_transform(self, X, Y, epochs=1, batch_size=256, shuffle=False, drop_last=False):
        self.fit(X, Y, epochs=epochs, batch_size=batch_size,
                 shuffle=shuffle, drop_last=drop_last)
        return self.transform(X)

    def predict(self, X, batch_size=256):
        """Run inference and return logits as numpy array."""
        X = tensor(X, dtype=torch.float32)
        if self.to_cuda:
            X = X.cuda()
        dl = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
        self.model.eval()
        result = None
        with torch.no_grad():
            for (x,) in dl:
                y_hat = self.model(x).cpu().detach().numpy()
                result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
        return result

    def get_shapelets(self):
        return self.model.get_shapelets().clone().cpu().detach().numpy()

    def get_weights_linear_layer(self):
        return (
            self.model.linear.weight.data.clone().cpu().detach().numpy(),
            self.model.linear.bias.data.clone().cpu().detach().numpy(),
        )
