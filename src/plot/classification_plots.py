"""
Classification visualization functions.

- Confusion matrix plots
- Per-class barycenter overlay plots (river, 4 methods)

All plots are designed for IEEE publications with Times New Roman fonts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, List, Union, Sequence, Mapping
from pathlib import Path
from sklearn.metrics import confusion_matrix


# =============================================================================
# Global settings for IEEE-compatible plots
# =============================================================================

def setup_ieee_style():
    """Configure matplotlib for IEEE publication style."""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'STIXGeneral']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = 8
    rcParams['axes.labelsize'] = 9
    rcParams['axes.titlesize'] = 9
    rcParams['xtick.labelsize'] = 7
    rcParams['ytick.labelsize'] = 7
    rcParams['legend.fontsize'] = 7
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.format'] = 'pdf'
    rcParams['axes.linewidth'] = 0.5
    rcParams['grid.linewidth'] = 0.5
    rcParams['lines.linewidth'] = 1.0


# =============================================================================
# Hydro-regime name mapping and plot constants
# =============================================================================

REGIMES = {
    "Pluvial modérément contrasté": {"code": "PM"},
    "Pluvial contrasté":             {"code": "PC"},
    "Pluvio-nival":                  {"code": "PN"},
    "Nivo-pluvial":                  {"code": "NP"},
    "Nival & nivo-glaciaire":        {"code": "NN"},
    "Nival":                         {"code": "NG"},
    "nivo-glaciaire":                {"code": "NG"},
}

# First-match reverse map: short code → full French name
_CODE_TO_NAME: Dict[str, str] = {}
for _regime_name, _regime_info in REGIMES.items():
    _code = _regime_info["code"]
    if _code not in _CODE_TO_NAME:
        _CODE_TO_NAME[_code] = _regime_name

# English names used in publication figures
_CODE_TO_NAME_EN: Dict[str, str] = {
    "PM": "Moderately contrasted pluvial",
    "PC": "Contrasted pluvial",
    "PN": "Pluvio-nival",
    "NP": "Nivo-pluvial",
    "NN": "Nival & nivo-glacial",
    "NG": "Nival",
}

# Methods displayed in per-class barycenter plots (in order) — current
# method_defs.py keys, div-mode variants except STA (see method_defs.make_softdtw_bary).
_PAIR_METHODS = ['wasps', 'eucl_params', 'eucl_raw', 'sta']

_METHOD_COLORS = {
    'wasps':       '#2ca02c',  # green
    'eucl_params': '#ff7f0e',  # orange
    'eucl_raw':    '#1f77b4',  # blue
    'sta':         '#d62728',  # red
}

_METHOD_LABELS = {
    'wasps':       'WaSPS (params)',
    'eucl_params': 'SoftDTW (params)',
    'eucl_raw':    'SoftDTW (raw)',
    'sta':         'STA',
}


# =============================================================================
# Confusion matrix plots
# =============================================================================

def plot_confusion_matrices(results: Dict, Y_test: np.ndarray,
                            idx_to_regime: Dict[int, str],
                            output_dir: str = None, save_pdf: bool = True):
    """
    Plot confusion matrices for available classification methods.

    Args:
        results: Dictionary with classification results
        Y_test: True test labels
        idx_to_regime: Mapping from label index to regime code
        output_dir: Output directory for saving
        save_pdf: Whether to save as PDF
    """
    setup_ieee_style()

    labels = sorted(idx_to_regime.keys())
    target_names = [idx_to_regime[i] for i in labels]
    method_candidates = [
        ('eucl_raw', 'Soft-DTW Euclidean\n(Raw Data)'),
        ('eucl_params', 'Soft-DTW Euclidean\n(Parameters)'),
        ('wasps', 'WaSPS-DTW\n(Parameters)'),
        ('sta', 'STA\n(Raw Data)'),
    ]
    methods = [(k, n) for k, n in method_candidates if k in results]
    if not methods:
        return

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(2.15 * n_methods + 0.8, 2.3))
    if n_methods == 1:
        axes = [axes]

    for idx, (method_key, method_name) in enumerate(methods):
        Y_pred = results[method_key]['predictions']
        cm = confusion_matrix(Y_test, Y_pred, labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.where(row_sums > 0,
                                  cm.astype('float') / np.maximum(row_sums, 1),
                                  0.0)

        ax = axes[idx]

        # Plot heatmap
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

        # Add text annotations
        thresh = 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm_normalized[i, j] > thresh else 'black'
                ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color=color, fontsize=8)

        ax.set_xticks(np.arange(len(target_names)))
        ax.set_yticks(np.arange(len(target_names)))
        ax.set_xticklabels(target_names, rotation=45, ha='right')
        ax.set_yticklabels(target_names)

        ax.set_title(method_name, fontsize=9)

        if idx == 0:
            ax.set_ylabel('True Class')
        ax.set_xlabel('Predicted Class')

        # Add F1 score below title
        f1 = results[method_key]['f1_weighted']
        ax.text(0.5, -0.35, f'F1={f1:.3f}', transform=ax.transAxes,
                ha='center', fontsize=8)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if save_pdf:
            plt.savefig(output_path / "confusion_matrices.pdf", bbox_inches='tight', dpi=300)

    plt.close()


# =============================================================================
# Per-class barycenter overlay plots
# =============================================================================

def plot_class_pair_barycenters(barycenters_by_method: Dict[str, Dict[int, np.ndarray]],
                                 X_train_params: List[np.ndarray],
                                 Y_train: np.ndarray,
                                 idx_to_regime: Dict[int, str],
                                 output_dir: str = None,
                                 save_pdf: bool = True,
                                 n_samples: int = 10,
                                 show_legend: Union[bool, Sequence[bool], Mapping[Union[int, str], bool]] = True):
    """
    Plot one figure per class (all 4 methods overlaid) with training sample traces.

    One PDF per class, saved under ``<output_dir>/class_pairs/``.

    Args:
        barycenters_by_method: Dict {method_key: {class_label: barycenter}} — method
            keys are current method_defs.py names (wasps, eucl_params, eucl_raw, sta).
            Every barycenter must already be in PARAMETER space, shape (T, 1) or (T,)
            — a rate β trajectory. For eucl_raw/sta (natively raw-sample barycenters),
            the caller is responsible for fitting a distribution on top of the raw
            barycenter beforehand (this function does not do that itself).
        X_train_params: List of per-sample parameter arrays, each shape (T, 1) — the
            background sample traces are drawn from these regardless of which
            method's barycenter is on top (they represent the underlying data, not
            any one method's fit).
        Y_train: Integer class labels for every training sample.
        idx_to_regime: {class_label: regime_code}, e.g. {0: 'PM', 1: 'PC'}.
        output_dir: Base output directory; figures saved under ``<output_dir>/class_pairs/``.
        save_pdf: Whether to save figures as PDF.
        n_samples: Number of sample traces to draw per class.
        show_legend: Legend visibility control.
            - bool: same value for all classes.
            - Sequence[bool]: one flag per class (ordered by sorted class labels).
            - Mapping[int|str, bool]: keyed by class label or regime code.
    """
    setup_ieee_style()

    def _sanitize_filename(text: str) -> str:
        safe = ''.join(ch if (ch.isalnum() or ch in ['_', '-']) else '_' for ch in text.strip())
        while '__' in safe:
            safe = safe.replace('__', '_')
        return safe.strip('_') or 'class'

    all_classes = sorted(next(iter(barycenters_by_method.values())).keys())
    T = X_train_params[0].shape[0]
    time_axis = np.arange(T)

    legend_map = None
    legend_seq = None
    if isinstance(show_legend, Mapping):
        legend_map = dict(show_legend)
    elif isinstance(show_legend, (list, tuple, np.ndarray)) and not isinstance(show_legend, (str, bytes)):
        legend_seq = [bool(v) for v in show_legend]

    if output_dir:
        output_path = Path(output_dir) / "class_pairs"
        output_path.mkdir(parents=True, exist_ok=True)

    fig_width = 8./3.
    fig_height = 4./3.

    for class_label in all_classes:
        code = idx_to_regime.get(class_label, str(class_label))
        class_name = _CODE_TO_NAME_EN.get(code, code)

        if legend_map is not None:
            class_show_legend = bool(
                legend_map.get(class_label, legend_map.get(str(class_label), legend_map.get(code, True)))
            )
        elif legend_seq is not None:
            class_pos = all_classes.index(class_label)
            class_show_legend = legend_seq[class_pos] if class_pos < len(legend_seq) else True
        else:
            class_show_legend = bool(show_legend)

        class_indices = [i for i in range(len(X_train_params)) if Y_train[i] == class_label]
        rng_pre = np.random.RandomState(42 + class_label)
        rng_pre.shuffle(class_indices)

        y_sample_vals = []
        for idx in class_indices[:n_samples]:
            s = X_train_params[idx]
            vals = 1.0 / np.maximum(s[:, 0], 1e-10)
            y_sample_vals.append(vals)

        if y_sample_vals:
            all_sv = np.concatenate(y_sample_vals)
            pos = all_sv[all_sv > 0]
            y_lo = float(np.nanpercentile(pos, 1)) * 0.85
            y_hi = float(np.nanpercentile(pos, 99)) * 1.15
            y_lo = max(y_lo, 1e-3)
        else:
            y_lo, y_hi = 0.1, 1000.0

        _lw_sp = min(0.2, 0.7 * (5.0 / max(n_samples, 5)) ** 0.23)
        _al_sp = min(0.10, 0.5 * (5.0 / max(n_samples, 5)) ** 0.65)

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        fig.subplots_adjust(bottom=0.28 if class_show_legend else 0.16, top=0.95, left=0.10, right=0.99)

        for idx in class_indices[:n_samples]:
            s = X_train_params[idx]
            ax.plot(time_axis, 1.0 / np.maximum(s[:, 0], 1e-10),
                    color='black', alpha=_al_sp, linewidth=_lw_sp)

        for method_key in _PAIR_METHODS:
            if method_key not in barycenters_by_method:
                continue
            bary = barycenters_by_method[method_key].get(class_label)
            if bary is None:
                continue

            b = np.asarray(bary)
            beta = b[:, 0] if b.ndim > 1 else b
            b_plot = 1.0 / np.maximum(beta, 1e-10)

            ax.plot(time_axis, b_plot,
                    color=_METHOD_COLORS[method_key],
                    linewidth=1.2,
                    zorder=5)

        ax.set_yscale('log')
        ax.set_ylim(0.25 * y_lo, 10 * y_hi)
        ax.tick_params(axis='x', rotation=0, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_ylabel(r'$\lambda\,(m^3/s)$', fontsize=8)
        ax.set_xlabel('Time step', fontsize=8)
        ax.grid(True, alpha=0.3)

        if class_show_legend:
            legend_handles = [
                plt.Line2D([0], [0],
                           color=_METHOD_COLORS[m],
                           linewidth=1.2,
                           label=_METHOD_LABELS[m])
                for m in _PAIR_METHODS
                if m in barycenters_by_method
            ]
            legend_handles.append(
                plt.Line2D([0], [0], color='black', lw=0.8, alpha=0.5, label='Samples')
            )
            fig.legend(
                handles=legend_handles,
                loc='lower center',
                ncol=len(legend_handles),
                fontsize=6.5,
                bbox_to_anchor=(0.5, 0.03),
                frameon=True,
            )

        if output_dir and save_pdf:
            filename = _sanitize_filename(class_name)
            plt.savefig(output_path / f"{filename}.pdf", bbox_inches='tight', dpi=300)

        plt.close()
