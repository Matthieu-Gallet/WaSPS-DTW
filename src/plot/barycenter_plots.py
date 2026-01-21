"""
Barycenter visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import sys
from pathlib import Path

# Add parent directory for sdtw access
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from sdtw.distance import SquaredEuclidean
from sdtw.soft_dtw import SoftDTW


def plot_with_correspondences(ax, lambda1, lambda2, lambda_bary, title, gamma, threshold=0.1):
    """
    Plot two series with their barycenter and DTW correspondences.
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    lambda1 : array
        First lambda series
    lambda2 : array
        Second lambda series
    lambda_bary : array
        Barycenter lambda series
    title : str
        Plot title
    gamma : float
        Soft-DTW gamma parameter used
    threshold : float
        Weight threshold for correspondence lines
    """
    ax.set_yscale('log')
    
    coords1 = np.arange(len(lambda1))
    coords2 = np.arange(len(lambda2))
    coords_bary = np.arange(len(lambda_bary))
    
    # Plot series
    ax.plot(coords1, lambda1, 'o-', color='blue', linewidth=2.5, markersize=10,
            label='Series 1', path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])
    ax.plot(coords2, lambda2, 's-', color='red', linewidth=2.5, markersize=10,
            label='Series 2', path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])
    ax.plot(coords_bary, lambda_bary, 'd-', color='green', linewidth=3, markersize=12,
            label='Barycenter', path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
    
    # Compute alignments to visualize correspondences with weights
    # Series 1 -> Barycenter
    D1 = SquaredEuclidean(lambda1[:, np.newaxis], lambda_bary[:, np.newaxis])
    sdtw1 = SoftDTW(D1, gamma=gamma)
    sdtw1.compute()
    E1 = sdtw1.grad()  # Alignment weight matrix
    
    # Normalize weights for visualization
    E1_norm = E1 / (np.max(E1) + 1e-8)
    
    # Plot all correspondences with weight > threshold
    for i in range(len(lambda1)):
        for j in range(len(lambda_bary)):
            weight = E1_norm[i, j]
            if weight > threshold:
                ax.plot([coords1[i], coords_bary[j]], [lambda1[i], lambda_bary[j]],
                       'k-', alpha=weight*0.5, linewidth=weight*2)
    
    # Series 2 -> Barycenter
    D2 = SquaredEuclidean(lambda2[:, np.newaxis], lambda_bary[:, np.newaxis])
    sdtw2 = SoftDTW(D2, gamma=gamma)
    sdtw2.compute()
    E2 = sdtw2.grad()
    
    # Normalize weights
    E2_norm = E2 / (np.max(E2) + 1e-8)
    
    # Plot all correspondences with weight > threshold
    for i in range(len(lambda2)):
        for j in range(len(lambda_bary)):
            weight = E2_norm[i, j]
            if weight > threshold:
                ax.plot([coords2[i], coords_bary[j]], [lambda2[i], lambda_bary[j]],
                       'k-', alpha=weight*0.5, linewidth=weight*2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Temporal index', fontsize=12)
    ax.set_ylabel('Parameter lambda (log scale)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, which='both')


def plot_multiple_series(ax, lambda_list, lambda_bary, title, gamma, threshold=0.15):
    """
    Plot multiple series with their barycenter and DTW correspondences.
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    lambda_list : list
        List of lambda series
    lambda_bary : array
        Barycenter lambda series
    title : str
        Plot title
    gamma : float
        Soft-DTW gamma parameter used
    threshold : float
        Weight threshold for correspondence lines
    """
    ax.set_yscale('log')
    
    # Color palette for series
    colors = ['blue', 'red', 'orange', 'purple']
    markers = ['o', 's', '^', 'v']
    
    coords_bary = np.arange(len(lambda_bary))
    
    # Plot all series
    for idx, (lam, color, marker) in enumerate(zip(lambda_list, colors, markers)):
        coords = np.arange(len(lam))
        ax.plot(coords, lam, f'{marker}-', color=color, linewidth=2, markersize=8,
                label=f'Series {idx+1}', alpha=0.7,
                path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])
    
    # Plot barycenter
    ax.plot(coords_bary, lambda_bary, 'd-', color='green', linewidth=3, markersize=12,
            label='Barycenter', path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
    
    # Compute and plot correspondences for each series
    for idx, lam in enumerate(lambda_list):
        D = SquaredEuclidean(lam[:, np.newaxis], lambda_bary[:, np.newaxis])
        sdtw = SoftDTW(D, gamma=gamma)
        sdtw.compute()
        E = sdtw.grad()
        E_norm = E / (np.max(E) + 1e-8)
        
        # Plot correspondences in black
        coords = np.arange(len(lam))
        for i in range(len(lam)):
            for j in range(len(lambda_bary)):
                weight = E_norm[i, j]
                if weight > threshold:
                    ax.plot([coords[i], coords_bary[j]], [lam[i], lambda_bary[j]],
                           'k-', alpha=weight*0.3, linewidth=weight*1.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Temporal index', fontsize=12)
    ax.set_ylabel('Parameter lambda (log scale)', fontsize=12)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, which='both')


def create_barycenter_comparison_figure(lambda_est1, lambda_est2, lambda_bary_raw, 
                                        lambda_bary_est, lambda_bary_wass, gamma_value,
                                        output_path=None):
    """
    Create a figure comparing 3 barycenter methods.
    
    Parameters
    ----------
    lambda_est1 : array
        Estimated lambda values for series 1
    lambda_est2 : array
        Estimated lambda values for series 2
    lambda_bary_raw : array
        Barycenter from Euclidean on raw data
    lambda_bary_est : array
        Barycenter from Euclidean on estimated parameters
    lambda_bary_wass : array
        Barycenter from Wasserstein SGD
    gamma_value : float
        Gamma parameter used
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib figure
        The created figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # Plot 1: Euclidean on raw data
    plot_with_correspondences(axes[0], lambda_est1, lambda_est2, lambda_bary_raw,
                            f'Euclidean on raw data\n(gamma={gamma_value})', gamma_value)
    
    # Plot 2: Euclidean on estimated parameters
    plot_with_correspondences(axes[1], lambda_est1, lambda_est2, lambda_bary_est,
                            f'Euclidean on estimated parameters\n(gamma={gamma_value})', gamma_value)
    
    # Plot 3: Wasserstein SGD
    plot_with_correspondences(axes[2], lambda_est1, lambda_est2, lambda_bary_wass,
                            f'Wasserstein SGD (softplus)\n(gamma={gamma_value})', gamma_value)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved: {output_path}")
    
    return fig
