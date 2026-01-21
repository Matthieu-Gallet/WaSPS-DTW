"""
Time series and geographic zone visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path


def plot_zone(ax, lambda_series, bary, title_suffix, markers=None, colors=None):
    """
    Plot a geographic zone with its series and barycenter.
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    lambda_series : list
        List of lambda series for this zone
    bary : array
        Barycenter for this zone
    title_suffix : str
        Title suffix for the plot
    markers : list, optional
        List of markers for each series
    colors : list, optional
        List of colors for each series
    """
    if colors is None:
        colors = ['blue', 'red', 'orange', 'purple']
    if markers is None:
        markers = ['o', 's', '^', 'v']
    
    ax.set_yscale('log')

    # Plot individual series
    for series_idx, lambda_vals in enumerate(lambda_series):
        coords = np.arange(len(lambda_vals))
        marker = markers[series_idx % len(markers)]
        color = colors[series_idx % len(colors)]
        ax.plot(coords, lambda_vals, marker=marker,
               color=color, linewidth=1, markersize=3,
               alpha=0.25, label=f'Series {series_idx + 1}')

    # Plot barycenter
    coords_bary = np.arange(len(bary))
    ax.plot(coords_bary, bary, 'd-', color='green', linewidth=3,
           markersize=4, label='Barycenter',
           path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])

    ax.set_title(f'{title_suffix}')
    ax.set_xlabel('Temporal index')
    ax.set_ylabel('Parameter lambda (log)')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)


def create_geographic_barycenter_figures(results, gamma, figures_dir):
    """
    Create 3 figures with 3 subplots each for geographic barycenters.

    Parameters
    ----------
    results : dict
        Barycenter computation results
    gamma : float
        Gamma value used
    figures_dir : str
        Output directory for figures
    """
    # Colors and markers for different series in each zone
    colors = ['blue', 'red', 'orange', 'purple']
    markers = ['o', 's', '^', 'v']

    # Create 3 figures (one per barycenter method)
    fig_raw, axes_raw = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig_est, axes_est = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig_wass, axes_wass = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    for zone_idx, (_, zone_data) in enumerate(results.items()):
        ax_raw = axes_raw[zone_idx]
        ax_est = axes_est[zone_idx]
        ax_wass = axes_wass[zone_idx]

        lambda_series = zone_data['lambda_series']
        bary_raw = zone_data['bary_raw']
        bary_est = zone_data['bary_est']
        bary_wass = zone_data['bary_wass']
        center_lat, center_lon = zone_data['center']

        # Plot each method
        plot_zone(ax_raw, lambda_series, bary_raw, 
                 f'Zone {zone_idx + 1} ({center_lat}, {center_lon})',
                 markers, colors)
        plot_zone(ax_est, lambda_series, bary_est, 
                 f'Zone {zone_idx + 1} ({center_lat}, {center_lon})',
                 markers, colors)
        plot_zone(ax_wass, lambda_series, bary_wass, 
                 f'Zone {zone_idx + 1} ({center_lat}, {center_lon})',
                 markers, colors)

    # Configure and save figures
    fig_raw.suptitle(f'Geographic barycenters - Euclidean on raw data (gamma={gamma})', fontsize=16)
    fig_est.suptitle(f'Geographic barycenters - Euclidean on parameters (gamma={gamma})', fontsize=16)
    fig_wass.suptitle(f'Geographic barycenters - Wasserstein SGD (gamma={gamma})', fontsize=16)

    for fig, suffix in [(fig_raw, 'euclidean_raw'), (fig_est, 'euclidean_params'), (fig_wass, 'wasserstein_sgd')]:
        plt.figure(fig.number)
        plt.tight_layout()
        output_path = Path(figures_dir) / f'geographic_barycenters_{suffix}_gamma_{gamma}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Figure saved: {output_path}")
