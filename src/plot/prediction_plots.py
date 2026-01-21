"""
Prediction visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_predictions(X_test, Y_test, Y_pred_dict, train_length, predict_length,
                    n_plots=4, output_path=None):
    """
    Plot prediction results comparing multiple models.
    
    Parameters
    ----------
    X_test : array, shape (n_samples, input_dim)
        Test input data
    Y_test : array, shape (n_samples, output_dim)
        Test ground truth
    Y_pred_dict : dict
        Dictionary mapping model names to their predictions
    train_length : int
        Length of training input
    predict_length : int
        Length of prediction
    n_plots : int
        Number of example plots to show
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib figure
        The created figure
    """
    fig = plt.figure(figsize=(20, 6))
    pos = 220

    # Color map for different models
    colors = {'Soft-DTW': 'blue', 'Wasserstein': 'green', 'Euclidean': 'orange'}
    
    n_plots = min(n_plots, X_test.shape[0])
    for i in range(n_plots):
        pos += 1
        ax = fig.add_subplot(pos)
        
        # Concatenate input and target for complete series
        ground_truth = np.concatenate([X_test[i], Y_test[i]])[-train_length - predict_length:]
        len_input = len(X_test[i])
        
        # Plot predictions for each model
        for model_name, Y_pred in Y_pred_dict.items():
            color = colors.get(model_name, 'blue')
            ax.plot(range(len_input, len_input + len(Y_pred[i])),
                    Y_pred[i],
                    alpha=0.75,
                    lw=3,
                    label=model_name,
                    color=color,
                    zorder=10)
        
        # Plot ground truth
        ax.semilogy(ground_truth,
                    c="k",
                    alpha=0.3,
                    lw=3,
                    label='Ground truth',
                    zorder=5)
        
        # Vertical line separating input/output
        y_min, y_max = ax.get_ylim()
        ax.plot([len_input, len_input],
                [y_min, y_max],
                lw=3,
                ls="--",
                c="red",
                alpha=0.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value (log scale)')
        ax.set_ylim(max(np.min(ground_truth), 1e-1), y_max)
        ax.legend()

    fig.set_tight_layout(True)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saving figure to {output_path}")
    
    return fig


def plot_losses(losses_dict, output_path=None, log_scale=True):
    """
    Plot training loss curves for multiple models.
    
    Parameters
    ----------
    losses_dict : dict
        Dictionary mapping model names to their loss histories
    output_path : str, optional
        Path to save the figure
    log_scale : bool
        Whether to use log scale for y-axis
        
    Returns
    -------
    fig : matplotlib figure
        The created figure
    """
    fig = plt.figure(figsize=(10, 5))
    
    colors = {'Soft-DTW': 'blue', 'Wasserstein': 'green', 'Euclidean': 'orange'}
    
    for model_name, losses in losses_dict.items():
        color = colors.get(model_name, 'blue')
        plt.plot(losses, label=model_name, color=color)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if log_scale:
        plt.yscale('log')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saving loss figure to {output_path}")
    
    return fig


def plot_single_prediction(ax, X_input, Y_true, Y_pred, input_length, 
                          pred_label='Prediction', color='blue'):
    """
    Plot a single prediction on an axis.
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    X_input : array
        Input sequence
    Y_true : array
        Ground truth output
    Y_pred : array
        Predicted output
    input_length : int
        Length of input sequence
    pred_label : str
        Label for prediction
    color : str
        Color for prediction line
    """
    ground_truth = np.concatenate([X_input, Y_true])
    
    ax.semilogy(ground_truth, c="k", alpha=0.3, lw=3, label='Ground truth')
    ax.plot(range(input_length, input_length + len(Y_pred)),
            Y_pred, alpha=0.75, lw=3, label=pred_label, color=color)
    
    y_min, y_max = ax.get_ylim()
    ax.axvline(x=input_length, color='red', linestyle='--', alpha=0.5, lw=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
