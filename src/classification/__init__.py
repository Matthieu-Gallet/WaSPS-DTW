"""
Classification module for hydrological regime classification.
"""

from .preprocessing import (
    load_classification_dataset,
    preprocess_samples,
    estimate_parameters_for_samples
)

from .barycenter_methods import (
    compute_barycenter_euclidean_raw,
    compute_barycenter_euclidean_params,
    compute_barycenter_wasserstein_sgd
)

from .distance_computation import (
    compute_sdtw_distance_euclidean,
    compute_sdtw_distance_wasserstein
)

from .evaluation import (
    classify_by_nearest_barycenter,
    evaluate_classification,
    print_detailed_results,
    save_results_to_csv
)

from .kfold_evaluation import (
    run_kfold_classification
)

from .sensitivity_analysis import (
    run_gamma_sensitivity_analysis,
    run_sample_size_sensitivity_analysis
)

from .visualization import (
    plot_confusion_matrices,
    plot_barycenter_with_samples,
    plot_gamma_sensitivity,
    plot_sample_size_sensitivity
)

__all__ = [
    'load_classification_dataset',
    'preprocess_samples',
    'estimate_parameters_for_samples',
    'compute_barycenter_euclidean_raw',
    'compute_barycenter_euclidean_params',
    'compute_barycenter_wasserstein_sgd',
    'compute_sdtw_distance_euclidean',
    'compute_sdtw_distance_wasserstein',
    'classify_by_nearest_barycenter',
    'evaluate_classification',
    'print_detailed_results',
    'save_results_to_csv',
    'run_kfold_classification',
    'run_gamma_sensitivity_analysis',
    'run_sample_size_sensitivity_analysis',
    'plot_confusion_matrices',
    'plot_barycenter_with_samples',
    'plot_gamma_sensitivity',
    'plot_sample_size_sensitivity'
]
