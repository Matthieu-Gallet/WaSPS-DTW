#!/bin/bash

# Script pour lancer toutes les expériences séquentiellement avec plots

# Activation de l'environnement virtuel
source /home/mgallet/Documents/Codes/Python/3_DEVELOPPEMENT/WaSPS-DTW/WaSPS-DTW/venv/bin/activate

# Variables de sélection des classifications
classif1=True  # one-shot
classif2=True  # kfold
classif3=False  # gamma-sens
classif4=False # sample-sens

# Classifications conditionnelles
if [ "$classif1" = "True" ]; then
    echo "Lancement du mode one-shot avec plots..."
    python src/experiments/sdtw_barycenter_classification.py --mode one-shot --plot-barycenters --n-samples-plot 20
fi

if [ "$classif2" = "True" ]; then
    echo "Lancement du mode kfold..."
    python src/experiments/sdtw_barycenter_classification.py --mode kfold --n-splits 5
fi

if [ "$classif3" = "True" ]; then
    echo "Lancement du mode gamma-sens..."
    python src/experiments/sdtw_barycenter_classification.py --mode gamma-sens --gamma-values 0.001,0.01,0.1,1.0,10.0,100.0,1000.0 --n-splits 5
fi

if [ "$classif4" = "True" ]; then
    echo "Lancement du mode sample-sens..."
    python src/experiments/sdtw_barycenter_classification.py --mode sample-sens --sample-sizes 0.05,0.1,0.2,0.4,0.6,0.8,1.0 --n-splits 5
fi

# =============================================================================
# CPAZMaL SAR classification (Weibull)
# =============================================================================
cpazmal_compare=True   # 3-method comparison: euclidean_raw / euclidean_params / wasserstein_weibull
cpazmal_kmedoid=False  # Weibull-only kmedoid (already validated)
cpazmal_shapelet=False # Learning shapelets (slow, ~10 min)

if [ "$cpazmal_compare" = "True" ]; then
    echo "Lancement CPAZMaL compare (3 méthodes: euc_raw / euc_params / wasserstein_weibull)..."
    python src/experiments/cpazmal_classification.py --mode compare
fi

if [ "$cpazmal_kmedoid" = "True" ]; then
    echo "Lancement CPAZMaL kmedoid..."
    python src/experiments/cpazmal_classification.py --mode kmedoid
fi

if [ "$cpazmal_shapelet" = "True" ]; then
    echo "Lancement CPAZMaL shapelet..."
    python src/experiments/cpazmal_classification.py --mode shapelet
fi

# =============================================================================
# Model fit robustness (river discharge)
# =============================================================================
robustness=False  # Trace F1 vs. model-fit quality (varies exponential KS threshold)

if [ "$robustness" = "True" ]; then
    echo "Lancement model_fit_robustness..."
    python src/experiments/model_fit_robustness.py
fi

echo "Toutes les expériences sont terminées. Résultats dans results/regime_classification/ et autres dossiers results/"