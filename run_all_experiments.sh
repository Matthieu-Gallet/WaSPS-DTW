#!/bin/bash

# Script pour lancer toutes les expériences séquentiellement avec plots

# Activation de l'environnement virtuel
source /home/mgallet/Documents/Codes/Python/3_DEVELOPPEMENT/WaSPS-DTW/WaSPS-DTW/venv/bin/activate

# Variables de sélection des classifications (regime hydro — désactivé pour cette session CPAZMaL)
classif1=False  # one-shot
classif2=False  # kfold
classif3=False  # gamma-sens
classif4=False  # sample-sens

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
# Dataset:  W=8 (64 px), T=56 (Jan 2020 – Dec 2021), HH, excl. HAG+ICA
# Balance:  strict subsample to minority class count (LAC ~60 → all classes equal)
# SGD:      lr=0.05, epochs=20, batch_size=4
# Plots:    confusion matrices + barycenters at γ=1e-4 and γ=100
# =============================================================================
cpazmal_compare=True   # 3-method comparison: euclidean_raw / euclidean_params / wasserstein_weibull
cpazmal_kmedoid=False   # Weibull-only kmedoid (SGD + divergence)
cpazmal_shapelet=False  # Learning shapelets (~10-20 min)

# Strict balance: subsample each class to the count of the smallest class (LAC)
CPAZMAL_BALANCE="--balance-mode subsample"
CPAZMAL_SGD="--sgd-epochs 20 --sgd-lr 0.05"
CPAZMAL_DATA="--window-size 8 --train-end 20211231 --predict-start 20220101 --predict-end 20221231 --exclude-classes HAG,ICA"
CPAZMAL_PLOT="--plot-gammas 0.0001,100.0"

if [ "$cpazmal_compare" = "True" ]; then
    echo "Lancement CPAZMaL compare (3 méthodes: euc_raw / euc_params / wasserstein_weibull)..."
    python src/experiments/cpazmal_classification.py --mode compare \
        $CPAZMAL_DATA $CPAZMAL_BALANCE $CPAZMAL_SGD $CPAZMAL_PLOT
fi

if [ "$cpazmal_kmedoid" = "True" ]; then
    echo "Lancement CPAZMaL kmedoid..."
    python src/experiments/cpazmal_classification.py --mode kmedoid \
        $CPAZMAL_DATA $CPAZMAL_BALANCE $CPAZMAL_SGD
fi

if [ "$cpazmal_shapelet" = "True" ]; then
    echo "Lancement CPAZMaL shapelet..."
    python src/experiments/cpazmal_classification.py --mode shapelet \
        $CPAZMAL_DATA $CPAZMAL_BALANCE $CPAZMAL_SGD
fi

# =============================================================================
# CPAZMaL sensitivity analysis — ntrain and gamma sweeps only (no W re-extraction)
# =============================================================================
cpazmal_sens_ntrain=True  # n_train sweep: effect of training set size
cpazmal_sens_gamma=True   # gamma sweep: effect of Soft-DTW regularisation

if [ "$cpazmal_sens_ntrain" = "True" ]; then
    echo "Lancement CPAZMaL sensitivity — ntrain sweep…"
    python src/experiments/cpazmal_sensitivity.py --sub-exp ntrain \
        --sgd-epochs 20 --n-seeds 1
fi

if [ "$cpazmal_sens_gamma" = "True" ]; then
    echo "Lancement CPAZMaL sensitivity — gamma sweep…"
    python src/experiments/cpazmal_sensitivity.py --sub-exp gamma \
        --sgd-epochs 20
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