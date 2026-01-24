#!/bin/bash

# Script pour lancer toutes les expériences séquentiellement avec plots

# Activation de l'environnement virtuel
source /home/mgallet/Documents/Codes/Python/3_DEVELOPPEMENT/WaSPS-DTW/WaSPS-DTW/venv/bin/activate

# Variables de sélection des classifications
classif1=True  # one-shot
classif2=True  # kfold
classif3=True  # gamma-sens
classif4=True  # sample-sens

# Variables de sélection des simulations
simu1=True  # barycenter_rmse_analysis.py
simu2=True  # geographic_barycenter.py
simu3=True  # simu_complex.py
simu4=True  # simu_simple.py

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

# Simulations conditionnelles
if [ "$simu1" = "True" ]; then
    echo "Lancement de la simulation 1 : barycenter_rmse_analysis.py..."
    python src/experiments/barycenter_rmse_analysis.py
fi

if [ "$simu2" = "True" ]; then
    echo "Lancement de la simulation 2 : geographic_barycenter.py..."
    python src/experiments/geographic_barycenter.py
fi

if [ "$simu3" = "True" ]; then
    echo "Lancement de la simulation 3 : simu_complex.py..."
    python src/experiments/simu_complex.py
fi

if [ "$simu4" = "True" ]; then
    echo "Lancement de la simulation 4 : simu_simple.py..."
    python src/experiments/simu_simple.py
fi

echo "Toutes les expériences sont terminées. Résultats dans results/regime_classification/ et autres dossiers results/"