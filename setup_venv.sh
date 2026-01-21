#!/bin/bash

# Script pour créer un environnement virtuel Python et installer les dépendances

# Nom de l'environnement virtuel
VENV_NAME="venv"

# Créer l'environnement virtuel
echo "Création de l'environnement virtuel '$VENV_NAME'..."
python3 -m venv $VENV_NAME

# Activer l'environnement virtuel
echo "Activation de l'environnement virtuel..."
source $VENV_NAME/bin/activate

# Mettre à jour pip
echo "Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances depuis requirements.txt
echo "Installation des dépendances..."
pip install -r requirements.txt

# Installer les dépendances supplémentaires pour soft-dtw
echo "Installation des dépendances supplémentaires pour soft-dtw..."
pip install cython nose pytest

# Cloner et installer soft-dtw
echo "Installation de soft-dtw..."
if [ ! -d "soft-dtw" ]; then
    git clone https://github.com/mblondel/soft-dtw.git
fi
cd soft-dtw
python setup.py build_ext --inplace
python setup.py install
cd ..

echo ""
echo "=== Installation terminée ==="
echo "Pour utiliser soft-dtw dans vos scripts Python, ajoutez cette ligne au début :"
echo "import sys; sys.path.insert(0, '$PWD/soft-dtw')"
echo ""
echo "Ou activez l'environnement virtuel avec : source venv/bin/activate"
echo "Puis configurez PYTHONPATH avec : export PYTHONPATH=\"$PWD/soft-dtw:\$PYTHONPATH\""
echo ""
echo "Test de l'installation :"
echo "  source venv/bin/activate"
echo "  export PYTHONPATH=\"$PWD/soft-dtw:\$PYTHONPATH\""
echo "  cd soft-dtw && python -m pytest sdtw/tests/test_soft_dtw.py sdtw/tests/test_path.py -v"
echo ""
echo "Note: Le test test_chainer_func.py peut échouer à cause d'incompatibilités avec NumPy 2.0"