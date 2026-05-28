#!/bin/bash
set -e

echo "Navigeren naar de scripts-map..."
# Ga naar de juiste map toe
cd fuzzing-project/scripts

echo "Starten met het uitvoeren van de Python-scripts..."

python3 Extract.py
python3 Transform.py
python3 XGBoost_model.py

echo "Alle scripts zijn succesvol afgerond!"