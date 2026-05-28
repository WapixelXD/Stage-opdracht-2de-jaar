#!/bin/bash
set -e

echo "Navigeren naar de scripts-map..."
# Ga naar de juiste map toe
cd fuzzing-project/scripts

echo "Starten met het uitvoeren van de Python-scripts..."

python3 Extract.py
echo "Data Extractie voltooid. Nu de data transformatie uitvoeren..."

python3 Transform.py

echo "Data Transformatie voltooid. Nu het XGBoost model trainen..."
python3 XGBoost_Model.py

echo "Alle scripts zijn succesvol afgerond!"