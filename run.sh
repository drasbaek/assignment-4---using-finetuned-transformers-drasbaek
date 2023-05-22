#!/usr/bin/env bash
# create virtual environment called lang_modelling_env
python3 -m venv emotion_classification_env

# activate virtual environment
source ./emotion_classification_env/bin/activate

# install requirements
echo "[INFO] Installing requirements..."
python3 -m pip install -r requirements.txt

# run classification with default model
echo "[INFO] Running classification..."
python3 src/classify.py -m "j-hartmann/emotion-english-distilroberta-base"

# create visualizations
echo "[INFO] Creating visualizations..."
python3 src/visualize.py -m "j-hartmann/emotion-english-distilroberta-base"

# deactivate virtual environment
deactivate