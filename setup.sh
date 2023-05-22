#!/usr/bin/env bash
# create virtual environment called lang_modelling_env
python3 -m venv emotion_classification_env

# activate virtual environment
source ./emotion_classification_env/bin/activate

# install requirements
python3 -m pip install -r requirements.txt