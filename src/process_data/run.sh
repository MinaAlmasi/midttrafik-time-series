#!/bin/bash

# get the dir 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# venv 
source "$SCRIPT_DIR/../../env/bin/activate"

# change script dir 
cd "$SCRIPT_DIR"

# run the script
echo "[INFO:] Subsetting direction of 1A and saving by running 'subset_direction.py' ..."
python subset_direction.py

echo "[INFO:] Subsetting five stops of bus 1A by running 'subset_stops.py' ..."
python subset_stops.py

echo "[INFO:] Cleaning the five stops by running 'clean_stops.py' ..."
python clean_stops.py

echo "[INFO:] DONE!"
