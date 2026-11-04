#!/usr/bin/env bash
# Downloads the Spaceship Titanic competition files into data/.
# Requires Kaggle credentials (KAGGLE_API_TOKEN, or ~/.kaggle/kaggle.json)
# and having accepted the competition rules on kaggle.com.
set -euo pipefail
cd "$(dirname "$0")/.."
kaggle competitions download -c spaceship-titanic -p data
unzip -o -q data/spaceship-titanic.zip -d data
rm -f data/spaceship-titanic.zip
ls -l data/*.csv
