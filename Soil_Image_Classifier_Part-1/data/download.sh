#!/bin/bash

# Download and unzip dataset

mkdir -p ./data/raw
kaggle competitions download -c soil-classification
unzip soil-classification.zip -d ./data/raw