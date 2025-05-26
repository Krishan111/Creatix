#!/bin/bash

# Download and unzip dataset

mkdir -p ./data/raw_2
kaggle competitions download -c soil-classification-part-2
unzip soil-classification.zip -d ./data/raw_2