#!/bin/bash
# This script demonstrates multiple commands

echo "Download raw data"
wget -P datasets/phoNER_COVID19 https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/syllable/train_syllable.json
wget -P datasets/phoNER_COVID19 https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/syllable/dev_syllable.json
wget -P datasets/phoNER_COVID19 https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/syllable/test_syllable.json
