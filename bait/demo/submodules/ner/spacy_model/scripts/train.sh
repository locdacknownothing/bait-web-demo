#!/bin/bash
# This script demonstrates multiple commands

echo "Create config file"
python3 -m spacy init fill-config ./ner_config.cfg config.cfg

echo "Training ..."
mkdir ./result
python3 -m spacy train ./config.cfg --output ./result --paths.train ./train.spacy --paths.dev ./test.spacy --gpu-id 0
