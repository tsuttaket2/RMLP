## Overview

This repository contains Python code implementing the Rational Multi-Layer Perceptrons (RMLP) model, along with experimentation on the MIMIC-III dataset. This implementation accompanies the paper:

Suttaket, T., & Kok, S. (2024). Interpretable Predictive Models for Healthcare via Rational Multi-Layer Perceptrons. ACM Transactions on Management Information Systems.

## Usage

### Data Preparation

1. Download the MIMIC-III dataset from [PhysioNet](https://mimic.physionet.org/).
2. Preprocess the data as required and save it in the `data/` directory.

### Training the Model

To train the RMLP model, run the following command:

    ```sh
    python train_rmlp.py --data_path data/mimic-iii/ --output_path results/
    ```
