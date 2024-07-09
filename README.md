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
    python -u main_ihm.py --log_likelihood_fn=loglikelihood_filename --deep_supervision=1 --pattern_specs="3-7_6-4_9-7_12-7_15-2_18-7" --batch_size=128 --data=data_folder --dropout=0.0 --clip=0 --mlp_hidden_dim=0 --num_mlp_layers=1 --mlp_pattern_NN="30" --target_repl_coef=0.5 --epochs=200 --file_name=output_filename
    ```
