# Project Title

## Overview

This project is structured to handle various tasks related to data analysis, dataset management, and training models. Below is an overview of the directory structure and the purpose of each file and folder.

## Directory Structure

```
config_dfc_unet.py
dfc.py
environment.yml
Fraunhofer.jl
micromamba_setup.sh
analysis/
    cloud_teaser.ipynb
    CloudTarget_eval.ipynb
    error_plots.ipynb
    plotting_tools.py
    quant_cnn_svm.ipynb
    synthetic_hologram_plots.ipynb
    utils.py
csfm/
    dataset_original.py
    dfc.py
    train_dfc.py
    utils_augmented.py
det_head/
    CNOModule.py
    holodataset.py
    mask_preprocess.py
    train_fcplusunet.py
    utils.py
```

### Root Directory

- `config_dfc_unet.py`: Configuration file for the DFC UNet model.
- `dfc.py`: Main script for DFC operations.
- `environment.yml`: Conda environment configuration file.
- `Fraunhofer.jl`: Julia script for Fraunhofer diffraction calculations.
- `micromamba_setup.sh`: Shell script for setting up Micromamba.

### analysis/

Contains Jupyter notebooks and utility scripts for data analysis and plotting.

- `cloud_teaser.ipynb`: Notebook for cloud teaser analysis.
- `CloudTarget_eval.ipynb`: Notebook for evaluating CloudTarget.
- `error_plots.ipynb`: Notebook for generating error plots.
- `plotting_tools.py`: Python script with tools for plotting.
- `quant_cnn_svm.ipynb`: Notebook for quantifying CNN and SVM results.
- `synthetic_hologram_plots.ipynb`: Notebook for plotting synthetic holograms.
- `utils.py`: Utility functions for analysis.

### csfm/

Contains scripts related to the CSFM dataset and training.

- `dataset_original.py`: Script for handling the original dataset.
- `dfc.py`: Script for DFC operations specific to CSFM.
- `train_dfc.py`: Script for training the DFC model.
- `utils_augmented.py`: Utility functions for augmented data.

### det_head/

Contains scripts related to detection head operations and training.

- `CNOModule.py`: Script for the CNO module.
- `holodataset.py`: Script for handling holographic datasets.
- `mask_preprocess.py`: Script for preprocessing masks.
- `train_fcplusunet.py`: Script for training the FC+UNet model.
- `utils.py`: Utility functions for detection head operations.

## Setup

To set up the environment, run the following command:

```sh
conda env create -f environment.yml
```

To activate the environment, use:

```sh
conda activate <environment_name>
```

## Usage

Provide specific usage instructions for your scripts and notebooks here.

## Contributing

Provide guidelines for contributing to the project.

## License

Specify the license under which the project is distributed.
