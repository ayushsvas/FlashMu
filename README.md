# [FLASHµ: Fast Localizing And Sizing of Holographic Microparticles](https://arxiv.org/abs/2503.11538)

## NOTE
Repository is still under development -- 

## Setup
To set up the environment, run the following command:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/ayushsvas/FlashMu.git
   cd FlashMu
   ```
2. **Micromamba Installation [OPTIONAL]**
   ```bash
    chmod +x micromamba_setup.sh
    ./micromamba_setup
   ```
3. **Get the dependencies**:
   ```bash
   micromamba create -f environment.yml
   micromamba activate FlashMu
   ```
   OR, if you have conda installed use conda in place of micromamba 

## Usage
-- 

## TODO

1. Need a common training script for `csfm` and `det_head`. The folders have to merge.
2. Usage
3. `config_dfc_unet.py` needs cleaning.
4. Add information on how to run `Fraunhofer.jl`.
5. Create a Python version of `Fraunhofer.jl`.
   
## BIBTEX 
```bibtex
@misc{paliwal2025flashmufastlocalizingsizing,
      title={FLASH{\mu}: Fast Localizing And Sizing of Holographic Microparticles}, 
      author={Ayush Paliwal and Oliver Schlenczek and Birte Thiede and Manuel Santos Pereira and Katja Stieger and Eberhard Bodenschatz and Gholamhossein Bagheri and Alexander Ecker},
      year={2025},
      eprint={2503.11538},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.11538}, 
}
```
