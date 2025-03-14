# FLASHµ: Fast Localizing And Sizing of Holographic Microparticles

## Setup 
1. **Clone the repository**:
   ```bash
   git clone https://github.com/ayushsvas/flashmu.git
   cd FlashMu
   ```

2. **Micromambda Installation [OPTIONAL]**
   ```bash
    chmod +x micromamba_setup.sh
    ./micromamba_setup
   ```
   
3. **Get the dependencies**:
   ```bash
   micromamba create -f environment.yml
   ```
   OR, if you have conda installed use conda inplace of micromamba 
   

For training and inference, we maintain a configuration file (config.py). train.py and infer.py scripts require path through config.py to training and test data, respectively. For your own data, make sure to type in the path to the folders containing them. 

## Training 
```bash
  python3 train.py
```

## Inference 
```bash
  python3 infer.py
```

