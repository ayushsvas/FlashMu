#!/bin/bash
# Check if the system is macOS or Linux
OS="$(uname -s)"
case "$OS" in
Darwin) OS="macOS" ;;
Linux) OS="Linux" ;;
*) echo "Unsupported OS: $OS" ; exit 1 ;;
esac
echo "Detected OS: $OS"

# Prompt the user to specify the directory for MicroMamba installation
read -p "Enter the directory where you want to install MicroMamba (default: $HOME/micromamba): " CUSTOM_MICROMAMBA_DIR
# Set the default directory if the user leaves the input blank
if [[ -z "$CUSTOM_MICROMAMBA_DIR" ]]; then
    CUSTOM_MICROMAMBA_DIR="$HOME/micromamba"
fi


# Install MicroMamba
echo "Installing MicroMamba..."
MAMBA_ROOT_PREFIX="$CUSTOM_MICROMAMBA_DIR" bash <(curl -L micro.mamba.pm/install.sh)

# Add MicroMamba to the PATH
if [[ "$OS" == "macOS" ]]; then
    echo "export MAMBA_ROOT_PREFIX=$CUSTOM_MICROMAMBA_DIR" >> ~/.zshrc
    echo "alias micromamba=$CUSTOM_MICROMAMBA_DIR/bin/micromamba" >> ~/.zshrc
    source ~/.zshrc
elif [[ "$OS" == "Linux" ]]; then
    echo "export MAMBA_ROOT_PREFIX=$CUSTOM_MICROMAMBA_DIR" >> ~/.bashrc
    echo "alias micromamba=$CUSTOM_MICROMAMBA_DIR/bin/micromamba" >> ~/.bashrc
    source ~/.bashrc
fi

# Path to the environment.yml file
ENV_YML_PATH="environment.yml"

# Check if the environment.yml file exists
if [[ ! -f "$ENV_YML_PATH" ]]; then
    echo "Error: $ENV_YML_PATH file not found!"
    exit 1
fi

# Create environment using environment.yml
echo "Creating the environment using $ENV_YML_PATH..."
micromamba create -f "$ENV_YML_PATH"

echo "Installation complete. To activate your environment, run:"
echo "micromamba activate myenv"