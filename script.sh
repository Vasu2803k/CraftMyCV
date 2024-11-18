#!/bin/bash

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Initialize conda for shell interaction
eval "$(conda shell.bash hook)"

# Get the environment name from environment.yml
ENV_NAME=$(grep "name:" environment.yml | cut -d' ' -f2)

# Check if environment exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME exists, activating..."
    conda activate "$ENV_NAME"
else
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
    conda activate "$ENV_NAME"
fi

# Install required packages if not already installed
echo "Ensuring all dependencies are installed..."
pip install -q crewai
pip install -q pdf2image
pip install -q python-docx
pip install -q pytesseract
pip install -q olefile

# Run the main application
python src/main.py --resume "/media/vasu/Hard Disk/Projects/CraftMyCV/data/Trial.pdf" --job-description "/media/vasu/Hard Disk/Projects/CraftMyCV/data/jd.txt"

# Deactivate conda environment
conda deactivate

