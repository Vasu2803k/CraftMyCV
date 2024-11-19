#!/bin/bash

# Function to display usage information
show_usage() {
    echo "Usage: $0 [--resume RESUME_PATH] [--job-description JD_PATH] [--output OUTPUT_PATH]"
    echo
    echo "Options:"
    echo "  --resume            Path to your resume file (PDF, DOCX, DOC, or TXT)"
    echo "  --job-description   Path to job description file or the job description text"
    echo "  --output           Optional: Path to save the output (default: output.json)"
    echo
    echo "Example:"
    echo "  $0 --resume ./data/resume.pdf --job-description ./data/jd.txt"
}

# Parse command line arguments
RESUME="/media/vasu/Hard Disk/Projects/CraftMyCV/data/Trial.pdf"
JD="/media/vasu/Hard Disk/Projects/CraftMyCV/data/jd.txt"
OUTPUT="output.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --job-description)
            JD="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$RESUME" ]] || [[ -z "$JD" ]]; then
    echo "Error: Both --resume and --job-description are required"
    show_usage
    exit 1
fi

# Check if files exist
if [[ ! -f "$RESUME" ]]; then
    echo "Error: Resume file not found: $RESUME"
    exit 1
fi

if [[ -f "$JD" ]] && [[ ! -r "$JD" ]]; then
    echo "Error: Job description file exists but is not readable: $JD"
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Check if environment.yml exists
if [[ ! -f "environment.yml" ]]; then
    echo "Error: environment.yml not found in current directory"
    exit 1
fi

# Initialize conda for shell interaction
eval "$(conda shell.bash hook)"

# Get the environment name from environment.yml
ENV_NAME=$(grep "name:" environment.yml | cut -d' ' -f2)

if [[ -z "$ENV_NAME" ]]; then
    echo "Error: Could not determine environment name from environment.yml"
    exit 1
fi

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME exists, activating..."
else
    echo "Creating conda environment from environment.yml..."
    if ! conda env create -f environment.yml; then
        echo "Error: Failed to create conda environment"
        exit 1
    fi
fi

# Activate the environment
echo "Activating environment $ENV_NAME..."
if ! conda activate "$ENV_NAME"; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

# Install required packages if not already installed
echo "Installing/updating dependencies..."
pip install -q llama-index-core llama-index-llms-anthropic llama-index-llms-openai llama-index-embeddings-openai crewai pdf2image python-docx pytesseract olefile pyyaml python-dotenv || {
    echo "Error: Failed to install required packages"
    conda deactivate
    exit 1
}

# Run the main application
echo "Starting CV customization process..."
python src/craft_my_cv_workflow.py --resume "$RESUME" --job-description "$JD" --output "$OUTPUT"

# Check if the script ran successfully
if [[ $? -eq 0 ]]; then
    echo "CV customization completed successfully!"
    echo "Output saved to: $OUTPUT"
else
    echo "Error: CV customization process failed"
    conda deactivate
    exit 1
fi

# Deactivate conda environment
conda deactivate

