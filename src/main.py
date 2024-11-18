import argparse
from pathlib import Path
from crew import CraftMyCV
import json
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_file_path(file_path: str) -> Path:
    """Validate that the file exists and is readable"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    return path

def save_output(output: dict, output_path: str):
    """Save the crew's output to a file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Output saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save output: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create a customized CV using AI agents')
    parser.add_argument(
        '--resume',
        type=str,
        required=True,
        help='Path to the input resume file (PDF, DOCX, DOC, or TXT)'
    )
    parser.add_argument(
        '--job-description',
        type=str,
        required=True,
        help='Job description text or path to job description file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.json',
        help='Path to save the output (default: output.json)'
    )

    args = parser.parse_args()

    try:
        # Validate resume file path
        resume_path = validate_file_path(args.resume)
        
        # Handle job description (either file or direct text)
        job_description = ""
        if Path(args.job_description).exists():
            with open(args.job_description, 'r', encoding='utf-8') as f:
                job_description = f.read()
        else:
            job_description = args.job_description

        # Initialize the CV creation crew
        logger.info("Initializing CV creation crew...")
        cv_creator = CraftMyCV()

        # Run the CV creation process
        logger.info("Starting CV creation process...")
        result = cv_creator.create_cv(
            resume_path=str(resume_path),
            job_description=job_description
        )

        # Save the output
        logger.info("CV creation completed. Saving results...")
        save_output(result, args.output)

        logger.info("Process completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise SystemExit(1)
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
