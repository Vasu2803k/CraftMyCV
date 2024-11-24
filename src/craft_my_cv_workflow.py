import yaml
import traceback
import os
import json
import dotenv
from llama_index.core.workflow import (
    Workflow, 
    step, 
    Event,
    Context,
    StartEvent,
    StopEvent
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core.llms import LLM
from fallback_llm import FallbackLLM
import warnings
from pathlib import Path
import logging
import argparse
from tools.text_extractor_tool import TextExtractionTool
import asyncio
from time import time
from typing import Any

# Ignore warnings
warnings.filterwarnings("ignore")
# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def validate_file_path(file_path: str) -> Path:
    """Validate that the file exists and is readable"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    return path

async def save_output(output: Any, output_path: str):
    """Save the output to a file"""
    try:
        # Extract data based on the type of output
        if hasattr(output, 'result'):  # StopEvent case
            data_to_save = output.result
        elif hasattr(output, '_data'):  # Event case
            data_to_save = output._data
        elif isinstance(output, dict):  # Dictionary case
            data_to_save = output
        else:
            logger.error(f"Unexpected output type: {type(output)}")
            logger.error(f"Output attributes: {dir(output)}")
            raise ValueError(f"Cannot handle output of type {type(output)}")
            
        # Add validation and logging
        if not data_to_save:
            logger.error("Empty data received")
            logger.error(f"Original output type: {type(output)}")
            logger.error(f"Original output content: {str(output)}")
            raise ValueError("No data to save")
            
        logger.debug(f"Attempting to save output: {json.dumps(data_to_save, indent=2)[:200]}...")  # Log first 200 chars
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
        # Verify the file was written
        if os.path.getsize(output_path) == 0:
            raise ValueError("Output file is empty after writing")
            
        logger.info(f"Output successfully saved to: {output_path}")
        logger.debug(f"Output file size: {os.path.getsize(output_path)} bytes")
        
    except Exception as e:
        logger.error(f"Failed to save output: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        logger.error(f"Output data type: {type(output)}")
        logger.error(f"Output content preview: {str(output)[:500]}")  # Log first 500 chars
        raise

class CraftMyCVWorkflow(Workflow):
    def __init__(self):
        super().__init__()
        self.steps = [
            self.setup_workflow_context,
            self.extract_resume_insights,
            self.parse_job_requirements,
            self.match_and_tailor_skills,
            self.generate_tailored_resume,
            self.craft_professional_summary,
            self.generate_latex_document,
            self.perform_quality_assessment
        ]
        try:
            # Load configurations
            with open('src/config/agents.yaml', 'r') as f:
                agent_config_yaml = yaml.safe_load(f)
            self.agent_config = agent_config_yaml['agents']
            llm_config = agent_config_yaml['llm_config']
            fallback_config = agent_config_yaml['fallback_llm']
            
            # Validate environment variables
            openai_api_key = os.getenv('OPENAI_API_KEY')
            claude_api_key = os.getenv('CLAUDE_API_KEY')
            fallback_api_key = os.getenv('FALLBACK_LLM_API_KEY')
            
            if not all([openai_api_key, claude_api_key, fallback_api_key]):
                raise ValueError("Missing required API keys in environment variables")
            
            # Initialize LLMs
            self.openai_llm1 = OpenAI(
                model=llm_config['openai_llm']['model_1'],
                api_key=openai_api_key,
                temperature=llm_config['openai_llm']['temperature_1']
            )
            
            self.openai_llm2 = OpenAI(
                model=llm_config['openai_llm']['model_2'],
                api_key=openai_api_key,
                temperature=llm_config['openai_llm']['temperature_2']
            )
            
            self.claude_llm1 = Anthropic(
                model=llm_config['claude_llm']['model_1'],
                api_key=claude_api_key,
                temperature=llm_config['claude_llm']['temperature_1']
            )
            
            self.claude_llm2 = Anthropic(
                model=llm_config['claude_llm']['model_2'],
                api_key=claude_api_key,
                temperature=llm_config['claude_llm']['temperature_2']
            )
            
            # Initialize fallback LLM
            self.fallback_llm = OpenAI(
                model=fallback_config['model'],
                api_key=fallback_api_key,
                temperature=fallback_config['temperature']
            )
            
            # Create FallbackLLM instances
            self.openai_llm1_with_fallback = FallbackLLM(
                primary_llm=self.openai_llm1,
                fallback_llm=self.fallback_llm,
                timeout=360
            )
            
            self.openai_llm2_with_fallback = FallbackLLM(
                primary_llm=self.openai_llm2,
                fallback_llm=self.fallback_llm,
                timeout=360
            )
            
            self.claude_llm1_with_fallback = FallbackLLM(
                primary_llm=self.claude_llm1,
                fallback_llm=self.fallback_llm,
                timeout=360
            )
            
            self.claude_llm2_with_fallback = FallbackLLM(
                primary_llm=self.claude_llm2,
                fallback_llm=self.fallback_llm,
                timeout=360
            )

        except Exception as e:
            logger.error(f"Failed to initialize workflow: {str(e)}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    def _build_agent_prompt(self, agent_config, **inputs):
        """Helper method to build prompts for agents"""
        prompt = f"""Role: {agent_config['role']}
        Goal: {agent_config['goal']}

        System Instructions:
        {agent_config['system_prompt']}

        CRITICAL JSON RESPONSE REQUIREMENTS:
        You MUST return a single, valid JSON object. Follow these rules exactly:
        1. Start with a single opening curly brace {{
        2. End with a single closing curly brace }}
        3. Use double quotes for ALL keys and string values
        4. Do not include ANY explanatory text before or after the JSON
        5. Do not include markdown formatting (like ```json or ```)
        6. Do not include multiple JSON objects
        7. Ensure all arrays and objects are properly closed
        8. No trailing commas after the last item in arrays/objects
        9. No comments within the JSON
        10. No line breaks within string values

        COMMON ERRORS TO AVOID:
        INCORRECT FORMAT 1:    ```json
        {{
            "key": "value"
        }}    ```

        INCORRECT FORMAT 2:
        Here's the analysis:
        {{
            "key": "value"
        }}

        CORRECT FORMAT:
        {{
            "key": "value"
        }}

        Expected Output Format:
        {agent_config['expected_output']}

        Input Data:
        """
        for key, value in inputs.items():
            prompt += f"\n{key}: {value}"
            
        prompt += """

        FINAL REMINDER:
        1. Return ONLY the JSON object
        2. No text before or after
        3. No markdown formatting
        4. No explanation or comments
        5. Must be valid, parseable JSON
        """
        
        return prompt

    def _validate_llm_response(self, response_text: str, step_name: str) -> dict:
        """Validate and parse LLM response, handling markdown code blocks"""
        if not response_text or not response_text.strip():
            raise ValueError(f"{step_name}: Received empty response from LLM")
        
        # Log the raw response for debugging
        logger.debug(f"{step_name} raw response:\n{response_text}")
        
        try:
            # First attempt: try direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError as e1:
            # If direct parsing fails, try to handle markdown code blocks
            try:
                # Handle ```json\n...\n``` pattern
                if response_text.startswith('```') and response_text.endswith('```'):
                    # Split by ``` and take the content
                    parts = response_text.split('```')
                    if len(parts) >= 3:  # Should have at least 3 parts: before, content, after
                        # Get the middle part (content) and remove any "json" language identifier
                        content = parts[1]
                        if content.lower().startswith('json'):
                            content = content[4:].strip()
                        else:
                            content = content.strip()
                        
                        # Try to parse the extracted content
                        return json.loads(content)
                
                # If we get here, show detailed error information
                logger.error(f"{step_name}: Failed to parse JSON response")
                logger.error(f"Original error: {str(e1)}")
                logger.error("Response text:")
                logger.error(response_text)
                
                # Show the problematic line and position
                lines = response_text.split('\n')
                if e1.lineno <= len(lines):
                    error_line = lines[e1.lineno - 1]
                    logger.error(f"Error at line {e1.lineno}:")
                    logger.error(error_line)
                    logger.error(' ' * (e1.colno - 1) + '^')
                
                raise ValueError(
                    f"{step_name}: Failed to parse LLM response as JSON. "
                    f"Error at line {e1.lineno}, column {e1.colno}: {e1.msg}"
                )
                
            except Exception as e:
                logger.error(f"{step_name}: Could not process response: {str(e)}")
                raise ValueError(f"{step_name}: Failed to parse response: {str(e)}")

    def _log_step_output(self, step_name: str, output: Any, tokens_used: int = None, duration: float = None):
        """Log detailed output for each workflow step"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Step Completed: {step_name}")
        logger.info(f"Duration: {duration:.2f} seconds" if duration else "Duration: Unknown")
        logger.info(f"Tokens Used: {tokens_used}" if tokens_used else "Tokens Used: Unknown")
        logger.info("\nFull Output:")
        
        try:
            if isinstance(output, dict):
                # Convert dictionary to formatted JSON string with indentation
                formatted_output = json.dumps(output, indent=2, ensure_ascii=False)
            else:
                # Handle non-JSON output (like LaTeX)
                formatted_output = str(output)
            
            # Split the output into lines and log each line
            for line in formatted_output.split('\n'):
                logger.info(line)
        except Exception as e:
            logger.error(f"Error formatting output: {str(e)}")
            logger.info(f"Raw output: {output}")
        
        logger.info(f"\n{'='*80}\n")

    @step
    async def setup_workflow_context(self, event: StartEvent) -> Event:
        """Setup initial workflow context and metrics tracking"""
        ctx = Context(workflow=self)
        
        # Initialize metrics and state
        ctx.steps_completed = 0
        ctx.tokens_used = 0
        ctx.total_steps = 7
        ctx.workflow_state = "initialized"
        ctx.errors = []
        ctx.step_timings = {}  # Add this to track step timings
        
        # Initialize workflow data containers
        ctx.resume_text = event.data.get("resume_text")
        ctx.job_description = event.data.get("job_description")
        
        # Initialize result containers
        ctx.resume_analysis = None
        ctx.job_analysis = None
        ctx.skill_customization = None
        ctx.resume_customization = None
        ctx.summary_customization = None
        ctx.latex_resume = None
        ctx.quality_check_result = None
        
        return Event(data={"context": ctx})

    @step
    async def extract_resume_insights(self, event: Event) -> Event:
        """Extract and analyze key information from the input resume"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "analyzing_resume"
            agent_config = self.agent_config['resume_analyzer_agent']
            
            if not ctx.resume_text:
                raise ValueError("Missing resume text input")
            
            logger.info("Starting resume analysis...")
            prompt = self._build_agent_prompt(
                agent_config,
                resume_text=ctx.resume_text
            )
            
            response = await self.openai_llm1_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            ctx.resume_analysis = self._validate_llm_response(
                response.text,
                "Resume Analysis"
            )
            
            ctx.steps_completed += 1
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            if tokens_used:
                ctx.tokens_used += tokens_used
            
            duration = time() - start_time
            ctx.step_timings['analyze_resume'] = duration
            
            # Log step output
            self._log_step_output(
                "Resume Analysis",
                ctx.resume_analysis,
                tokens_used,
                duration
            )
            
            return Event(data={"context": ctx})
        except Exception as e:
            error_msg = f"Resume analysis failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def parse_job_requirements(self, event: Event) -> Event:
        """Parse and analyze job description requirements"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "analyzing_job_description"
            agent_config = self.agent_config['job_description_analyzer_agent']
            
            logger.info("Starting job description analysis...")
            if not ctx.job_description:
                raise ValueError("Missing job description input")
            
            prompt = self._build_agent_prompt(
                agent_config,
                job_description=ctx.job_description
            )
            
            response = await self.openai_llm1_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            ctx.job_analysis = self._validate_llm_response(
                response.text,
                "Job Analysis"
            )
            
            ctx.steps_completed += 1
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            if tokens_used:
                ctx.tokens_used += tokens_used
            
            duration = time() - start_time
            ctx.step_timings['analyze_job_description'] = duration
            
            # Log step output
            self._log_step_output(
                "Job Description Analysis",
                ctx.job_analysis,
                tokens_used,
                duration
            )
            
            return Event(data={"context": ctx})
        except Exception as e:
            error_msg = f"Job description analysis failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def match_and_tailor_skills(self, event: Event) -> Event:
        """Match and customize skills based on job requirements"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "customizing_skills"
            agent_config = self.agent_config['skill_customizer_agent']
            
            logger.info("Starting skill customization...")
            # Validate input data
            if not ctx.resume_analysis or not ctx.job_analysis:
                raise ValueError("Missing required resume or job analysis data")
            
            resume_skills = ctx.resume_analysis.get('resume_categorization', {}).get('skills', {})
            job_requirements = ctx.job_analysis.get('job_analysis', {})
            
            if not resume_skills or not job_requirements:
                raise ValueError("Missing skills or job requirements data")
            
            prompt = self._build_agent_prompt(
                agent_config,
                resume_skills=json.dumps(resume_skills),
                job_requirements=json.dumps(job_requirements)
            )
            
            response = await self.openai_llm2_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            ctx.skill_customization = self._validate_llm_response(
                response.text, 
                "Skill Customization"
            )
            
            ctx.steps_completed += 1
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            if tokens_used:
                ctx.tokens_used += tokens_used
            
            duration = time() - start_time
            ctx.step_timings['customize_skills'] = duration
            
            # Log step output
            self._log_step_output(
                "Skill Customization",
                ctx.skill_customization,
                tokens_used,
                duration
            )
            
            return Event(data={"context": ctx})
        except Exception as e:
            error_msg = f"Skill customization failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def generate_tailored_resume(self, event: Event) -> Event:
        """Generate a customized resume content based on job requirements"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "customizing_resume"
            agent_config = self.agent_config['resume_customizer_agent']
            
            logger.info("Starting resume customization...")
            if not all([ctx.resume_analysis, ctx.job_analysis, ctx.skill_customization]):
                raise ValueError("Missing required input data for resume customization")
            
            prompt = self._build_agent_prompt(
                agent_config,
                resume_analysis=json.dumps(ctx.resume_analysis),
                job_analysis=json.dumps(ctx.job_analysis),
                skill_customization=json.dumps(ctx.skill_customization)
            )
            
            response = await self.openai_llm2_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            ctx.resume_customization = self._validate_llm_response(
                response.text,
                "Resume Customization"
            )
            
            ctx.steps_completed += 1
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            if tokens_used:
                ctx.tokens_used += tokens_used
            
            duration = time() - start_time
            ctx.step_timings['customize_resume'] = duration
            
            # Log step output
            self._log_step_output(
                "Resume Customization",
                ctx.resume_customization,
                tokens_used,
                duration
            )
            
            return Event(data={"context": ctx})
        except Exception as e:
            error_msg = f"Resume customization failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def craft_professional_summary(self, event: Event) -> Event:
        """Create a targeted professional summary"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "customizing_summary"
            agent_config = self.agent_config['summary_customizer_agent']
            
            logger.info("Starting professional summary creation...")
            if not all([ctx.resume_customization, ctx.job_analysis]):
                raise ValueError("Missing required input data for summary customization")
            
            prompt = self._build_agent_prompt(
                agent_config,
                resume_customization=json.dumps(ctx.resume_customization),
                job_analysis=json.dumps(ctx.job_analysis)
            )
            
            response = await self.openai_llm1_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            ctx.summary_customization = self._validate_llm_response(
                response.text,
                "Summary Customization"
            )
            
            ctx.steps_completed += 1
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            if tokens_used:
                ctx.tokens_used += tokens_used
            
            duration = time() - start_time
            ctx.step_timings['customize_summary'] = duration
            
            # Log step output
            self._log_step_output(
                "Professional Summary",
                ctx.summary_customization,
                tokens_used,
                duration
            )
            
            return Event(data={"context": ctx})
        except Exception as e:
            error_msg = f"Summary customization failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def generate_latex_document(self, event: Event) -> Event:
        """Convert resume content to LaTeX format"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "converting_format"
            agent_config = self.agent_config['format_converter_agent']
            
            logger.info("Starting LaTeX conversion...")
            if not ctx.resume_customization:
                raise ValueError("Missing customized resume data")
            
            prompt = self._build_agent_prompt(
                agent_config,
                customized_resume=json.dumps(ctx.resume_customization)
            )
            
            response = await self.openai_llm1_with_fallback.acomplete(prompt)
            
            # Extract LaTeX content from response if needed
            latex_content = response.text
            if latex_content.startswith('```') and latex_content.endswith('```'):
                parts = latex_content.split('```')
                if len(parts) >= 3:
                    latex_content = parts[1]
                    if latex_content.lower().startswith('latex'):
                        latex_content = latex_content[5:].strip()
                    else:
                        latex_content = latex_content.strip()
            
            ctx.latex_resume = latex_content
            
            ctx.steps_completed += 1
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            if tokens_used:
                ctx.tokens_used += tokens_used
            
            duration = time() - start_time
            ctx.step_timings['convert_format'] = duration
            
            # Log step output
            self._log_step_output(
                "LaTeX Generation",
                ctx.latex_resume,
                tokens_used,
                duration
            )
            
            return Event(data={"context": ctx})
        except Exception as e:
            error_msg = f"Format conversion failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def perform_quality_assessment(self, event: Event) -> StopEvent:
        """Perform final quality check and validation"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "quality_check"
            agent_config = self.agent_config['quality_controller_agent']
            
            logger.info("Starting quality assessment...")
            if not all([ctx.latex_resume, ctx.job_analysis]):
                raise ValueError("Missing required input data for quality assessment")
            
            prompt = self._build_agent_prompt(
                agent_config,
                latex_resume=ctx.latex_resume,
                job_analysis=json.dumps(ctx.job_analysis)
            )
            
            response = await self.openai_llm2_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            ctx.quality_check_result = self._validate_llm_response(
                response.text,
                "Quality Assessment"
            )
            
            ctx.steps_completed += 1
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            if tokens_used:
                ctx.tokens_used += tokens_used
            
            duration = time() - start_time
            ctx.step_timings['quality_check'] = duration
            
            # Log step output
            self._log_step_output(
                "Quality Assessment",
                ctx.quality_check_result,
                tokens_used,
                duration
            )
            
            # Prepare final output
            final_output = {
                "resume_analysis": ctx.resume_analysis,
                "job_analysis": ctx.job_analysis,
                "skill_customization": ctx.skill_customization,
                "resume_customization": ctx.resume_customization,
                "summary_customization": ctx.summary_customization,
                "latex_resume": ctx.latex_resume,
                "quality_check": ctx.quality_check_result,
                "workflow_metrics": {
                    "steps_completed": ctx.steps_completed,
                    "total_steps": ctx.total_steps,
                    "tokens_used": ctx.tokens_used,
                    "final_state": ctx.workflow_state,
                    "errors": ctx.errors,
                    "step_timings": ctx.step_timings
                }
            }
            
            ctx.workflow_state = "completed"
            
            # Changed: Return StopEvent with the final output in the event data
            return StopEvent(final_output)
            
        except Exception as e:
            error_msg = f"Quality check failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

async def main():
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
        resume_path = await validate_file_path(args.resume)
        
        # Handle job description
        job_description = ""
        if Path(args.job_description).exists():
            with open(args.job_description, 'r', encoding='utf-8') as f:
                job_description = f.read()
        else:
            job_description = args.job_description

        # Extract resume text
        logger.info("Extracting text from resume...")
        text_extractor = TextExtractionTool()
        resume_text = text_extractor._run(str(resume_path))

        # Initialize workflow
        logger.info("Initializing CV creation workflow...")
        workflow = CraftMyCVWorkflow()
        
        # Create StartEvent with input data
        event = StartEvent(data={
            "resume_text": resume_text,
            "job_description": job_description
        })
        
        # Run each step in sequence
        current_event = event
        for step in workflow.steps:
            logger.info(f"Executing step: {step.__name__.replace('_', ' ').title()}")
            try:
                current_event = await step(current_event)
            except Exception as e:
                logger.error(f"Step {step.__name__} failed: {str(e)}")
                raise
        
        # Add debug logging before saving
        logger.debug(f"Final event type: {type(current_event)}")
        logger.debug(f"Final event attributes: {dir(current_event)}")
        if hasattr(current_event, 'result'):
            logger.debug(f"Result attribute preview: {str(current_event.result)[:200]}")
        
        # Save the output
        await save_output(current_event, args.output)

        # Log completion metrics - updated to use result instead of _data
        if hasattr(current_event, 'result') and 'workflow_metrics' in current_event.result:
            metrics = current_event.result['workflow_metrics']
            logger.info(f"Total steps completed: {metrics['steps_completed']}")
            logger.info(f"Total tokens used: {metrics['tokens_used']}")
            
            if metrics['errors']:
                logger.warning("Errors occurred during processing:")
                for error in metrics['errors']:
                    logger.warning(f"- {error}")
        
            # Log timing information
            logger.info("Step timing information:")
            for step_name, duration in metrics['step_timings'].items():
                logger.info(f"- {step_name}: {duration:.2f} seconds")
        
        logger.info("Process completed successfully!")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise SystemExit(1)

if __name__ == "__main__":
    asyncio.run(main())