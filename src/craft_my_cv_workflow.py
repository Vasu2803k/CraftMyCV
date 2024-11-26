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
        # Simplified workflow steps based on flowchart
        self.steps = [
            self.setup_workflow_context,
            self.resume_analyzer,
            self.job_description_analyzer,
            self.resume_customizer,
            self.summary_customizer,
            self.pre_latex_quality_controller,
            self.latex_formatter,
            self.post_latex_quality_controller
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

    def _build_latex_prompt(self, agent_config, recommendations=None, **inputs):
        """Build prompt for generating raw LaTeX code"""
        prompt = f"""Role: {agent_config['role']}
        Goal: {agent_config['goal']}

        System Instructions:
        {agent_config['system_prompt']}
        """
        
        if recommendations and 'issues_and_improvements' in recommendations:
            prompt += "\nQuality Improvement Recommendations:\n"
            for issue in recommendations['issues_and_improvements']:
                prompt += f"- Issue: {issue['issue']}\n"
                prompt += f"  Section: {issue['section']}\n"
                prompt += f"  Resolution: {issue['resolution']}\n"
                
        prompt += """
        CRITICAL REQUIREMENTS:
        1. Generate ONLY raw LaTeX code
        2. Include all necessary package imports
        3. Use professional formatting and layout
        4. Ensure ATS compatibility
        5. Follow proper LaTeX syntax
        6. Include all sections in a logical order
        7. Use appropriate commands for formatting
        8. Handle special characters correctly
        
        DO NOT:
        1. Include any JSON wrapping
        2. Add any explanatory text
        3. Use markdown formatting
        4. Include any non-LaTeX content
        
        You will be provided with the required input data.
        Input Data:
        """
        
        for key, value in inputs.items():
            prompt += f"\n{key}: {value}"
            
        prompt += """
        
        FINAL REMINDER:
        - Return ONLY the raw LaTeX code
        - Start with \\documentclass
        - Include all necessary packages
        - End with \\end{document}
        """
        return prompt
    
    def _build_content_prompt(self, agent_config, recommendations=None, **inputs):
        """Helper method to build prompts for agents with quality recommendations"""
        prompt = f"""Role: {agent_config['role']}
        Goal: {agent_config['goal']}

        System Instructions:
        {agent_config['system_prompt']}
        """
        
        # Add quality recommendations if available
        if recommendations and 'issues_and_improvements' in recommendations:
            prompt += "\nQuality Improvement Recommendations:\n"
            for agent_type, issues in recommendations['issues_and_improvements'].items():
                prompt += f"\nIssues for {agent_type}:\n"
                for issue in issues:
                    prompt += f"- Issue: {issue['issue']}\n"
                    prompt += f"  Section: {issue['section']}\n"
                    prompt += f"  Resolution: {issue['resolution']}\n"
                    prompt += f"  Benefit: {issue['benefit']}\n"
        
        prompt += f"""
        CRITICAL JSON RESPONSE REQUIREMENTS:
        You MUST return a single, valid JSON object. Follow these rules exactly:
        1. Start with a single opening curly brace
        2. End with a single closing curly brace
        3. Use double quotes for ALL keys and string values
        4. Do not include ANY explanatory text before or after the JSON
        5. Do not include markdown formatting (like ```json or ```)
        6. Do not include multiple JSON objects
        7. Ensure all arrays and objects are properly closed
        8. No trailing commas after the last item in arrays/objects
        9. No comments within the JSON
        10. No line breaks within string values

        Expected Output Format:
        {agent_config['expected_output']}

        You will be provided with the required input data.
        Input Data:
        """
        
        for key, value in inputs.items():
            if key != 'recommendations':  # Skip printing recommendations in input data
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

    def _validate_llm_response(self, response_text: str) -> dict | str:
        """Validate and parse LLM response, handling markdown code blocks"""
        try:
            # Remove any markdown code block formatting if present
            if response_text.startswith("```") and response_text.endswith("```"):
                # Extract content between code block markers
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])  # Remove first and last lines
            
            # Try to parse as JSON first
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If not JSON, check if it's LaTeX
                if response_text.strip().startswith("\\documentclass"):
                    return response_text.strip()
                else:
                    raise ValueError("Response is neither valid JSON nor LaTeX")
                
        except Exception as e:
            logger.error(f"Failed to validate LLM response: {str(e)}")
            logger.error(f"Response text:\n{response_text}")
            raise ValueError(f"Invalid response format: {str(e)}")

    def _log_step_output(self, step_name: str, output: Any, duration: float = None):
        """Log detailed output for each workflow step"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Step Completed: {step_name}")
        logger.info(f"Duration: {duration:.2f} seconds" if duration else "Duration: Unknown")
        logger.info("\nFull Output:")
        
        try:
            if isinstance(output, dict):
                formatted_output = json.dumps(output, indent=2, ensure_ascii=False)
            else:
                formatted_output = str(output)
            
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
        ctx.total_steps = len(self.steps)
        ctx.workflow_state = "initialized"
        ctx.errors = []
        ctx.step_timings = {}
        
        # Initialize workflow data containers
        ctx.resume_text = event.data.get("resume_text")
        ctx.job_description = event.data.get("job_description")
        
        # Initialize result containers based on agent output structure
        ctx.resume_analysis = None  # Will contain resume_analyzer_agent output
        ctx.job_analysis = None    # Will contain job_description_analyzer_agent output
        ctx.resume_customization = None  # Will contain resume_customizer_agent output
        ctx.summary_customization = None  # Will contain summary_customizer_agent output
        ctx.latex_resume = None    # Will contain format_converter_agent output
        
        # Quality check containers
        ctx.pre_latex_quality_check = None  # Will contain pre_latex_quality_controller_agent output
        ctx.post_latex_quality_check = None  # Will contain post_latex_quality_controller_agent output
        ctx.content_quality_check = None  # Will contain resume_feedback_agent output
        
        # Validation containers
        ctx.validation_status = {
            "resume_analyzer": False,
            "job_description_analyzer": False,
            "resume_customizer": False,
            "summary_customizer": False,
            "latex_formatter": False,
            "quality_controllers": False  # Covers all quality control steps
        }
        
        return Event(data={"context": ctx})

    @step
    async def resume_analyzer(self, event: StartEvent) -> Event:
        """Extract and analyze key information from the input resume"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "resume_analyzer"
            agent_config = self.agent_config['resume_analyzer_agent']
            
            if not ctx.resume_text:
                raise ValueError("Missing resume text input")
            
            logger.info("Starting resume analysis...")
            
            prompt = self._build_content_prompt(
                agent_config,
                resume_text=ctx.resume_text
            )
            
            response = await self.openai_llm1_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            parsed_response = self._validate_llm_response(response.text)
            
            ctx.resume_analysis = parsed_response
            
            ctx.steps_completed += 1
            
            duration = time() - start_time
            ctx.step_timings['resume_analyzer'] = duration
            
            # Log step output
            self._log_step_output(
                "Resume Analysis",
                ctx.resume_analysis,
                duration=duration
            )
            
            return Event(data={"context": ctx})
        except Exception as e:
            error_msg = f"Resume analysis failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def job_description_analyzer(self, event: Event) -> Event:
        """Parse and analyze job description requirements"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "job_description_analyzer"
            agent_config = self.agent_config['job_description_analyzer_agent']
            
            logger.info("Starting job description analysis...")
            if not ctx.job_description:
                raise ValueError("Missing job description input")
            
            prompt = self._build_content_prompt(
                agent_config,
                job_description=ctx.job_description
            )
            
            response = await self.openai_llm1_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            parsed_response = self._validate_llm_response(response.text)
            
            ctx.job_analysis = parsed_response
            
            ctx.steps_completed += 1
            
            duration = time() - start_time
            ctx.step_timings['job_description_analyzer'] = duration
            
            # Log step output
            self._log_step_output(
                "Job Description Analysis",
                ctx.job_analysis,
                duration=duration
            )
            
            return Event(data={"context": ctx})
        except Exception as e:
            error_msg = f"Job description analysis failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def resume_customizer(self, event: Event) -> Event:
        """Generate a customized resume content based on job requirements"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "resume_customizer"
            agent_config = self.agent_config['resume_customizer_agent']
            
            logger.info("Starting resume customization...")
            if not all([ctx.resume_analysis, ctx.job_analysis]):
                raise ValueError("Missing required input data for resume customization")
            
            resume_data = ctx.resume_analysis
            if not resume_data:
                raise ValueError("Missing resume analysis data")
            
            # Build prompt with correct data structure
            prompt = self._build_content_prompt(
                agent_config,
                resume_data=json.dumps(resume_data),
                job_analysis=json.dumps(ctx.job_analysis)
            )
            
            response = await self.openai_llm2_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            ctx.resume_customization = self._validate_llm_response(response.text)

            ctx.steps_completed += 1
            
            duration = time() - start_time
            ctx.step_timings['resume_customizer'] = duration
            
            # Log step output
            self._log_step_output(
                "Resume Customization",
                ctx.resume_customization,
                duration=duration
            )
            
            return Event(data={"context": ctx})
        except Exception as e:
            error_msg = f"Resume customization failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def summary_customizer(self, event: Event) -> Event:
        """Create a targeted professional summary"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "summary_customizer"
            agent_config = self.agent_config['summary_customizer_agent']
            
            logger.info("Starting professional summary creation...")
            
            # More flexible data validation
            if not ctx.resume_analysis or not ctx.job_analysis:
                raise ValueError("Missing basic resume or job analysis data")
            
            # Extract resume data with safer fallbacks
            resume_data = ctx.resume_analysis
            work_summary = resume_data.get('work_summary', {})
            experience = resume_data.get('experience', [])
            skills = resume_data.get('skills', {})
            
            # Get customized skills data with fallback
            customized_skills = ctx.skill_customization
            job_analysis = ctx.job_analysis
            
            # Build input data structure with available information
            input_data = {
                'resume_data': {
                    'work_summary': work_summary,
                    'experience': experience,
                    'skills': skills
                },
                'customized_skills': customized_skills,
                'job_analysis': job_analysis
            }
            
            prompt = self._build_content_prompt(
                agent_config,
                resume_data=json.dumps(input_data['resume_data']),
                customized_skills=json.dumps(input_data['customized_skills']),
                job_analysis=json.dumps(input_data['job_analysis'])
            )
            
            response = await self.openai_llm1_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            parsed_response = self._validate_llm_response(response.text)
            
            ctx.summary_customization = parsed_response
            
            ctx.steps_completed += 1
            
            duration = time() - start_time
            ctx.step_timings['summary_customizer'] = duration
            
            # Log step output
            self._log_step_output(
                "Professional Summary",
                ctx.summary_customization,
                duration=duration
            )
            
            return Event(data={"context": ctx})
            
        except Exception as e:
            error_msg = f"Summary customization failed: {str(e)}"
            if not hasattr(ctx, 'errors'):
                ctx.errors = []
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def pre_latex_quality_controller(self, event: Event) -> Event:
        """Perform quality check on resume content before LaTeX generation"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "pre_latex_quality_controller"
            agent_config = self.agent_config['pre_latex_quality_controller_agent']
            
            logger.info("Starting pre-LaTeX quality assessment...")
            
            # Prepare input data for quality check
            input_data = {
                "resume_analysis": ctx.resume_analysis,
                "job_analysis": ctx.job_analysis,
                "customized_skills": ctx.skill_customization,
                "customized_resume": ctx.resume_customization,
                "customized_summary": ctx.summary_customization
            }
            
            prompt = self._build_content_prompt(
                agent_config,
                **input_data
            )
            
            response = await self.openai_llm2_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            ctx.pre_latex_quality_check = self._validate_llm_response(response.text)
            
            # Update validation status
            ctx.validation_status["quality_checks"] = True
            
            duration = time() - start_time
            ctx.step_timings['pre_latex_quality_controller'] = duration
            
            self._log_step_output(
                "Pre-LaTeX Quality Check",
                ctx.pre_latex_quality_check,
                duration=duration
            )
            
            return Event(data={"context": ctx})
            
        except Exception as e:
            error_msg = f"Pre-LaTeX quality check failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def latex_formatter(self, event: Event) -> Event:
        """Convert resume content to LaTeX format"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "latex_formatter"
            agent_config = self.agent_config['latex_formatting_agent']
            
            logger.info("Starting LaTeX conversion...")
            if not ctx.resume_customization.get('customized_resume'):
                raise ValueError("Missing customized resume data")
            
            # Prepare input data for LaTeX conversion
            resume_data = {
                "customized_skills": ctx.skill_customization.get('customized_skills', {}),
                "customized_resume": ctx.resume_customization.get('customized_resume', {}),
                "customized_summary": ctx.summary_customization.get('customized_summary', {})
            }
            
            # Use the specialized LaTeX prompt builder
            prompt = self._build_latex_prompt(
                agent_config,
                resume_data=json.dumps(resume_data)
            )
            
            response = await self.openai_llm1_with_fallback.acomplete(prompt)
            
            # Validate response and ensure it has latex_document structure
            parsed_response = self._validate_llm_response(response.text)
            
            if 'latex_document' not in parsed_response:
                raise ValueError("Missing 'latex_document' key in LaTeX response")
            
            # Additional LaTeX-specific validation
            latex_doc = parsed_response['latex_document']
            required_sections = ['content', 'metadata', 'validation']
            for section in required_sections:
                if section not in latex_doc:
                    raise ValueError(f"Missing required section '{section}' in latex_document")
            
            # Validate content structure
            content = latex_doc['content']
            if not all(key in content for key in ['preamble', 'document_class', 'main_content']):
                raise ValueError("Missing required content structure in latex_document")
            
            ctx.latex_resume = parsed_response['latex_document']
            
            ctx.steps_completed += 1
            
            duration = time() - start_time
            ctx.step_timings['latex_formatter'] = duration
            
            # Log step output
            self._log_step_output(
                "LaTeX Generation",
                ctx.latex_resume,
                duration=duration
            )
            
            return Event(data={"context": ctx})
        except Exception as e:
            error_msg = f"Format conversion failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def post_latex_quality_controller(self, event: Event) -> Event:
        """Perform quality check specifically for LaTeX output"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "post_latex_quality_controller"
            agent_config = self.agent_config['latex_quality_controller_agent']
            
            logger.info("Starting LaTeX quality assessment...")
            
            if not ctx.latex_resume:
                raise ValueError("Missing LaTeX document for quality check")

            prompt = self._build_content_prompt(
                agent_config,
                latex_document=ctx.latex_resume,
                job_requirements=json.dumps(ctx.job_analysis)
            )
            
            response = await self.openai_llm2_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            ctx.latex_quality_check = self._validate_llm_response(response.text)
            
            duration = time() - start_time
            ctx.step_timings['post_latex_quality_controller'] = duration
            
            # Log step output
            self._log_step_output(
                "LaTeX Quality Check",
                ctx.latex_quality_check,
                duration=duration
            )
            
            return Event(data={"context": ctx})
            
        except Exception as e:
            error_msg = f"LaTeX quality check failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def content_quality_controller(self, event: Event) -> StopEvent:
        """Perform final quality check and validation"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "content_quality_controller"
            agent_config = self.agent_config['content_quality_controller_agent']
            
            logger.info("Starting quality assessment...")
            if not all([ctx.latex_resume, ctx.job_analysis]):
                raise ValueError("Missing required input data for quality assessment")
            
            prompt = self._build_content_prompt(
                agent_config,
                latex_resume=ctx.latex_resume,
                job_analysis=json.dumps(ctx.job_analysis)
            )
            
            response = await self.openai_llm2_with_fallback.acomplete(prompt)
            
            # Validate and parse response
            ctx.content_quality_check = self._validate_llm_response(response.text)
            
            ctx.steps_completed += 1
            
            duration = time() - start_time
            ctx.step_timings['content_quality_controller'] = duration
            
            # Log step output
            self._log_step_output(
                "Content Quality Assessment",
                ctx.content_quality_check,
                duration=duration
            )
            
            # Prepare final output with correct structure
            final_output = {
                "resume_analysis": ctx.resume_analysis.get('resume_analysis', {}),
                "job_analysis": ctx.job_analysis.get('job_analysis', {}),
                "customized_skills": ctx.skill_customization.get('customized_skills', {}),
                "customized_resume": ctx.resume_customization.get('customized_resume', {}),
                "customized_summary": ctx.summary_customization.get('customized_summary', {}),
                "latex_document": ctx.latex_resume.get('latex_document', {}),
                "content_quality_check": ctx.content_quality_check.get('content_quality_check', {}),
                "workflow_metrics": {
                    "steps_completed": ctx.steps_completed,
                    "total_steps": ctx.total_steps,
                    "final_state": ctx.workflow_state,
                    "errors": ctx.errors,
                    "step_timings": ctx.step_timings,
                    "validation_status": {
                        "resume_analysis": "verified" if ctx.resume_analysis.get('resume_analysis') else "incomplete",
                        "latex_format": "verified" if ctx.latex_resume.get('latex_document', {}).get('validation', {}).get('structure_check') == "pass" else "incomplete",
                        "content_quality_check": "verified" if ctx.content_quality_check.get('content_quality_check') else "incomplete"
                    }
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

async def apply_quality_recommendations(event: Event | StopEvent, recommendations: dict, workflow: Workflow) -> Event:
    """Apply quality check recommendations and improvements"""
    try:
        # Handle both Event and StopEvent types
        if isinstance(event, StopEvent):
            ctx = Context(workflow=workflow)
            # Copy relevant data from StopEvent result
            for key, value in event.result.items():
                setattr(ctx, key, value)
        else:
            ctx = event.data.get("context")
            
        if not ctx:
            raise ValueError("Missing context in event data")

        logger.info("Applying quality check recommendations...")
        
        if not recommendations or not isinstance(recommendations, dict):
            logger.warning("No valid recommendations to apply")
            return event

        # Track improvements
        improvements_applied = []
        improvement_results = {}

        # Check if these are pre-latex recommendations (agent-wise) or post-latex recommendations
        if 'issues_and_improvements' in recommendations:
            # Pre-latex case: Handle agent-specific improvements
            for agent_type, issues in recommendations['issues_and_improvements'].items():
                logger.info(f"\nProcessing improvements for {agent_type}...")
                
                for issue in issues:
                    issue_section = issue.get('section')
                    if not issue_section:
                        continue
                        
                    logger.info(f"\nApplying improvement for {issue_section}")
                    resolution = issue.get('resolution', 'No resolution provided')
                    benefit = issue.get('benefit', 'No benefit description')
                    logger.info(f"Resolution: {resolution}")
                    logger.info(f"Expected benefit: {benefit}")
                    
                    try:
                        # Create new event with current context state
                        improvement_event = Event(data={
                            "context": ctx,
                            "recommendations": {
                                "issues_and_improvements": {agent_type: [issue]}
                            }
                        })

                        # Map agent types to workflow steps
                        result = None
                        if agent_type == 'skill_customizer_agent':
                            if hasattr(ctx, 'resume_analysis'):
                                result = await workflow.skill_customizer(improvement_event)
                                
                        elif agent_type == 'summary_customizer_agent':
                            if hasattr(ctx, 'resume_analysis') and hasattr(ctx, 'job_analysis'):
                                result = await workflow.summary_customizer(improvement_event)
                                
                        elif agent_type == 'resume_customizer_agent':
                            if all(hasattr(ctx, attr) for attr in ['resume_analysis', 'job_analysis', 'skill_customization']):
                                result = await workflow.resume_customizer(improvement_event)

                        if result and (isinstance(result, Event) or isinstance(result, StopEvent)):
                            # Update context with improvement results
                            if isinstance(result, Event):
                                ctx = result.data.get("context", ctx)
                            else:  # StopEvent
                                for key, value in result.result.items():
                                    setattr(ctx, key, value)
                                    
                            improvements_applied.append(f"{agent_type}:{issue_section}")
                            improvement_results[f"{agent_type}:{issue_section}"] = {
                                'status': 'success',
                                'resolution': resolution,
                                'benefit': benefit
                            }
                        else:
                            logger.warning(f"No changes applied for {agent_type}:{issue_section}")
                            improvement_results[f"{agent_type}:{issue_section}"] = {
                                'status': 'skipped',
                                'reason': 'No changes required or applicable'
                            }

                    except Exception as e:
                        error_msg = f"Error applying {agent_type}:{issue_section} improvement: {str(e)}"
                        logger.error(error_msg)
                        logger.error(f"Stack trace:\n{traceback.format_exc()}")
                        improvement_results[f"{agent_type}:{issue_section}"] = {
                            'status': 'failed',
                            'error': str(e)
                        }
                        continue

        else:
            # Post-latex case: Handle direct improvements
            for issue in recommendations.get('recommendations', {}).get('issues_and_improvements', []):
                issue_section = issue.get('section')
                if not issue_section:
                    continue
                    
                logger.info(f"\nApplying improvement for {issue_section}")
                resolution = issue.get('resolution', 'No resolution provided')
                logger.info(f"Resolution: {resolution}")
                
                try:
                    # Create new event with current context state
                    improvement_event = Event(data={
                        "context": ctx,
                        "recommendations": {"current_issue": issue}
                    })

                    # Map sections to workflow steps
                    result = None
                    if issue_section in ['document_formatting', 'typography', 'section_headers']:
                        if hasattr(ctx, 'resume_customization'):
                            result = await workflow.latex_formatter(improvement_event)
                            
                    if result and (isinstance(result, Event) or isinstance(result, StopEvent)):
                        if isinstance(result, Event):
                            ctx = result.data.get("context", ctx)
                        else:  # StopEvent
                            for key, value in result.result.items():
                                setattr(ctx, key, value)
                                
                        improvements_applied.append(issue_section)
                        improvement_results[issue_section] = {
                            'status': 'success',
                            'resolution': resolution
                        }
                    else:
                        logger.warning(f"No changes applied for {issue_section}")
                        improvement_results[issue_section] = {
                            'status': 'skipped',
                            'reason': 'No changes required or applicable'
                        }

                except Exception as e:
                    error_msg = f"Error applying {issue_section} improvement: {str(e)}"
                    logger.error(error_msg)
                    logger.error(f"Stack trace:\n{traceback.format_exc()}")
                    improvement_results[issue_section] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    continue

        # Update workflow metrics with improvement results
        if not hasattr(ctx, 'workflow_metrics'):
            ctx.workflow_metrics = {}
        
        # Calculate success rate only if there are improvement results
        success_rate = 0
        if improvement_results:
            successful_improvements = len([r for r in improvement_results.values() if r.get('status') == 'success'])
            success_rate = (successful_improvements / len(improvement_results)) * 100
            
        ctx.workflow_metrics['quality_improvements'] = {
            'improvements_applied': improvements_applied,
            'improvement_results': improvement_results,
            'total_improvements': len(improvements_applied),
            'success_rate': f"{success_rate:.2f}%"
        }

        # Log improvement summary
        logger.info("\nQuality Improvement Summary:")
        logger.info(f"Total improvements attempted: {len(improvement_results)}")
        logger.info(f"Successful improvements: {len(improvements_applied)}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        
        # Create new event with updated context
        return Event(data={"context": ctx})

    except Exception as e:
        logger.error(f"Failed to apply quality recommendations: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        # Return original event if improvements fail
        return event

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
        
        # Run workflow steps
        current_event = event
        for step in workflow.steps:
            logger.info(f"Executing step: {step.__name__.replace('_', ' ').title()}")
            try:
                current_event = await step(current_event)
            except Exception as e:
                logger.error(f"Step {step.__name__} failed: {str(e)}")
                raise
        
        # Quality improvement loop
        max_improvement_iterations = 3
        current_iteration = 0
        
        while current_iteration < max_improvement_iterations:
            logger.info(f"\nStarting quality check iteration {current_iteration + 1}/{max_improvement_iterations}")
            
            # Get quality check result
            quality_check = None
            if isinstance(current_event, StopEvent):
                quality_check = current_event.result.get('content_quality_check', {})
            else:
                quality_check = current_event.data.get("context", {}).get('content_quality_check', {})

            if not quality_check:
                logger.warning("No quality check results available")
                break
            
            # Check if improvements are needed
            needs_improvements = False
            if 'improvement_priority' in quality_check:
                for priority in ['high', 'medium', 'low']:
                    if quality_check['improvement_priority'].get(priority):
                        needs_improvements = True
                        break
            
            if needs_improvements:
                logger.info("Content improvements needed")
                
                # Apply improvements
                try:
                    improved_event = await apply_quality_recommendations(
                        current_event,
                        quality_check,
                        workflow
                    )
                    
                    if improved_event != current_event:
                        # Perform quality check again
                        current_event = await workflow.content_quality_controller(improved_event)
                        current_iteration += 1
                    else:
                        logger.info("No improvements were applied")
                        break
                        
                except Exception as e:
                    logger.error(f"Error in improvement iteration {current_iteration + 1}: {str(e)}")
                    logger.error(f"Stack trace:\n{traceback.format_exc()}")
                    break
            else:
                logger.info("No content improvements needed")
                break
            
        if current_iteration == max_improvement_iterations:
            logger.warning(f"Reached maximum improvement iterations ({max_improvement_iterations})")
        
        # Save the final output
        await save_output(current_event, args.output)
        
        # Log completion summary
        logger.info("\nWorkflow Completion Summary:")
        if isinstance(current_event, StopEvent):
            metrics = current_event.result.get('workflow_metrics', {})
        else:
            metrics = current_event.data.get("context", {}).get('workflow_metrics', {})
            
        if metrics:
            logger.info(f"Steps completed: {metrics.get('steps_completed', 'Unknown')}")
            logger.info(f"Total duration: {sum(metrics.get('step_timings', {}).values()):.2f} seconds")
            
            if 'quality_improvements' in metrics:
                qi_metrics = metrics['quality_improvements']
                logger.info("\nQuality Improvement Results:")
                logger.info(f"Total improvements: {qi_metrics.get('total_improvements', 0)}")
                logger.info(f"Success rate: {qi_metrics.get('success_rate', '0%')}")
        
        logger.info("\nProcess completed successfully!")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise SystemExit(1)

if __name__ == "__main__":
    asyncio.run(main())