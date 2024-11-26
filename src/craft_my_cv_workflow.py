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
    if not os.access(path, os.R_OK):
        raise PermissionError(f"File not readable: {file_path}")
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
            
        logger.debug(f"Attempting to save output: {json.dumps(data_to_save, indent=2)[:200]}...")
        
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
        logger.error(f"Output content preview: {str(output)[:500]}")
        raise

class CraftMyCVWorkflow(Workflow):
    def __init__(self):
        super().__init__()
        # Updated workflow steps based on flowchart
        self.steps = [
            self.setup_workflow_context,
            self.resume_analyzer,
            self.job_description_analyzer,
            self.resume_customizer,
            self.summary_customizer,
            self.pre_latex_quality_controller,
            self.latex_formatter,
            self.post_latex_quality_controller,
            self.finalize_output
        ]
        
        try:
            # Load configurations
            with open('src/config/agents.yaml', 'r') as f:
                agent_config_yaml = yaml.safe_load(f)
            self.agent_config = agent_config_yaml['agents']
            llm_config = agent_config_yaml['llm_config']
            fallback_config = agent_config_yaml['fallback_llm']
            
            # Validate environment variables
            required_keys = {
                'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
                'CLAUDE_API_KEY': os.getenv('CLAUDE_API_KEY'),
                'FALLBACK_LLM_API_KEY': os.getenv('FALLBACK_LLM_API_KEY')
            }
            
            missing_keys = [k for k, v in required_keys.items() if not v]
            if missing_keys:
                raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
            
            # Initialize LLMs with improved error handling
            self.llms = {
                'openai_llm1': self._init_llm(OpenAI, {
                    'model': llm_config['openai_llm']['model_1'],
                    'api_key': required_keys['OPENAI_API_KEY'],
                    'temperature': llm_config['openai_llm']['temperature_1']
                }),
                'openai_llm2': self._init_llm(OpenAI, {
                    'model': llm_config['openai_llm']['model_2'],
                    'api_key': required_keys['OPENAI_API_KEY'],
                    'temperature': llm_config['openai_llm']['temperature_2']
                }),
                'claude_llm1': self._init_llm(Anthropic, {
                    'model': llm_config['claude_llm']['model_1'],
                    'api_key': required_keys['CLAUDE_API_KEY'],
                    'temperature': llm_config['claude_llm']['temperature_1']
                }),
                'claude_llm2': self._init_llm(Anthropic, {
                    'model': llm_config['claude_llm']['model_2'],
                    'api_key': required_keys['CLAUDE_API_KEY'],
                    'temperature': llm_config['claude_llm']['temperature_2']
                })
            }
            
            # Initialize fallback LLM
            self.fallback_llm = self._init_llm(OpenAI, {
                'model': fallback_config['model'],
                'api_key': required_keys['FALLBACK_LLM_API_KEY'],
                'temperature': fallback_config['temperature']
            })
            
            # Create FallbackLLM instances with unified timeout
            self.llm_with_fallback = {
                name: FallbackLLM(
                    primary_llm=llm,
                    fallback_llm=self.fallback_llm,
                    timeout=60
                ) for name, llm in self.llms.items()
            }

        except Exception as e:
            logger.error(f"Failed to initialize workflow: {str(e)}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    def _init_llm(self, llm_class, config: dict) -> LLM:
        """Initialize LLM with error handling"""
        try:
            return llm_class(**config)
        except Exception as e:
            logger.error(f"Failed to initialize {llm_class.__name__}: {str(e)}")
            raise

    def _build_latex_prompt(self, agent_config, recommendations=None, **inputs):
        """Build prompt for generating raw LaTeX code"""
        prompt = f"""Role: {agent_config['role']}
        Goal: {agent_config['goal']}

        System Instructions:
        {agent_config['system_prompt']}
        """
        
        if recommendations:
            prompt += f"\nQuality Improvement Recommendations:\n{recommendations}"
        
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
        
        if recommendations:
            prompt += f"\nQuality Improvement Recommendations:\n{recommendations}"
        
        prompt += f"""
        CRITICAL JSON RESPONSE REQUIREMENTS:
        You MUST return a single, valid JSON object. Follow these rules exactly:
        1. Start with a single opening curly brace
        2. End with a single closing curly brace
        3. Use double quotes for ALL keys and string values
        4. Do not include ANY explanatory text before or after the JSON
        5. Do not include markdown formatting
        6. Do not include multiple JSON objects
        7. Ensure all arrays and objects are properly closed
        8. No trailing commas after the last item
        9. No comments within the JSON
        10. No line breaks within string values

        Expected Output Format:
        {agent_config['expected_output']}

        Input Data:
        """
        
        for key, value in inputs.items():
            if key != 'recommendations':
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
        """Validate and parse LLM response with enhanced error handling"""
        try:
            # Remove any markdown code block formatting if present
            if response_text.startswith("```") and response_text.endswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])
            
            # Try to parse as JSON first
            try:
                parsed = json.loads(response_text)
                self._validate_json_structure(parsed)
                return parsed
            except json.JSONDecodeError:
                # If not JSON, check if it's LaTeX
                if response_text.strip().startswith("\\documentclass"):
                    self._validate_latex_structure(response_text.strip())
                    return response_text.strip()
                else:
                    raise ValueError("Response is neither valid JSON nor LaTeX")
                
        except Exception as e:
            logger.error(f"Failed to validate LLM response: {str(e)}")
            logger.error(f"Response text:\n{response_text}")
            raise ValueError(f"Invalid response format: {str(e)}")

    def _validate_json_structure(self, data: dict):
        """Validate JSON structure meets expected format"""
        if not isinstance(data, dict):
            raise ValueError("Response must be a dictionary")
            
        # Add specific validation rules based on response type
        if 'resume_analysis' in data:
            required_fields = ['personal_info', 'professional_summary', 'work_experience']
            missing = [f for f in required_fields if f not in data['resume_analysis']]
            if missing:
                raise ValueError(f"Missing required fields in resume analysis: {missing}")
                
        # Add more validation rules for other response types

    def _validate_latex_structure(self, latex_text: str):
        """Validate LaTeX document structure"""
        required_elements = [
            "\\documentclass",
            "\\begin{document}",
            "\\end{document}"
        ]
        
        missing = [elem for elem in required_elements if elem not in latex_text]
        if missing:
            raise ValueError(f"Missing required LaTeX elements: {missing}")

    def _log_step_output(self, step_name: str, output: Any, duration: float = None):
        """Log detailed output for each workflow step"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Step Completed: {step_name}")
        logger.info(f"Duration: {duration:.2f} seconds" if duration else "Duration: Unknown")
        
        try:
            if isinstance(output, dict):
                formatted_output = json.dumps(output, indent=2, ensure_ascii=False)
            else:
                formatted_output = str(output)
            
            logger.info("\nOutput Summary:")
            logger.info(formatted_output)
                
        except Exception as e:
            logger.error(f"Error formatting output: {str(e)}")
            logger.info(f"Raw output: {str(output)[:500]}")
        
        logger.info(f"\n{'='*80}\n")

    async def _apply_quality_recommendations(self, ctx: Context, recommendations: dict) -> bool:
        """Apply quality recommendations and determine if improvements are needed"""
        try:
            # Extract quality metrics
            quality_scores = recommendations.get('quality_validation', {}).get('sections', {})
            formatting_issues = recommendations.get('formatting_validation', {}).get('consistency', {})
            latex_readiness = recommendations.get('latex_readiness', {}).get('status', 'Fail')
            
            # Map of possible section names to standardized names
            section_name_map = {
                'summary_section': ['summary_section', 'professional_summary', 'summary'],
                'experience_section': ['experience_section', 'experience', 'work_experience'],
                'skills_section': ['skills_section', 'skills', 'technical_skills'],
                'education_section': ['education_section', 'education'],
                'personal_info': ['personal_info', 'contact_info', 'contact'],
                'projects_section': ['projects_section', 'projects'],
                'certifications_section': ['certifications_section', 'certifications']
            }
            
            # Calculate overall quality score with section name mapping
            section_scores = []
            critical_issues = []
            
            # Check for specific formatting issues
            date_format_issues = formatting_issues.get('date_formats', {})
            if date_format_issues.get('status') == 'Fail':
                critical_issues.append("Date format inconsistencies detected")
            
            # Check for missing durations in projects and experience
            for section_scores_key, score_data in quality_scores.items():
                # Check if the section name matches any of our mapped names
                for standard_name, variants in section_name_map.items():
                    if section_scores_key in variants:
                        if isinstance(score_data, dict):
                            # Convert completeness score to float if it's a string
                            completeness = score_data.get('completeness', 0)
                            if isinstance(completeness, str):
                                # Remove any '%' symbol and convert to float
                                completeness = float(completeness.replace('%', ''))
                            section_scores.append(completeness)
                            
                            # Check for specific issues in the section
                            issues = score_data.get('issues', [])
                            for issue in issues:
                                if 'Duration' in issue or 'duration' in issue:
                                    critical_issues.append(f"Duration issue in {standard_name}: {issue}")
                            break
            
            avg_quality = sum(section_scores) / len(section_scores) if section_scores else 0
            
            # Check formatting issues with more specific checks
            has_formatting_issues = (
                any(issue.get('status', 'Fail') == 'Fail' for issue in formatting_issues.values()) or
                bool(critical_issues)  # Consider critical issues as formatting issues
            )
            
            # Determine if improvements needed
            needs_improvement = (
                avg_quality < 85 or  # Quality threshold
                has_formatting_issues or
                latex_readiness == 'Fail'
            )
            
            if needs_improvement:
                logger.info("Quality check indicates improvements needed:")
                if avg_quality < 85:
                    logger.info(f"- Average quality score ({avg_quality:.1f}) below threshold")
                if has_formatting_issues:
                    logger.info("- Formatting issues detected:")
                    for issue in critical_issues:
                        logger.info(f"  * {issue}")
                    if formatting_issues:
                        logger.info("  * General formatting issues found")
                if latex_readiness == 'Fail':
                    logger.info("- LaTeX readiness check failed")
                
                # Log specific recommendations for improvement
                logger.info("\nRecommended improvements:")
                if critical_issues:
                    for i, issue in enumerate(critical_issues, 1):
                        logger.info(f"{i}. {issue}")
                if avg_quality < 85:
                    logger.info(f"- Improve content quality in sections with scores below 85%")
                
            return needs_improvement
            
        except Exception as e:
            logger.error(f"Error applying quality recommendations: {str(e)}")
            logger.error(f"Recommendations data: {json.dumps(recommendations, indent=2)}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return True  # Conservative approach: assume improvements needed on error

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
        ctx.retry_count = 0
        ctx.max_retries = 3
        
        # Initialize workflow data containers
        ctx.resume_text = event.data.get("resume_text")
        ctx.job_description = event.data.get("job_description")
        
        # Initialize result containers
        ctx.resume_analysis = None
        ctx.job_analysis = None
        ctx.resume_customization = None
        ctx.summary_customization = None
        ctx.latex_resume = None
        
        # Quality check containers
        ctx.pre_latex_quality_check = None
        ctx.post_latex_quality_check = None
        
        # Validation status tracking
        ctx.validation_status = {
            "resume_analyzer": False,
            "job_description_analyzer": False,
            "resume_customizer": False,
            "summary_customizer": False,
            "latex_formatter": False,
            "quality_controllers": False
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
            
            response = await self.llm_with_fallback['openai_llm1'].acomplete(prompt)
            parsed_response = self._validate_llm_response(response.text)
            
            ctx.resume_analysis = parsed_response
            ctx.steps_completed += 1
            
            duration = time() - start_time
            ctx.step_timings['resume_analyzer'] = duration
            
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
            
            if not ctx.job_description:
                raise ValueError("Missing job description input")
            
            if len(ctx.job_description.strip()) < 50:
                raise ValueError("Job description is too short or empty")
            
            prompt = self._build_content_prompt(
                agent_config,
                job_description=ctx.job_description
            )
            
            response = await self.llm_with_fallback['openai_llm1'].acomplete(prompt)
            parsed_response = self._validate_llm_response(response.text)
            
            ctx.job_analysis = parsed_response
            ctx.steps_completed += 1
            
            duration = time() - start_time
            ctx.step_timings['job_description_analyzer'] = duration
            
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
        """Customize resume content based on job requirements"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "resume_customizer"
            agent_config = self.agent_config['resume_customizer_agent']
            
            if not all([ctx.resume_analysis, ctx.job_analysis]):
                raise ValueError("Missing required input data for resume customization")
            
            resume_data = ctx.resume_analysis.get('resume_analysis', {})
            if not resume_data:
                raise ValueError("Missing resume analysis data")
            
            recommendations = None
            if hasattr(ctx, 'pre_latex_quality_check'):
                recommendations = json.dumps(ctx.pre_latex_quality_check)
            
            prompt = self._build_content_prompt(
                agent_config,
                resume_data=json.dumps(resume_data),
                job_analysis=json.dumps(ctx.job_analysis),
                recommendations=recommendations
            )
            
            response = await self.llm_with_fallback['openai_llm2'].acomplete(prompt)
            parsed_response = self._validate_llm_response(response.text)
            
            if 'content_customization' not in parsed_response:
                raise ValueError("Missing 'content_customization' in response")
            
            ctx.resume_customization = parsed_response
            ctx.steps_completed += 1
            
            duration = time() - start_time
            ctx.step_timings['resume_customizer'] = duration
            
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
            
            if not ctx.resume_analysis or not ctx.job_analysis:
                raise ValueError("Missing required analysis data")
            
            resume_data = ctx.resume_analysis
            work_summary = resume_data.get('resume_analysis', {}).get('work_summary', {})
            experience = resume_data.get('resume_analysis', {}).get('work_experience', [])
            skills = resume_data.get('resume_analysis', {}).get('skills', {})
            
            customized_resume = ctx.resume_customization.get('content_customization', {})
            
            recommendations = None
            if hasattr(ctx, 'pre_latex_quality_check'):
                recommendations = json.dumps(ctx.pre_latex_quality_check)
            
            input_data = {
                'resume_data': {
                    'work_summary': work_summary,
                    'experience': experience,
                    'skills': skills
                },
                'customized_resume': customized_resume,
                'job_analysis': ctx.job_analysis
            }
            
            prompt = self._build_content_prompt(
                agent_config,
                resume_data=json.dumps(input_data['resume_data']),
                customized_resume=json.dumps(input_data['customized_resume']),
                job_analysis=json.dumps(input_data['job_analysis']),
                recommendations=recommendations
            )
            
            response = await self.llm_with_fallback['openai_llm1'].acomplete(prompt)
            parsed_response = self._validate_llm_response(response.text)
            
            ctx.summary_customization = parsed_response
            ctx.steps_completed += 1
            
            duration = time() - start_time
            ctx.step_timings['summary_customizer'] = duration
            
            self._log_step_output(
                "Professional Summary",
                ctx.summary_customization,
                duration=duration
            )
            
            return Event(data={"context": ctx})
            
        except Exception as e:
            error_msg = f"Summary customization failed: {str(e)}"
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
            
            required_inputs = {
                "resume_analysis": ctx.resume_analysis,
                "job_analysis": ctx.job_analysis,
                "customized_resume": ctx.resume_customization,
                "customized_summary": ctx.summary_customization
            }
            
            missing_inputs = [k for k, v in required_inputs.items() if not v]
            if missing_inputs:
                raise ValueError(f"Missing required inputs: {missing_inputs}")
            
            input_data = {
                "customized_resume": ctx.resume_customization.get('content_customization', {}),
                "customized_summary": ctx.summary_customization.get('summary_analysis', {})
            }
            
            for key, value in input_data.items():
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid {key} structure: expected dictionary")
            
            prompt = self._build_content_prompt(
                agent_config,
                **input_data
            )
            
            response = await self.llm_with_fallback['openai_llm2'].acomplete(prompt)
            parsed_response = self._validate_llm_response(response.text)
            
            required_sections = ['quality_validation', 'formatting_validation', 'latex_readiness']
            missing_sections = [s for s in required_sections if s not in parsed_response]
            if missing_sections:
                raise ValueError(f"Missing required quality check sections: {missing_sections}")
            
            # Store quality check results
            ctx.pre_latex_quality_check = parsed_response
            
            # Determine if improvements are needed
            needs_improvement = await self._apply_quality_recommendations(ctx, parsed_response)
            
            if needs_improvement and ctx.retry_count < ctx.max_retries:
                ctx.retry_count += 1
                logger.info(f"Quality check indicates improvements needed. Attempt {ctx.retry_count}/{ctx.max_retries}")
                # Return to resume customizer step
                ctx.workflow_state = "resume_customizer"
                return await self.resume_customizer(Event(data={"context": ctx}))
            
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
        """Convert structured content to LaTeX format"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "latex_formatter"
            agent_config = self.agent_config['latex_formatting_agent']
            
            logger.info("Starting LaTeX conversion...")
            
            # Get customized content
            customized_resume = ctx.resume_customization.get('content_customization', {})
            customized_summary = ctx.summary_customization.get('summary_analysis', {})
            job_requirements = ctx.job_analysis
            
            if not all([customized_resume, customized_summary, job_requirements]):
                raise ValueError("Missing required content for LaTeX conversion")
            
            required_sections = ['personal_info', 'summary_section', 'experience_section', 'skills_section']
            missing_sections = [s for s in required_sections if s not in customized_resume]
            if missing_sections:
                raise ValueError(f"Missing required resume sections: {missing_sections}")
            
            recommendations = None
            if hasattr(ctx, 'pre_latex_quality_check'):
                recommendations = json.dumps(ctx.pre_latex_quality_check)
            
            resume_data = {
                "customized_resume": customized_resume,
                "customized_summary": customized_summary,
                "job_requirements": job_requirements
            }
            
            prompt = self._build_latex_prompt(
                agent_config,
                resume_data=json.dumps(resume_data),
                recommendations=recommendations
            )
            
            response = await self.llm_with_fallback['openai_llm1'].acomplete(prompt)
            latex_content = self._validate_llm_response(response.text)
            
            if isinstance(latex_content, str):
                latex_content = latex_content.strip()
                
                # Validate LaTeX structure
                required_latex_elements = [
                    "\\documentclass",
                    "\\begin{document}",
                    "\\end{document}"
                ]
                
                missing_elements = [
                    elem for elem in required_latex_elements 
                    if elem not in latex_content
                ]
                
                if missing_elements:
                    raise ValueError(f"Invalid LaTeX structure. Missing: {missing_elements}")
                
                ctx.latex_resume = {
                    "latex_document": {
                        "content": latex_content,
                        "metadata": {
                            "timestamp": time(),
                            "version": "1.0",
                            "generator": "latex_formatting_agent"
                        },
                        "validation": {
                            "structure_check": "pass",
                            "required_elements": required_latex_elements,
                            "validation_timestamp": time()
                        }
                    }
                }
            else:
                raise ValueError("Invalid LaTeX response format: expected string")
            
            ctx.steps_completed += 1
            
            duration = time() - start_time
            ctx.step_timings['latex_formatter'] = duration
            
            self._log_step_output(
                "LaTeX Generation",
                {
                    "status": "success",
                    "latex_length": len(latex_content),
                    "validation": ctx.latex_resume["latex_document"]["validation"]
                },
                duration=duration
            )
            
            return Event(data={"context": ctx})
            
        except Exception as e:
            error_msg = f"LaTeX formatting failed: {str(e)}"
            ctx.errors.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    @step
    async def post_latex_quality_controller(self, event: Event) -> Event:
        """Perform final quality check on LaTeX output"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "post_latex_quality_controller"
            agent_config = self.agent_config['post_latex_quality_controller_agent']
            
            logger.info("Starting LaTeX quality assessment...")
            
            if not ctx.latex_resume:
                raise ValueError("Missing LaTeX document for quality check")

            prompt = self._build_content_prompt(
                agent_config,
                latex_document=ctx.latex_resume,
                job_requirements=json.dumps(ctx.job_analysis)
            )
            
            response = await self.llm_with_fallback['openai_llm2'].acomplete(prompt)
            parsed_response = self._validate_llm_response(response.text)
            
            ctx.latex_quality_check = parsed_response
            
            # Check if LaTeX needs improvement
            latex_quality = parsed_response.get('compilation_validation', {}).get('compilation_status', {})
            if latex_quality.get('success') != "Yes" and ctx.retry_count < ctx.max_retries:
                ctx.retry_count += 1
                logger.info(f"LaTeX quality check failed. Attempt {ctx.retry_count}/{ctx.max_retries}")
                return await self.latex_formatter(Event(data={"context": ctx}))
            
            duration = time() - start_time
            ctx.step_timings['post_latex_quality_controller'] = duration
            
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
    async def finalize_output(self, event: Event) -> StopEvent:
        """Prepare final output and complete workflow"""
        try:
            ctx = event.data["context"]
            
            # Prepare final output structure
            output = {
                "workflow_summary": {
                    "status": "completed",
                    "steps_completed": ctx.steps_completed,
                    "total_steps": ctx.total_steps,
                    "total_duration": sum(ctx.step_timings.values()),
                    "step_timings": ctx.step_timings,
                    "errors": ctx.errors
                },
                "resume_output": {
                    "latex_document": ctx.latex_resume["latex_document"],
                    "quality_validation": {
                        "pre_latex": ctx.pre_latex_quality_check,
                        "post_latex": ctx.latex_quality_check
                    }
                },
                "metadata": {
                    "timestamp": time(),
                    "version": "1.0"
                }
            }
            
            return StopEvent(result=output)
            
        except Exception as e:
            error_msg = f"Output finalization failed: {str(e)}"
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
        
        # Initialize text processor
        text_processor = TextExtractionTool()
        
        # Extract resume text with OCR fallback
        logger.info("Extracting text from resume...")
        resume_text = text_processor.run(resume_path)
        
        # Handle job description
        job_description = ""
        if Path(args.job_description).exists():
            job_path = await validate_file_path(args.job_description)
            job_description = text_processor.run(job_path)
        else:
            job_description = args.job_description

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
        try:
            for step in workflow.steps:
                logger.info(f"Executing step: {step.__name__.replace('_', ' ').title()}")
                current_event = await step(current_event)
                
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise
        
        # Save the final output
        await save_output(current_event, args.output)
        logger.info("\nProcess completed successfully!")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise SystemExit(1)

if __name__ == "__main__":
    asyncio.run(main())