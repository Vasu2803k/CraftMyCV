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
from setup_logging import setup_logging
import argparse
from tools.text_extractor_tool import TextExtractionTool
import asyncio
from time import time
from typing import Any
from datetime import datetime

# Ignore warnings
warnings.filterwarnings("ignore")
# Load environment variables
dotenv.load_dotenv()

# Configure logging using setup_logging with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logging(
    log_level="INFO",
    log_dir="data/logs",
    log_file_name=f"craft_my_cv_{timestamp}",
    logger_name="craft_my_cv_workflow"
)

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

class CraftMyCVWorkflow(Workflow):
    def __init__(self):
        super().__init__()
        # Updated workflow steps based on flowchart
        self.steps = [
            self.setup_workflow_context,
            self.resume_analyzer,
            self.job_description_analyzer,
            self.suggestion_agent,
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
            with open('src/config/templates.yaml', 'r') as f:
                template_config_yaml = yaml.safe_load(f)
            with open('src/config/projects.yaml', 'r') as f:
                projects_config_yaml = yaml.safe_load(f)
            
            self.agent_config = agent_config_yaml['agents']
            self.template_config = template_config_yaml['templates']
            self.projects_config = projects_config_yaml['projects']

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

    def _suggestion_prompt(self, agent_config, **inputs):
        """Build prompt for suggestion agent"""
        prompt = f"""
        Generate a valid JSON object based on the input data according to the system instructions.

        **System Instructions**: {agent_config['system_prompt']}

        **Input Data**:
        """
        for key, value in inputs.items():
            prompt += f"\n{key}: {value}"

        prompt += """
        # Expected Output Format
        Return exactly one JSON object with the following structure:

        {
            "suggestions": {
                "missing_skills": "Include the skills that are missing from the resume as a list",
                "suggested_projects": "Include the projects that are relevant to the job requirements espcially for the missing skills from the provided projects list as a list"
            }
        }
        """
        prompt += f"""
        **Projects List**:
        The projects list is given in yaml format as follows:
        {self.projects_config}
        """

        prompt += """
        # Notes
        - Do not include any text before or after the JSON output
        - Ensure proper JSON formatting with double quotes
        - Do not include ```json before the JSON output
        - Return exactly one JSON object
        """
        return prompt
    
    def _build_latex_prompt(self, agent_config, template_config, recommendations=None, **inputs):
        """Build prompt for generating raw LaTeX code"""
        prompt = f"""
        Generate LaTeX code based on the input data according to the system instructions.

        **System Instructions**: {agent_config['system_prompt']}

        **Input Data**:
        """
        for key, value in inputs.items():
            prompt += f"\n{key}: {value}"

        if recommendations:
            prompt += f"""
            Ensure that all recommendations provided are applied to improve the output.
            **Quality Improvement Recommendations**: {recommendations}
            """

        prompt += f"""
        **Expected Output Format**:
        Strictly follow the formatting style of the template provided with a little flexibility when necessary to improve the output.
        - {template_config['content']}

        # Notes

        - The output must be LaTeX code only, including any additional formatting or styling
        - Follow consistent indentation and spacing in the LaTeX code
        - Use proper line breaks between sections and environments
        - Apply consistent capitalization for LaTeX commands and environments
        - Include all necessary package imports and configurations
        - Do not include ```latex before the LaTeX code
        - Follow professional formatting and layout standards
        - Ensure proper LaTeX syntax and ATS compatibility
        - Structure content in logical sections
        - Handle special characters correctly

        # Final Notes
        - Ensure the final latex code contains only the sections specified in the expected output format
        - The chronological order of experiences must be maintained
        - The order of sections must be maintained as specified in the expected output format
        - Strictly follow the formatting rules specified in the expected output format
        - Adhere to the notes provided.
        """
        return prompt
    
    def _build_content_prompt(self, agent_config, recommendations=None, **inputs):
        """Helper method to build prompts for agents with quality recommendations"""
        prompt = f"""
        Generate a valid JSON object based on the input data according to the system instructions.

        **System Instructions**: {agent_config['system_prompt']}

        **Input Data**:
        """
        for key, value in inputs.items():
            prompt += f"\n{key}: {value}"

        # Add specific instructions for resume customizer if skill/project suggestions exist
        if 'suggestions' in inputs:
            prompt += """
            # Skill and Project Integration Instructions
            
            1. Review the suggested skills and projects carefully
            2. Add skills and relevant projects from the suggestions to the resume
            3. Maintain authenticity while incorporating new elements
            4. Add new skills to the skills section and add the projects to the projects section
            5. Ensure project descriptions demonstrate the suggested skills
            6. Maintain balance between existing and new content
            7. Keep focus on job-relevant skills and projects
            """

        if recommendations:
            prompt += f"""
            Ensure that all recommendations provided, which might be added during a retry mechanism, are applied to improve the output.
            **Quality Improvement Recommendations**: {recommendations}
            """

        prompt += f"""
        # Steps

        - Ensure that the JSON object starts with a single opening curly brace and ends with a single closing curly brace.
        - Use double quotes for all keys and string values within the JSON object.
        - Avoid any extraneous text, markdown, or multiple JSON objects.
        - Properly close all arrays and objects within the JSON.
        - Do not include trailing commas or comments within the JSON content.
        - Maintain string values without line breaks.

        # Output Format

        Return exactly one JSON object as specified in the agent configuration.
        
        **Expected Output Format**:

        - {agent_config['expected_output']}
        
        # Notes
        - Do not include any text, explanation, comments, or markdown before or after the JSON output.
        - Do not include ```json or ```latex before the JSON output.
        - Ensure consistency with the formatting rules, as any deviation might cause errors in JSON parsing.
        - Carefully check the structure of arrays and objects to prevent errors in JSON syntax.
        - Ensure that the keys and values are properly quoted and escaped.
        """
        return prompt

    def _validate_llm_response(self, response_text: str) -> dict | str:
        """Validate and parse LLM response with enhanced error handling"""
        try:
            # Remove any markdown code block formatting if present
            if response_text.startswith("```") and response_text.endswith("```"):
                lines = response_text.split("\n")
                # Handle potential language identifier in code block
                if lines[0].lower().startswith("```json"):
                    lines = lines[1:-1]
                elif lines[0].strip() == "```":
                    lines = lines[1:-1]
                response_text = "\n".join(lines)
            
            # Clean the response text
            response_text = response_text.strip()
            
            # Try to parse as JSON first
            try:
                # Handle potential multiple JSON objects by taking the first valid one
                possible_json = response_text.split('\n\n')[0]
                parsed = json.loads(possible_json)
                self._validate_json_structure(parsed)
                return parsed
            except json.JSONDecodeError:
                # If not JSON, check if it's LaTeX
                if "\\documentclass" in response_text:
                    self._validate_latex_structure(response_text)
                    return response_text
                else:
                    # Log the problematic response for debugging
                    logger.error(f"Invalid response format. Response text:\n{response_text[:500]}...")
                    logger.error("Response is neither valid JSON nor LaTeX")
                    raise ValueError(f"Response validation failed. Expected JSON or LaTeX, got:\n{response_text[:100]}...")
                
        except Exception as e:
            logger.error(f"Failed to validate LLM response: {str(e)}")
            logger.error(f"Response text preview:\n{response_text[:500]}...")
            raise ValueError(f"Invalid response format: {str(e)}")

    def _validate_json_structure(self, data: dict):
        """Validate JSON structure meets expected format"""
        if not isinstance(data, dict):
            raise ValueError("Response must be a dictionary")
            
        # Resume analyzer output validation
        if 'resume_analysis' in data:  # Keep 'resume_analysis' as the top-level key
            if not isinstance(data['resume_analysis'], dict):
                raise ValueError("resume_analysis must be a dictionary")

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

    async def _apply_quality_recommendations(self, ctx: Context, recommendations: dict, workflow_state: str = None) -> bool:
        """
        Apply quality recommendations and determine if improvements needed
        Args:
            ctx: Workflow context
            recommendations: Quality check recommendations
            workflow_state: Current workflow state to determine quality check type
        """
        try:
            # Track specific issues for retry context
            ctx.quality_issues = {
                'content': [],
                'formatting': [],
                'latex': []
            }
            
            # Determine which quality check to use based on workflow state
            if workflow_state == 'latex_formatter_retry':
                quality_data = recommendations.get('post_latex_quality_check', {})
                validation_key = 'compilation_validation'
                readiness_key = 'latex_readiness'
            else:  # resume_customizer_retry or default
                quality_data = recommendations.get('pre_latex_quality_check', {})
                validation_key = 'quality_validation'
                readiness_key = 'latex_readiness'
            
            # Extract quality validation data with safe gets
            quality_validation = quality_data.get(validation_key, {})
            formatting_validation = quality_data.get('formatting_validation', {})
            latex_readiness = quality_data.get(readiness_key, {})
            
            # Initialize flags for different types of issues
            has_quality_issues = False
            has_formatting_issues = False
            has_latex_issues = False
            critical_issues = []
            
            if workflow_state == 'latex_formatter_retry':
                # Handle LaTeX-specific quality checks
                compilation_status = quality_validation.get('compilation_status', {})
                if compilation_status.get('success') != "Yes":
                    has_latex_issues = True
                    for issue in compilation_status.get('issues', []):
                        latex_issue = f"LaTeX Compilation: {issue}"
                        critical_issues.append(latex_issue)
                        ctx.quality_issues['latex'].append(latex_issue)
                    
                # Check LaTeX structure and formatting
                for category in ['structure', 'formatting', 'hierarchy']:
                    category_data = latex_readiness.get(category, {})
                    if category_data.get('status') == 'Fail':
                        has_latex_issues = True
                        for issue in category_data.get('issues', []):
                            if isinstance(issue, dict):
                                if issue.get('requires_fix'):
                                    latex_issue = f"LaTeX {category}: {issue.get('description', 'Unknown issue')}"
                                    critical_issues.append(latex_issue)
                                    ctx.quality_issues['latex'].append(latex_issue)
                            else:
                                latex_issue = f"LaTeX {category}: {issue}"
                                critical_issues.append(latex_issue)
                                ctx.quality_issues['latex'].append(latex_issue)
            else:
                # Handle content and formatting checks for resume customization
                # Check section quality
                sections = quality_validation.get('sections', {})
                for section_name, section_data in sections.items():
                    if section_data.get('status') == 'Fail':
                        has_quality_issues = True
                        completeness = float(str(section_data.get('completeness', '0')).replace('%', ''))
                        if completeness < 95:
                            issue = f"{section_name} section is incomplete ({completeness}%)"
                            critical_issues.append(issue)
                            ctx.quality_issues['content'].append(issue)
                        
                        # Add specific section issues
                        section_issues = section_data.get('issues', [])
                        for issue in section_issues:
                            critical_issues.append(f"{section_name}: {issue}")
                            ctx.quality_issues['content'].append(f"{section_name}: {issue}")
                
                # Check relationship validation
                relationships = quality_validation.get('relationships', {})
                for rel_type, rel_data in relationships.items():
                    if rel_data.get('status') == 'Fail':
                        has_quality_issues = True
                        for issue in rel_data.get('issues', []):
                            critical_issues.append(f"Relationship ({rel_type}): {issue}")
                            ctx.quality_issues['content'].append(f"Relationship ({rel_type}): {issue}")
                
                # Check formatting consistency
                consistency = formatting_validation.get('consistency', {})
                for format_type, format_data in consistency.items():
                    if format_data.get('status') == 'Fail':
                        has_formatting_issues = True
                        for issue in format_data.get('issues', []):
                            critical_issues.append(f"Formatting ({format_type}): {issue}")
                            ctx.quality_issues['formatting'].append(f"Formatting ({format_type}): {issue}")
            
            # Store retry context
            ctx.retry_context = {
                'workflow_state': workflow_state,
                'has_quality_issues': has_quality_issues,
                'has_formatting_issues': has_formatting_issues,
                'has_latex_issues': has_latex_issues,
                'critical_issues': critical_issues,
                'retry_count': ctx.retry_count,
                'max_retries': ctx.max_retries,
                'previous_issues': getattr(ctx, 'previous_issues', set()),
                'timestamp': time()
            }
            
            # Track persistent issues across retries
            current_issues = set(critical_issues)
            if hasattr(ctx, 'previous_issues'):
                persistent_issues = current_issues.intersection(ctx.previous_issues)
                if persistent_issues:
                    logger.warning(f"\nDetected persistent issues across retries ({workflow_state}):")
                    for issue in persistent_issues:
                        logger.warning(f"- {issue}")
            ctx.previous_issues = current_issues
            
            # Determine if improvements needed
            needs_improvement = has_quality_issues or has_formatting_issues or has_latex_issues
            
            if needs_improvement:
                logger.info(f"\nQuality check indicates improvements needed ({workflow_state}):")
                
                if workflow_state == 'latex_formatter_retry':
                    logger.info("\nLaTeX Issues:")
                    for issue in ctx.quality_issues['latex']:
                        logger.info(f"- {issue}")
                else:
                    if has_quality_issues:
                        logger.info("\nContent Quality Issues:")
                        for issue in ctx.quality_issues['content']:
                            logger.info(f"- {issue}")
                    
                    if has_formatting_issues:
                        logger.info("\nFormatting Issues:")
                        for issue in ctx.quality_issues['formatting']:
                            logger.info(f"- {issue}")
                
                logger.info(f"\nRetry attempt {ctx.retry_count + 1}/{ctx.max_retries}")
                logger.info("\nRecommended Actions:")
                if workflow_state == 'latex_formatter_retry':
                    logger.info("1. Fix LaTeX compilation and structure issues")
                else:
                    if has_quality_issues:
                        logger.info("1. Review and enhance content completeness")
                    if has_formatting_issues:
                        logger.info("2. Address formatting inconsistencies")
            
            return needs_improvement
            
        except Exception as e:
            logger.error(f"Error applying quality recommendations ({workflow_state}): {str(e)}")
            logger.error(f"Recommendations data: {json.dumps(recommendations, indent=2)}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            # Store error in retry context
            ctx.retry_context = {
                'error': str(e),
                'workflow_state': workflow_state,
                'retry_count': ctx.retry_count,
                'max_retries': ctx.max_retries,
                'timestamp': time()
            }
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
        ctx.suggestions = None

        # Quality check containers
        ctx.pre_latex_quality_check = None
        ctx.post_latex_quality_check = None
        
        # Validation status tracking
        ctx.validation_status = {
            "resume_analyzer": False,
            "job_description_analyzer": False,
            "suggestion_agent": False,
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
            logger.info(f"Parsed response: {parsed_response}")
            # Validate expected output structure
            if 'resume_analysis' not in parsed_response:
                raise ValueError("Missing 'resume_analysis' in response")
            
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
            
            # Validate expected output structure
            if 'job_analysis' not in parsed_response:
                raise ValueError("Missing 'job_analysis' in response")
            
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
    async def suggestion_agent(self, event: Event) -> Event:
        """Suggest skills and projects based on job requirements"""
        try:
            start_time = time()
            ctx = event.data["context"]
            ctx.workflow_state = "suggestion_agent"
            agent_config = self.agent_config['suggestion_agent']
            
            if not ctx.resume_analysis or not ctx.job_analysis:
                raise ValueError("Missing required analysis data for suggestions")
            
            resume_data = ctx.resume_analysis.get('resume_analysis', {})
            job_data = ctx.job_analysis.get('job_analysis', {})
            
            logger.info("Starting suggestion analysis...")
            
            prompt = self._suggestion_prompt(
                agent_config,
                resume_data=json.dumps(resume_data),
                job_analysis=json.dumps(job_data)
            )
            
            response = await self.llm_with_fallback['openai_llm2'].acomplete(prompt)
            parsed_response = self._validate_llm_response(response.text)
            
            if 'suggestions' not in parsed_response:
                raise ValueError("Missing 'suggestions' in response")
            
            ctx.suggestions = parsed_response
            ctx.steps_completed += 1
            
            duration = time() - start_time
            ctx.step_timings['suggestion_agent'] = duration
            
            self._log_step_output(
                "Suggestions",
                ctx.suggestions,
                duration=duration
            )
            
            return Event(data={"context": ctx})
            
        except Exception as e:
            error_msg = f"Suggestion analysis failed: {str(e)}"
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
            
            if not all([ctx.resume_analysis, ctx.job_analysis, ctx.suggestions]):
                raise ValueError("Missing required input data for resume customization")
            
            resume_data = ctx.resume_analysis.get('resume_analysis', {})
            job_data = ctx.job_analysis.get('job_analysis', {})
            suggestions = ctx.suggestions.get('suggestions', None)
            
            # Get suggestion list from the user 
            if not suggestions:
                raise ValueError("Missing suggestions data")

            recommendations = None
            if hasattr(ctx, 'pre_latex_quality_check'):
                recommendations = json.dumps(ctx.pre_latex_quality_check)
            
            prompt = self._build_content_prompt(
                agent_config,
                resume_data=json.dumps(resume_data),
                job_analysis=json.dumps(job_data),
                suggestions=json.dumps(suggestions),
                recommendations=recommendations
            )
            
            response = await self.llm_with_fallback['openai_llm1'].acomplete(prompt)
            parsed_response = self._validate_llm_response(response.text)
            
            # Validate expected output structure
            if 'customized_resume' not in parsed_response:
                raise ValueError("Missing 'customized_resume' in response")
            
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
            
            resume_data = ctx.resume_analysis.get('resume_analysis', {})
            job_analysis = ctx.job_analysis.get('job_analysis', {})
            customized_resume = ctx.resume_customization.get('customized_resume', {})

            if not customized_resume:
                raise ValueError("Missing customized resume data")
            if not job_analysis:
                raise ValueError("Missing job analysis data")
            if not resume_data:
                raise ValueError("Missing resume data")
            
            recommendations = None
            if hasattr(ctx, 'pre_latex_quality_check'):
                recommendations = json.dumps(ctx.pre_latex_quality_check)
            
            input_data = {
                'resume_data': resume_data,
                'customized_resume': customized_resume,
                'job_analysis': job_analysis
            }
            
            prompt = self._build_content_prompt(
                agent_config,
                **input_data,
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
            workflow_state = ctx.workflow_state
            
            agent_config = self.agent_config['pre_latex_quality_controller_agent']
            
            logger.info("Starting pre-LaTeX quality assessment...")
            
            resume_analysis = ctx.resume_analysis.get('resume_analysis', {})
            job_analysis = ctx.job_analysis.get('job_analysis', {})
            customized_resume = ctx.resume_customization.get('customized_resume', {})
            summary_analysis = ctx.summary_customization.get('summary_analysis', {})
            
            input_data = {
                "resume_analysis": resume_analysis,
                "job_analysis": job_analysis,
                "customized_resume": customized_resume,
                "summary_analysis": summary_analysis
            }
            
            for key, value in input_data.items():
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid {key} structure: expected dictionary")
            
            prompt = self._build_content_prompt(
                agent_config,
                **input_data
            )
            
            response = await self.llm_with_fallback['openai_llm1'].acomplete(prompt)
            parsed_response = self._validate_llm_response(response.text)
            
            # Check if improvements needed
            needs_improvement = await self._apply_quality_recommendations(
                ctx, 
                parsed_response, 
                workflow_state=workflow_state
            )
            
            if needs_improvement and ctx.retry_count < ctx.max_retries:
                ctx.retry_count += 1
                logger.info(f"Quality check indicates improvements needed. Attempt {ctx.retry_count}/{ctx.max_retries}")
                ctx.workflow_state = "resume_customizer_retry"
                return await self.resume_customizer(Event(data={"context": ctx}))
            
            # Make sure retry count is reset to 0
            ctx.retry_count = 0
            
            # Store quality check results
            ctx.pre_latex_quality_check = parsed_response
            
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
            template_config = self.template_config['resume_template_1']
            
            logger.info("Starting LaTeX conversion...")
            
            # Get customized content with updated keys
            customized_resume = ctx.resume_customization.get('customized_resume', {})
            summary_analysis = ctx.summary_customization.get('summary_analysis', {})
            job_analysis = ctx.job_analysis.get('job_analysis', {})
            
            if not all([customized_resume, summary_analysis, job_analysis]):
                raise ValueError("Missing required content for LaTeX conversion")
            
            recommendations = None
            if hasattr(ctx, 'pre_latex_quality_check'):
                recommendations = json.dumps(ctx.pre_latex_quality_check)
            
            resume_data = {
                "customized_resume": customized_resume,
                "summary_analysis": summary_analysis,
                "job_analysis": job_analysis
            }
            
            prompt = self._build_latex_prompt(
                agent_config,
                template_config,
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
            workflow_state = ctx.workflow_state
            
            agent_config = self.agent_config['post_latex_quality_controller_agent']
            
            logger.info("Starting LaTeX quality assessment...")
            
            if not ctx.latex_resume:
                raise ValueError("Missing LaTeX document for quality check")

            prompt = self._build_content_prompt(
                agent_config,
                latex_document=ctx.latex_resume
            )
            
            response = await self.llm_with_fallback['openai_llm1'].acomplete(prompt)
            parsed_response = self._validate_llm_response(response.text)
            
            # Check if LaTeX needs improvement
            needs_improvement = await self._apply_quality_recommendations(
                ctx, 
                parsed_response,
                workflow_state=workflow_state
            )
            
            if needs_improvement and ctx.retry_count < ctx.max_retries:
                ctx.retry_count += 1
                logger.info(f"LaTeX quality check failed. Attempt {ctx.retry_count}/{ctx.max_retries}")
                ctx.workflow_state = "latex_formatter_retry"
                return await self.latex_formatter(Event(data={"context": ctx}))
            
            ctx.latex_quality_check = parsed_response
            
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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Update key access for personal info
            resume_data = ctx.resume_analysis.get('resume_analysis', {})
            personal_info = resume_data.get('personal_info', {})
            full_name = personal_info.get('name', 'unnamed').lower().replace(' ', '_')
            position_overview = ctx.job_analysis.get('job_analysis', {}).get('position_overview', {})
            role = position_overview.get('title', 'role').lower().replace(' ', '_')
            
            # Create output directory if it doesn't exist
            output_dir = Path('data/output')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save LaTeX file
            latex_content = ctx.latex_resume["latex_document"]["content"]
            tex_filename = output_dir / f"{full_name}_{role}_{timestamp}.tex"
            
            try:
                tex_filename.write_text(latex_content, encoding='utf-8')
                logger.info(f"LaTeX file saved as: {tex_filename}")
            except Exception as e:
                error_msg = f"Failed to save LaTeX file: {str(e)}"
                ctx.errors.append(error_msg)
                logger.error(error_msg)
                logger.error(f"Stack trace:\n{traceback.format_exc()}")
            
            # Prepare final output structure
            output = {
                "workflow_summary": {
                    "status": "completed",
                    "steps_completed": ctx.steps_completed,
                    "total_steps": ctx.total_steps,
                    "total_duration": sum(ctx.step_timings.values()),
                    "step_timings": ctx.step_timings,
                    "errors": ctx.errors,
                    "latex_file": str(tex_filename)
                },
                "resume_output": {
                    "latex_document": ctx.latex_resume["latex_document"],
                    "quality_validation": {
                        "pre_latex": ctx.pre_latex_quality_check,
                        "post_latex": ctx.latex_quality_check
                    }
                },
                "analysis": {
                    "resume_analysis": ctx.resume_analysis,
                    "job_analysis": ctx.job_analysis,
                    "customization": {
                        "resume": ctx.resume_customization,
                        "summary": ctx.summary_customization
                    }
                },
                "metadata": {
                    "timestamp": timestamp,
                    "version": "1.0",
                    "input_files": {
                        "resume": getattr(ctx, 'resume_file', None),
                        "job_description": getattr(ctx, 'job_description_file', None)
                    }
                }
            }
            
            # Save detailed output JSON
            json_filename = output_dir / f"{full_name}_{role}_{timestamp}_detailed.json"
            try:
                json_filename.write_text(
                    json.dumps(output, indent=2, ensure_ascii=False),
                    encoding='utf-8'
                )
                logger.info(f"Detailed output saved as: {json_filename}")
            except Exception as e:
                error_msg = f"Failed to save detailed output: {str(e)}"
                ctx.errors.append(error_msg)
                logger.error(error_msg)
                logger.error(f"Stack trace:\n{traceback.format_exc()}")
            
            # Save summary output JSON
            summary_output = {
                "status": output["workflow_summary"]["status"],
                "steps_completed": output["workflow_summary"]["steps_completed"],
                "total_duration": output["workflow_summary"]["total_duration"],
                "errors": output["workflow_summary"]["errors"],
                "files": {
                    "latex": str(tex_filename),
                    "detailed_output": str(json_filename)
                },
                "timestamp": output["metadata"]["timestamp"]
            }
            
            summary_filename = output_dir / f"{full_name}_{role}_{timestamp}_summary.json"
            try:
                summary_filename.write_text(
                    json.dumps(summary_output, indent=2, ensure_ascii=False),
                    encoding='utf-8'
                )
                logger.info(f"Summary output saved as: {summary_filename}")
            except Exception as e:
                error_msg = f"Failed to save summary output: {str(e)}"
                ctx.errors.append(error_msg)
                logger.error(error_msg)
                logger.error(f"Stack trace:\n{traceback.format_exc()}")
            
            logger.info("\nWorkflow completed successfully!")
            logger.info(f"Output files saved in: {output_dir}")
            logger.info(f"LaTeX file: {tex_filename.name}")
            logger.info(f"Detailed output: {json_filename.name}")
            logger.info(f"Summary output: {summary_filename.name}")
            
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
        logger.info("\nProcess completed successfully!")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise SystemExit(1)

if __name__ == "__main__":
    asyncio.run(main())