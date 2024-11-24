import yaml
import traceback
import os
import json
import dotenv
from llama_index.core.workflow import (
    Workflow, 
    step, 
    Event,
    Context
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

async def save_output(output: dict, output_path: str):
    """Save the output to a file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Output saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save output: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise

class CraftMyCVWorkflow(Workflow):
    def __init__(self):
        super().__init__()
        self.steps = [
            self.initialize_workflow,
            self.analyze_resume,
            self.analyze_job_description,
            self.customize_skills,
            self.customize_resume,
            self.customize_summary,
            self.convert_format,
            self.quality_check
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
                timeout=30
            )
            
            self.openai_llm2_with_fallback = FallbackLLM(
                primary_llm=self.openai_llm2,
                fallback_llm=self.fallback_llm,
                timeout=30
            )
            
            self.claude_llm1_with_fallback = FallbackLLM(
                primary_llm=self.claude_llm1,
                fallback_llm=self.fallback_llm,
                timeout=30
            )
            
            self.claude_llm2_with_fallback = FallbackLLM(
                primary_llm=self.claude_llm2,
                fallback_llm=self.fallback_llm,
                timeout=30
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

        Expected Output Format:
        {agent_config['expected_output']}

        Input Data:
        """
        for key, value in inputs.items():
            prompt += f"\n{key}: {value}"
            
        return prompt

    @step
    async def initialize_workflow(self, event: Event) -> Event:
        """Initialize workflow context with all necessary attributes"""
        ctx = Context(workflow=self)
        
        # Initialize metrics and state
        ctx.steps_completed = 0
        ctx.tokens_used = 0
        ctx.total_steps = 7
        ctx.workflow_state = "initialized"
        ctx.errors = []
        
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
    async def analyze_resume(self, event: Event) -> Event:
        """Resume analysis step using the resume analyzer agent"""
        try:
            ctx = event.data["context"]
            ctx.workflow_state = "analyzing_resume"
            agent_config = self.agent_config['resume_analyzer_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                resume_text=ctx.resume_text
            )
            
            response = await self.openai_llm1_with_fallback.acomplete(prompt)
            ctx.resume_analysis = json.loads(response.text)
            
            ctx.steps_completed += 1
            if hasattr(response, 'usage'):
                ctx.tokens_used += response.usage.total_tokens
            
            return Event(data={"context": ctx})
        except Exception as e:
            ctx.errors.append(f"Resume analysis failed: {str(e)}")
            raise

    @step
    async def analyze_job_description(self, event: Event) -> Event:
        """Job description analysis step"""
        try:
            ctx = event.data["context"]
            ctx.workflow_state = "analyzing_job_description"
            agent_config = self.agent_config['job_description_analyzer_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                job_description=ctx.job_description
            )
            
            response = await self.claude_llm1_with_fallback.acomplete(prompt)
            ctx.job_analysis = json.loads(response.text)
            
            ctx.steps_completed += 1
            if hasattr(response, 'usage'):
                ctx.tokens_used += response.usage.total_tokens
            
            return Event(data={"context": ctx})
        except Exception as e:
            ctx.errors.append(f"Job description analysis failed: {str(e)}")
            raise

    @step
    async def customize_skills(self, event: Event) -> Event:
        """Skill customization step"""
        try:
            ctx = event.data["context"]
            ctx.workflow_state = "customizing_skills"
            agent_config = self.agent_config['skill_customizer_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                resume_skills=json.dumps(ctx.resume_analysis.get('resume_categorization', {}).get('skills', {})),
                job_requirements=json.dumps(ctx.job_analysis.get('job_analysis', {}))
            )
            
            response = await self.openai_llm2_with_fallback.acomplete(prompt)
            ctx.skill_customization = json.loads(response.text)
            
            ctx.steps_completed += 1
            if hasattr(response, 'usage'):
                ctx.tokens_used += response.usage.total_tokens
            
            return Event(data={"context": ctx})
        except Exception as e:
            ctx.errors.append(f"Skill customization failed: {str(e)}")
            raise

    @step
    async def customize_resume(self, event: Event) -> Event:
        """Resume customization step"""
        try:
            ctx = event.data["context"]
            ctx.workflow_state = "customizing_resume"
            agent_config = self.agent_config['resume_customizer_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                resume_analysis=json.dumps(ctx.resume_analysis),
                job_analysis=json.dumps(ctx.job_analysis),
                skill_customization=json.dumps(ctx.skill_customization)
            )
            
            response = await self.claude_llm2_with_fallback.acomplete(prompt)
            ctx.resume_customization = json.loads(response.text)
            
            ctx.steps_completed += 1
            if hasattr(response, 'usage'):
                ctx.tokens_used += response.usage.total_tokens
            
            return Event(data={"context": ctx})
        except Exception as e:
            ctx.errors.append(f"Resume customization failed: {str(e)}")
            raise

    @step
    async def customize_summary(self, event: Event) -> Event:
        """Summary customization step"""
        try:
            ctx = event.data["context"]
            ctx.workflow_state = "customizing_summary"
            agent_config = self.agent_config['summary_customizer_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                resume_customization=json.dumps(ctx.resume_customization),
                job_analysis=json.dumps(ctx.job_analysis)
            )
            
            response = await self.openai_llm1_with_fallback.acomplete(prompt)
            ctx.summary_customization = json.loads(response.text)
            
            ctx.steps_completed += 1
            if hasattr(response, 'usage'):
                ctx.tokens_used += response.usage.total_tokens
            
            return Event(data={"context": ctx})
        except Exception as e:
            ctx.errors.append(f"Summary customization failed: {str(e)}")
            raise

    @step
    async def convert_format(self, event: Event) -> Event:
        """Format conversion step"""
        try:
            ctx = event.data["context"]
            ctx.workflow_state = "converting_format"
            agent_config = self.agent_config['format_converter_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                customized_resume=json.dumps(ctx.resume_customization)
            )
            
            response = await self.claude_llm1_with_fallback.acomplete(prompt)
            ctx.latex_resume = response.text
            
            ctx.steps_completed += 1
            if hasattr(response, 'usage'):
                ctx.tokens_used += response.usage.total_tokens
            
            return Event(data={"context": ctx})
        except Exception as e:
            ctx.errors.append(f"Format conversion failed: {str(e)}")
            raise

    @step
    async def quality_check(self, event: Event) -> Event:
        """Quality control step"""
        try:
            ctx = event.data["context"]
            ctx.workflow_state = "quality_check"
            agent_config = self.agent_config['quality_controller_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                latex_resume=ctx.latex_resume,
                job_analysis=json.dumps(ctx.job_analysis)
            )
            
            response = await self.claude_llm2_with_fallback.acomplete(prompt)
            ctx.quality_check_result = json.loads(response.text)
            
            ctx.steps_completed += 1
            if hasattr(response, 'usage'):
                ctx.tokens_used += response.usage.total_tokens
            
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
                    "errors": ctx.errors
                }
            }
            
            ctx.workflow_state = "completed"
            return Event(data=final_output)
        except Exception as e:
            ctx.errors.append(f"Quality check failed: {str(e)}")
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
        
        # Create event with input data
        event = Event(data={
            "resume_text": resume_text,
            "job_description": job_description
        })
        
        # Run each step in sequence
        current_event = event
        for step in workflow.steps:
            logger.info(f"Executing step: {step.__name__}")
            try:
                current_event = await step(current_event)
            except Exception as e:
                logger.error(f"Step {step.__name__} failed: {str(e)}")
                raise
        
        final_result = current_event
        
        # Save the output
        logger.info("CV creation completed. Saving results...")
        await save_output(final_result.data, args.output)

        # Log completion metrics
        logger.info(f"Workflow completed successfully")
        if 'workflow_metrics' in final_result.data:
            metrics = final_result.data['workflow_metrics']
            logger.info(f"Total steps completed: {metrics['steps_completed']}")
            logger.info(f"Total tokens used: {metrics['tokens_used']}")
            
            if metrics['errors']:
                logger.warning("Errors occurred during processing:")
                for error in metrics['errors']:
                    logger.warning(f"- {error}")
        
        logger.info("Process completed successfully!")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise SystemExit(1)

if __name__ == "__main__":
    asyncio.run(main())