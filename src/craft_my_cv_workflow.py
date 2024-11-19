import yaml
import traceback
import os
import json
import dotenv
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Context
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core.llms import LLM
from fallback_llm import FallbackLLM
import warnings
from pathlib import Path
import logging
import argparse
from tools.text_extractor_tool import TextExtractionTool
# Ignore warnings
warnings.filterwarnings("ignore")
# Load environment variables
dotenv.load_dotenv()

from tools.text_extractor_tool import TextExtractionTool
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
    """Save the crew's output to a file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Output saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save output: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise

# CraftMyCVWorkflow class
class CraftMyCVWorkflow(Workflow):
    def __init__(self):
        super().__init__()
        try:
            # Load configurations using correct paths relative to src directory
            with open('src/config/agents.yaml', 'r') as f:
                agent_config_yaml = yaml.safe_load(f)
            # Load agents configuration
            self.agent_config = agent_config_yaml['agents']
            # Load LLM configuration
            llm_config = agent_config_yaml['llm_config']
            # Load fallback LLM configuration
            fallback_config = agent_config_yaml['fallback_llm']
            
            # Validate environment variables
            openai_api_key = os.getenv('OPENAI_API_KEY')
            claude_api_key = os.getenv('CLAUDE_API_KEY')
            fallback_api_key = os.getenv('FALLBACK_LLM_API_KEY')
            
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            if not claude_api_key:
                raise ValueError("CLAUDE_API_KEY environment variable is not set")
            if not fallback_api_key:
                raise ValueError("FALLBACK_LLM_API_KEY environment variable is not set")
            
            # Create primary OpenAI LLMs
            self.openai_llm1 = OpenAI(
                model=llm_config['openai_llm']['model_1'],
                api_key=openai_api_key,
                temperature=llm_config['openai_llm']['temperature_1'],
                max_tokens=llm_config['openai_llm']['max_tokens']
            )
            
            self.openai_llm2 = OpenAI(
                model=llm_config['openai_llm']['model_2'],
                api_key=openai_api_key,
                temperature=llm_config['openai_llm']['temperature_2'],
                max_tokens=llm_config['openai_llm']['max_tokens']
            )
            
            # Create primary Claude LLMs
            self.claude_llm1 = Anthropic(
                model=llm_config['claude_llm']['model_1'],
                api_key=claude_api_key,
                temperature=llm_config['claude_llm']['temperature_1'],
                max_tokens=llm_config['claude_llm']['max_tokens']
            )
            
            self.claude_llm2 = Anthropic(
                model=llm_config['claude_llm']['model_2'],
                api_key=claude_api_key,
                temperature=llm_config['claude_llm']['temperature_2'],
                max_tokens=llm_config['claude_llm']['max_tokens']
            )
            
            # Create fallback LLM (using OpenAI as fallback)
            self.fallback_llm = OpenAI(
                model=fallback_config['model'],
                api_key=fallback_api_key,
                temperature=fallback_config['temperature'],
                max_tokens=fallback_config['max_tokens']
            )
            
            # Create FallbackLLM instances for each primary LLM
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

        except FileNotFoundError as e:
            print(f"Stack trace:\n{traceback.format_exc()}")
            raise ValueError(f"Configuration file not found: {str(e)}")
        except yaml.YAMLError as e:
            print(f"Stack trace:\n{traceback.format_exc()}")
            raise ValueError(f"Invalid YAML configuration: {str(e)}")
        except Exception as e:
            print(f"Stack trace:\n{traceback.format_exc()}")
            raise ValueError(f"Failed to load configuration: {str(e)}")

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
        # Add any input data to the prompt
        for key, value in inputs.items():
            prompt += f"\n{key}: {value}"
            
        return prompt

    @step()
    async def analyze_resume(self, ev: StartEvent, ctx: Context) -> StopEvent:
        """Resume analysis step using the resume analyzer agent"""
        try:
            resume_text = ev.data.get("resume_text")
            agent_config = self.agent_config['resume_analyzer_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                resume_text=resume_text
            )
            
            response = await self.openai_llm1_with_fallback.acomplete(prompt)
            result = json.loads(response.text)
            ctx.metrics["steps_completed"] += 1
            
            return StopEvent(result=result)
        except Exception as e:
            print(f"Resume analysis failed: {str(e)}")
            raise

    @step()
    async def analyze_job_description(self, ev: StartEvent, ctx: Context) -> StopEvent:
        """Job description analysis step"""
        try:
            job_description = ev.data.get("job_description")
            agent_config = self.agent_config['job_description_analyzer_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                job_description=job_description
            )
            
            response = await self.claude_llm1_with_fallback.acomplete(prompt)
            result = json.loads(response.text)
            ctx.metrics["steps_completed"] += 1
            
            return StopEvent(result=result)
        except Exception as e:
            print(f"Job description analysis failed: {str(e)}")
            raise

    @step()
    async def customize_skills(self, ev: StopEvent, ctx: Context) -> StopEvent:
        """Skill customization step"""
        try:
            # Get data from previous steps' results
            resume_analysis = ev.data.get("resume_analysis").result
            job_analysis = ev.data.get("job_analysis").result
            agent_config = self.agent_config['skill_customizer_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                resume_skills=json.dumps(resume_analysis.get('resume_categorization', {}).get('skills', {})),
                job_requirements=json.dumps(job_analysis.get('job_analysis', {}))
            )
            
            response = await self.openai_llm2_with_fallback.acomplete(prompt)
            result = json.loads(response.text)
            ctx.metrics["steps_completed"] += 1
            
            return StopEvent(result=result)
        except Exception as e:
            print(f"Skill customization failed: {str(e)}")
            raise

    @step()
    async def customize_resume(self, ev: StopEvent, ctx: Context) -> StopEvent:
        """Resume customization step"""
        try:
            # Get data from previous steps' results
            resume_analysis = ev.data.get("resume_analysis").result
            job_analysis = ev.data.get("job_analysis").result
            skill_customization = ev.data.get("skill_customization").result
            agent_config = self.agent_config['resume_customizer_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                resume_analysis=json.dumps(resume_analysis),
                job_analysis=json.dumps(job_analysis),
                skill_customization=json.dumps(skill_customization)
            )
            
            response = await self.claude_llm2_with_fallback.acomplete(prompt)
            result = json.loads(response.text)
            ctx.metrics["steps_completed"] += 1
            
            return StopEvent(result=result)
        except Exception as e:
            print(f"Resume customization failed: {str(e)}")
            raise

    @step()
    async def customize_summary(self, ev: StopEvent, ctx: Context) -> StopEvent:
        """Summary customization step"""
        try:
            # Get data from previous steps' results
            resume_customization = ev.data.get("resume_customization").result
            job_analysis = ev.data.get("job_analysis").result
            agent_config = self.agent_config['summary_customizer_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                resume_customization=json.dumps(resume_customization),
                job_analysis=json.dumps(job_analysis)
            )
            
            response = await self.openai_llm1_with_fallback.acomplete(prompt)
            result = json.loads(response.text)
            ctx.metrics["steps_completed"] += 1
            
            return StopEvent(result=result)
        except Exception as e:
            print(f"Summary customization failed: {str(e)}")
            raise

    @step()
    async def convert_format(self, ev: StopEvent, ctx: Context) -> StopEvent:
        """Format conversion step"""
        try:
            # Get data from previous steps' results
            customized_resume = ev.data.get("customized_resume").result
            agent_config = self.agent_config['format_converter_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                customized_resume=json.dumps(customized_resume)
            )
            
            response = await self.claude_llm1_with_fallback.acomplete(prompt)
            ctx.metrics["steps_completed"] += 1
            
            return StopEvent(result=response.text)  # Return LaTeX formatted string
        except Exception as e:
            print(f"Format conversion failed: {str(e)}")
            raise

    @step()
    async def quality_check(self, ev: StopEvent, ctx: Context) -> StopEvent:
        """Quality control step"""
        try:
            # Get data from previous steps' results
            latex_resume = ev.data.get("latex_resume").result
            job_analysis = ev.data.get("job_analysis").result
            agent_config = self.agent_config['quality_controller_agent']
            
            prompt = self._build_agent_prompt(
                agent_config,
                latex_resume=latex_resume,
                job_analysis=json.dumps(job_analysis)
            )
            
            response = await self.claude_llm2_with_fallback.acomplete(prompt)
            result = json.loads(response.text)
            ctx.metrics["steps_completed"] += 1
            
            return StopEvent(result=result)
        except Exception as e:
            print(f"Quality check failed: {str(e)}")
            raise

    @step()
    async def initialize_workflow(self, ev: StartEvent) -> Context:
        # Initialize metrics in Context
        return Context(
            metrics={
                "steps_completed": 0,
                "tokens_used": 0
            }
        )

async def main():
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
        resume_path = await validate_file_path(args.resume)
        
        # Handle job description (either file or direct text)
        job_description = ""
        if Path(args.job_description).exists():
            with open(args.job_description, 'r', encoding='utf-8') as f:
                job_description = f.read()
        else:
            job_description = args.job_description

        # Initialize text extractor and get resume text
        logger.info("Extracting text from resume...")
        text_extractor = TextExtractionTool()
        resume_text = await text_extractor._run(str(resume_path))

        # Initialize workflow and context
        logger.info("Initializing CV creation workflow...")
        workflow = CraftMyCVWorkflow()
    
        # Initialize workflow with input data
        start_event = StartEvent(data={
            "resume_text": resume_text,
            "job_description": job_description
        })
        ctx = await workflow.initialize_workflow(start_event)
        
        # Execute workflow steps
        final_result = await workflow.run(start_event, ctx)
        # Save the output
        logger.info("CV creation completed. Saving results...")
        await save_output(final_result.result, args.output)

        # Log completion metrics
        logger.info(f"Total steps completed: {ctx.metrics['steps_completed']}")
        logger.info(f"Total tokens used: {ctx.metrics['tokens_used']}")
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
    import asyncio
    asyncio.run(main())