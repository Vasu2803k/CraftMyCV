from crewai import Crew, Agent, Task, Process, LLM
from pathlib import Path
import warnings
import os
import yaml
import json
import dotenv
from typing import List, Dict, Any, Union, Optional
from tools.text_extractor_tool import TextExtractionTool
from fallback_llm import FallbackLLM
import traceback

warnings.filterwarnings('ignore')

# Load environment variables
dotenv.load_dotenv()

class CraftMyCV:
    """Craft my CV with a team of specialized AI agents"""

    def __init__(self):
        self.text_extractor = TextExtractionTool()
        
        try:
            # Load configurations using correct paths relative to src directory
            with open('src/config/agents.yaml', 'r') as f:
                agent_config_yaml = yaml.safe_load(f)
                self.agent_config = agent_config_yaml['agents']
            with open('src/config/tasks.yaml', 'r') as f:
                task_config_yaml = yaml.safe_load(f)
                self.task_config = task_config_yaml['tasks']
                
            # Initialize LLMs
            llm_config = agent_config_yaml['llm_config']
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
            self.openai_llm1 = LLM(
                model=llm_config['openai_llm_1']['model'], 
                api_key=openai_api_key,
                temperature=llm_config['openai_llm_1']['temperature'],
                top_p=llm_config['openai_llm_1']['top_p'],
                max_tokens=llm_config['openai_llm_1']['max_tokens']
            )
            
            self.openai_llm2 = LLM(
                model=llm_config['openai_llm_2']['model'],
                api_key=openai_api_key,
                temperature=llm_config['openai_llm_2']['temperature'],
                top_p=llm_config['openai_llm_2']['top_p'],
                max_tokens=llm_config['openai_llm_2']['max_tokens']
            )
            
            # Create primary Claude LLMs
            self.claude_llm1 = LLM(
                model=llm_config['claude_llm_1']['model'],
                api_key=claude_api_key,
                temperature=llm_config['claude_llm_1']['temperature'],
                top_p=llm_config['claude_llm_1']['top_p'],
                max_tokens=llm_config['claude_llm_1']['max_tokens']
            )
            
            self.claude_llm2 = LLM(
                model=llm_config['claude_llm_2']['model'],
                api_key=claude_api_key,
                temperature=llm_config['claude_llm_2']['temperature'],
                top_p=llm_config['claude_llm_2']['top_p'],
                max_tokens=llm_config['claude_llm_2']['max_tokens']
            )
            
            # Create fallback LLM
            fallback_llm = LLM(
                model=fallback_config['model'],
                api_key=fallback_api_key,
                temperature=fallback_config['temperature'],
                top_p=fallback_config['top_p'],
                max_tokens=fallback_config['max_tokens']
            )
            
            # Create FallbackLLM instances for each primary LLM
            self.openai_llm1_with_fallback = FallbackLLM(
                primary_llm=self.openai_llm1,
                fallback_llm=fallback_llm,
                timeout=30
            )
            
            self.openai_llm2_with_fallback = FallbackLLM(
                primary_llm=self.openai_llm2,
                fallback_llm=fallback_llm,
                timeout=30
            )
            
            self.claude_llm1_with_fallback = FallbackLLM(
                primary_llm=self.claude_llm1,
                fallback_llm=fallback_llm,
                timeout=30
            )
            
            self.claude_llm2_with_fallback = FallbackLLM(
                primary_llm=self.claude_llm2,
                fallback_llm=fallback_llm,
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

    def resume_analyzer(self) -> Agent:
        """Create the Resume Analyzer agent with fallback LLM"""
        config = self.agent_config['resume_analyzer_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            llm=self.openai_llm1_with_fallback,
            system_template=config['system_prompt'],
            verbose=True,
            tools=[self.text_extractor]
        )

    def job_description_analyzer(self) -> Agent:
        """Create the Job Description Analyzer agent with fallback LLM"""
        config = self.agent_config['job_description_analyzer_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            llm=self.openai_llm2_with_fallback,
            system_template=config['system_prompt'],
            verbose=True
        )

    def skill_customizer(self) -> Agent:
        """Create the Skill Customizer agent"""
        config = self.agent_config['skill_customizer_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            llm=self.claude_llm1_with_fallback,
            system_template=config['system_prompt'],
            verbose=True
        )

    def resume_customizer(self) -> Agent:
        """Create the Resume Customizer agent"""
        config = self.agent_config['resume_customizer_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            llm=self.claude_llm2_with_fallback,
            system_template=config['system_prompt'],
            verbose=True
        )

    def information_synthesizer(self) -> Agent:
        """Create the Information Synthesizer agent"""
        config = self.agent_config['information_synthesizer_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            llm=self.claude_llm2_with_fallback,
            system_template=config['system_prompt'],
            verbose=True
        )

    def format_converter(self) -> Agent:
        """Create the Format Converter agent"""
        config = self.agent_config['format_converter_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            llm=self.claude_llm2_with_fallback,
            system_template=config['system_prompt'],
            verbose=True
        )

    def quality_controller(self) -> Agent:
        """Create the Quality Controller agent"""
        config = self.agent_config['quality_controller_agent']
        return Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            llm=self.claude_llm2_with_fallback,
            system_template=config['system_prompt'],
            verbose=True
        )

    def analyze_resume(self, resume_text: str) -> Task:
        """Create a resume analysis task"""
        task_cfg = self.task_config['resume_analysis_task']
        return Task(
            description=task_cfg['description'],
            agent=self.resume_analyzer(),
            expected_output=task_cfg['expected_output'],
            context={'input': resume_text}
        )

    def analyze_job_description(self, job_description: str) -> Task:
        """Create a job description analysis task"""
        task_cfg = self.task_config['job_description_analysis_task']
        return Task(
            description=task_cfg['description'],
            agent=self.job_description_analyzer(),
            expected_output=task_cfg['expected_output'],
            context={'input': job_description}
        )
    
    def customize_resume(self, resume_data: str, job_analysis: str) -> Task:
        """Create a resume customization task"""
        task_cfg = self.task_config['resume_customization_task']
        return Task(
            description=task_cfg['description'],
            agent=self.resume_customizer(),
            expected_output=task_cfg['expected_output'],
            context={
                'resume_data': resume_data,
                'job_analysis': job_analysis
            }
        )
    
    def customize_skills(self, resume_data: str, job_analysis: str) -> Task:
        """Create a skill customization task"""
        task_cfg = self.task_config['skill_customization_task']
        return Task(
            description=task_cfg['description'],
            agent=self.skill_customizer(),
            expected_output=task_cfg['expected_output'],
            context={
                'resume_skills': resume_data,
                'job_requirements': job_analysis
            }
        )

    def synthesize_information(self, resume_data: str, job_analysis: str, optimized_skills: str) -> Task:
        """Create an information synthesis task"""
        task_cfg = self.task_config['information_synthesis_task']
        return Task(
            description=task_cfg['description'],
            agent=self.information_synthesizer(),
            expected_output=task_cfg['expected_output'],
            context={
                'resume_data': resume_data,
                'job_analysis': job_analysis,
                'optimized_skills': optimized_skills
            }
        )

    def convert_format(self, synthesized_data: str) -> Task:
        """Create a format conversion task"""
        task_cfg = self.task_config['format_conversion_task']
        return Task(
            description=task_cfg['description'],
            agent=self.format_converter(),
            expected_output=task_cfg['expected_output'],
            context={'json_data': synthesized_data}
        )

    def quality_check(self, latex_resume: str, job_requirements: str) -> Task:
        """Create a quality control task"""
        task_cfg = self.task_config['quality_control_task']
        return Task(
            description=task_cfg['description'],
            agent=self.quality_controller(),
            expected_output=task_cfg['expected_output'],
            context={
                'latex_resume': latex_resume,
                'job_requirements': job_requirements
            }
        )

    def create_crew(self) -> Crew:
        """Create and configure the CV creation crew"""
        return Crew(
            agents=[
                self.resume_analyzer(),
                self.job_description_analyzer(),
                self.skill_customizer(),
                self.information_synthesizer(),
                self.resume_customizer(),
                self.format_converter(),
                self.quality_controller()
            ],
            tasks=[
                self.analyze_resume("{{resume_text}}"),
                self.analyze_job_description("{{job_description}}"),
                self.customize_skills("{{analyze_resume.output}}", "{{analyze_job_description.output}}"),
                self.synthesize_information(
                    "{{analyze_resume.output}}", 
                    "{{analyze_job_description.output}}", 
                    "{{customize_skills.output}}"
                ),
                self.customize_resume(
                    "{{analyze_resume.output}}", 
                    "{{analyze_job_description.output}}"
                ),
                self.convert_format("{{synthesize_information.output}}"),
                self.quality_check("{{convert_format.output}}", "{{analyze_job_description.output}}")
            ],
            verbose=True,
            process=Process.sequential
        )

    def create_cv(self, resume_path: Optional[str] = None, job_description: str = "") -> Dict[str, Any]:
        """
        Main method to create a customized CV
        """
        try:
            # Initialize crew
            crew = self.create_crew()
            
            # Extract text from resume if provided
            resume_text = ""
            if resume_path:
                if not Path(resume_path).exists():
                    raise FileNotFoundError(f"Resume file not found: {resume_path}")
                resume_text = self.text_extractor._run(resume_path)
            
            # Execute the CV creation process
            result = crew.kickoff({
                'resume_text': resume_text,
                'job_description': job_description
            })
            
            return result
            
        except FileNotFoundError as e:
            print(f"Stack trace:\n{traceback.format_exc()}")
            raise
        except Exception as e:
            print(f"Stack trace:\n{traceback.format_exc()}")
            raise ValueError(f"Error in CV creation process: {str(e)}")

if __name__ == "__main__":
    cv_creator = CraftMyCV()
    try:
        # Process with an existing file
        final_cv = cv_creator.create_cv(
            resume_path="path/to/existing/resume.pdf", 
            job_description="Software Engineer"
        )
        print("CV created successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")