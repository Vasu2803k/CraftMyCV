from typing import Dict, List, Optional
import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import json
import asyncio
from pathlib import Path
from datetime import datetime
import logging

class CVCustomizationSystem:
    def __init__(self, config_path: str):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up logging
        self.logger = self._setup_logger()
        
        # Initialize LLM configurations
        self.llm_config = {
            "config_list": self.config["llm_config"],
            "cache_seed": 42  # For reproducibility
        }
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize group chats
        self.group_chats = self._initialize_group_chats()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger("CVCustomization")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize all agents with their specific roles"""
        agents = {
            "project_manager": AssistantAgent(
                name="project_manager",
                system_message="""You are the project manager overseeing the CV customization process.
                Your responsibilities:
                1. Coordinate between different agent groups
                2. Ensure quality and consistency
                3. Handle exceptions and recovery
                4. Track progress and report status
                5. Make high-level decisions about workflow""",
                llm_config=self.llm_config
            ),

            "resume_analyzer": AssistantAgent(
                name="resume_analyzer",
                system_message="""You are an expert in analyzing resumes.
                Your responsibilities:
                1. Extract key information from resumes
                2. Identify skills and experiences
                3. Categorize content
                4. Assess resume structure
                5. Provide detailed analysis reports""",
                llm_config=self.llm_config
            ),

            "job_analyzer": AssistantAgent(
                name="job_analyzer",
                system_message="""You are an expert in analyzing job descriptions.
                Your responsibilities:
                1. Extract key requirements
                2. Identify must-have vs nice-to-have skills
                3. Understand company culture and values
                4. Analyze role responsibilities
                5. Identify key success metrics""",
                llm_config=self.llm_config
            ),

            "skills_customizer": AssistantAgent(
                name="skills_customizer",
                system_message="""You are an expert in customizing skills for job applications.
                Your responsibilities:
                1. Map candidate skills to job requirements
                2. Identify skill gaps
                3. Suggest skill presentation strategies
                4. Optimize keyword matching
                5. Ensure authentic representation""",
                llm_config=self.llm_config
            ),

            "content_agent": AssistantAgent(
                name="content_agent",
                system_message="""You are an expert in resume content optimization.
                Your responsibilities:
                1. Customize content for specific roles
                2. Improve impact statements
                3. Ensure ATS compatibility
                4. Maintain professional tone
                5. Optimize content structure""",
                llm_config=self.llm_config
            ),

            "latex_formatter": AssistantAgent(
                name="latex_formatter",
                system_message="""You are an expert in LaTeX formatting.
                Your responsibilities:
                1. Create professional LaTeX templates
                2. Format content in LaTeX
                3. Ensure proper styling
                4. Handle special characters
                5. Optimize layout""",
                llm_config=self.llm_config
            ),

            "quality_controller": AssistantAgent(
                name="quality_controller",
                system_message="""You are an expert in quality control.
                Your responsibilities:
                1. Verify content accuracy
                2. Check formatting consistency
                3. Validate ATS compatibility
                4. Ensure requirement alignment
                5. Provide improvement suggestions""",
                llm_config=self.llm_config
            ),

            "user_proxy": UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                code_execution_config={"work_dir": "temp"}
            )
        }

        self.logger.info("Agents initialized successfully")
        return agents

    def _initialize_group_chats(self) -> Dict[str, GroupChat]:
        """Initialize specialized group chats for different phases"""
        group_chats = {
            "analysis": GroupChat(
                agents=[
                    self.agents["project_manager"],
                    self.agents["resume_analyzer"],
                    self.agents["job_analyzer"],
                    self.agents["user_proxy"]
                ],
                messages=[],
                max_round=5
            ),

            "customization": GroupChat(
                agents=[
                    self.agents["project_manager"],
                    self.agents["skills_customizer"],
                    self.agents["content_agent"],
                    self.agents["user_proxy"]
                ],
                messages=[],
                max_round=5
            ),

            "formatting": GroupChat(
                agents=[
                    self.agents["project_manager"],
                    self.agents["latex_formatter"],
                    self.agents["quality_controller"],
                    self.agents["user_proxy"]
                ],
                messages=[],
                max_round=5
            )
        }

        self.logger.info("Group chats initialized successfully")
        return group_chats

    async def process_resume(
        self,
        resume_text: str,
        job_description: str,
        output_path: Path
    ) -> Dict:
        """Process resume through the agent system"""
        try:
            self.logger.info("Starting resume processing")

            # Phase 1: Document Analysis
            analysis_manager = GroupChatManager(
                groupchat=self.group_chats["analysis"],
                llm_config=self.llm_config
            )

            analysis_result = await analysis_manager.run(
                f"""Analyze the following resume and job description:
                
                Resume:
                {resume_text}

                Job Description:
                {job_description}

                Provide detailed analysis of both documents."""
            )

            # Phase 2: Content Customization
            customization_manager = GroupChatManager(
                groupchat=self.group_chats["customization"],
                llm_config=self.llm_config
            )

            customization_result = await customization_manager.run(
                f"""Customize the resume content based on the following analysis:
                
                Analysis Results:
                {analysis_result}

                Provide optimized and customized content."""
            )

            # Phase 3: Formatting and Quality Control
            formatting_manager = GroupChatManager(
                groupchat=self.group_chats["formatting"],
                llm_config=self.llm_config
            )

            final_result = await formatting_manager.run(
                f"""Format the following content in LaTeX and perform quality check:
                
                Customized Content:
                {customization_result}

                Provide formatted content and quality assessment."""
            )

            # Prepare final output
            output = {
                "analysis": analysis_result,
                "customization": customization_result,
                "final_document": final_result,
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }

            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)

            self.logger.info(f"Processing completed. Output saved to {output_path}")
            return output

        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}")
            raise

async def main():
    # Initialize system
    system = CVCustomizationSystem("config.json")
    
    # Set up paths
    current_dir = Path(__file__).parent
    output_path = current_dir / "outputs" / f"result_{datetime.now():%Y%m%d_%H%M%S}.json"
    
    # Read input files
    with open(current_dir / "resume.txt", 'r') as f:
        resume_text = f.read()
    
    with open(current_dir / "job_description.txt", 'r') as f:
        job_description = f.read()
    
    # Process resume
    try:
        result = await system.process_resume(
            resume_text=resume_text,
            job_description=job_description,
            output_path=output_path
        )
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())