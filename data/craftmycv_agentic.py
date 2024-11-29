from typing import Dict, List, Optional, Any, Union
import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import json
import asyncio
from pathlib import Path
from datetime import datetime
import logging
from retry_mechanism import RetryMechanism
from tools.text_extractor_tool import TextExtractionTool
from setup_logging import setup_logging
import yaml
import argparse
import aiofiles
import aioconsole

class CVCustomizationSystem:
    """A system for customizing CVs using multiple specialized agents."""
    
    async def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration from YAML file asynchronously."""
        try:
            config_dir = Path(config_path).parent
            
            # Load YAML files asynchronously
            async with aiofiles.open(config_dir / 'agents.yaml', 'r') as f:
                agents_content = await f.read()
                agents_config = yaml.safe_load(agents_content)
            
            async with aiofiles.open(config_dir / 'templates.yaml', 'r') as f:
                templates_content = await f.read()
                templates_config = yaml.safe_load(templates_content)
                
            async with aiofiles.open(config_dir / 'projects.yaml', 'r') as f:
                projects_content = await f.read()
                projects_config = yaml.safe_load(projects_content)
            
            config = {
                "agent_config": agents_config["agents"],
                "llm_config": agents_config["llm_config"],
                "templates": templates_config["templates"],
                "projects": projects_config["projects"]
            }
            
            return config
            
        except FileNotFoundError as e:
            self.logger.error(f"Configuration file not found: {e.filename}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in configuration file: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise

    async def __init__(self, config_path: str):
        """Initialize the CV customization system asynchronously."""
        # Load configuration
        self.config = await self._load_config(config_path)
        
        # Set up logging with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = setup_logging(
            log_level="INFO",
            log_dir="data/logs",
            log_file_name=f"cv_customization_{timestamp}",
            logger_name="cv_customization"
        )
        
        # Initialize other components
        await self._initialize_components()

    async def _initialize_components(self):
        """Initialize system components asynchronously."""
        # Initialize LLM configurations
        self.llm_config = {
            "config_list": self.config["llm_config"],
            "cache_seed": 42,
            "temperature": 0.3,
            "retry_mechanism": RetryMechanism(
                max_retries=3,
                base_delay=1,
                max_delay=10,
                exponential_base=2
            )
        }
        
        # Initialize agents and group chats
        self.agents = await self._initialize_agents()
        self.group_chats = await self._initialize_group_chats()
        
        # Initialize metrics tracking
        self.metrics = {
            "steps_completed": 0,
            "total_steps": len(self._get_workflow_steps()),
            "start_time": None,
            "end_time": None,
            "step_timings": {},
            "errors": []
        }

    async def _get_user_input(self, prompt: str) -> str:
        """Get user input asynchronously."""
        return await aioconsole.ainput(prompt)

    async def _handle_user_interaction(self, result: Dict, phase: str) -> Dict:
        """Handle user interaction asynchronously."""
        print(f"\n=== {phase} Results ===")
        print(json.dumps(result, indent=2))
        
        while True:
            choice = await self._get_user_input(
                f"\nDo you want to proceed with these {phase.lower()} results? (yes/no/modify): "
            )
            choice = choice.lower()
            
            if choice == 'yes':
                break
            elif choice == 'no':
                raise ValueError(f"User cancelled the process during {phase}")
            elif choice == 'modify':
                modifications = await self._get_user_input("\nPlease enter your modifications (JSON format):\n")
                try:
                    modifications = json.loads(modifications)
                    result.update(modifications)
                    break
                except json.JSONDecodeError:
                    print("Invalid JSON format. Please try again.")
            
            return result
            
    async def _save_output(self, output: Dict, output_path: Path):
        """Save output asynchronously."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(json.dumps(output, indent=2))

    async def process_resume(
        self,
        resume_text: str,
        job_description: str,
        output_path: Path,
        interactive: bool = True
    ) -> Dict:
        """Process resume through the agent system asynchronously."""
        try:
            self.logger.info("Starting resume processing")
            self.metrics["start_time"] = datetime.now()

            # Phase 1: Document Analysis
            analysis_result = await self._run_group_chat(
                "analysis",
                f"""Analyze the following resume and job description:
                
                Resume:
                {resume_text}

                Job Description:
                {job_description}

                Provide a detailed analysis of both documents.""",
                "analysis_phase"
            )

            if interactive:
                analysis_result = await self._handle_user_interaction(
                    analysis_result, 
                    "Analysis"
                )

            # Phase 2: Content Customization
            customization_result = await self._run_group_chat(
                "customization",
                f"""Customize the resume content based on the following analysis:
                
                Analysis Results:
                {analysis_result}""",
                "customization_phase"
            )

            if interactive:
                customization_result = await self._handle_user_interaction(
                    customization_result, 
                    "Customization"
                )

            # Phase 3: Formatting and Quality Control
            final_result = await self._run_group_chat(
                "formatting",
                f"""Format the following content in LaTeX and perform quality check:
                
                Customized Content:
                {customization_result}""",
                "formatting_phase"
            )

            # Prepare and save output
            output = {
                "analysis": analysis_result,
                "customization": customization_result,
                "final_document": final_result,
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "metrics": self.metrics
                }
            }

            await self._save_output(output, output_path)

            self.metrics["end_time"] = datetime.now()
            self.logger.info(f"Processing completed. Output saved to {output_path}")
            
            return output

        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}")
            self.metrics["errors"].append({
                "step": "process_resume",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise

    async def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize all agents with their specific roles asynchronously."""
        try:
            self.logger.info("Initializing agents...")
            
            # Create agents with fallback mechanism
            agents = {}
            agent_configs = self.config["agent_config"]
            
            # Initialize primary and fallback LLMs
            primary_llm = self.llm_config["config_list"][0]
            fallback_llm = self.llm_config["config_list"][1]
            
            # Create FallbackLLM for each agent
            fallback_wrapper = FallbackLLM(
                primary_llm=primary_llm,
                fallback_llm=fallback_llm,
                timeout=60,
                max_retries=3
            )
            
            # Common LLM config for all agents
            base_llm_config = {
                "llm": fallback_wrapper,
                "cache_seed": self.llm_config["cache_seed"],
                "temperature": self.llm_config["temperature"]
            }
            
            # Initialize each agent with error handling
            agent_roles = [
                "project_manager",
                "resume_analyzer",
                "job_analyzer",
                "skills_customizer",
                "content_agent",
                "summary_customizer",
                "latex_formatter",
                "quality_controller"
            ]
            
            for role in agent_roles:
                try:
                    if role not in agent_configs:
                        raise ValueError(f"Missing configuration for agent: {role}")
                        
                    agent_config = agent_configs[role]
                    
                    # Validate required config fields
                    required_fields = ["system_prompt", "expected_output"]
                    missing_fields = [f for f in required_fields if f not in agent_config]
                    if missing_fields:
                        raise ValueError(f"Missing required fields for {role}: {missing_fields}")
                    
                    agents[role] = AssistantAgent(
                        name=role,
                        system_message=agent_config["system_prompt"],
                        llm_config=base_llm_config.copy()
                    )
                    
                    self.logger.info(f"Successfully initialized agent: {role}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize agent {role}: {str(e)}")
                    raise
            
            # Initialize user proxy agent
            try:
                agents["user_proxy"] = UserProxyAgent(
                    name="user_proxy",
                    human_input_mode="NEVER",
                    code_execution_config={
                        "work_dir": "temp",
                        "use_docker": False,
                        "timeout": 60
                    },
                    llm_config=base_llm_config.copy()
                )
                
                self.logger.info("Successfully initialized user proxy agent")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize user proxy agent: {str(e)}")
                raise
            
            return agents
            
        except Exception as e:
            self.logger.error(f"Error in agent initialization: {str(e)}")
            raise

    async def _initialize_group_chats(self) -> Dict[str, GroupChat]:
        """Initialize specialized group chats for different phases asynchronously."""
        try:
            self.logger.info("Initializing group chats...")
            
            # Define group chat configurations
            chat_configs = {
                "analysis": {
                    "members": [
                        "project_manager",
                        "resume_analyzer",
                        "job_analyzer",
                        "user_proxy"
                    ],
                    "max_round": 5,
                    "description": "Resume and job analysis phase"
                },
                "customization": {
                    "members": [
                        "project_manager",
                        "skills_customizer",
                        "content_agent",
                        "summary_customizer",
                        "user_proxy"
                    ],
                    "max_round": 5,
                    "description": "Content customization phase"
                },
                "formatting": {
                    "members": [
                        "project_manager",
                        "latex_formatter",
                        "quality_controller",
                        "user_proxy"
                    ],
                    "max_round": 5,
                    "description": "LaTeX formatting and quality control phase"
                }
            }
            
            group_chats = {}
            
            # Initialize each group chat with error handling
            for chat_name, config in chat_configs.items():
                try:
                    # Validate member existence
                    missing_agents = [
                        member for member in config["members"] 
                        if member not in self.agents
                    ]
                    
                    if missing_agents:
                        raise ValueError(
                            f"Missing required agents for {chat_name}: {missing_agents}"
                        )
                    
                    # Create group chat
                    group_chats[chat_name] = GroupChat(
                        agents=[self.agents[member] for member in config["members"]],
                        messages=[],
                        max_round=config["max_round"],
                        description=config["description"]
                    )
                    
                    self.logger.info(f"Successfully initialized group chat: {chat_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize group chat {chat_name}: {str(e)}")
                    raise
            
            return group_chats
            
        except Exception as e:
            self.logger.error(f"Error in group chat initialization: {str(e)}")
            raise

    async def _run_group_chat(
        self,
        chat_name: str,
        prompt: str,
        message_id: str
    ) -> Dict:
        """Run a group chat with robust error handling and retry mechanism."""
        try:
            if chat_name not in self.group_chats:
                raise ValueError(f"Invalid group chat name: {chat_name}")
            
            self.logger.info(f"Starting group chat: {chat_name}")
            start_time = datetime.now()
            
            # Create chat manager with retry mechanism
            chat_manager = GroupChatManager(
                groupchat=self.group_chats[chat_name],
                llm_config=self.llm_config
            )
            
            # Set up retry mechanism
            retry_count = 0
            max_retries = 3
            base_delay = 1
            
            while retry_count < max_retries:
                try:
                    result = await chat_manager.run(
                        prompt,
                        message_id=message_id
                    )
                    
                    # Validate result
                    if not result:
                        raise ValueError("Empty result from group chat")
                    
                    duration = (datetime.now() - start_time).total_seconds()
                    self.metrics["step_timings"][message_id] = duration
                    
                    self.logger.info(
                        f"Group chat {chat_name} completed successfully in {duration:.2f}s"
                    )
                    return result
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
                        self.logger.warning(
                            f"Attempt {retry_count} failed for {chat_name}: {str(e)}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
            
        except Exception as e:
            self.logger.error(f"Error in group chat {chat_name}: {str(e)}")
            self.metrics["errors"].append({
                "step": chat_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "message_id": message_id
            })
            raise

    def _get_workflow_steps(self) -> List[str]:
        """Get list of workflow steps in order."""
        return [
            "initialization",
            "document_analysis",
            "content_customization",
            "summary_creation",
            "latex_formatting",
            "quality_control",
            "final_output"
        ]

async def main():
    """Main entry point for the CV customization system."""
    parser = argparse.ArgumentParser(description='Create a customized CV using AI agents')
    parser.add_argument(
        '--resume',
        type=str,
        required=True,
        help='Path to the input resume file'
    )
    parser.add_argument(
        '--job-description',
        type=str,
        required=True,
        help='Job description text or path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.json',
        help='Path to save output'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='src/config',
        help='Directory containing YAML config files'
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Run in non-interactive mode'
    )

    args = parser.parse_args()

    try:
        # Initialize text processor
        text_processor = TextExtractionTool()
        
        # Process files asynchronously
        async def process_file(file_path: Path) -> str:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            return await asyncio.to_thread(text_processor.run, file_path)

        # Process resume and job description concurrently
        resume_path = Path(args.resume)
        job_path = Path(args.job_description)
        
        resume_text, job_description = await asyncio.gather(
            process_file(resume_path),
            process_file(job_path) if job_path.exists() else asyncio.sleep(0, args.job_description)
        )

        # Initialize system
        system = await CVCustomizationSystem(args.config_dir)
        output_path = Path(args.output)
        
        # Process resume
        result = await system.process_resume(
            resume_text=resume_text,
            job_description=job_description,
            output_path=output_path,
            interactive=not args.non_interactive
        )
        
        print("Processing completed successfully!")
        return result

    except Exception as e:
        print(f"Error: {str(e)}")
        raise SystemExit(1)

if __name__ == "__main__":
    asyncio.run(main())