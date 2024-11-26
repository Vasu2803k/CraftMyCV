from typing import Optional, Any, Dict
from crewai.llm import LLM
import time
import logging
from langchain.schema import BaseMessage

logger = logging.getLogger(__name__)

class FallbackLLM(LLM):
    """A wrapper class that implements fallback logic for LLMs"""
    
    def __init__(
        self,
        primary_llm: LLM,
        fallback_llm: LLM,
        timeout: int = 60,
        max_retries: int = 5
    ):
        # Initialize parent class
        super().__init__(
            model=primary_llm.model,
            temperature=primary_llm.temperature,
            max_tokens=primary_llm.max_tokens
        )
        
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, messages: list[BaseMessage], **kwargs: Any) -> str:
        """
        Override chat method to implement fallback logic
        """
        start_time = time.time()
        attempts = 0
        last_error = None

        while attempts < self.max_retries:
            try:
                # First try primary LLM
                if time.time() - start_time < self.timeout:
                    return self.primary_llm.chat(messages, **kwargs)
                
            except Exception as e:
                logger.warning(f"Primary LLM failed: {str(e)}")
                last_error = e
            
            try:
                # Fallback to secondary LLM
                logger.info("Attempting fallback LLM...")
                return self.fallback_llm.chat(messages, **kwargs)
                
            except Exception as e:
                logger.warning(f"Fallback LLM failed: {str(e)}")
                last_error = e
            
            attempts += 1
            time.sleep(1)  # Brief pause between retries
        
        # If we get here, both LLMs failed all retry attempts
        raise Exception(f"All LLM attempts failed. Last error: {str(last_error)}")

    def call(self, *args: Any, **kwargs: Any) -> str:
        """
        Implement call method for compatibility
        """
        start_time = time.time()
        attempts = 0
        last_error = None

        while attempts < self.max_retries:
            try:
                # First try primary LLM
                if time.time() - start_time < self.timeout:
                    return self.primary_llm.call(*args, **kwargs)
                
            except Exception as e:
                logger.warning(f"Primary LLM failed: {str(e)}")
                last_error = e
            
            try:
                # Fallback to secondary LLM
                logger.info("Attempting fallback LLM...")
                return self.fallback_llm.call(*args, **kwargs)
                
            except Exception as e:
                logger.warning(f"Fallback LLM failed: {str(e)}")
                last_error = e
            
            attempts += 1
            time.sleep(1)  # Brief pause between retries
        
        # If we get here, both LLMs failed all retry attempts
        raise Exception(f"All LLM attempts failed. Last error: {str(last_error)}")

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Implement generate_text method for compatibility
        """
        return self.call(prompt, **kwargs)

    def get_model_name(self) -> str:
        """Return the model name for compatibility"""
        return self.model

    def __getattr__(self, name: str) -> Any:
        """Delegate any unknown attributes/methods to the primary LLM"""
        return getattr(self.primary_llm, name)