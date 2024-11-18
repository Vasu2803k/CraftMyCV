from typing import Any, Dict, Optional
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from crewai import LLM
import traceback

class FallbackLLM:
    """A wrapper class that implements fallback logic between two LLMs"""
    
    def __init__(
        self, 
        primary_llm: LLM, 
        fallback_llm: LLM, 
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm
        self.timeout = timeout
        self.max_retries = max_retries

    def _generate_with_retry(self, llm: LLM, prompt: str) -> str:
        """Attempt to generate response with retry logic"""
        @retry(
            stop=stop_after_attempt(self.max_retries), 
            wait=wait_exponential(multiplier=1, min=4, max=10)
        )
        def _generate():
            try:
                return llm.generate(prompt)
            except Exception as e:
                print(f"Stack trace:\n{traceback.format_exc()}")
                print(f"Error with LLM: {str(e)}")
                raise e
        
        return _generate()

    def generate(self, prompt: str) -> str:
        """Generate response with fallback logic"""
        start_time = time.time()
        
        # Try primary LLM first
        try:
            result = self._generate_with_retry(self.primary_llm, prompt)
            if time.time() - start_time <= self.timeout:
                return result
        except Exception as e:
            print(f"Stack trace:\n{traceback.format_exc()}")
            print(f"Primary LLM failed: {str(e)}")

        # If primary fails or times out, use fallback with same prompt
        print("Switching to fallback LLM...")
        try:
            return self._generate_with_retry(self.fallback_llm, prompt)
        except Exception as e:
            print(f"Stack trace:\n{traceback.format_exc()}")
            raise Exception(f"Both primary and fallback LLMs failed: {str(e)}")

    def __getattr__(self, name: str) -> Any:
        """Delegate any unknown attributes/methods to the primary LLM"""
        return getattr(self.primary_llm, name) 