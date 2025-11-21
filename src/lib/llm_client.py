from abc import ABC, abstractmethod

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model: str, instructions: str, temperature: float = 0.0):
        self.model = model
        self.instructions = instructions
        self.temperature = temperature
    
    @abstractmethod
    def get_response(self, input: str) -> str:
        """Get a response from the LLM."""
        pass
