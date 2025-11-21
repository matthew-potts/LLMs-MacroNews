from src.lib.llm_client import LLMClient
from src.lib.openai_client import OpenAIClient
from src.lib.nebius_client import NebiusClient
from src.lib.anthropic_client import AnthropicClient

class LLMClientFactory:
    """Factory class for creating LLM client instances."""
    
    # Registry of model names to client classes
    _MODEL_REGISTRY = {
        "gpt-4o": OpenAIClient,
        "DeepSeek-R1-0528": NebiusClient,
        "claude-sonnet-4-5-20250929": AnthropicClient,
    }
    
    @classmethod
    def create(cls, model: str, instructions: str, temperature: float = 0.0) -> LLMClient:
        
        client_class = cls._MODEL_REGISTRY.get(model)
        if client_class is None:
            supported_models = ", ".join(cls._MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model: {model}. Supported models: {supported_models}")
        
        return client_class(model, instructions, temperature)
