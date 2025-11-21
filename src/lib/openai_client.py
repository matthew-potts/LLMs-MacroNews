import os
from src.lib.llm_client import LLMClient
from openai import OpenAI

class OpenAIClient(LLMClient):
    """Client for OpenAI API (including gpt-4o)."""
    
    def __init__(self, model: str, instructions: str, temperature: float = 0.0):
        super().__init__(model, instructions, temperature)
        self.client = OpenAI(api_key=os.getenv("OPEN_AI_SECRET_KEY"))
    
    def get_response(self, input: str) -> str:
        if self.model == "gpt-4o":
            response = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                input=input,
                temperature=self.temperature
            )
        elif self.model == "gpt-5":
            response = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                input=input
            )

        return response.output_text