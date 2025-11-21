import os
from openai import OpenAI
from src.lib.llm_client import LLMClient

class NebiusClient(LLMClient):
    """Client for Nebius API, including open-source models"""
    
    def __init__(self, model: str, instructions: str, temperature: float = 0.0):
        super().__init__(model, instructions)
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.getenv("NEBIUS_SECRET_KEY"),
        )
        self.temperature = temperature
    
    def get_response(self, input: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages = [
                {
                    "role": "user",
                    "content": f"{self.instructions}\n\n{input}"
                }
            ],
            temperature=self.temperature
        )
        return response.choices[0].message.content

