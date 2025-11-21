import os
from anthropic import Anthropic
from src.lib.llm_client import LLMClient

class AnthropicClient(LLMClient):    
    def __init__(self, model: str, instructions: str):
        super().__init__(model, instructions)
        self.client = Anthropic(api_key=os.getenv("CLAUDE_SECRET_KEY"))
    
    def get_response(self, input: str) -> str:
        response = self.client.messages.create(
            max_tokens=1024,
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": input
                }
            ]
        )
        return response.content[0].text
