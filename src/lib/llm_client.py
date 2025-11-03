import os
from openai import OpenAI
from anthropic import Anthropic

class LLMClient():
    def __init__(self, model: str, instructions: str):
        self.model = model
        self.instructions = instructions
        self.client = self.resolve_llm_client()
        

    def resolve_llm_client(self):
        if self.model=="gpt-4o":
            return OpenAI(api_key=os.getenv("OPEN_AI_SECRET_KEY"))
        elif self.model=="DeepSeek-R1-0528":
            return OpenAI(
                base_url="https://api.studio.nebius.com/v1/",
                api_key=os.getenv("NEBIUS_SECRET_KEY"),
            )
        elif self.model=="claude-sonnet-4-5-20250929":
            return Anthropic(
                api_key=os.getenv("CLAUDE_SECRET_KEY")
            )
        return None
    
    def get_response(self, input: str):
        if type(self.client) is OpenAI:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                input=input
            )
            return response.output_text
        elif type(self.client) is Anthropic:
            response = self.client.messages.create(
            max_tokens=1024,
            model="claude-sonnet-4-5-20250929",
            messages=[
                {
                    "role": "user",
                    "content": input
                }
            ]
        )
            return response.content[0].text
