import os
import re
from openai import OpenAI
from src.lib.llm_client import LLMClient

class NebiusClient(LLMClient):
    """Client for Nebius API, including open-source models"""
    
    def __init__(self, model: str, instructions: str, temperature: float = 0.0):
        super().__init__(model, instructions, temperature)
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.getenv("NEBIUS_SECRET_KEY"),
        )
    
    def get_response(self, input: str) -> str:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                input=input,
                temperature=self.temperature
            )
            return response.output_text
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages = [
        #         {
        #             "role": "user",
        #             "content": f"{self.instructions}\n\n{input}"
        #         }
        #     ],
        #     temperature=self.temperature
        # )
        
        # content = response.choices[0].message.content
        # numbers = re.findall(r'-?\d+\.?\d*', content)
        # if numbers:
        #     return numbers[-1]
        # return content

