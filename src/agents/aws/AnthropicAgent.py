import os

import anthropic

from src.core.agent import Agent


class AnthropicAgent(Agent):
    def __init__(self, model_id: str, api_key: str, verbose: bool = False):
        super().__init__(verbose)
        self.model_id = model_id

        if not api_key or os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("Anthropic API key is required")

        self.client = anthropic.Anthropic(
            api_key=api_key if api_key else os.getenv("ANTHROPIC_API_KEY")
        )
