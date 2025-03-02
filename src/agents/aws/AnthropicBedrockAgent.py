from typing import List

import anthropic

from src.core.agent import Agent
from src.core.tool import Tool


class AnthropicBedrockAgent(Agent):
    def __init__(
        self,
        model_id: str,
        aws_region: str = "us-east-1",
        verbose: bool = False,
        tools: List[Tool] = None,
    ):
        """
        Initialize Anthropic Bedrock agent with tools

        Args:
            model_id: The model ID to use
            aws_region: AWS region
            verbose: Whether to print verbose output
            tools: List of Tool objects to register automatically
        """
        super().__init__(verbose=verbose, tools=tools)
        self.model = model_id  # İnvokelamak için kullanılacak model ID'si
        self.model_id = model_id
        self.client = anthropic.AnthropicBedrock(aws_region=aws_region)
