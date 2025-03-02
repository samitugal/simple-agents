from typing import Any, Dict, List, Optional

import json
import anthropic

from src.core.agent import Agent
from src.core.tool import Tool
from src.agents.aws.AnthropicAgent import AnthropicAgent

from pydantic import BaseModel
class AnthropicBedrockAgent(AnthropicAgent):
    def __init__(
        self,
        agent_name: str,
        model_id: str,
        aws_region: str = "us-east-1",
        verbose: bool = False,
        tools: List[Tool] = None,
        team: List[Agent] = None,
        system_prompt: str = None,
        instructions: str = None,
        output_format: str | BaseModel = None,
        thinking: bool = False,
        max_iterations: int = 10,
        max_tokens: int = 4096,
        temperature: float = None,
        betas: List[str] = [],
    ):
        """
        Initialize Anthropic Bedrock agent with tools

        Args:
            model_id: The model ID to use
            aws_region: AWS region
            verbose: Whether to print verbose output
            tools: List of Tool objects to register automatically
            thinking: Whether to enable thinking
            max_iterations: Maximum number of iterations
            max_tokens: Maximum number of tokens
            temperature: Temperature
        """
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            instructions=instructions,
            output_format=output_format,
            team=team,
            verbose=verbose,
            tools=tools,
            thinking=thinking,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            temperature=temperature,
            betas=betas,
            model_id=model_id
        )
        self.client = anthropic.AnthropicBedrock(aws_region=aws_region)