import json
import sys
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.core.tool import Tool
from src.tools.registry import ToolRegistry


class Agent:
    """Agent that invokes Claude with tools and handles the full conversation cycle"""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        output_format: Optional[str | BaseModel] = None,
        verbose: bool = False,
        tools: List[Tool] = None,
    ):
        """
        Initialize the agent with optional tools

        Args:
            verbose: Whether to print verbose output
            tools: List of Tool objects to register automatically
        """
        self.system_prompt = system_prompt
        self.instructions = instructions
        self.output_format = output_format
        self.verbose = verbose
        self.registry = ToolRegistry()

        # Register tools if provided
        if tools:
            for tool in tools:
                self.registry.register(tool)

    def invoke(self, prompt: str) -> Dict[str, Any]:
        """
        Invoke the agent with a prompt, handling the full conversation cycle

        Args:
            prompt: The prompt to invoke the agent with
        """
        pass

    def standalone_call(self, prompt: str) -> Dict[str, Any]:
        """
        Invoke the agent with a prompt, handling the full conversation cycle

        Args:
            prompt: The prompt to invoke the agent with
        """
        pass
