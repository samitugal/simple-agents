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
        system_prompt: str,
        instructions: str,
        output_format: str | BaseModel,
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

