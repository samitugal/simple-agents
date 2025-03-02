from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Tool(ABC):
    """Abstract base class for all tools"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the tool"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of the tool"""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """Get the input schema for the tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters"""
        pass

    def as_dict(self) -> Dict[str, Any]:
        """Convert the tool to a dict for Claude API"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
