from typing import Any, Dict, Literal

from src.core.tool import Tool


class WeatherTool(Tool):
    """Tool for getting weather information"""

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "Get the current weather in a given location"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                },
            },
            "required": ["location"],
        }

    def execute(self, location: str, unit: str = "celsius") -> Dict[str, Any]:
        """Get weather for the specified location"""
        # In a real implementation, this would call a weather API
        return {
            "weather": "sunny",
            "temperature": 22 if unit == "celsius" else 72,
            "location": location,
            "unit": unit,
        }
