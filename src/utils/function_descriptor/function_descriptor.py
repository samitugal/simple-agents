import inspect
from typing import Any, Callable, Dict, List, Optional, get_type_hints


class FunctionDescriptor:
    """Class to convert a Python function to a Claude API tool description"""

    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        self.description = func.__doc__ or f"Execute {self.name} function"
        self.type_hints = get_type_hints(func)
        self.signature = inspect.signature(func)

    def as_dict(self) -> Dict[str, Any]:
        """Convert the function to a Claude API tool descriptor"""
        properties = {}
        required = []

        # Process parameters
        for param_name, param in self.signature.parameters.items():
            # Skip self for methods
            if param_name == "self":
                continue

            param_type = self.type_hints.get(param_name, Any)
            param_desc = self._extract_param_description(param_name)

            # Create parameter definition
            properties[param_name] = {
                "type": self._get_json_type(param_type),
                "description": param_desc,
            }

            # Handle enum types (Literal)
            if hasattr(param_type, "__args__") and all(
                isinstance(arg, str) for arg in param_type.__args__
            ):
                properties[param_name]["enum"] = list(param_type.__args__)

            # Check if parameter is required
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _get_json_type(self, py_type: Any) -> str:
        """Convert Python type to JSON schema type"""
        if py_type in (str, type(None)):
            return "string"
        elif py_type in (int, float):
            return "number"
        elif py_type == bool:
            return "boolean"
        elif py_type == list or getattr(py_type, "__origin__", None) == list:
            return "array"
        elif py_type == dict or getattr(py_type, "__origin__", None) == dict:
            return "object"
        else:
            return "string"  # Default to string for complex types

    def _extract_param_description(self, param_name: str) -> str:
        """Extract parameter description from docstring"""
        if not self.func.__doc__:
            return ""

        # Simple docstring parser
        docstring = self.func.__doc__
        lines = docstring.split("\n")
        param_section = False
        current_param = None

        for line in lines:
            line = line.strip()

            # Find parameters section
            if line.lower().startswith("args:") or line.lower().startswith(
                "parameters:"
            ):
                param_section = True
                continue

            # End of parameters section
            if (
                param_section
                and line
                and line.endswith(":")
                and not line.startswith(" ")
            ):
                param_section = False

            # Find parameter
            if param_section and ":" in line:
                parts = line.split(":", 1)
                if parts[0].strip() == param_name:
                    return parts[1].strip()

        return ""
