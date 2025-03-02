import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, get_type_hints

from src.core.tool import Tool
from src.tools.registry import ToolRegistry


class FunctionTool(Tool):
    """Tool implementation for a decorated function"""

    def __init__(self, func: Callable, custom_description: Optional[str] = None):
        self.func = func
        self.func_name = func.__name__
        self._description = (
            custom_description or func.__doc__ or f"Execute {self.func_name} function"
        )
        self.type_hints = get_type_hints(func)
        self.signature = inspect.signature(func)

    @property
    def name(self) -> str:
        return self.func_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> Dict[str, Any]:
        properties = {}
        required = []

        for param_name, param in self.signature.parameters.items():
            # Skip self parameter for methods
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

        return {"type": "object", "properties": properties, "required": required}

    def execute(self, **kwargs) -> Any:
        """Execute the function with the given parameters"""
        return self.func(**kwargs)

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


def tool(func: Optional[Callable] = None, *, description: Optional[str] = None):
    """
    Decorator to register a function as a Claude tool.

    Args:
        func: The function to decorate
        description: Optional description to override function docstring

    Usage:
        @tool
        def my_function(...):
            ...

        @tool(description="Custom description")
        def my_function(...):
            ...
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        # Create and register the tool
        function_tool = FunctionTool(f, description)
        registry = ToolRegistry()
        registry.register(function_tool)

        return wrapper

    # Handle both @tool and @tool(description="...") usage
    if func is None:
        return decorator
    return decorator(func)
