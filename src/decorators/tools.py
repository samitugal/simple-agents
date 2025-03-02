from functools import wraps
from typing import Callable, Optional

from src.tools import FunctionDescriptor, registered_tools


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

        # Create descriptor
        descriptor = FunctionDescriptor(f)

        # Override description if provided
        if description:
            descriptor.description = description

        # Register the tool
        tool_dict = descriptor.as_dict()
        if tool_dict not in registered_tools:
            registered_tools.append(tool_dict)

        return wrapper

    # Handle both @tool and @tool(description="...") usage
    if func is None:
        return decorator
    return decorator(func)
