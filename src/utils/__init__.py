from .function_descriptor.function_descriptor import FunctionDescriptor

registered_tools = []


def get_registered_tools():
    return registered_tools


__all__ = ["FunctionDescriptor", "get_registered_tools"]
