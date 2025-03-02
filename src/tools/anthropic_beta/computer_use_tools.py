from typing import Any, Dict, List

from src.core.tool import Tool


class ComputerUseTool(Tool):
    def __init__(self):
        super().__init__()
        self.name = "computer_use"
        self.description = "Use the computer to perform a task"
