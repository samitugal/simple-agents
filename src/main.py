import os
import sys
from typing import Any, Dict, Literal

from dotenv import load_dotenv

from src.agents.aws import AnthropicBedrockAgent
from src.decorators.tool_decorator import tool
from src.tools.bash.bash_tool import BashTool
from src.tools.computer_use.computer_tool import ComputerTool
from src.tools.registry import ToolRegistry
from src.tools.weather.weather_tool import WeatherTool

from pydantic import BaseModel

load_dotenv()

class OutputFormat(BaseModel):
    answer: str
    reasoning: str

def main():
    """Main entry point"""
    tools = [WeatherTool(), BashTool()]

    agent = AnthropicBedrockAgent(
        model_id=os.getenv("ANTHROPIC_MODEL"),
        verbose=True,
        tools=tools,
        system_prompt="You are a helpful assistant that can use tools to get information.",
        instructions="You can use the following tools to get information: {tools}",
        output_format=OutputFormat,
    )
    prompt = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What's the weather like in San Francisco? Also show me the files in the current directory."
    )

    response = agent.invoke(prompt)

    print("\n=== FINAL RESULT ===")
    for content_block in response.content:
        if content_block.type == "text":
            print(content_block.text)


if __name__ == "__main__":
    main()
