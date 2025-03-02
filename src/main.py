import os
import sys

from dotenv import load_dotenv

from src.agents.aws import AnthropicBedrockAgent
from src.tools.bash.bash_tool import BashTool
from src.tools.computer_use.computer_tool import ComputerTool
from src.tools.registry import ToolRegistry
from src.tools.weather.weather_tool import WeatherTool
from src.tools.web import TavilySearchTool

from pydantic import BaseModel

load_dotenv()

class OutputFormat(BaseModel):
    answer: str
    reasoning: str

def main():
    """Main entry point"""
    basic_tools = [BashTool()]
    weather_tools = [WeatherTool()]
    web_tools = [TavilySearchTool()]
    
    research_agent = AnthropicBedrockAgent(
        model_id=os.getenv("ANTHROPIC_MODEL"),
        verbose=True,
        tools=web_tools,
        system_prompt="You are a research expert. You can use the web tool to get the information for a given topic.",
        instructions="Use the web tool to get the information for the given topic.",
        agent_name="Research Expert",
    )
    
    weather_agent = AnthropicBedrockAgent(
        model_id=os.getenv("ANTHROPIC_MODEL"),
        verbose=True,
        tools=weather_tools,
        system_prompt="You are a weather expert. You can use the weather tool to get the weather information for a given location.",
        instructions="Use the weather tool to get the weather information for the given location.",
        agent_name="Weather Expert",
    )
    
    manager_agent = AnthropicBedrockAgent(
        model_id=os.getenv("ANTHROPIC_MODEL"),
        verbose=True,
        tools=basic_tools,
        system_prompt="""You are a manager. You can delegate tasks to the team members.
        Research Expert is good at research, Weather Expert is good at weather.""",
        instructions="Delegate the task to the appropriate team member based on the user's question.",
        agent_name="Manager Agent",
        team=[research_agent, weather_agent],
    )
    
    # Kullanıcı sorusu
    prompt = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the weather in San Francisco today?"
    )
    
    print(f"\n=== MANAGER AGENT PROCESSING: '{prompt}' ===")
    response = manager_agent.invoke(prompt)

    print("\n=== FINAL RESULT ===")
    for content_block in response.content:
        if content_block.type == "text":
            print(content_block.text)


if __name__ == "__main__":
    main()
