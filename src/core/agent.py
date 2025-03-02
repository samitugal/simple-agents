import json
import sys
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.core.tool import Tool
from src.tools.registry import ToolRegistry

class Agent:
    """Agent that invokes Claude with tools and handles the full conversation cycle"""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        output_format: Optional[str | BaseModel] = None,
        verbose: bool = False,
        tools: List[Tool] = None,
        team: Optional[List["Agent"]] = None,
        agent_name: Optional[str] = None,
    ):
        """
        Initialize the agent with optional tools

        Args:
            system_prompt: System prompt for the agent
            instructions: Instructions for the agent
            output_format: Expected output format
            verbose: Whether to print verbose output
            tools: List of Tool objects to register automatically
            team: List of Agent objects that this agent can delegate tasks to
            agent_name: Name of this agent (helps with team identification)
        """
        self.system_prompt = system_prompt
        self.instructions = instructions
        self.output_format = output_format
        self.verbose = verbose
        self.team = team or []
        self.agent_name = agent_name or str(uuid.uuid4())
        self.registry = ToolRegistry()

        # Register tools if provided
        if tools:
            for tool in tools:
                self.registry.register(tool)
                
        # Register team delegation tool if team is provided
        if self.team:
            self.register_team_tool()

    def register_team_tool(self) -> None:
        """Register a tool for team delegation"""
        
        # Create a tool for delegating to team members
        team_tool = TeamDelegationTool(self)
        self.registry.register(team_tool)
    
    def get_team_capabilities(self) -> str:
        """
        Get a description of each team member's capabilities
        
        Returns:
            String describing team member capabilities
        """
        if not self.team:
            return "No team members available."
            
        capabilities = []
        for idx, agent in enumerate(self.team):
            name = getattr(agent, 'agent_name', f"Agent {idx}")
            role = agent.system_prompt.split(".")[0] if agent.system_prompt else "No specific role"
            capabilities.append(f"{idx}: {name} - {role}")
            
        return "\n".join(capabilities)
        
    def delegate_to_team(self, task: str, agent_idx: int) -> Dict[str, Any]:
        """
        Delegate a task to a specific team member
        
        Args:
            task: The task to delegate
            agent_idx: Index of the team member in the team list
            
        Returns:
            The response from the team member
        """
        if not self.team:
            raise ValueError("No team members available")
            
        if agent_idx < 0 or agent_idx >= len(self.team):
            raise ValueError(f"Invalid team member index: {agent_idx}. Must be between 0 and {len(self.team)-1}")
        
        # Get the appropriate team member
        team_member = self.team[agent_idx]
        
        if self.verbose:
            name = getattr(team_member, 'agent_name', f"Agent {agent_idx}")
            print(f"\n--- Delegating task to {name} ---")
            print(f"Task: {task}")
        
        # Invoke the team member with the task
        response = team_member.invoke(task)
        
        if self.verbose:
            print(f"\n--- Response from team member {agent_idx} ---")
            # Handle response format differences
            if hasattr(response, 'content'):
                for block in response.content:
                    if hasattr(block, 'text'):
                        print(block.text)
            else:
                print(str(response))
        
        return response

    def invoke(self, prompt: str) -> Dict[str, Any]:
        """
        Invoke the agent with a prompt, handling the full conversation cycle

        Args:
            prompt: The prompt to invoke the agent with
        """
        pass

    def standalone_call(self, prompt: str) -> Dict[str, Any]:
        """
        Invoke the agent with a prompt, handling the full conversation cycle

        Args:
            prompt: The prompt to invoke the agent with
        """
        pass


class TeamDelegationTool(Tool):
    """Tool for delegating tasks to team members"""
    
    def __init__(self, parent_agent: Agent):
        self.parent_agent = parent_agent
    
    @property
    def name(self) -> str:
        return "delegate_to_team"
    
    @property
    def description(self) -> str:
        return "Delegate a task to a team member with specialized capabilities"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        team_capabilities = self.parent_agent.get_team_capabilities()
        
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to delegate to a team member",
                },
                "agent_idx": {
                    "type": "integer",
                    "description": f"The index of the team member to delegate to. Available team members:\n{team_capabilities}",
                },
            },
            "required": ["task", "agent_idx"],
        }
    
    def execute(self, task: str, agent_idx: int) -> Any:
        """Execute the delegation to a team member"""
        return self.parent_agent.delegate_to_team(task, agent_idx)