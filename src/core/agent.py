import json
import sys
from typing import Any, Dict, List, Optional

import anthropic

from src.core.tool import Tool
from src.tools.registry import ToolRegistry


class Agent:
    """Agent that invokes Claude with tools and handles the full conversation cycle"""

    def __init__(self, verbose: bool = False, tools: List[Tool] = None):
        """
        Initialize the agent with optional tools

        Args:
            verbose: Whether to print verbose output
            tools: List of Tool objects to register automatically
        """
        self.verbose = verbose
        self.registry = ToolRegistry()

        # Register tools if provided
        if tools:
            for tool in tools:
                self.registry.register(tool)

    def invoke(
        self, prompt: str, max_iterations: int = 10, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Invoke the agent with a prompt, handling the full cycle of tool uses

        Args:
            prompt: The user prompt
            max_iterations: Maximum number of tool use iterations
            system_prompt: Optional system prompt

        Returns:
            Final response from the model
        """
        # Initial messages
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        custom_tools = self.registry.get_all_tools()

        seen_tool_names = set()
        all_tools = []

        for tool in custom_tools:
            tool_name = tool["name"]
            if tool_name not in seen_tool_names:
                seen_tool_names.add(tool_name)
                all_tools.append(tool)

        iterations = 0
        final_response = None

        while iterations < max_iterations:
            iterations += 1

            if self.verbose:
                print(f"\n--- Iteration {iterations} ---")
                print(
                    f"Sending messages to Claude: {json.dumps(messages[-1], indent=2)}"
                )

            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4096,
                tools=all_tools,
                messages=messages,
                betas=["computer-use-2025-01-24"],
                thinking={"type": "enabled", "budget_tokens": 1024},
            )

            if self.verbose and any(
                block.type == "thinking" for block in response.content
            ):
                print("\n--- Claude's Thinking ---")
                for content_block in response.content:
                    if content_block.type == "thinking":
                        print(content_block.thinking)

            if response.stop_reason == "tool_use":
                tool_use_found = False

                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_use_found = True
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_use_id = content_block.id

                        if self.verbose:
                            print(f"\n--- Tool Use Requested ---")
                            print(f"Tool: {tool_name}")
                            print(f"Input: {json.dumps(tool_input, indent=2)}")

                        try:
                            tool_result = self.registry.execute_tool(
                                tool_name, **tool_input
                            )
                        except Exception as e:
                            tool_result = {
                                "error": f"Error executing tool {tool_name}: {str(e)}"
                            }

                        if self.verbose:
                            print("\n--- Tool Result ---")
                            print(
                                json.dumps(tool_result, indent=2)
                                if isinstance(tool_result, dict)
                                else tool_result
                            )

                        messages.append(
                            {"role": "assistant", "content": response.content}
                        )

                        tool_result_content = (
                            json.dumps(tool_result)
                            if isinstance(tool_result, dict)
                            else tool_result
                        )

                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use_id,
                                        "content": tool_result_content,
                                    }
                                ],
                            }
                        )

                        break

                if not tool_use_found:
                    if self.verbose:
                        print("Expected tool use but none found in response")
                    break
            else:
                # Final response from model
                final_response = response

                if self.verbose:
                    print("\n--- Final Response ---")
                    for content_block in response.content:
                        if content_block.type == "text":
                            print(content_block.text)

                break

        if final_response is None:
            final_response = response
            if self.verbose:
                print("\n--- Reached maximum iterations ---")
                print("Using last response as final")

        return final_response
