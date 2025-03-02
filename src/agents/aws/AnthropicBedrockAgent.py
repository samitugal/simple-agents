from typing import Any, Dict, List

import json
import anthropic

from src.core.agent import Agent
from src.core.tool import Tool

from pydantic import BaseModel
class AnthropicBedrockAgent(Agent):
    def __init__(
        self,
        model_id: str,
        aws_region: str = "us-east-1",
        verbose: bool = False,
        tools: List[Tool] = None,
        system_prompt: str = None,
        instructions: str = None,
        output_format: str | BaseModel = None,
        thinking: bool = False,
        max_iterations: int = 10,
        max_tokens: int = 4096,
        temperature: float = None,
        betas: List[str] = [],
    ):
        """
        Initialize Anthropic Bedrock agent with tools

        Args:
            model_id: The model ID to use
            aws_region: AWS region
            verbose: Whether to print verbose output
            tools: List of Tool objects to register automatically
            thinking: Whether to enable thinking
            max_iterations: Maximum number of iterations
            max_tokens: Maximum number of tokens
            temperature: Temperature
        """
        super().__init__(
            system_prompt=system_prompt,
            instructions=instructions,
            output_format=output_format,
            verbose=verbose,
            tools=tools,
        )
        self.client = anthropic.AnthropicBedrock(aws_region=aws_region)
        self.model = model_id
        self.model_id = model_id 
        self.thinking = thinking
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.betas = betas

        if thinking:
            if "claude-3-7" in model_id or "claude-3-5-sonnet" in model_id:
                self.thinking = {"type": "enabled", "budget_tokens": 1024}
            else:
                raise ValueError(
                    f"Thinking is not supported for this model: {model_id}"
                )
            
            if self.temperature is None:
                self.temperature = 1 

            print(f"Temperature: {self.temperature}")

            if self.temperature != 1:
                raise ValueError("Temperature may only be set to 1 when thinking is enabled.")

        if self.temperature is None:
            self.temperature = 0.5

    def invoke(self, prompt: str, max_iterations: int = 10) -> Dict[str, Any]:
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

        if self.system_prompt:
            messages.append({"role": "user", "content": f"<system>{self.system_prompt}</system>"})

        if self.instructions:
            messages.append({"role": "user", "content": f"<instructions>{self.instructions}</instructions>"})

        if self.output_format:
            if isinstance(self.output_format, BaseModel):
                messages.append(
                    {
                        "role": "user",
                        "content": f"<output_format>{self.output_format.model_dump_json()}</output_format>",
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": f"<output_format>{self.output_format}</output_format>",
                    }
                )

        messages.append({"role": "user", "content": f"{prompt}"})

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

        while iterations < self.max_iterations:
            iterations += 1

            if self.verbose:
                print(f"\n--- Iteration {iterations} ---")
                print(
                    f"Sending messages to Claude: {json.dumps(messages[-1], indent=2)}"
                )

            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                tools=all_tools,
                messages=messages,
                betas=self.betas,
                thinking= self.thinking if self.thinking else {"type": "disabled"},
                temperature=self.temperature,
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