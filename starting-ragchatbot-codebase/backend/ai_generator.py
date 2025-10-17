import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **You may use the search tool up to 2 times per query** to gather comprehensive information
- Common multi-step patterns:
  - First search: Get course outline or lesson details
  - Second search: Use information from first search to find related content
- Synthesize all search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search as needed (up to 2 searches), then answer
- **Multi-step queries**: Break down the problem and search sequentially
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "I will search"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_tool_rounds: int = 2) -> str:
        """
        Generate AI response with support for up to 2 sequential tool-calling rounds.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum sequential tool calling rounds (default: 2)

        Returns:
            Generated response as string
        """

        # Build system content with optional conversation history
        system_content = self._build_system_content(conversation_history)

        # Initialize message history with user query
        messages = [{"role": "user", "content": query}]

        # Track rounds completed
        rounds_completed = 0

        # Iterative loop for multi-round tool calling
        while rounds_completed < max_tool_rounds:
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            response = self.client.messages.create(**api_params)
            rounds_completed += 1

            # TERMINATION CONDITION 1: No tool use (natural termination)
            if response.stop_reason != "tool_use":
                return self._extract_text_from_response(response)

            # Add assistant's tool use response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls and collect results
            try:
                tool_results = self._execute_tool_calls(response.content, tool_manager)
            except Exception as e:
                # TERMINATION CONDITION 3: Tool execution error (immediate termination)
                return f"Error executing tool: {str(e)}"

            # Add tool results to messages
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # TERMINATION CONDITION 2: Max rounds reached
            if rounds_completed >= max_tool_rounds:
                # Force final response by making API call without tools
                final_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": system_content
                    # Explicitly no tools - forces text response
                }
                final_response = self.client.messages.create(**final_params)
                return self._extract_text_from_response(final_response)

        # Safety fallback (should never reach here)
        return "Unable to generate response after maximum rounds"
    
    def _build_system_content(self, conversation_history: Optional[str]) -> str:
        """Build system prompt with optional conversation history"""
        return (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

    def _execute_tool_calls(self, content_blocks, tool_manager) -> List[Dict]:
        """
        Execute all tool calls in a response and return formatted results.
        Raises exception on tool execution error.

        Args:
            content_blocks: List of content blocks from API response
            tool_manager: Manager to execute tools

        Returns:
            List of tool result dictionaries
        """
        tool_results = []
        for content_block in content_blocks:
            if content_block.type == "tool_use":
                # Execute tool - let exceptions propagate
                tool_result = tool_manager.execute_tool(
                    content_block.name,
                    **content_block.input
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })

        return tool_results

    def _extract_text_from_response(self, response) -> str:
        """Extract text content from response, handling mixed content blocks"""
        if response.content:
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
        return "No response generated"