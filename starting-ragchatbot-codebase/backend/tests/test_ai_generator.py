"""
Tests for AIGenerator class
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add backend directory to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from vector_store import SearchResults


class TestAIGenerator:
    """Test AIGenerator class"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator instance for testing"""
        return AIGenerator(api_key="test_key", model="claude-3-sonnet-20240229")

    @pytest.fixture
    def mock_tool_manager(self, mock_vector_store):
        """Create mock tool manager"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Mock search results
        mock_vector_store.search.return_value = SearchResults(
            documents=["Machine learning content"],
            metadata=[{"course_title": "ML Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = None

        return manager

    def test_initialization(self, ai_generator):
        """Test AIGenerator initialization"""
        assert ai_generator.model == "claude-3-sonnet-20240229"
        assert ai_generator.base_params['model'] == "claude-3-sonnet-20240229"
        assert ai_generator.base_params['temperature'] == 0
        assert ai_generator.base_params['max_tokens'] == 800

    def test_system_prompt_exists(self, ai_generator):
        """Test that system prompt is defined"""
        assert AIGenerator.SYSTEM_PROMPT is not None
        assert len(AIGenerator.SYSTEM_PROMPT) > 0
        assert "search tool" in AIGenerator.SYSTEM_PROMPT.lower()

    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client):
        """Test generating response without tools (direct response)"""
        ai_generator.client = mock_anthropic_client

        response = ai_generator.generate_response(
            query="What is 2+2?",
            conversation_history=None,
            tools=None,
            tool_manager=None
        )

        assert response == "This is a test response"
        mock_anthropic_client.messages.create.assert_called_once()

        # Check call parameters
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]['messages'][0]['content'] == "What is 2+2?"
        assert 'tools' not in call_args[1]

    def test_generate_response_with_conversation_history(self, ai_generator, mock_anthropic_client):
        """Test generating response with conversation history"""
        ai_generator.client = mock_anthropic_client

        history = "User: Hello\nAssistant: Hi there!"
        response = ai_generator.generate_response(
            query="How are you?",
            conversation_history=history,
            tools=None,
            tool_manager=None
        )

        # Check that history is included in system prompt
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]['system']
        assert "Previous conversation:" in system_content
        assert "Hello" in system_content

    def test_generate_response_with_tools_no_tool_use(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test generating response with tools available but not used"""
        ai_generator.client = mock_anthropic_client

        # Mock response that doesn't use tools
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Direct answer", type="text")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        response = ai_generator.generate_response(
            query="What is Python?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        assert response == "Direct answer"

        # Check that tools were provided in the call
        call_args = mock_anthropic_client.messages.create.call_args
        assert 'tools' in call_args[1]
        assert call_args[1]['tool_choice'] == {"type": "auto"}

    def test_generate_response_with_tool_use(self, ai_generator, mock_anthropic_client, mock_tool_manager, mock_tool_use_response):
        """Test generating response that uses tools"""
        ai_generator.client = mock_anthropic_client

        # First call returns tool_use
        # Second call returns final answer
        final_response = MagicMock()
        final_response.content = [MagicMock(text="Machine learning is a subset of AI", type="text")]
        final_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_use_response,
            final_response
        ]

        response = ai_generator.generate_response(
            query="What is machine learning?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Should have made 2 API calls
        assert mock_anthropic_client.messages.create.call_count == 2
        assert response == "Machine learning is a subset of AI"

    def test_helper_methods(self, ai_generator):
        """Test helper methods for building content and extracting text"""
        # Test _build_system_content without history
        content = ai_generator._build_system_content(None)
        assert content == AIGenerator.SYSTEM_PROMPT

        # Test _build_system_content with history
        history = "User: Hello\nAssistant: Hi there"
        content_with_history = ai_generator._build_system_content(history)
        assert "Previous conversation:" in content_with_history
        assert "Hello" in content_with_history

        # Test _extract_text_from_response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response", type="text")]
        text = ai_generator._extract_text_from_response(mock_response)
        assert text == "Test response"

        # Test _extract_text_from_response with empty content
        empty_response = MagicMock()
        empty_response.content = []
        text = ai_generator._extract_text_from_response(empty_response)
        assert text == "No response generated"

    def test_tool_execution_with_multiple_tools(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test handling multiple tool calls in one response"""
        ai_generator.client = mock_anthropic_client

        # Create response with multiple tool uses
        tool_response = MagicMock()
        tool_response.stop_reason = "tool_use"

        tool_block_1 = MagicMock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "first query"}

        tool_block_2 = MagicMock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "search_course_content"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"query": "second query"}

        tool_response.content = [tool_block_1, tool_block_2]

        final_response = MagicMock()
        final_response.content = [MagicMock(text="Combined answer", type="text")]

        mock_anthropic_client.messages.create.side_effect = [
            tool_response,
            final_response
        ]

        response = ai_generator.generate_response(
            query="Complex query needing multiple searches",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        assert response == "Combined answer"

        # Check that final call included both tool results
        final_call_args = mock_anthropic_client.messages.create.call_args
        tool_results_message = final_call_args[1]['messages'][-1]
        assert len(tool_results_message['content']) == 2

    def test_generate_response_handles_api_error(self, ai_generator, mock_anthropic_client):
        """Test that API errors are raised properly"""
        ai_generator.client = mock_anthropic_client

        # Mock API error
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            ai_generator.generate_response(
                query="test",
                tools=None,
                tool_manager=None
            )

    def test_base_params_configuration(self, ai_generator):
        """Test that base parameters are correctly configured"""
        assert ai_generator.base_params['temperature'] == 0
        assert ai_generator.base_params['max_tokens'] == 800
        assert 'model' in ai_generator.base_params

    def test_tool_choice_auto_when_tools_provided(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test that tool_choice is set to auto when tools provided"""
        ai_generator.client = mock_anthropic_client

        ai_generator.generate_response(
            query="test",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]['tool_choice'] == {"type": "auto"}

    def test_two_sequential_tool_calls(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test the main use case: two sequential tool calls"""
        ai_generator.client = mock_anthropic_client

        # Round 1: Claude makes first tool call
        round1_response = MagicMock()
        round1_response.stop_reason = "tool_use"
        tool_block_1 = MagicMock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "course X outline"}
        round1_response.content = [tool_block_1]

        # Round 2: Claude makes second tool call
        round2_response = MagicMock()
        round2_response.stop_reason = "tool_use"
        tool_block_2 = MagicMock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "search_course_content"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"query": "neural networks courses"}
        round2_response.content = [tool_block_2]

        # Round 3: Final answer (no tool use)
        final_response = MagicMock()
        final_response.content = [MagicMock(text="Found 3 courses about neural networks", type="text")]
        final_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response
        ]

        response = ai_generator.generate_response(
            query="Find courses about the same topic as lesson 4 of course X",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_tool_rounds=2
        )

        # Verify 3 API calls made (2 rounds + final answer)
        assert mock_anthropic_client.messages.create.call_count == 3
        assert response == "Found 3 courses about neural networks"

        # Verify message history accumulated correctly
        final_call_messages = mock_anthropic_client.messages.create.call_args_list[2][1]['messages']
        # Should have: user query, assistant tool_use, user tool_result, assistant tool_use, user tool_result
        assert len(final_call_messages) == 5
        assert final_call_messages[0]['role'] == 'user'
        assert final_call_messages[1]['role'] == 'assistant'
        assert final_call_messages[2]['role'] == 'user'
        assert final_call_messages[3]['role'] == 'assistant'
        assert final_call_messages[4]['role'] == 'user'

    def test_max_rounds_forces_final_call(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test that hitting max rounds forces a final response without tools"""
        ai_generator.client = mock_anthropic_client

        # Both rounds return tool_use (Claude keeps trying to use tools)
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test"}
        tool_use_response.content = [tool_block]

        # Final forced response without tools
        forced_response = MagicMock()
        forced_response.content = [MagicMock(text="Based on available information...", type="text")]
        forced_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            tool_use_response,  # Round 1
            tool_use_response,  # Round 2
            forced_response     # Forced final call
        ]

        response = ai_generator.generate_response(
            query="Complex query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_tool_rounds=2
        )

        # Verify 3 API calls made
        assert mock_anthropic_client.messages.create.call_count == 3
        assert response == "Based on available information..."

        # Verify final call has NO tools parameter
        final_call = mock_anthropic_client.messages.create.call_args_list[2][1]
        assert 'tools' not in final_call

    def test_tool_execution_error_terminates_immediately(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test that tool errors terminate immediately and return error message"""
        ai_generator.client = mock_anthropic_client

        # Mock tool to raise exception
        mock_tool_manager.execute_tool = Mock(side_effect=ValueError("Database connection failed"))

        # Round 1: Tool call that will fail
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test"}
        tool_use_response.content = [tool_block]

        mock_anthropic_client.messages.create.return_value = tool_use_response

        response = ai_generator.generate_response(
            query="Search query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_tool_rounds=2
        )

        # Should terminate immediately with error message
        assert "Error executing tool" in response
        assert "Database connection failed" in response

        # Should have only made 1 API call (no second round)
        assert mock_anthropic_client.messages.create.call_count == 1

    def test_conversation_history_preserved_across_rounds(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Verify conversation history is included in system prompt for all rounds"""
        ai_generator.client = mock_anthropic_client

        history = "User: What is ML?\nAssistant: Machine learning is AI subset"

        # Round 1: Tool use
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_1"
        tool_block.input = {"query": "test"}
        tool_use_response.content = [tool_block]

        # Round 2: Final answer
        final_response = MagicMock()
        final_response.content = [MagicMock(text="Answer...", type="text")]
        final_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [tool_use_response, final_response]

        ai_generator.generate_response(
            query="Follow-up question",
            conversation_history=history,
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_tool_rounds=2
        )

        # Verify both API calls included history in system prompt
        for call in mock_anthropic_client.messages.create.call_args_list:
            system_content = call[1]['system']
            assert "Previous conversation:" in system_content
            assert "What is ML?" in system_content


class TestAIGeneratorIntegration:
    """Integration tests with real tool manager"""

    @pytest.fixture
    def real_setup(self, mock_anthropic_client, mock_vector_store):
        """Setup with real components"""
        ai_gen = AIGenerator(api_key="test_key", model="claude-3-sonnet-20240229")
        ai_gen.client = mock_anthropic_client

        tool_manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(tool)

        return ai_gen, tool_manager, mock_anthropic_client, mock_vector_store

    def test_full_tool_calling_flow(self, real_setup):
        """Test complete flow from query to tool execution to final answer"""
        ai_gen, tool_manager, mock_client, mock_store = real_setup

        # Setup mock responses
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "what is machine learning"}
        tool_use_response.content = [tool_block]

        final_response = MagicMock()
        final_response.content = [MagicMock(text="ML is a field of AI", type="text")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Setup mock search results
        mock_store.search.return_value = SearchResults(
            documents=["Machine learning content"],
            metadata=[{"course_title": "ML Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_store.get_lesson_link.return_value = None

        # Execute
        response = ai_gen.generate_response(
            query="What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify
        assert response == "ML is a field of AI"
        assert mock_store.search.called
        assert mock_client.messages.create.call_count == 2
