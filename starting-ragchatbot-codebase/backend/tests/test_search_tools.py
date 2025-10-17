"""
Tests for search_tools module (CourseSearchTool and ToolManager)
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# Add backend directory to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from search_tools import CourseSearchTool, ToolManager, Tool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test CourseSearchTool class"""

    def test_initialization(self, mock_vector_store):
        """Test tool initialization"""
        tool = CourseSearchTool(mock_vector_store)
        assert tool.store == mock_vector_store
        assert tool.last_sources == []

    def test_get_tool_definition(self, mock_vector_store):
        """Test tool definition format"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition['name'] == 'search_course_content'
        assert 'description' in definition
        assert 'input_schema' in definition
        assert definition['input_schema']['type'] == 'object'
        assert 'query' in definition['input_schema']['properties']
        assert 'course_name' in definition['input_schema']['properties']
        assert 'lesson_number' in definition['input_schema']['properties']
        assert definition['input_schema']['required'] == ['query']

    def test_execute_successful_search(self, mock_vector_store):
        """Test successful search execution"""
        tool = CourseSearchTool(mock_vector_store)

        # Mock successful search results
        mock_vector_store.search.return_value = SearchResults(
            documents=["Machine learning is a subset of AI"],
            metadata=[{"course_title": "ML Course", "lesson_number": 1}],
            distances=[0.3],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        result = tool.execute(query="what is machine learning")

        assert isinstance(result, str)
        assert "ML Course" in result
        assert "Lesson 1" in result
        assert "Machine learning is a subset of AI" in result
        mock_vector_store.search.assert_called_once_with(
            query="what is machine learning",
            course_name=None,
            lesson_number=None
        )

    def test_execute_with_course_filter(self, mock_vector_store):
        """Test execution with course name filter"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Content about Python"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 2}],
            distances=[0.2],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson2"

        result = tool.execute(
            query="python functions",
            course_name="Python Basics"
        )

        assert "Python Basics" in result
        mock_vector_store.search.assert_called_once_with(
            query="python functions",
            course_name="Python Basics",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test execution with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson3"

        result = tool.execute(
            query="test query",
            lesson_number=3
        )

        assert "Lesson 3" in result
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=3
        )

    def test_execute_with_both_filters(self, mock_vector_store):
        """Test execution with both course and lesson filters"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{"course_title": "Advanced Course", "lesson_number": 5}],
            distances=[0.15],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson5"

        result = tool.execute(
            query="advanced topic",
            course_name="Advanced Course",
            lesson_number=5
        )

        mock_vector_store.search.assert_called_once_with(
            query="advanced topic",
            course_name="Advanced Course",
            lesson_number=5
        )

    def test_execute_with_error(self, mock_vector_store):
        """Test execution when search returns error"""
        tool = CourseSearchTool(mock_vector_store)

        # Mock error result
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="No course found matching 'Nonexistent Course'"
        )

        result = tool.execute(
            query="test",
            course_name="Nonexistent Course"
        )

        assert result == "No course found matching 'Nonexistent Course'"

    def test_execute_empty_results(self, mock_vector_store):
        """Test execution when no results found"""
        tool = CourseSearchTool(mock_vector_store)

        # Mock empty results (no error, just no matches)
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_empty_results_with_course_filter(self, mock_vector_store):
        """Test execution when no results found with course filter"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )

        result = tool.execute(
            query="nonexistent topic",
            course_name="Test Course"
        )

        assert "No relevant content found" in result
        assert "Test Course" in result

    def test_execute_empty_results_with_lesson_filter(self, mock_vector_store):
        """Test execution when no results found with lesson filter"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )

        result = tool.execute(
            query="test",
            lesson_number=99
        )

        assert "No relevant content found" in result
        assert "lesson 99" in result

    def test_format_results_tracks_sources(self, mock_vector_store):
        """Test that formatting results tracks sources with links"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/a/lesson1",
            "https://example.com/b/lesson2"
        ]

        tool.execute(query="test")

        # Check that sources were tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]['text'] == "Course A - Lesson 1"
        assert tool.last_sources[0]['url'] == "https://example.com/a/lesson1"
        assert tool.last_sources[1]['text'] == "Course B - Lesson 2"
        assert tool.last_sources[1]['url'] == "https://example.com/b/lesson2"

    def test_format_results_without_lesson_numbers(self, mock_vector_store):
        """Test formatting results when lesson numbers are missing"""
        tool = CourseSearchTool(mock_vector_store)

        mock_vector_store.search.return_value = SearchResults(
            documents=["General content"],
            metadata=[{"course_title": "General Course", "lesson_number": None}],
            distances=[0.1],
            error=None
        )

        result = tool.execute(query="test")

        assert "[General Course]" in result
        assert "Lesson" not in result or "None" in result


class TestToolManager:
    """Test ToolManager class"""

    def test_initialization(self):
        """Test ToolManager initialization"""
        manager = ToolManager()
        assert manager.tools == {}

    def test_register_tool(self, mock_vector_store):
        """Test registering a tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool

    def test_register_tool_without_name(self, mock_vector_store):
        """Test registering tool without name raises error"""
        manager = ToolManager()

        # Create a mock tool with no name
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {"description": "test"}

        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]['name'] == 'search_course_content'

    def test_execute_tool(self, mock_vector_store):
        """Test executing a registered tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        mock_vector_store.search.return_value = SearchResults(
            documents=["Test result"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = None

        result = manager.execute_tool("search_course_content", query="test query")

        assert isinstance(result, str)
        assert "Test" in result

    def test_execute_nonexistent_tool(self):
        """Test executing non-existent tool returns error"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store):
        """Test retrieving last sources from tools"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute a search to populate sources
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/test"

        manager.execute_tool("search_course_content", query="test")

        sources = manager.get_last_sources()
        assert len(sources) > 0
        assert sources[0]['text'] == "Test Course - Lesson 1"

    def test_get_last_sources_empty(self):
        """Test getting sources when no search has been performed"""
        manager = ToolManager()
        sources = manager.get_last_sources()
        assert sources == []

    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Populate sources
        tool.last_sources = [{"text": "Test", "url": "https://example.com"}]

        # Reset
        manager.reset_sources()

        assert tool.last_sources == []
        assert manager.get_last_sources() == []
