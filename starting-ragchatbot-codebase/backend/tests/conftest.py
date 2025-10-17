"""
Shared pytest fixtures for RAG system tests
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import tempfile
import shutil

# Add backend directory to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Jane Smith",
        lessons=[
            Lesson(lesson_number=0, title="Course Overview", lesson_link="https://example.com/ml-course/lesson0"),
            Lesson(lesson_number=1, title="Linear Regression", lesson_link="https://example.com/ml-course/lesson1"),
            Lesson(lesson_number=2, title="Neural Networks", lesson_link="https://example.com/ml-course/lesson2")
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Course Introduction to Machine Learning Lesson 0 content: This course covers machine learning fundamentals.",
            course_title="Introduction to Machine Learning",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Course Introduction to Machine Learning Lesson 1 content: Linear regression is a supervised learning algorithm.",
            course_title="Introduction to Machine Learning",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Course Introduction to Machine Learning Lesson 2 content: Neural networks are composed of layers of neurons.",
            course_title="Introduction to Machine Learning",
            lesson_number=2,
            chunk_index=2
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)

    # Mock search method to return sample results
    mock_store.search.return_value = SearchResults(
        documents=["This is sample content about machine learning"],
        metadata=[{"course_title": "Introduction to Machine Learning", "lesson_number": 1}],
        distances=[0.5],
        error=None
    )

    # Mock get_lesson_link
    mock_store.get_lesson_link.return_value = "https://example.com/ml-course/lesson1"

    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = MagicMock()

    # Mock a standard text response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="This is a test response", type="text")]
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_tool_use_response():
    """Create a mock response with tool use for testing"""
    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"

    # Create tool use content block
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.id = "tool_123"
    tool_block.input = {"query": "what is machine learning"}

    mock_response.content = [tool_block]

    return mock_response


@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def real_vector_store(temp_chroma_db):
    """Create a real VectorStore instance with temp database"""
    return VectorStore(
        chroma_path=temp_chroma_db,
        embedding_model="all-MiniLM-L6-v2",
        max_results=5
    )
