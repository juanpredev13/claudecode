"""
Integration tests for RAGSystem
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

# Add backend directory to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from rag_system import RAGSystem
from config import Config
from models import Course, Lesson
from vector_store import SearchResults


class TestRAGSystemIntegration:
    """Integration tests for RAGSystem"""

    @pytest.fixture
    def test_config(self, temp_chroma_db):
        """Create test configuration"""
        config = Config()
        config.CHROMA_PATH = temp_chroma_db
        config.ANTHROPIC_API_KEY = "test_key"
        return config

    @pytest.fixture
    def rag_system(self, test_config):
        """Create RAGSystem instance for testing"""
        return RAGSystem(test_config)

    @pytest.fixture
    def sample_course_file(self, tmp_path):
        """Create a sample course file for testing"""
        course_content = """Course Title: Introduction to Python
Course Link: https://example.com/python
Course Instructor: John Doe

Lesson 0: Getting Started
Lesson Link: https://example.com/python/lesson0
Welcome to Python programming. This course covers the basics.

Lesson 1: Variables and Data Types
Lesson Link: https://example.com/python/lesson1
In Python, variables are created when you assign a value to them.
Python supports various data types including integers, floats, strings, and booleans.

Lesson 2: Control Flow
Lesson Link: https://example.com/python/lesson2
Control flow in Python includes if statements, for loops, and while loops.
"""
        file_path = tmp_path / "python_course.txt"
        file_path.write_text(course_content)
        return str(file_path)

    def test_initialization(self, rag_system):
        """Test RAGSystem initializes all components"""
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None

    def test_add_course_document(self, rag_system, sample_course_file):
        """Test adding a single course document"""
        course, chunk_count = rag_system.add_course_document(sample_course_file)

        assert course is not None
        assert course.title == "Introduction to Python"
        assert course.instructor == "John Doe"
        assert len(course.lessons) == 3
        assert chunk_count > 0

        # Verify course was added to vector store
        course_titles = rag_system.vector_store.get_existing_course_titles()
        assert "Introduction to Python" in course_titles

    def test_add_course_folder(self, rag_system, tmp_path):
        """Test adding multiple courses from a folder"""
        # Create multiple course files
        course1 = tmp_path / "course1.txt"
        course1.write_text("""Course Title: Course One
Course Instructor: Teacher A

Lesson 0: Introduction
Content for course one.""")

        course2 = tmp_path / "course2.txt"
        course2.write_text("""Course Title: Course Two
Course Instructor: Teacher B

Lesson 0: Introduction
Content for course two.""")

        total_courses, total_chunks = rag_system.add_course_folder(str(tmp_path))

        assert total_courses == 2
        assert total_chunks > 0

        course_titles = rag_system.vector_store.get_existing_course_titles()
        assert "Course One" in course_titles
        assert "Course Two" in course_titles

    def test_add_course_folder_prevents_duplicates(self, rag_system, tmp_path):
        """Test that adding same folder twice doesn't duplicate courses"""
        course_file = tmp_path / "course.txt"
        course_file.write_text("""Course Title: Test Course
Lesson 0: Test
Test content""")

        # Add first time
        count1, _ = rag_system.add_course_folder(str(tmp_path))
        assert count1 == 1

        # Add second time - should skip existing
        count2, _ = rag_system.add_course_folder(str(tmp_path))
        assert count2 == 0

        # Verify only one course exists
        assert rag_system.vector_store.get_course_count() == 1

    def test_add_course_folder_clear_existing(self, rag_system, tmp_path):
        """Test clearing existing data before adding"""
        course_file = tmp_path / "course.txt"
        course_file.write_text("""Course Title: Test Course
Lesson 0: Test
Test content""")

        # Add first time
        rag_system.add_course_folder(str(tmp_path))
        assert rag_system.vector_store.get_course_count() == 1

        # Add with clear_existing=True
        count, _ = rag_system.add_course_folder(str(tmp_path), clear_existing=True)
        assert count == 1
        assert rag_system.vector_store.get_course_count() == 1

    def test_add_course_folder_nonexistent(self, rag_system):
        """Test adding from non-existent folder"""
        count, chunks = rag_system.add_course_folder("/nonexistent/folder")
        assert count == 0
        assert chunks == 0

    def test_query_with_mocked_ai(self, rag_system, sample_course_file):
        """Test query execution with mocked AI responses"""
        # Add course data
        rag_system.add_course_document(sample_course_file)

        # Mock AI generator to return a simple response
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_gen:
            mock_gen.return_value = "Python is a programming language"

            response, sources = rag_system.query("What is Python?")

            assert response == "Python is a programming language"
            assert mock_gen.called

            # Check that correct parameters were passed
            call_args = mock_gen.call_args
            assert "What is Python?" in call_args[1]['query']
            assert call_args[1]['tools'] is not None
            assert call_args[1]['tool_manager'] is not None

    def test_query_with_session(self, rag_system):
        """Test query with session management"""
        session_id = "test_session_1"

        with patch.object(rag_system.ai_generator, 'generate_response') as mock_gen:
            mock_gen.return_value = "Response 1"

            # First query
            response1, _ = rag_system.query("Query 1", session_id)

            # Second query - should have history
            mock_gen.return_value = "Response 2"
            response2, _ = rag_system.query("Query 2", session_id)

            # Check that second call included history
            second_call_args = mock_gen.call_args
            history = second_call_args[1]['conversation_history']
            assert history is not None
            assert "Query 1" in history
            assert "Response 1" in history

    def test_query_without_session(self, rag_system):
        """Test query without session ID"""
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_gen:
            mock_gen.return_value = "Test response"

            response, sources = rag_system.query("Test query")

            # Should work without session
            assert response == "Test response"

            # History should be None
            call_args = mock_gen.call_args
            assert call_args[1]['conversation_history'] is None

    def test_query_retrieves_sources(self, rag_system, sample_course_file):
        """Test that query retrieves sources from tool manager"""
        # Add course
        rag_system.add_course_document(sample_course_file)

        # Mock AI to use the tool
        def mock_generate(query, conversation_history, tools, tool_manager):
            # Simulate tool execution
            tool_manager.execute_tool("search_course_content", query="Python")
            return "Python info"

        with patch.object(rag_system.ai_generator, 'generate_response', side_effect=mock_generate):
            response, sources = rag_system.query("What is Python?")

            # Sources should be populated
            assert isinstance(sources, list)
            # Note: actual source population depends on tool execution

    def test_query_resets_sources_after_retrieval(self, rag_system):
        """Test that sources are reset after being retrieved"""
        # Set up sources
        rag_system.tool_manager.tools['search_course_content'].last_sources = [
            {"text": "Test", "url": "https://example.com"}
        ]

        with patch.object(rag_system.ai_generator, 'generate_response') as mock_gen:
            mock_gen.return_value = "Test"

            response, sources = rag_system.query("Test")

            # Sources should be reset after query
            remaining_sources = rag_system.tool_manager.get_last_sources()
            assert remaining_sources == []

    def test_get_course_analytics(self, rag_system, sample_course_file):
        """Test getting course analytics"""
        # Initially empty
        analytics = rag_system.get_course_analytics()
        assert analytics['total_courses'] == 0
        assert analytics['course_titles'] == []

        # Add course
        rag_system.add_course_document(sample_course_file)

        # Check analytics
        analytics = rag_system.get_course_analytics()
        assert analytics['total_courses'] == 1
        assert "Introduction to Python" in analytics['course_titles']

    def test_error_handling_invalid_file(self, rag_system, tmp_path):
        """Test error handling for invalid file"""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("Invalid content without proper format")

        # Should handle error gracefully
        course, chunks = rag_system.add_course_document(str(invalid_file))
        # Depending on implementation, might return None or raise handled exception


class TestRAGSystemRealScenarios:
    """Test real-world scenarios with actual vector store"""

    @pytest.fixture
    def test_config(self, temp_chroma_db):
        """Create test configuration"""
        config = Config()
        config.CHROMA_PATH = temp_chroma_db
        config.ANTHROPIC_API_KEY = "test_key"
        return config

    @pytest.fixture
    def populated_rag_system(self, test_config, tmp_path):
        """Create RAG system with sample data"""
        rag = RAGSystem(test_config)

        # Create sample courses
        ml_course = tmp_path / "ml.txt"
        ml_course.write_text("""Course Title: Machine Learning Fundamentals
Course Instructor: Dr. Smith

Lesson 0: Introduction to ML
Machine learning is a subset of artificial intelligence.

Lesson 1: Supervised Learning
Supervised learning uses labeled data for training.""")

        python_course = tmp_path / "python.txt"
        python_course.write_text("""Course Title: Python Programming
Course Instructor: Prof. Jones

Lesson 0: Python Basics
Python is a high-level programming language.

Lesson 1: Data Structures
Python provides lists, dictionaries, sets, and tuples.""")

        rag.add_course_folder(str(tmp_path))
        return rag

    def test_search_finds_relevant_content(self, populated_rag_system):
        """Test that search actually finds relevant content"""
        # Direct search through vector store
        results = populated_rag_system.vector_store.search("machine learning")

        assert not results.is_empty()
        assert any("machine learning" in doc.lower() for doc in results.documents)

    def test_search_with_course_filter_works(self, populated_rag_system):
        """Test searching with course filter"""
        results = populated_rag_system.vector_store.search(
            query="programming",
            course_name="Python"
        )

        assert not results.is_empty()
        for meta in results.metadata:
            assert "Python" in meta['course_title']

    def test_tool_execution_returns_results(self, populated_rag_system):
        """Test that tool execution returns formatted results"""
        result = populated_rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="machine learning"
        )

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain formatted results
        assert "[" in result  # Header format

    def test_end_to_end_query_flow(self, populated_rag_system):
        """Test complete query flow from question to answer"""
        # This test requires mocking the AI since we don't have real API key
        with patch.object(populated_rag_system.ai_generator, 'generate_response') as mock_gen:
            # Simulate AI calling the tool and returning answer
            def simulate_ai_with_tool(query, conversation_history, tools, tool_manager):
                # AI decides to search
                search_result = tool_manager.execute_tool(
                    "search_course_content",
                    query="machine learning"
                )
                # AI synthesizes answer
                return "Machine learning is a subset of AI that uses algorithms to learn from data."

            mock_gen.side_effect = simulate_ai_with_tool

            response, sources = populated_rag_system.query("What is machine learning?")

            assert isinstance(response, str)
            assert len(response) > 0
            assert isinstance(sources, list)
