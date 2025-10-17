# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) chatbot system for querying course materials. It uses ChromaDB for vector storage, Anthropic's Claude API for AI responses, and FastAPI for the backend API.

## Setup and Running

### Initial Setup
```bash
# Install dependencies
uv sync

# Create .env file with your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Running the Application

**IMPORTANT**: Do NOT automatically run the server using `./run.sh` or any other command. The user will start the server themselves.

```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

Access at http://localhost:8000 (web interface) or http://localhost:8000/docs (API docs).

## Architecture Overview

### RAG Pipeline Flow

**Query Processing Chain:**
1. **Frontend** (JavaScript) → POST `/api/query` with user question
2. **FastAPI Endpoint** (`app.py`) → Receives request, delegates to RAGSystem
3. **RAGSystem** (`rag_system.py`) → Orchestrates entire RAG workflow
4. **AIGenerator** (`ai_generator.py`) → Makes 2 API calls to Claude:
   - First call: Claude decides to use `search_course_content` tool
   - Tool execution: Searches vector store via ToolManager
   - Second call: Claude synthesizes final answer from search results
5. **SessionManager** (`session_manager.py`) → Stores conversation history
6. **Response** → Returns answer + sources to frontend

### Document Processing Pipeline

**On Startup** (`app.py:88-98`):
1. Loads all `.txt`/`.pdf`/`.docx` files from `docs/` folder
2. Processes via `DocumentProcessor` → extracts course metadata and lessons
3. Creates sentence-based chunks (800 chars, 100 char overlap)
4. Adds contextual prefixes: `"Course {title} Lesson {N} content: {chunk}"`
5. Generates embeddings via sentence-transformers (all-MiniLM-L6-v2)
6. Stores in ChromaDB with two collections:
   - `course_catalog`: Course-level metadata for fuzzy matching
   - `course_content`: Actual text chunks for semantic search

### Vector Store Architecture

**Two-Collection Design** (`vector_store.py`):
- **course_catalog**: High-level course metadata (title, instructor, lessons JSON)
  - Used for course name resolution via semantic search
  - ID = course title
- **course_content**: Searchable content chunks with metadata
  - Filtered by course_title and/or lesson_number
  - ID = `{course_title}_{chunk_index}`

**Search Flow** (`vector_store.py:61-100`):
1. If `course_name` provided → fuzzy match via catalog semantic search
2. Build ChromaDB filter dict (AND logic for course + lesson)
3. Query content collection with embeddings
4. Return top 5 results (configurable via `MAX_RESULTS`)

### Tool-Based Search Pattern

The system uses Anthropic's tool calling (function calling) instead of direct RAG:
- **Tools defined** in `search_tools.py` → `CourseSearchTool`
- **Tool registration** in `rag_system.py:22-25` → ToolManager
- **AI decides** when to call tools (not hardcoded)
- **Sources tracked** via `last_sources` attribute on search tool
- **Retrieved after response** via `tool_manager.get_last_sources()`

### Session Management

**Conversation Context** (`session_manager.py`):
- Each session has a unique ID (`session_1`, `session_2`, etc.)
- Stores last N exchanges (default: 2, configurable via `MAX_HISTORY`)
- History formatted as: `"User: {msg}\nAssistant: {msg}"`
- Appended to system prompt in `ai_generator.py:62-64`

## Key Configuration

All settings in `backend/config.py`:
- `CHUNK_SIZE`: 800 chars (affects retrieval granularity)
- `CHUNK_OVERLAP`: 100 chars (context preservation between chunks)
- `MAX_RESULTS`: 5 (top-k results returned from vector search)
- `MAX_HISTORY`: 2 (number of conversation turns remembered)
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514

## Document Format

Course files in `docs/` must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[content...]

Lesson 1: [title]
Lesson Link: [url]
[content...]
```

Processed by `document_processor.py:97-259`. Lesson links are optional but lesson markers are required for proper chunking.

## Important Implementation Details

### Preventing Duplicate Documents
`rag_system.py:76-96` checks existing course titles before processing. Only new courses are added to prevent re-indexing on app restart.

### AI System Prompt
`ai_generator.py:8-30` contains strict instructions:
- Use search tool only for course-specific questions
- Maximum one search per query
- No meta-commentary in responses
- Direct answers only

### ChromaDB Persistence
Database stored at `./backend/chroma_db/` (relative to backend directory). Persists across restarts. Use `clear_existing=True` in `add_course_folder()` to rebuild.

### Frontend-Backend Communication
- Frontend uses relative URLs (`/api`) for proxy compatibility
- CORS configured for `*` origins in development
- Markdown rendering via `marked.js` library
- Source attribution in collapsible `<details>` element

## Modifying the System

### Adding New Tools
1. Create tool class inheriting from `Tool` in `search_tools.py`
2. Implement `get_tool_definition()` and `execute()` methods
3. Register in `rag_system.py` via `tool_manager.register_tool()`

### Changing Embedding Model
Modify `EMBEDDING_MODEL` in `config.py`. Must be compatible with sentence-transformers library. Requires rebuilding the vector database.

### Adjusting Response Length
Change `max_tokens` in `ai_generator.py:40` (currently 800). Claude will be cut off if exceeding this limit.

### Custom Document Types
Extend `document_processor.py:97` to handle additional file types. Current support: `.txt`, `.pdf`, `.docx`.
- dont run the server using ./run.sh I will start it myself