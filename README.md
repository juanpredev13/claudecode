# Claude Code Learning Repository

This repository contains projects and exercises from the [**Claude Code: A Highly Agentic Coding Assistant**](https://learn.deeplearning.ai/courses/claude-code-a-highly-agentic-coding-assistant/) course by DeepLearning.AI in partnership with Anthropic.

## About the Course

**Claude Code** represents a significant advancement in AI-powered coding assistants, offering a higher degree of agency compared to previous tools. The course teaches developers how to explore, build, and refine codebases using systematic best practices with Claude Code.

### What Makes Claude Code Different?

- **Extended Autonomous Operation**: Works independently for many minutes on complex tasks
- **Planning & Thinking Modes**: Strategizes before executing
- **Parallel Session Management**: Handles multiple workflows simultaneously
- **Local Codebase Processing**: Works directly with your code without requiring semantic indexing
- **Git Integration**: Native support for Git worktrees and version control
- **MCP Server Integration**: Connects with Model Context Protocol servers for enhanced capabilities

### Course Topics

The course covers practical applications across key areas:

- **AI Coding & Software Development**: Building features from frontend to backend
- **Agents & Chatbots**: Creating intelligent conversational systems
- **Data Processing & Analysis**: Working with Jupyter notebooks and dashboards
- **Prompt Engineering & RAG**: Retrieval-Augmented Generation systems
- **LLMOps & Task Automation**: Deployment and operational workflows
- **Testing & Evaluation**: Code quality and monitoring practices

**Course Duration**: ~111 minutes of video content

## Repository Structure

```
claudecode/
├── starting-ragchatbot-codebase/    # Main project: RAG Chatbot System
│   ├── backend/                     # FastAPI backend with Claude integration
│   │   ├── ai_generator.py         # Multi-round tool calling AI engine
│   │   ├── vector_store.py         # ChromaDB vector storage
│   │   ├── search_tools.py         # Course search tools
│   │   ├── rag_system.py           # RAG orchestration
│   │   └── tests/                  # Test infrastructure
│   ├── frontend/                    # Interactive web interface
│   │   ├── index.html              # Main UI
│   │   ├── script.js               # Frontend logic with session management
│   │   └── style.css               # Modern dark theme styling
│   ├── docs/                        # Course materials (text files)
│   └── CLAUDE.md                    # Project-specific Claude Code instructions
├── demo/                            # Demo files and experiments
├── backend-tool-refactor.md         # Documentation on tool architecture
└── CLAUDE.md                        # Repository-level Claude Code guidance
```

## Featured Project: RAG Chatbot for Course Materials

The main project is a production-ready **Retrieval-Augmented Generation (RAG) chatbot** designed to answer questions about course content using semantic search and AI.

### Key Features

#### 🧠 **Advanced AI Architecture**
- **Multi-Turn Tool Calling**: Up to 2 sequential searches per query for complex information retrieval
- **Anthropic Claude Integration**: Using Claude Sonnet 4 for intelligent responses
- **Context-Aware Responses**: Maintains conversation history for coherent multi-turn dialogues

#### 🔍 **Semantic Search with ChromaDB**
- **Dual-Collection Design**:
  - `course_catalog`: High-level course metadata with fuzzy matching
  - `course_content`: Sentence-level chunks for granular search
- **Intelligent Chunking**: 800-character chunks with 100-character overlap
- **Contextual Prefixes**: "Course {title} Lesson {N} content: {chunk}"

#### 🎨 **Modern Web Interface**
- **Session Management**: "New Chat" functionality with backend session clearing
- **Interactive Source Attribution**: Clickable lesson links displayed as styled pills
- **Markdown Rendering**: Rich text formatting for AI responses
- **Dark Mode UI**: Professional, accessible interface

#### 🧪 **Testing & Quality**
- **pytest Infrastructure**: Comprehensive test suite
- **Modular Architecture**: Separation of concerns (tools, storage, AI generation)

### How It Works

```
User Query → FastAPI → RAGSystem → AIGenerator (Claude API)
                                        ↓
                                   Tool Calling
                                        ↓
                                  CourseSearchTool
                                        ↓
                                   VectorStore (ChromaDB)
                                        ↓
                            Search Results → Claude Synthesis
                                        ↓
                                Response + Sources
```

### Getting Started with the RAG Chatbot

#### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Anthropic API key

#### Quick Start

```bash
# Navigate to project directory
cd starting-ragchatbot-codebase

# Install dependencies
uv sync

# Set up environment
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Run the application
./run.sh
```

Access the application at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

#### Adding Course Materials

Place course documents in the `docs/` folder with this format:

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

Supported formats: `.txt`, `.pdf`, `.docx`

### Configuration

All settings are centralized in `backend/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 800 | Characters per chunk (affects retrieval granularity) |
| `CHUNK_OVERLAP` | 100 | Overlap for context preservation |
| `MAX_RESULTS` | 5 | Top-k search results returned |
| `MAX_HISTORY` | 2 | Conversation turns to remember |
| `MAX_TOOL_ROUNDS` | 2 | Sequential tool calls per query |
| `ANTHROPIC_MODEL` | claude-sonnet-4-20250514 | Claude model version |

### API Endpoints

- `POST /api/query` - Submit questions (with session_id)
- `GET /api/courses` - List all indexed courses
- `GET /api/stats` - Course statistics
- `POST /api/session/clear` - Clear conversation history

## Technologies Used

- **Backend**: FastAPI, Python 3.13
- **AI/ML**: Anthropic Claude API, sentence-transformers
- **Vector Database**: ChromaDB
- **Frontend**: Vanilla JavaScript, marked.js for Markdown
- **Testing**: pytest
- **Package Management**: uv

## Learning Outcomes

Working through this repository demonstrates:

1. **Agentic AI Integration**: Implementing multi-round tool calling patterns
2. **RAG System Design**: Dual-collection vector stores with semantic chunking
3. **API Development**: RESTful services with FastAPI
4. **Session Management**: Stateful conversation handling
5. **Full-Stack Development**: Backend Python + Frontend JavaScript
6. **Testing Practices**: pytest for AI-powered applications
7. **Vector Search**: ChromaDB for semantic similarity
8. **Prompt Engineering**: System prompts for reliable AI behavior

## Course Link

**Enroll in the course**: [Claude Code: A Highly Agentic Coding Assistant](https://learn.deeplearning.ai/courses/claude-code-a-highly-agentic-coding-assistant/lesson/66b35/introduction)

## Development Environment

- **Operating System**: Linux (6.14.0-33-generic)
- **Python Version**: 3.13
- **Package Manager**: uv

## Future Projects

This repository will expand with additional projects from the course:
- Data analysis with Jupyter notebooks
- Dashboard creation for e-commerce data
- Frontend development from Figma mockups
- MCP server integrations

## Contributing

This is a personal learning repository. Feel free to fork and experiment with the code for your own learning journey.

## License

Educational purposes only. Course content and structure are property of DeepLearning.AI and Anthropic.

## Acknowledgments

- **DeepLearning.AI** for the comprehensive course curriculum
- **Anthropic** for Claude Code and Claude API
- **Instructors**: For the practical, hands-on approach to AI-assisted development

---

**Built with Claude Code** - Demonstrating the power of highly agentic AI coding assistants
