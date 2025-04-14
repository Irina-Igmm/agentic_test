# Agentic RAG System

An advanced Retrieval-Augmented Generation system with agent capabilities that combines local document knowledge with web search to intelligently answer user queries.

## Features

- **Knowledge Router**: Intelligently determines whether to use local knowledge or web search
- **Multi-format Document Support**: Processes PDFs, CSVs, and text files
- **Scalable Vector Database**: Efficiently stores and retrieves document embeddings
- **Web Search Integration**: Utilizes CrewAI for intelligent web search and content extraction
- **Customizable URL Sources**: Option to provide specific URLs for targeted information retrieval
- **Command Line Interface**: Easy-to-use CLI for both single queries and interactive sessions

## Installation

### Using UV (Recommended)

```bash
uv venv
uv pip install -e .
```

### Using pip

```bash
pip install -e .
```

## Environment Setup

Create a `.env` file in the project root with the following variables:

```bash
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Command Line Interface

Process a single query:

```bash
agentic-rag --query "What is Retrieval-Augmented Generation?"
```

Interactive mode:

```bash
agentic-rag --interactive
```

Specify document files:

```bash
agentic-rag --files data/document1.pdf data/document2.csv --query "What are the key metrics?"
```

Use specific URLs for web search:

```bash
agentic-rag --urls https://example.com/page1 https://example.com/page2 --query "Latest developments in RAG"
```

### Python API

```python
from agentic_rag import AgenticRAG

# Initialize with specific files
rag = AgenticRAG(["data/document1.pdf", "data/document2.csv"])

# Process a query
answer = rag.process_query("What are the key metrics?")
print(answer)

# Process a query with specific URLs
answer = rag.process_query(
    "Latest developments in RAG",
    specific_urls=["https://example.com/page1", "https://example.com/page2"]
)
print(answer)
```

## Development

### Install Development Dependencies

```bash
uv pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
black src tests
isort src tests
```

### Code Linting

```bash
ruff check src tests
mypy src tests
```

## Project Structure

```
agentic_rag/
├── src/
│   └── agentic_rag/
│       ├── __init__.py
│       ├── cli.py
│       ├── main.py
│       ├── config/
│       │   └── __init__.py
│       ├── knowledge/
│       │   ├── __init__.py
│       │   ├── knowledge_router.py
│       │   ├── local_knowledge.py
│       │   └── web_knowledge.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── answer_generator.py
│       │   └── llm_provider.py
│       └── utils/
├── data/
├── notebooks/
├── tests/
│   ├── test_knowledge_router.py
│   ├── test_answer_generator.py
│   └── test_local_knowledge.py
├── .env.example
├── pyproject.toml
└── README.md
```

## License

MIT