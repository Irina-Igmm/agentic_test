# Agentic Datacamp Project

A CrewAI-based project for autonomous data research, analysis, and report generation.

## Project Structure

```
├── config/             # Configuration files
├── data/               # Data files and storage
├── notebooks/          # Jupyter notebooks
├── src/                # Source code
│   ├── agents.py       # Agent definitions
│   ├── config.py       # Configuration settings
│   ├── main.py         # Main entry point
│   └── tasks.py        # Task definitions
├── .env                # Environment variables (don't commit this file)
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys
4. Run the main script:
   ```
   python src/main.py
   ```

## Usage

By default, the agents will research, analyze, and create a report on "data science trends in 2025".
You can specify a different topic when running the script:

```python
from src.main import main

main("machine learning applications in healthcare")
```

## Agents

The project uses three primary agents:
1. **Researcher Agent** - Gathers information on the specified topic
2. **Analyst Agent** - Processes and analyzes the gathered information
3. **Report Writer Agent** - Creates a final report based on the analysis

## License

MIT