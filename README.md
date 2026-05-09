## Analytica Deep Agent

Analytica is a data-analysis agent built around `deepagents.create_deep_agent`.
It accepts CSV/DataFrame data, inspects schema before choosing columns, executes
controlled pandas analysis code, supports read-only SQL-style queries over the
current DataFrame, and can save markdown report artifacts.

## Setup

```bash
make setup
make update
cp .env_example .env
```

Set the LLM key for the provider configured in `llm_config.yaml`, for example:

```bash
GEMINI_API_KEY="..."
```

Important config values can be changed through `.env`:

```bash
ANALYTICA_DEFAULT_DATA_PATH="data/train.csv"
ANALYTICA_ARTIFACT_DIR="artifacts"
ANALYTICA_SQL_TABLE_NAME="data"
ANALYTICA_MAX_SQL_ROWS="100"
ANALYTICA_DEFAULT_CLI_QUERY="Опиши структуру данных и предложи направления анализа."
ANALYTICA_MPLBACKEND="Agg"
ANALYTICA_LLM_PROVIDER="openrouter"
ANALYTICA_LLM_MODEL="openai/gpt-4o-mini"
ANALYTICA_FALLBACK_LLM_PROVIDER=""
ANALYTICA_FALLBACK_LLM_MODEL=""
ANALYTICA_CHECKPOINTER_TYPE="memory"
ANALYTICA_CHECKPOINTER_PATH=".analytica/checkpoints.sqlite"
ANALYTICA_THREAD_ID=""
```

`ANALYTICA_FALLBACK_LLM_PROVIDER` and `ANALYTICA_FALLBACK_LLM_MODEL` are optional.
When set, LangChain will try the fallback model if the primary model call raises
a provider/API error. Leaving them empty keeps the previous single-model behavior.

Persistent agent memory is optional. By default the project keeps the current
safe in-process memory saver:

```bash
ANALYTICA_CHECKPOINTER_TYPE="memory"
```

To persist LangGraph/Deep Agent checkpoints between process restarts:

```bash
ANALYTICA_CHECKPOINTER_TYPE="sqlite"
ANALYTICA_CHECKPOINTER_PATH=".analytica/checkpoints.sqlite"
ANALYTICA_THREAD_ID="demo"
```

Use a stable `ANALYTICA_THREAD_ID` when you want a later run to continue the
same agent thread. If SQLite cannot start, the app returns a clear checkpointer
error instead of silently switching storage modes.

Optional LangSmith tracing stays disabled unless enabled through the environment:

```bash
ANALYTICA_LANGSMITH_TRACING="true"
LANGSMITH_API_KEY="..."
LANGSMITH_PROJECT="analytica-demo"
```

## Run

Streamlit demo:

```bash
make run
```

CLI:

```bash
python3 run.py --csv path/to/file.csv --query "Опиши структуру данных"
```

FastAPI:

```bash
uvicorn main:app --reload
```

Then send either JSON records:

```json
{
  "query": "Покажи топ-5 категорий по метрике",
  "data": [{"category": "A", "value": 10}]
}
```

or a CSV path relative to the project root:

```json
{
  "query": "Через SQL посчитай количество строк",
  "csv_path": "path/to/file.csv"
}
```

## Notebook Demo

Open and run:

```bash
jupyter notebook notebooks/analytica_agent_demo.ipynb
```

The notebook shows CSV loading, schema/profile inspection, SQL-style querying,
agent execution, generated code/SQL, plot rendering, and report artifact output.

## How to Test Before Demo

Run static import/syntax checks:

```bash
python3 -m compileall source app.py main.py run.py
python3 -m json.tool notebooks/analytica_agent_demo.ipynb
```

Run unit tests:

```bash
pytest
```

Smoke-test the CLI with your CSV:

```bash
python3 run.py --csv path/to/file.csv --query "Опиши структуру данных"
```

Launch the Streamlit demo:

```bash
streamlit run app.py
```

Open the notebook demo:

```bash
jupyter notebook notebooks/analytica_agent_demo.ipynb
```

## Architecture

- `source/agent.py` builds the Deep Agent, system prompt, streaming wrapper, structured report schema, and optional LangSmith metadata.
- `source/checkpointing.py` selects the LangGraph checkpointer: in-memory by default, optional SQLite persistence via env.
- `source/skills/registry.py` exposes progressive-disclosure skill metadata and full skill content lookup.
- `source/tools/analytics_tools.py` exposes schema, Python analysis, SQL, visualization, and report tools.
- `source/tools/skill_tools.py` exposes `list_available_skills` and `load_skill` so the agent loads full skill content only on demand.
- `source/func.py` contains controlled code execution and result previews.
- `source/dataframe.py` contains CSV loading, DataFrame profiling, and read-only SQLite querying.
- `source/skills/*/SKILL.md` stores skill metadata (`name`, `description`, `allowed-tools`) plus full instructions for data analysis, CSV/DataFrame work, SQL querying, visualization, reporting, business analysis, and code execution safety.
- `app.py` shows streaming progress when available and falls back to the stable `run_once` path if streaming is unavailable.
- `run.py` and `main.py` keep the CLI and API entrypoints compatible with the existing non-streaming call.

## Output Shape

The legacy markdown answer remains available as `final_answer`. Newer clients can
also read `structured_report`:

```text
summary
key_findings
limitations
artifacts
next_steps
generated_code
loaded_skills
tool_timeline
sql_metadata
```
