from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def env_int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_project_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    """Resolve user/config paths without baking absolute local paths into the app."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


DEFAULT_DATA_PATH = PROJECT_ROOT / env_str("ANALYTICA_DEFAULT_DATA_PATH", "data/train.csv")
SKILL_DIR = PROJECT_ROOT / env_str("ANALYTICA_SKILL_DIR", "source/skills")
ARTIFACT_DIR = PROJECT_ROOT / env_str("ANALYTICA_ARTIFACT_DIR", "artifacts")
MPLCONFIGDIR = Path(env_str("ANALYTICA_MPLCONFIGDIR", "/tmp/analytica-matplotlib"))
MPLBACKEND = env_str("ANALYTICA_MPLBACKEND", "Agg")
LLM_CONFIG_PATH = resolve_project_path(env_str("ANALYTICA_LLM_CONFIG_PATH", "llm_config.yaml"))
OPENROUTER_BASE_URL = env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OLLAMA_BASE_URL = env_str("OLLAMA_BASE_URL", "http://localhost:11434")
ANALYTICA_HTTP_REFERER = env_str("ANALYTICA_HTTP_REFERER", "http://localhost:8501")
ANALYTICA_APP_TITLE = env_str("ANALYTICA_APP_TITLE", "Analytica")

THREAD_PREFIX = env_str("ANALYTICA_THREAD_PREFIX", "analytica")
ANALYTICA_THREAD_ID = env_str("ANALYTICA_THREAD_ID", "")
ANALYTICA_CHECKPOINTER_TYPE = env_str("ANALYTICA_CHECKPOINTER_TYPE", "memory").lower()
ANALYTICA_CHECKPOINTER_PATH = resolve_project_path(env_str("ANALYTICA_CHECKPOINTER_PATH", ".analytica/checkpoints.sqlite"))
SQL_TABLE_NAME = env_str("ANALYTICA_SQL_TABLE_NAME", "data")
DEFAULT_CLI_QUERY = env_str(
    "ANALYTICA_DEFAULT_CLI_QUERY",
    "Опиши структуру данных и предложи направления анализа.",
)
SCHEMA_SUGGESTION_PROMPT = env_str(
    "ANALYTICA_SCHEMA_SUGGESTION_PROMPT",
    "Опиши структуру данных и возможные направления анализа",
)
TOP_N_SUGGESTION_TEMPLATE = env_str(
    "ANALYTICA_TOP_N_SUGGESTION_TEMPLATE",
    "Покажи топ-5 значений `{dimension}` по `{metric}`",
)
BAR_CHART_SUGGESTION_TEMPLATE = env_str(
    "ANALYTICA_BAR_CHART_SUGGESTION_TEMPLATE",
    "Построй bar chart: `{dimension}` по `{metric}`",
)
SQL_TOP_N_SUGGESTION_TEMPLATE = env_str(
    "ANALYTICA_SQL_TOP_N_SUGGESTION_TEMPLATE",
    "Через SQL найди топ-5 `{dimension}` по сумме `{metric}`",
)
MAX_HISTORY_MESSAGES = env_int("ANALYTICA_MAX_HISTORY_MESSAGES", 12, minimum=0, maximum=100)
MAX_TOOL_ROWS = env_int("ANALYTICA_MAX_TOOL_ROWS", 50, minimum=1, maximum=500)
DEFAULT_TOP_N = env_int("ANALYTICA_DEFAULT_TOP_N", 10, minimum=1, maximum=500)
MAX_TOP_N = env_int("ANALYTICA_MAX_TOP_N", 50, minimum=1, maximum=5000)
MAX_SQL_ROWS = env_int("ANALYTICA_MAX_SQL_ROWS", 100, minimum=1, maximum=1000)
LANGSMITH_TRACING_ENABLED = (
    env_bool("ANALYTICA_LANGSMITH_TRACING", False)
    or env_bool("LANGSMITH_TRACING", False)
    or env_bool("LANGCHAIN_TRACING_V2", False)
)
LANGSMITH_PROJECT = env_str("LANGSMITH_PROJECT", "")

ALLOWED_AGGREGATIONS = tuple(
    part.strip()
    for part in env_str("ANALYTICA_ALLOWED_AGGREGATIONS", "sum,mean,median,min,max,count").split(",")
    if part.strip()
)

ALLOWED_CODE_IMPORTS = tuple(
    part.strip()
    for part in env_str("ANALYTICA_ALLOWED_CODE_IMPORTS", "math,statistics,numpy,pandas,matplotlib,matplotlib.pyplot").split(",")
    if part.strip()
)
