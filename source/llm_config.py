import os
from dataclasses import dataclass
from typing import Any, Optional, Literal, Dict
import yaml

Provider = Literal["gemini", "openai", "ollama"]
@dataclass(frozen=True)
class LLMDefaults:
    provider: Provider
    model: str
    temperature: float
    max_tokens: Optional[int]
    timeout: Optional[float]
    max_retries: int


@dataclass(frozen=True)
class LLMConfig:
    defaults: LLMDefaults
    providers: Dict[str, Dict[str, Any]]


def _as_optional_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if v == "":
        return None
    return int(v)


def _as_optional_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if v == "":
        return None
    return float(v)


def load_llm_config(path: str = "llm_config.yaml") -> LLMConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    d = raw.get("defaults", {}) or {}
    provider = (d.get("provider") or "gemini").strip().lower()
    if provider not in ("gemini", "openai", "ollama"):
        provider = "gemini"

    defaults = LLMDefaults(
        provider=provider,
        model=str(d.get("model") or "gemini-2.5-flash"),
        temperature=float(d.get("temperature") if d.get("temperature") is not None else 1.0),
        max_tokens=_as_optional_int(d.get("max_tokens")),
        timeout=_as_optional_float(d.get("timeout")),
        max_retries=int(d.get("max_retries") or 2),
    )

    providers = raw.get("providers", {}) or {}
    return LLMConfig(defaults=defaults, providers=providers)


def resolve_api_key(providers_cfg: Dict[str, Dict[str, Any]], provider: str) -> str:
    cfg = providers_cfg.get(provider, {}) or {}
    env_name = (cfg.get("api_key_env") or "").strip()
    if not env_name:
        return ""
    return (os.getenv(env_name) or "").strip()
