from typing import Dict, Optional, Literal
from pydantic import BaseModel, Field, model_validator
import yaml
import os

from source.config import LLM_CONFIG_PATH, env_str

EngineName = Literal["pandas", "polars", "spark", "auto", ""]

class NodeLLMOverride(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None

class LLMDefaults(BaseModel):
    provider: str = Field(default_factory=lambda: env_str("ANALYTICA_LLM_PROVIDER", ""))
    model: str = Field(default_factory=lambda: env_str("ANALYTICA_LLM_MODEL", ""))
    fallback_provider: str = Field(default_factory=lambda: env_str("ANALYTICA_FALLBACK_LLM_PROVIDER", ""))
    fallback_model: str = Field(default_factory=lambda: env_str("ANALYTICA_FALLBACK_LLM_MODEL", ""))
    temperature: float = Field(default=1.0)
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    max_retries: int = Field(default=2)
    engine: EngineName = Field(default="pandas")


class ProviderConfig(BaseModel):
    api_key_env: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None


class LLMConfig(BaseModel):
    defaults: LLMDefaults
    providers: Dict[str, ProviderConfig]
    node_overrides: Dict[str, NodeLLMOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_provider(self):
        if self.defaults.provider not in self.providers:
            raise ValueError(
                f"Provider '{self.defaults.provider}' not found in config.providers"
            )
        return self


def load_llm_config(path: str | os.PathLike[str] | None = None) -> LLMConfig:
    resolved_path = path or LLM_CONFIG_PATH
    with open(resolved_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = raw.setdefault("defaults", {})
    defaults["provider"] = env_str("ANALYTICA_LLM_PROVIDER", str(defaults.get("provider", "")))
    defaults["model"] = env_str("ANALYTICA_LLM_MODEL", str(defaults.get("model", "")))
    defaults["fallback_provider"] = env_str(
        "ANALYTICA_FALLBACK_LLM_PROVIDER",
        str(defaults.get("fallback_provider", "")),
    )
    defaults["fallback_model"] = env_str(
        "ANALYTICA_FALLBACK_LLM_MODEL",
        str(defaults.get("fallback_model", "")),
    )
    return LLMConfig(**raw)


def resolve_api_key(cfg: LLMConfig, provider_name: Optional[str] = None) -> str:
    prov = provider_name or cfg.defaults.provider
    provider_cfg = cfg.providers.get(prov)
    if not provider_cfg or not provider_cfg.api_key_env:
        return ""
    value = (os.getenv(provider_cfg.api_key_env) or "").strip()
    if value:
        return value

    fallback_envs = {
        "gemini": ("GOOGLE_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "openrouter": ("OPENROUTER_API_KEY",),
        "ollama": ("OLLAMA_API_KEY",),
    }
    for env_name in fallback_envs.get(prov, ()):
        value = (os.getenv(env_name) or "").strip()
        if value:
            return value
    return ""
