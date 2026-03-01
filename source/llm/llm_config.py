from typing import Dict, Optional
from pydantic import BaseModel, Field, model_validator
import yaml
import os

class LLMDefaults(BaseModel):
    provider: str = Field(default="gemini")
    model: str = Field(default="gemini-2.5-flash")
    temperature: float = Field(default=1.0)
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    max_retries: int = Field(default=2)


class ProviderConfig(BaseModel):
    api_key_env: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None


class LLMConfig(BaseModel):
    defaults: LLMDefaults
    providers: Dict[str, ProviderConfig]

    @model_validator(mode="after")
    def _validate_provider(self):
        if self.defaults.provider not in self.providers:
            raise ValueError(
                f"Provider '{self.defaults.provider}' not found in config.providers"
            )
        return self


def load_llm_config(path: str = "config.yaml") -> LLMConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return LLMConfig(**raw)


def resolve_api_key(cfg: LLMConfig) -> str:
    provider_cfg = cfg.providers.get(cfg.defaults.provider)
    if not provider_cfg or not provider_cfg.api_key_env:
        return ""
    return (os.getenv(provider_cfg.api_key_env) or "").strip()
