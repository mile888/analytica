from typing import Optional
from source.llm.llm_config import LLMConfig, load_llm_config, resolve_api_key

_config: Optional[LLMConfig] = None

def _get_config() -> LLMConfig:
    global _config
    if _config is None:
        _config = load_llm_config()
    return _config

def make_llm(node_name: str = "general", config: Optional[LLMConfig] = None):
    """
    Create an LLM instance for the given pipeline node.

    Provider and model are read from llm_config.yaml.
    Per-node temperature overrides are applied from node_overrides section.

    node_name: deep_agent | general
    config: optional pre-loaded LLMConfig (uses cached singleton if None)
    """
    cfg = config or _get_config()
    d = cfg.defaults

    override = cfg.node_overrides.get(node_name)
    provider = (override.provider if override and override.provider else None) or d.provider

    p_cfg = cfg.providers.get(provider)
    p_model = getattr(p_cfg, "model", None) if p_cfg else None
    model = (override.model if override and getattr(override, "model", None) else None) or p_model or d.model

    temperature = d.temperature
    if override and override.temperature is not None:
        temperature = override.temperature

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = resolve_api_key(cfg, "gemini")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY. Set it in .env or environment.")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=d.max_tokens,
            timeout=d.timeout,
            max_retries=d.max_retries,
            api_key=api_key,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = resolve_api_key(cfg, "openai")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY or OPEN_API_TOKEN. Set it in .env or environment.")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=d.max_tokens,
            timeout=d.timeout,
            max_retries=d.max_retries,
            api_key=api_key,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        api_key = resolve_api_key(cfg, "openrouter")
        if not api_key:
            raise ValueError("Missing OPENROUTER_API_KEY. Set it in .env or environment.")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=d.max_tokens,
            timeout=d.timeout,
            max_retries=d.max_retries,
            api_key=api_key,
            base_url=getattr(p_cfg, "base_url", "https://openrouter.ai/api/v1"),
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Analytica",
            },
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        ollama_cfg = cfg.providers.get("ollama")
        base_url = getattr(ollama_cfg, "base_url", "http://localhost:11434") if ollama_cfg else "http://localhost:11434"
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
