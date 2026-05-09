from typing import Optional
from source.llm.llm_config import LLMConfig, load_llm_config, resolve_api_key
from source.config import ANALYTICA_APP_TITLE, ANALYTICA_HTTP_REFERER, OLLAMA_BASE_URL, OPENROUTER_BASE_URL

_config: Optional[LLMConfig] = None

def _get_config() -> LLMConfig:
    global _config
    if _config is None:
        _config = load_llm_config()
    return _config


def _node_provider_model(cfg: LLMConfig, node_name: str) -> tuple[str, str, float]:
    d = cfg.defaults
    override = cfg.node_overrides.get(node_name)
    provider = (override.provider if override and override.provider else None) or d.provider

    p_cfg = cfg.providers.get(provider)
    p_model = getattr(p_cfg, "model", None) if p_cfg else None
    model = (override.model if override and getattr(override, "model", None) else None) or p_model or d.model

    temperature = d.temperature
    if override and override.temperature is not None:
        temperature = override.temperature
    return provider, model, temperature


def _fallback_provider_model(cfg: LLMConfig, primary_provider: str, primary_model: str) -> tuple[str, str] | None:
    provider = (cfg.defaults.fallback_provider or "").strip()
    if not provider:
        return None
    p_cfg = cfg.providers.get(provider)
    model = (cfg.defaults.fallback_model or "").strip() or (getattr(p_cfg, "model", None) if p_cfg else "") or cfg.defaults.model
    if provider == primary_provider and model == primary_model:
        return None
    return provider, model


def _make_provider_llm(cfg: LLMConfig, provider: str, model: str, temperature: float):
    d = cfg.defaults
    p_cfg = cfg.providers.get(provider)
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
            base_url=getattr(p_cfg, "base_url", None) or OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": ANALYTICA_HTTP_REFERER,
                "X-Title": ANALYTICA_APP_TITLE,
            },
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        ollama_cfg = cfg.providers.get("ollama")
        base_url = (
            getattr(ollama_cfg, "base_url", None)
            if ollama_cfg
            else None
        ) or OLLAMA_BASE_URL
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def make_llm(node_name: str = "general", config: Optional[LLMConfig] = None):
    """
    Create an LLM instance for the given pipeline node.

    Provider/model are read from llm_config.yaml plus optional env overrides.
    If ANALYTICA_FALLBACK_LLM_PROVIDER is set, the returned Runnable uses
    LangChain fallbacks so transient provider/API failures can try the fallback.
    """
    cfg = config or _get_config()
    provider, model, temperature = _node_provider_model(cfg, node_name)
    primary = _make_provider_llm(cfg, provider, model, temperature)

    fallback = _fallback_provider_model(cfg, provider, model)
    if not fallback:
        return primary

    fallback_provider, fallback_model = fallback
    fallback_llm = _make_provider_llm(cfg, fallback_provider, fallback_model, temperature)
    if hasattr(primary, "with_fallbacks"):
        return primary.with_fallbacks([fallback_llm])
    return primary
