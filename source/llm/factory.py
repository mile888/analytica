from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from source.llm.llm_config import load_llm_config, resolve_api_key
from dotenv import load_dotenv
load_dotenv()

def make_llm():
    cfg = load_llm_config()
    d = cfg.defaults
    provider = d.provider
    provider_cfg = cfg.providers[provider]
    model = provider_cfg.model or d.model

    if provider == "gemini":
        api_key = resolve_api_key(cfg)
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY. Put it into .env")

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=d.temperature,
            max_tokens=d.max_tokens,
            timeout=d.timeout,
            max_retries=d.max_retries,
            api_key=api_key,
        )

    if provider == "openai":
        api_key = resolve_api_key(cfg)
        if not api_key:
            raise ValueError("Missing OPEN_API_TOKEN (OpenAI). Put it into .env")

        return ChatOpenAI(
            model=model,
            temperature=d.temperature,
            max_tokens=d.max_tokens,
            timeout=d.timeout,
            max_retries=d.max_retries,
            api_key=api_key,
        )

    if provider == "ollama":
        base_url = provider_cfg.base_url or "http://localhost:11434"
        return ChatOllama(
            model=model,
            temperature=d.temperature,
            base_url=base_url,
        )

    raise ValueError(f"Unsupported provider: {provider}")
