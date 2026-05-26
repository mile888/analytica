from source.llm import factory
from source.llm.llm_config import LLMConfig, LLMDefaults, ProviderConfig


class DummyLLM:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.fallbacks = []

    def with_fallbacks(self, fallbacks):
        self.fallbacks = fallbacks
        return self


def test_make_llm_adds_optional_fallback(monkeypatch):
    cfg = LLMConfig(
        defaults=LLMDefaults(
            provider="openai",
            model="primary-model",
            fallback_provider="gemini",
            fallback_model="fallback-model",
        ),
        providers={
            "openai": ProviderConfig(model="primary-model"),
            "gemini": ProviderConfig(model="fallback-model"),
        },
    )

    monkeypatch.setattr(
        factory,
        "_make_provider_llm",
        lambda cfg, provider, model, temperature: DummyLLM(provider, model),
    )

    llm = factory.make_llm("deep_agent", config=cfg)

    assert llm.provider == "openai"
    assert llm.model == "primary-model"
    assert len(llm.fallbacks) == 1
    assert llm.fallbacks[0].provider == "gemini"
    assert llm.fallbacks[0].model == "fallback-model"


def test_make_llm_without_fallback_keeps_primary(monkeypatch):
    cfg = LLMConfig(
        defaults=LLMDefaults(provider="openai", model="primary-model"),
        providers={"openai": ProviderConfig(model="primary-model")},
    )

    monkeypatch.setattr(
        factory,
        "_make_provider_llm",
        lambda cfg, provider, model, temperature: DummyLLM(provider, model),
    )

    llm = factory.make_llm("deep_agent", config=cfg)

    assert llm.provider == "openai"
    assert llm.fallbacks == []
