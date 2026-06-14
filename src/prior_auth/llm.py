from langchain_anthropic import ChatAnthropic

from prior_auth import config


def get_llm(temperature: float = 0) -> ChatAnthropic:
    """Single factory for the Claude client, so every node configures
    the model the same way (and future changes -- tracing, model swaps --
    happen in one place)."""
    return ChatAnthropic(
        model=config.DEFAULT_MODEL,
        temperature=temperature,
        api_key=config.ANTHROPIC_API_KEY,
    )
