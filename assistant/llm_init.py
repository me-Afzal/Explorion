"""
llm_init.py
-----------
Centralised LLM factory.

Manager agent    → qwen2.5:14b  (reliable structured output + instruction following)
Research/Analyze → qwen2.5:7b   (strong reasoning for web research)
"""

import os
from langchain_openai import ChatOpenAI


def _get_ollama_base_url() -> str:
    # Explicit override always wins (set in .env or shell)
    if url := os.getenv("OLLAMA_BASE_URL"):
        return url
    # /.dockerenv is created by Docker in every container — reliable detection
    if os.path.exists("/.dockerenv"):
        return "http://host.docker.internal:11434/v1"
    return "http://localhost:11434/v1"


OLLAMA_BASE_URL = _get_ollama_base_url()
OLLAMA_API_KEY  = "ollama"


def _make_ollama_llm(model: str, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        temperature=temperature,
    )


def get_manager_llm() -> ChatOpenAI:
    return _make_ollama_llm("qwen2.5:7b")


def get_agent_llm() -> ChatOpenAI:
    return _make_ollama_llm("qwen2.5:7b")
