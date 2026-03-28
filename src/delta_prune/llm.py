"""LLM adapter layer. Supports multiple backends."""

from __future__ import annotations

import subprocess
from typing import Protocol


class LLM(Protocol):
    """Minimal LLM interface: text in, text out."""

    def generate(self, prompt: str) -> str | None: ...


class ClaudeCLI:
    """Use Anthropic Claude via the `claude` CLI (subscription, $0 API cost)."""

    def __init__(self, model: str = "sonnet") -> None:
        self._model = model

    def generate(self, prompt: str) -> str | None:
        try:
            result = subprocess.run(
                ["claude", "-p", "--model", self._model, prompt],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None


class OllamaLLM:
    """Use a local Ollama model."""

    def __init__(self, model: str = "gemma3:27b", timeout: int = 120) -> None:
        self._model = model
        self._timeout = timeout

    def generate(self, prompt: str) -> str | None:
        try:
            import ollama as _ollama

            response = _ollama.generate(model=self._model, prompt=prompt)
            return response.get("response")
        except Exception:
            return None


class OpenAILLM:
    """Use OpenAI API."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model

    def generate(self, prompt: str) -> str | None:
        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception:
            return None
