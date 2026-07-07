"""Provider abstraction for LLM calls used by bochan.

OpenAI / Gemini などの SDK 差分はこのモジュールに閉じ込めます。
候補生成側は :class:`BaseLLMClient` の ``generate_json`` だけを使います。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .configs import LLMConfig, coerce_llm_config


@dataclass
class LLMResponse:
    """LLM からの応答。"""

    text: str
    raw: Any | None = None
    provider: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None


class BaseLLMClient(ABC):
    """Provider 非依存の最小 LLM client interface。"""

    @abstractmethod
    def generate_json(self, prompt: str, *, schema: dict[str, Any] | None = None) -> LLMResponse:
        """JSON 文字列を生成する。"""

        raise NotImplementedError


def _resolve_api_key(config: LLMConfig) -> str | None:
    if config.api_key:
        return config.api_key
    env_name = config.resolved_api_key_env()
    return os.environ.get(env_name) if env_name else None


def _missing_dependency(provider: str, package: str) -> ImportError:
    return ImportError(
        f"LLM provider {provider!r} requires optional dependency {package!r}. "
        "Install with `pip install -e .[llm]` or install the provider package directly."
    )


class OpenAIClient(BaseLLMClient):
    """OpenAI SDK backed client."""

    def __init__(self, config: LLMConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise _missing_dependency("openai", "openai") from exc

        api_key = _resolve_api_key(config)
        if not api_key:
            env_name = config.resolved_api_key_env() or "OPENAI_API_KEY"
            raise ValueError(f"OpenAI API key was not found. Set {env_name} or pass LLMConfig.api_key.")

        self.config = config
        kwargs: dict[str, Any] = {"api_key": api_key}
        if config.timeout is not None:
            kwargs["timeout"] = config.timeout
        self.client = OpenAI(**kwargs)

    def generate_json(self, prompt: str, *, schema: dict[str, Any] | None = None) -> LLMResponse:
        response_format: dict[str, Any]
        if schema is None:
            response_format = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "bochan_llm_response",
                    "schema": schema,
                    "strict": False,
                },
            }

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            response_format=response_format,
            **self.config.extra_kwargs,
        )
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            raw=response,
            provider="openai",
            model=self.config.model,
            usage=None if usage is None else dict(usage),
        )


class GeminiClient(BaseLLMClient):
    """Google Gemini SDK backed client."""

    def __init__(self, config: LLMConfig) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise _missing_dependency("gemini", "google-genai") from exc

        api_key = _resolve_api_key(config)
        if not api_key:
            env_name = config.resolved_api_key_env() or "GEMINI_API_KEY"
            raise ValueError(f"Gemini API key was not found. Set {env_name} or pass LLMConfig.api_key.")

        self.config = config
        self.client = genai.Client(api_key=api_key)

    def generate_json(self, prompt: str, *, schema: dict[str, Any] | None = None) -> LLMResponse:
        config_payload = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "response_mime_type": "application/json",
        }
        if schema is not None:
            config_payload["response_schema"] = schema
        config_payload.update(self.config.extra_kwargs)

        response = self.client.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config=config_payload,
        )
        text = getattr(response, "text", "") or ""
        return LLMResponse(
            text=text,
            raw=response,
            provider="gemini",
            model=self.config.model,
            usage=None,
        )


def make_llm_client(config: LLMConfig | dict[str, Any] | None) -> BaseLLMClient:
    """LLMConfig から provider client を生成する。"""

    resolved = coerce_llm_config(config)
    if resolved is None:
        raise ValueError("llm_config is required when no explicit candidate_set is supplied.")

    provider = str(resolved.provider).lower()
    if provider == "openai":
        return OpenAIClient(resolved)
    if provider == "gemini":
        return GeminiClient(resolved)
    raise ValueError(f"Unknown LLM provider: {resolved.provider!r}.")
