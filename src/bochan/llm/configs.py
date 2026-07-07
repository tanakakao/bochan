"""Configuration objects for LLM-assisted bochan workflows.

このモジュールは、LLM provider の設定、LLM に渡すドメイン文脈、
および optimizer 全体で共有する LLM 設定を定義します。
API key そのものは保存・ログ出力しない前提です。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

LLMProvider = Literal["openai", "gemini"] | str

DEFAULT_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


@dataclass
class LLMConfig:
    """LLM provider に依存しない呼び出し設定。

    Args:
        provider: LLM provider 名。初期実装では ``"openai"`` と ``"gemini"`` を想定します。
        model: provider 側の model 名。
        api_key_env: API key を読む環境変数名。未指定なら provider ごとの既定値を使います。
        api_key: 直接渡す API key。notebook / JSON 保存に残りやすいため通常は非推奨です。
        temperature: 生成温度。
        max_output_tokens: 出力 token 上限。
        timeout: provider client に渡す timeout 秒。
        extra_kwargs: provider 固有の追加引数。
    """

    provider: LLMProvider = "openai"
    model: str = "gpt-4.1-mini"
    api_key_env: str | None = None
    api_key: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 4096
    timeout: float | None = 60.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def resolved_api_key_env(self) -> str | None:
        """provider に応じた API key 環境変数名を返す。"""

        if self.api_key_env is not None:
            return self.api_key_env
        return DEFAULT_API_KEY_ENV.get(str(self.provider).lower())

    def safe_dict(self) -> dict[str, Any]:
        """ログや保存に使える secret をマスクした dict を返す。"""

        data = asdict(self)
        if data.get("api_key"):
            data["api_key"] = "***"
        return data


@dataclass
class LLMContextConfig:
    """LLM に渡す任意のドメイン文脈。

    目的の最大化 / 最小化や bounds など既存 config から分かる情報は、
    LLMContextConfig ではなく bochan 側の prompt builder が自動生成します。
    ここでは列名だけでは分からない意味・単位・実験上の注意を補足します。
    """

    variable_names: Sequence[str] | None = None
    target_names: Sequence[str] | None = None
    variable_descriptions: Mapping[str, str] = field(default_factory=dict)
    target_descriptions: Mapping[str, str] = field(default_factory=dict)
    domain_notes: Sequence[str] = field(default_factory=list)
    candidate_policy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalConfig:
    """自然言語 goal を bochan 設定へ変換するための軽量設定。

    初期実装では prompt builder 用の文脈として保持します。ObjectiveConfig や
    AcquisitionConfig の自動生成は、後続の planner 実装でこの設定を使って拡張します。
    """

    text: str
    infer_objective: bool = True
    infer_constraints: bool = True
    infer_acquisition: bool = True
    infer_optimizer: bool = False
    target_aliases: dict[str, str] = field(default_factory=dict)
    default_direction: str | None = None
    require_confirmation: bool = False


@dataclass
class LLMSettings:
    """BayesianOptimizer 全体で共有する LLM 実行文脈。

    ``ModelConfig(model_type="llm_selected")`` と
    ``OptimizeConfig(optimizer="llm_candidate_set")`` の両方から参照される
    共通設定です。個別の ``model_kwargs`` / ``optimizer_kwargs`` に同名キーを
    渡した場合は、個別設定が優先されます。
    """

    goal: str | GoalConfig | None = None
    llm_config: LLMConfig | Mapping[str, Any] | None = None
    llm_context: LLMContextConfig | Mapping[str, Any] | None = None

    # planner / model selection
    planner_response: Any | None = None
    planner_mode: str = "model_config"
    existing_model_config: dict[str, Any] | None = None
    existing_fit_config: dict[str, Any] | None = None

    # candidate generation
    candidate_set: Any | None = None
    n_llm_candidates: int | None = None
    history_summary: dict[str, Any] | None = None
    pending_candidates: Any | None = None
    fallback_sobol: bool | None = None
    seed: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def model_kwargs(self) -> dict[str, Any]:
        """LLM model selector に渡す既定 kwargs を返す。"""

        data: dict[str, Any] = {}
        if self.goal is not None:
            data["goal"] = self.goal
        if self.llm_config is not None:
            data["llm_config"] = self.llm_config
        if self.llm_context is not None:
            data["llm_context"] = self.llm_context
        if self.planner_response is not None:
            data["planner_response"] = self.planner_response
        if self.planner_mode is not None:
            data["mode"] = self.planner_mode
        if self.existing_model_config is not None:
            data["existing_model_config"] = self.existing_model_config
        if self.existing_fit_config is not None:
            data["existing_fit_config"] = self.existing_fit_config
        return data

    def optimizer_kwargs(self) -> dict[str, Any]:
        """LLM candidate-set optimizer に渡す既定 kwargs を返す。"""

        data: dict[str, Any] = {}
        if self.goal is not None:
            data["goal"] = self.goal
        if self.llm_config is not None:
            data["llm_config"] = self.llm_config
        if self.llm_context is not None:
            data["llm_context"] = self.llm_context
        if self.candidate_set is not None:
            data["candidate_set"] = self.candidate_set
        if self.n_llm_candidates is not None:
            data["n_llm_candidates"] = self.n_llm_candidates
        if self.history_summary is not None:
            data["history_summary"] = self.history_summary
        if self.pending_candidates is not None:
            data["pending_candidates"] = self.pending_candidates
        if self.fallback_sobol is not None:
            data["fallback_sobol"] = self.fallback_sobol
        if self.seed is not None:
            data["seed"] = self.seed
        return data

    def safe_dict(self) -> dict[str, Any]:
        """secret をマスクした保存・表示用 dict を返す。"""

        data = asdict(self)
        llm_config = coerce_llm_config(self.llm_config)
        if llm_config is not None:
            data["llm_config"] = llm_config.safe_dict()
        return data


def coerce_llm_config(value: LLMConfig | Mapping[str, Any] | None) -> LLMConfig | None:
    """dict / dataclass / None を :class:`LLMConfig` に正規化する。"""

    if value is None or isinstance(value, LLMConfig):
        return value
    return LLMConfig(**dict(value))


def coerce_llm_context(value: LLMContextConfig | Mapping[str, Any] | None) -> LLMContextConfig:
    """dict / dataclass / None を :class:`LLMContextConfig` に正規化する。"""

    if value is None:
        return LLMContextConfig()
    if isinstance(value, LLMContextConfig):
        return value
    return LLMContextConfig(**dict(value))


def coerce_goal_config(value: GoalConfig | Mapping[str, Any] | str | None) -> GoalConfig | None:
    """str / dict / dataclass / None を :class:`GoalConfig` に正規化する。"""

    if value is None or isinstance(value, GoalConfig):
        return value
    if isinstance(value, str):
        return GoalConfig(text=value)
    return GoalConfig(**dict(value))


def coerce_llm_settings(value: LLMSettings | Mapping[str, Any] | None) -> LLMSettings | None:
    """dict / dataclass / None を :class:`LLMSettings` に正規化する。"""

    if value is None or isinstance(value, LLMSettings):
        return value
    return LLMSettings(**dict(value))
