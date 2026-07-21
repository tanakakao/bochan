"""LLM-assisted configuration suggestions for :class:`BayesianOptimizer`.

This module keeps LLM planning separate from model fitting and candidate generation.
It installs a review-first API on the public ``BayesianOptimizer`` class so users can
request all settings together or model, acquisition, and optimizer settings
independently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Literal

from .acquisition_config import AcquisitionConfig, OutcomeConstraintConfig
from .configs import (
    CandidateRepairConfig,
    InputTransformConfig,
    ModelConfig,
    MultiOutputConfig,
    ObjectiveConfig,
)
from .fit_config import FitConfig
from .optimizer_api import OptimizeConfig

SuggestionMode = Literal["all", "model", "acquisition", "optimizer"]
_SECTION_NAMES = ("model", "acquisition", "optimizer")


def _normalize_mode(value: Any) -> SuggestionMode:
    normalized = "".join(character for character in str(value).lower() if character.isalnum())
    if normalized in {"all", "full", "config", "settings"}:
        return "all"
    if normalized in {"model", "modelconfig", "fit", "fitconfig"}:
        return "model"
    if normalized in {"acquisition", "acq", "acquisitionconfig", "acqconfig"}:
        return "acquisition"
    if normalized in {
        "optimizer",
        "optimization",
        "optimize",
        "optimizeconfig",
        "candidateoptimizer",
    }:
        return "optimizer"
    raise ValueError(
        "Unknown LLM suggestion mode. Expected 'all', 'model', "
        "'acquisition', or 'optimizer'."
    )


def _requested_sections(mode: SuggestionMode) -> list[str]:
    return list(_SECTION_NAMES) if mode == "all" else [mode]


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable(asdict(value))
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def _safe_config_repr(config: Any) -> Any:
    if config is None:
        return None
    return _to_jsonable(config)


def _coerce_model_config(value: Any) -> ModelConfig | None:
    if value is None or isinstance(value, ModelConfig):
        return value
    data = dict(value)
    if isinstance(data.get("input_transform_config"), Mapping):
        data["input_transform_config"] = InputTransformConfig(
            **dict(data["input_transform_config"])
        )
    if isinstance(data.get("multi_output_config"), Mapping):
        data["multi_output_config"] = MultiOutputConfig(
            **dict(data["multi_output_config"])
        )
    return ModelConfig(**data)


def _coerce_fit_config(value: Any) -> FitConfig | None:
    if value is None or isinstance(value, FitConfig):
        return value
    return FitConfig(**dict(value))


def _coerce_acquisition_config(value: Any) -> AcquisitionConfig | None:
    if value is None or isinstance(value, AcquisitionConfig):
        return value
    data = dict(value)
    if isinstance(data.get("objective_config"), Mapping):
        data["objective_config"] = ObjectiveConfig(**dict(data["objective_config"]))
    if isinstance(data.get("outcome_constraint_config"), Mapping):
        data["outcome_constraint_config"] = OutcomeConstraintConfig(
            **dict(data["outcome_constraint_config"])
        )
    return AcquisitionConfig(**data)


def _coerce_optimize_config(value: Any) -> OptimizeConfig | None:
    if value is None or isinstance(value, OptimizeConfig):
        return value
    data = dict(value)
    if isinstance(data.get("repair_config"), Mapping):
        data["repair_config"] = CandidateRepairConfig(**dict(data["repair_config"]))
    return OptimizeConfig(**data)


@dataclass
class BayesianOptimizerSuggestion:
    """Typed result returned by ``BayesianOptimizer.suggest_*`` methods."""

    mode: str
    plan: dict[str, Any]
    model_config: ModelConfig | None = None
    fit_config: FitConfig | None = None
    acq_config: AcquisitionConfig | None = None
    opt_config: OptimizeConfig | None = None
    warnings: list[Any] | None = None
    reasoning_summary: str = ""

    def __post_init__(self) -> None:
        self.warnings = list(self.warnings or [])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for review and logging."""

        return {
            "mode": self.mode,
            "plan": _to_jsonable(self.plan),
            "model_config": _safe_config_repr(self.model_config),
            "fit_config": _safe_config_repr(self.fit_config),
            "acq_config": _safe_config_repr(self.acq_config),
            "opt_config": _safe_config_repr(self.opt_config),
            "warnings": _to_jsonable(self.warnings),
            "reasoning_summary": self.reasoning_summary,
        }


def suggestion_from_plan(
    plan: Mapping[str, Any],
    *,
    mode: str,
) -> BayesianOptimizerSuggestion:
    """Convert a serializable planner response to typed public configs."""

    plan_dict = dict(plan)
    return BayesianOptimizerSuggestion(
        mode=mode,
        plan=plan_dict,
        model_config=_coerce_model_config(plan_dict.get("model_config")),
        fit_config=_coerce_fit_config(plan_dict.get("fit_config")),
        acq_config=_coerce_acquisition_config(
            plan_dict.get("acquisition_config") or plan_dict.get("acq_config")
        ),
        opt_config=_coerce_optimize_config(
            plan_dict.get("optimize_config") or plan_dict.get("opt_config")
        ),
        warnings=list(plan_dict.get("warnings") or []),
        reasoning_summary=str(plan_dict.get("reasoning_summary") or ""),
    )


def _section_prompt_map(
    *,
    mode: SuggestionMode,
    prompt: str | None,
    prompts: Mapping[str, str] | None,
    model_prompt: str | None,
    acquisition_prompt: str | None,
    optimizer_prompt: str | None,
) -> dict[str, str]:
    result = {
        str(key): str(value)
        for key, value in dict(prompts or {}).items()
        if value is not None
    }
    if model_prompt is not None:
        result["model"] = str(model_prompt)
    if acquisition_prompt is not None:
        result["acquisition"] = str(acquisition_prompt)
    if optimizer_prompt is not None:
        result["optimizer"] = str(optimizer_prompt)
    if prompt is not None:
        if mode == "all":
            result["overall"] = str(prompt)
        else:
            result[mode] = str(prompt)
    return result


def install_bayesian_optimizer_llm_api(optimizer_cls: type[Any]) -> None:
    """Install the public LLM suggestion methods on ``optimizer_cls`` once."""

    if getattr(optimizer_cls, "_bochan_llm_suggestion_api_installed", False):
        return

    original_fit = optimizer_cls.fit
    original_acquisition = optimizer_cls.acquisition
    original_candidate = optimizer_cls.candidate

    def suggest(
        self: Any,
        mode: str = "all",
        *,
        prompt: str | None = None,
        prompts: Mapping[str, str] | None = None,
        model_prompt: str | None = None,
        acquisition_prompt: str | None = None,
        optimizer_prompt: str | None = None,
        goal: Any | None = None,
        train_X: Any | None = None,
        train_Y: Any | None = None,
        bounds: Any | None = None,
        llm_config: Any | None = None,
        llm_context: Any | None = None,
        planner_response: Any | None = None,
        apply: bool = False,
    ) -> BayesianOptimizerSuggestion:
        """Ask an LLM for all or selected BayesianOptimizer settings.

        ``prompt`` applies to the selected section for single-section modes and is
        treated as an overall instruction in ``mode='all'``. Use ``model_prompt``,
        ``acquisition_prompt``, and ``optimizer_prompt`` to provide independent
        instructions in one all-settings request.
        """

        from bochan.llm import plan_configs

        normalized_mode = _normalize_mode(mode)
        sections = _requested_sections(normalized_mode)
        section_prompts = _section_prompt_map(
            mode=normalized_mode,
            prompt=prompt,
            prompts=prompts,
            model_prompt=model_prompt,
            acquisition_prompt=acquisition_prompt,
            optimizer_prompt=optimizer_prompt,
        )

        settings = getattr(self, "llm_settings", None)
        settings_kwargs = settings.model_kwargs() if settings is not None else {}
        resolved_goal = goal or settings_kwargs.get("goal")
        if resolved_goal is None:
            resolved_goal = prompt or "Configure BayesianOptimizer for the supplied data and objective."

        resolved_train_X = train_X if train_X is not None else getattr(self, "train_X", None)
        resolved_train_Y = train_Y if train_Y is not None else getattr(self, "train_Y", None)
        resolved_bounds = bounds if bounds is not None else getattr(self, "bounds", None)

        plan = plan_configs(
            goal=resolved_goal,
            llm_config=llm_config if llm_config is not None else settings_kwargs.get("llm_config"),
            llm_context=llm_context if llm_context is not None else settings_kwargs.get("llm_context"),
            train_X=resolved_train_X,
            train_Y=resolved_train_Y,
            bounds=resolved_bounds,
            mode="full" if normalized_mode == "all" else normalized_mode,
            planner_response=(
                planner_response
                if planner_response is not None
                else settings_kwargs.get("planner_response")
            ),
            requested_sections=sections,
            section_prompts=section_prompts,
            existing_model_config=_safe_config_repr(getattr(self, "model_config", None)),
            existing_fit_config=_safe_config_repr(getattr(self, "fit_config", None)),
            existing_acquisition_config=_safe_config_repr(getattr(self, "acq_config", None)),
            existing_optimize_config=_safe_config_repr(getattr(self, "opt_config", None)),
        )
        suggestion = suggestion_from_plan(plan, mode=normalized_mode)
        self.last_suggestion = suggestion
        if apply:
            self.apply_suggestion(suggestion)
        return suggestion

    def suggest_all(
        self: Any,
        *,
        prompt: str | None = None,
        model_prompt: str | None = None,
        acquisition_prompt: str | None = None,
        optimizer_prompt: str | None = None,
        **kwargs: Any,
    ) -> BayesianOptimizerSuggestion:
        """Suggest model, acquisition, and optimization settings together."""

        return self.suggest(
            "all",
            prompt=prompt,
            model_prompt=model_prompt,
            acquisition_prompt=acquisition_prompt,
            optimizer_prompt=optimizer_prompt,
            **kwargs,
        )

    def suggest_model(
        self: Any,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> BayesianOptimizerSuggestion:
        """Suggest only ``ModelConfig`` and ``FitConfig``."""

        return self.suggest("model", prompt=prompt, **kwargs)

    def suggest_acquisition(
        self: Any,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> BayesianOptimizerSuggestion:
        """Suggest only ``AcquisitionConfig`` and objective-related settings."""

        return self.suggest("acquisition", prompt=prompt, **kwargs)

    def suggest_optimizer(
        self: Any,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> BayesianOptimizerSuggestion:
        """Suggest only ``OptimizeConfig`` and candidate-search settings."""

        return self.suggest("optimizer", prompt=prompt, **kwargs)

    def apply_suggestion(
        self: Any,
        suggestion: BayesianOptimizerSuggestion | Mapping[str, Any],
        *,
        model_config: bool = True,
        fit_config: bool = True,
        acq_config: bool = True,
        opt_config: bool = True,
    ) -> Any:
        """Apply selected parts of an LLM suggestion to the optimizer defaults."""

        if not isinstance(suggestion, BayesianOptimizerSuggestion):
            suggestion = suggestion_from_plan(dict(suggestion), mode="all")

        model_changed = False
        if model_config and suggestion.model_config is not None:
            merge = getattr(self, "_merge_llm_settings_into_model_config", None)
            self.model_config = (
                merge(suggestion.model_config) if callable(merge) else suggestion.model_config
            )
            model_changed = True
        if fit_config and suggestion.fit_config is not None:
            self.fit_config = suggestion.fit_config
            model_changed = True
        if acq_config and suggestion.acq_config is not None:
            self.acq_config = suggestion.acq_config
        if opt_config and suggestion.opt_config is not None:
            self.opt_config = suggestion.opt_config

        self.last_suggestion = suggestion
        if model_changed and getattr(self, "bundle", None) is not None:
            self._llm_refit_required = True
        return self

    def fit_with_llm_state(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_fit(self, *args, **kwargs)
        self._llm_refit_required = False
        return result

    def acquisition_with_default(
        self: Any,
        acq_config: AcquisitionConfig | None = None,
        *,
        data_context: Any | None = None,
    ) -> Any:
        resolved = acq_config if acq_config is not None else getattr(self, "acq_config", None)
        if resolved is None:
            raise ValueError(
                "acq_config is required. Pass it explicitly or apply an LLM acquisition suggestion first."
            )
        if getattr(self, "_llm_refit_required", False):
            raise RuntimeError(
                "The LLM changed model or fit settings after fitting. Call fit() or refit() before building an acquisition."
            )
        return original_acquisition(self, resolved, data_context=data_context)

    def candidate_with_defaults(
        self: Any,
        acq_config: AcquisitionConfig | None = None,
        opt_config: OptimizeConfig | None = None,
        *,
        data_context: Any | None = None,
        bounds: Any | None = None,
        return_result: bool = False,
    ) -> Any:
        resolved_acq = acq_config if acq_config is not None else getattr(self, "acq_config", None)
        resolved_opt = opt_config if opt_config is not None else getattr(self, "opt_config", None)
        if resolved_acq is None:
            raise ValueError(
                "acq_config is required. Pass it explicitly or apply an LLM acquisition suggestion first."
            )
        if resolved_opt is None:
            raise ValueError(
                "opt_config is required. Pass it explicitly or apply an LLM optimizer suggestion first."
            )
        if getattr(self, "_llm_refit_required", False):
            raise RuntimeError(
                "The LLM changed model or fit settings after fitting. Call fit() or refit() before candidate()."
            )
        return original_candidate(
            self,
            resolved_acq,
            resolved_opt,
            data_context=data_context,
            bounds=bounds,
            return_result=return_result,
        )

    optimizer_cls.suggest = suggest
    optimizer_cls.suggest_all = suggest_all
    optimizer_cls.suggest_model = suggest_model
    optimizer_cls.suggest_acquisition = suggest_acquisition
    optimizer_cls.suggest_optimizer = suggest_optimizer
    optimizer_cls.apply_suggestion = apply_suggestion
    optimizer_cls.fit = fit_with_llm_state
    optimizer_cls.acquisition = acquisition_with_default
    optimizer_cls.candidate = candidate_with_defaults
    optimizer_cls._bochan_llm_suggestion_api_installed = True


__all__ = [
    "BayesianOptimizerSuggestion",
    "SuggestionMode",
    "install_bayesian_optimizer_llm_api",
    "suggestion_from_plan",
]
