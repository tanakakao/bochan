"""Early stopping, generation scheduling, and config defaults for studies.

This module extends :class:`bochan.api.study.BochanStudy` without changing the
ask/tell core.  The public study API accepts config dataclasses or serializable
mappings and supplies practical single-objective regression defaults when the
configs are omitted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from .acquisition_config import AcquisitionConfig
from .configs import (
    CandidateRepairConfig,
    DataContext,
    InputTransformConfig,
    ModelConfig,
    MultiObjectiveConfig,
    MultiOutputConfig,
    ObjectiveConfig,
    OutputConfig,
)
from .fit_config import FitConfig
from .optimizer_api import OptimizeConfig
from .study import (
    BochanStudy as _BaseBochanStudy,
    CandidateBatch,
    StudySnapshot,
    Trial,
    TrialState,
    _scalar_from_row,
    _to_jsonable,
)
from .study_results import install_study_result_api

Direction = Literal["maximize", "minimize"]
TargetMode = Literal["ge", "le", "abs_diff_le"]
ConfigLike = Mapping[str, Any]


@dataclass
class EarlyStoppingConfig:
    """BochanStudy の batch 単位 early stopping 設定。"""

    output_index: int = 0
    direction: Direction = "maximize"
    target: float | None = None
    target_mode: TargetMode = "ge"
    target_tolerance: float = 0.0
    target_patience: int = 1
    no_improvement_patience: int | None = None
    min_delta: float = 0.0
    min_completed_trials: int = 0
    enabled: bool = True


@dataclass
class StopDecision:
    """early stopping の判定結果。"""

    should_stop: bool = False
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationStep:
    """候補生成スケジュールの1区間。"""

    name: str | None = None
    num_trials: int | None = None
    until_completed: int | None = None
    q: int | None = None
    acq_config: AcquisitionConfig | ConfigLike | str | None = None
    opt_config: OptimizeConfig | ConfigLike | None = None
    data_context: DataContext | ConfigLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.acq_config = _coerce_acquisition_config(
            self.acq_config,
            model_config=None,
            use_default=False,
        )
        self.opt_config = _coerce_optimize_config(
            self.opt_config,
            use_default=False,
        )
        self.data_context = _coerce_data_context(
            self.data_context,
            bounds=None,
            use_default=False,
        )
        self.metadata = dict(self.metadata)


@dataclass
class GenerationSchedule:
    """completed trial 数に応じて GenerationStep を切り替える設定。"""

    steps: Sequence[GenerationStep | ConfigLike]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("GenerationSchedule requires at least one step.")
        self.steps = [
            step if isinstance(step, GenerationStep) else GenerationStep(**dict(step))
            for step in self.steps
        ]

    def resolve(self, n_completed: int) -> GenerationStep:
        """現在の completed trial 数に対応する step を返す。"""

        completed = int(n_completed)
        cumulative = 0
        last_step = self.steps[-1]
        for step in self.steps:
            if step.until_completed is not None:
                if completed < int(step.until_completed):
                    return step
                continue
            if step.num_trials is not None:
                cumulative += int(step.num_trials)
                if completed < cumulative:
                    return step
                continue
            return step
        return last_step


GenerationScheduleLike = (
    GenerationSchedule
    | Sequence[GenerationStep | ConfigLike]
    | ConfigLike
    | None
)


class BochanStudy(_BaseBochanStudy):
    """Config defaults, dictionaries, early stopping, and schedules.

    When omitted, the study uses ``ModelConfig()`` (regression/base),
    ``FitConfig()``, ``AcquisitionConfig(name="EI")``, and ``OptimizeConfig()``.
    Every config also accepts a mapping, including nested config mappings.
    """

    def __init__(
        self,
        *args: Any,
        generation_schedule: GenerationScheduleLike = None,
        early_stopping_config: EarlyStoppingConfig | ConfigLike | None = None,
        **kwargs: Any,
    ) -> None:
        bounds = kwargs.get("bounds")
        model_config = _coerce_model_config(
            kwargs.pop("model_config", None),
            use_default=True,
        )
        fit_config = _coerce_fit_config(
            kwargs.pop("fit_config", None),
            use_default=True,
        )
        acq_config = _coerce_acquisition_config(
            kwargs.pop("acq_config", None),
            model_config=model_config,
            use_default=True,
        )
        opt_config = _coerce_optimize_config(
            kwargs.pop("opt_config", None),
            use_default=True,
        )
        data_context = _coerce_data_context(
            kwargs.pop("data_context", None),
            bounds=bounds,
            use_default=True,
        )

        super().__init__(
            *args,
            model_config=model_config,
            fit_config=fit_config,
            acq_config=acq_config,
            opt_config=opt_config,
            data_context=data_context,
            **kwargs,
        )
        self.generation_schedule = _coerce_generation_schedule(generation_schedule)
        self.early_stopping_config = _coerce_early_stopping_config(early_stopping_config)
        self._early_stopping_state: dict[str, Any] = {
            "target_count": 0,
            "no_improvement_count": 0,
            "best_score": None,
        }
        self.stop_decision: StopDecision | None = None

    def ask(
        self,
        q: int | None = None,
        *,
        acq_config: AcquisitionConfig | ConfigLike | str | None = None,
        opt_config: OptimizeConfig | ConfigLike | None = None,
        data_context: DataContext | ConfigLike | None = None,
        mark_running: bool = False,
        return_batch: bool = False,
        fit: bool = True,
        generation_schedule: GenerationScheduleLike = None,
    ) -> Any | CandidateBatch:
        """次候補を生成する。明示値は generation step より優先する。"""

        step = self.current_generation_step(generation_schedule)
        if step is not None:
            if q is None and step.q is not None:
                q = int(step.q)
            if acq_config is None and step.acq_config is not None:
                acq_config = step.acq_config
            if opt_config is None and step.opt_config is not None:
                opt_config = step.opt_config
            if data_context is None and step.data_context is not None:
                data_context = step.data_context

        resolved_acq_config = _coerce_acquisition_config(
            acq_config,
            model_config=self.model_config,
            use_default=False,
        )
        resolved_opt_config = _coerce_optimize_config(
            opt_config,
            use_default=False,
        )
        resolved_data_context = _coerce_data_context(
            data_context,
            bounds=self.bounds,
            use_default=False,
        )

        result = super().ask(
            q=q,
            acq_config=resolved_acq_config,
            opt_config=resolved_opt_config,
            data_context=resolved_data_context,
            mark_running=mark_running,
            return_batch=return_batch,
            fit=fit,
        )
        if step is not None and self.last_candidate_batch is not None:
            self._annotate_generation_step(self.last_candidate_batch.trial_ids, step)
        return result

    def optimize(
        self,
        objective_func: Any,
        *,
        n_trials: int,
        q: int = 1,
        acq_config: AcquisitionConfig | ConfigLike | str | None = None,
        opt_config: OptimizeConfig | ConfigLike | None = None,
        save_path: str | Path | None = None,
        mark_running: bool = False,
        early_stopping_config: EarlyStoppingConfig | ConfigLike | None = None,
        generation_schedule: GenerationScheduleLike = None,
    ) -> "BochanStudy":
        """Python 関数を評価器として ask/tell ループを自動実行する。"""

        if n_trials <= 0:
            return self

        remaining = int(n_trials)
        while remaining > 0:
            step = self.current_generation_step(generation_schedule)
            step_q = step.q if step is not None and step.q is not None else q
            batch_q = min(int(step_q), remaining)
            batch = self.ask(
                q=batch_q,
                acq_config=acq_config,
                opt_config=opt_config,
                mark_running=mark_running,
                return_batch=True,
                generation_schedule=generation_schedule,
            )
            values = objective_func(batch.candidates)
            self.tell(batch, values)
            remaining -= len(batch.trial_ids)

            decision = self.check_early_stop(
                config=early_stopping_config,
                trial_ids=batch.trial_ids,
                update=True,
            )
            if save_path is not None:
                self.save(save_path)
            if decision.should_stop:
                break
        return self

    def current_generation_step(
        self,
        generation_schedule: GenerationScheduleLike = None,
    ) -> GenerationStep | None:
        """現在の completed trial 数に対応する generation step を返す。"""

        schedule = _coerce_generation_schedule(generation_schedule) or self.generation_schedule
        if schedule is None:
            return None
        return schedule.resolve(self.n_completed)

    def check_early_stop(
        self,
        *,
        config: EarlyStoppingConfig | ConfigLike | None = None,
        trial_ids: Sequence[int] | None = None,
        update: bool = True,
    ) -> StopDecision:
        """early stopping 条件を判定する。"""

        cfg = _coerce_early_stopping_config(config) or self.early_stopping_config
        if cfg is None or not cfg.enabled:
            decision = StopDecision(False, None, {"enabled": False})
            self.stop_decision = decision
            return decision

        if self.n_completed < int(cfg.min_completed_trials):
            decision = StopDecision(
                False,
                None,
                {
                    "n_completed": self.n_completed,
                    "min_completed_trials": int(cfg.min_completed_trials),
                },
            )
            self.stop_decision = decision
            return decision

        recent_trials = (
            self._completed_trials_from_ids(trial_ids)
            if trial_ids is not None
            else self.completed_trials()
        )
        recent_values = [
            _scalar_from_row(trial.y, cfg.output_index)
            for trial in recent_trials
            if trial.y is not None
        ]
        all_values = [
            _scalar_from_row(trial.y, cfg.output_index)
            for trial in self.completed_trials()
            if trial.y is not None
        ]
        if not recent_values or not all_values:
            decision = StopDecision(False, None, {"reason": "no_completed_values"})
            self.stop_decision = decision
            return decision

        details: dict[str, Any] = {
            "recent_values": recent_values,
            "n_completed": self.n_completed,
            "output_index": int(cfg.output_index),
        }

        target_hit = _target_hit(recent_values, cfg)
        if cfg.target is not None:
            target_count = int(self._early_stopping_state.get("target_count") or 0)
            target_count = target_count + 1 if target_hit else 0
            if update:
                self._early_stopping_state["target_count"] = target_count
            details.update(
                {
                    "target": float(cfg.target),
                    "target_mode": cfg.target_mode,
                    "target_hit": target_hit,
                    "target_count": target_count,
                    "target_patience": int(cfg.target_patience),
                }
            )
            if target_count >= int(cfg.target_patience):
                decision = StopDecision(True, "target_reached", details)
                self.stop_decision = decision
                return decision

        if cfg.no_improvement_patience is not None:
            best_score = _best_value(all_values, cfg.direction)
            prev_best = self._early_stopping_state.get("best_score")
            improved = prev_best is None or _is_improved(
                best_score,
                float(prev_best),
                direction=cfg.direction,
                min_delta=float(cfg.min_delta),
            )
            no_improvement_count = int(
                self._early_stopping_state.get("no_improvement_count") or 0
            )
            if improved:
                no_improvement_count = 0
                if update:
                    self._early_stopping_state["best_score"] = best_score
            else:
                no_improvement_count += 1
            if update:
                self._early_stopping_state["no_improvement_count"] = no_improvement_count
            details.update(
                {
                    "best_score": best_score,
                    "previous_best_score": prev_best,
                    "improved": improved,
                    "no_improvement_count": no_improvement_count,
                    "no_improvement_patience": int(cfg.no_improvement_patience),
                    "min_delta": float(cfg.min_delta),
                    "direction": cfg.direction,
                }
            )
            if no_improvement_count >= int(cfg.no_improvement_patience):
                decision = StopDecision(True, "no_improvement", details)
                self.stop_decision = decision
                return decision

        decision = StopDecision(False, None, details)
        self.stop_decision = decision
        return decision

    def reset_early_stopping_state(self) -> "BochanStudy":
        """early stopping の counter を初期化する。"""

        self._early_stopping_state = {
            "target_count": 0,
            "no_improvement_count": 0,
            "best_score": None,
        }
        self.stop_decision = None
        return self

    def to_snapshot(self) -> StudySnapshot:
        """JSON 保存用 snapshot を作る。"""

        snapshot = super().to_snapshot()
        snapshot.metadata["generation_schedule"] = _safe_asdict(self.generation_schedule)
        snapshot.metadata["early_stopping_config"] = _safe_asdict(
            self.early_stopping_config
        )
        snapshot.metadata["early_stopping_state"] = _to_jsonable(
            self._early_stopping_state
        )
        snapshot.metadata["stop_decision"] = _safe_asdict(self.stop_decision)
        return snapshot

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        model_config: ModelConfig | ConfigLike | None = None,
        acq_config: AcquisitionConfig | ConfigLike | str | None = None,
        opt_config: OptimizeConfig | ConfigLike | None = None,
        fit_config: FitConfig | ConfigLike | None = None,
        bounds: Any | None = None,
        data_context: DataContext | ConfigLike | None = None,
        n_initial_random: int = 0,
        model_registry: Mapping[Any, Any] | None = None,
        acquisition_registry: Mapping[str, Any] | None = None,
        generation_schedule: GenerationScheduleLike = None,
        early_stopping_config: EarlyStoppingConfig | ConfigLike | None = None,
    ) -> "BochanStudy":
        """save() した JSON から study を復元する。"""

        import json

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        study = cls(
            model_config=model_config,
            acq_config=acq_config,
            opt_config=opt_config,
            fit_config=fit_config,
            bounds=bounds,
            data_context=data_context,
            n_initial_random=n_initial_random,
            model_registry=model_registry,
            acquisition_registry=acquisition_registry,
            generation_schedule=generation_schedule,
            early_stopping_config=early_stopping_config,
            metadata=raw.get("metadata") or {},
        )
        study.next_trial_id = int(raw.get("next_trial_id", 0))
        study.trials = [Trial.from_dict(item) for item in raw.get("trials", [])]
        if study.trials:
            study.next_trial_id = max(
                study.next_trial_id,
                max(trial.trial_id for trial in study.trials) + 1,
            )
        state = (raw.get("metadata") or {}).get("early_stopping_state")
        if isinstance(state, Mapping):
            study._early_stopping_state.update(dict(state))
        return study

    def _completed_trials_from_ids(
        self,
        trial_ids: Sequence[int] | None,
    ) -> list[Trial]:
        if trial_ids is None:
            return self.completed_trials()
        by_id = {trial.trial_id: trial for trial in self.trials}
        trials: list[Trial] = []
        for trial_id in trial_ids:
            trial = by_id.get(int(trial_id))
            if trial is not None and trial.state == TrialState.COMPLETED:
                trials.append(trial)
        return trials

    def _annotate_generation_step(
        self,
        trial_ids: Sequence[int],
        step: GenerationStep,
    ) -> None:
        by_id = {trial.trial_id: trial for trial in self.trials}
        for trial_id in trial_ids:
            trial = by_id.get(int(trial_id))
            if trial is None:
                continue
            if step.name is not None:
                trial.metadata["generation_step"] = step.name
            if step.metadata:
                trial.metadata["generation_step_metadata"] = dict(step.metadata)


def _coerce_input_transform_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return InputTransformConfig(**dict(value))
    return value


def _coerce_fit_config(
    value: FitConfig | ConfigLike | None,
    *,
    use_default: bool,
) -> FitConfig | Any | None:
    if value is None:
        return FitConfig() if use_default else None
    if isinstance(value, Mapping):
        payload = dict(value)
        aliases = {
            "fit_method": "method",
            "fit_optimizer_kwargs": "optimizer_kwargs",
            "fit_beta": "beta",
        }
        for alias, target in aliases.items():
            if alias in payload and target not in payload:
                payload[target] = payload.pop(alias)
        return FitConfig(**payload)
    return value


def _coerce_output_config(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    payload = dict(value)
    if "input_transform_config" in payload:
        payload["input_transform_config"] = _coerce_input_transform_config(
            payload["input_transform_config"]
        )
    if "fit_config" in payload:
        payload["fit_config"] = _coerce_fit_config(
            payload["fit_config"],
            use_default=False,
        )
    return OutputConfig(**payload)


def _coerce_multi_output_config(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    payload = dict(value)
    output_configs = payload.get("output_configs")
    if output_configs is not None:
        payload["output_configs"] = [
            _coerce_output_config(item) for item in output_configs
        ]
    output_fit_configs = payload.get("output_fit_configs")
    if isinstance(output_fit_configs, Mapping):
        payload["output_fit_configs"] = _coerce_fit_config(
            output_fit_configs,
            use_default=False,
        )
    elif isinstance(output_fit_configs, (list, tuple)):
        payload["output_fit_configs"] = [
            _coerce_fit_config(item, use_default=False)
            for item in output_fit_configs
        ]
    return MultiOutputConfig(**payload)


def _coerce_model_config(
    value: ModelConfig | ConfigLike | None,
    *,
    use_default: bool,
) -> ModelConfig | Any | None:
    if value is None:
        return ModelConfig() if use_default else None
    if not isinstance(value, Mapping):
        return value
    payload = dict(value)
    if "input_transform_config" in payload:
        payload["input_transform_config"] = _coerce_input_transform_config(
            payload["input_transform_config"]
        )
    if "multi_output_config" in payload:
        payload["multi_output_config"] = _coerce_multi_output_config(
            payload["multi_output_config"]
        )
    return ModelConfig(**payload)


def _default_acquisition_name(model_config: ModelConfig | Any | None) -> str:
    task_type = str(getattr(model_config, "task_type", "regression")).lower()
    return "NEHVI" if task_type == "multi_objective" else "EI"


def _coerce_acquisition_config(
    value: AcquisitionConfig | ConfigLike | str | None,
    *,
    model_config: ModelConfig | Any | None,
    use_default: bool,
) -> AcquisitionConfig | Any | None:
    if value is None:
        if not use_default:
            return None
        return AcquisitionConfig(name=_default_acquisition_name(model_config))
    if isinstance(value, str):
        return AcquisitionConfig(name=value)
    if not isinstance(value, Mapping):
        return value

    payload = dict(value)
    if "acq_name" in payload and "name" not in payload:
        payload["name"] = payload.pop("acq_name")
    payload.setdefault("name", _default_acquisition_name(model_config))

    objective_aliases = {
        "objective_mode": "mode",
        "objective_output": "output",
        "objective_outputs": "outputs",
        "objective_specs": "specs",
        "objective_directions": "directions",
        "objective_weights": "weights",
        "objective_eq_targets": "eq_targets",
        "objective_direction": "direction",
        "objective_weight": "weight",
        "objective_eq_target": "eq_target",
        "objective_n_w": "n_w",
        "objective_risk_type": "risk_type",
        "objective_alpha": "alpha",
        "objective_maximize": "maximize",
        "objective_aggregate_mean_when_no_risk": (
            "aggregate_mean_when_no_risk"
        ),
        "objective_allow_unexpanded": "allow_unexpanded",
        "objective_utility_values": "utility_values",
        "objective_ordinal_likelihood": "ordinal_likelihood",
    }
    objective_value = payload.get("objective_config")
    objective_payload = (
        dict(objective_value)
        if isinstance(objective_value, Mapping)
        else {}
    )
    for alias, target in objective_aliases.items():
        if alias in payload:
            objective_payload[target] = payload.pop(alias)
    if objective_payload:
        payload["objective_config"] = ObjectiveConfig(**objective_payload)
    elif isinstance(objective_value, Mapping):
        payload["objective_config"] = ObjectiveConfig(**dict(objective_value))
    return AcquisitionConfig(**payload)


def _coerce_repair_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return CandidateRepairConfig(**dict(value))
    return value


def _coerce_optimize_config(
    value: OptimizeConfig | ConfigLike | None,
    *,
    use_default: bool,
) -> OptimizeConfig | Any | None:
    if value is None:
        return OptimizeConfig() if use_default else None
    if not isinstance(value, Mapping):
        return value
    payload = dict(value)
    if "repair_config" in payload:
        payload["repair_config"] = _coerce_repair_config(payload["repair_config"])
    return OptimizeConfig(**payload)


def _coerce_data_context(
    value: DataContext | ConfigLike | None,
    *,
    bounds: Any | None,
    use_default: bool,
) -> DataContext | Any | None:
    if value is None:
        return DataContext(bounds=bounds) if use_default else None
    if isinstance(value, Mapping):
        payload = dict(value)
        if isinstance(payload.get("multi_objective"), Mapping):
            payload["multi_objective"] = MultiObjectiveConfig(
                **dict(payload["multi_objective"])
            )
        if payload.get("bounds") is None and bounds is not None:
            payload["bounds"] = bounds
        return DataContext(**payload)
    if getattr(value, "bounds", None) is None and bounds is not None:
        try:
            return replace(value, bounds=bounds)
        except TypeError:
            pass
    return value


def _coerce_early_stopping_config(
    value: EarlyStoppingConfig | ConfigLike | None,
) -> EarlyStoppingConfig | None:
    if value is None or isinstance(value, EarlyStoppingConfig):
        return value
    if isinstance(value, Mapping):
        return EarlyStoppingConfig(**dict(value))
    raise TypeError(
        "early_stopping_config must be None, a mapping, or EarlyStoppingConfig. "
        f"Got {type(value).__name__}."
    )


def _coerce_generation_schedule(
    schedule: GenerationScheduleLike,
) -> GenerationSchedule | None:
    if schedule is None:
        return None
    if isinstance(schedule, GenerationSchedule):
        return schedule
    if isinstance(schedule, Mapping):
        return GenerationSchedule(**dict(schedule))
    return GenerationSchedule(steps=list(schedule))


def _best_value(values: Sequence[float], direction: Direction) -> float:
    if direction == "minimize":
        return min(float(value) for value in values)
    return max(float(value) for value in values)


def _is_improved(
    value: float,
    best: float,
    *,
    direction: Direction,
    min_delta: float,
) -> bool:
    if direction == "minimize":
        return float(value) < float(best) - float(min_delta)
    return float(value) > float(best) + float(min_delta)


def _target_hit(values: Sequence[float], config: EarlyStoppingConfig) -> bool:
    if config.target is None:
        return False
    target = float(config.target)
    tolerance = float(config.target_tolerance)
    if config.target_mode == "ge":
        return max(float(value) for value in values) >= target + tolerance
    if config.target_mode == "le":
        return min(float(value) for value in values) <= target - tolerance
    if config.target_mode == "abs_diff_le":
        return min(abs(float(value) - target) for value in values) <= tolerance
    raise ValueError(f"Unknown target_mode: {config.target_mode!r}")


def _safe_asdict(value: Any) -> Any:
    if value is None:
        return None
    try:
        return _to_jsonable(asdict(value))
    except Exception:
        return repr(value)


install_study_result_api(BochanStudy)
