"""Early stopping and generation scheduling for :mod:`bochan.api.study`.

This module extends the base :class:`bochan.api.study.BochanStudy` with two
study-level controls:

- early stopping based on target achievement or lack of improvement
- generation schedules that change q / acquisition / optimizer settings over time

The implementation intentionally keeps the original study implementation intact
and layers these controls on top of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .configs import AcquisitionConfig, DataContext, FitConfig, ModelConfig, OptimizeConfig
from .study import (
    BochanStudy as _BaseBochanStudy,
    CandidateBatch,
    StudySnapshot,
    Trial,
    TrialState,
    _scalar_from_row,
    _to_jsonable,
)

Direction = Literal["maximize", "minimize"]
TargetMode = Literal["ge", "le", "abs_diff_le"]


@dataclass
class EarlyStoppingConfig:
    """BochanStudy の batch 単位 early stopping 設定。

    Args:
        output_index: 停止判定に使う出力 index。
        direction: 改善判定で使う方向。``maximize`` なら最大値、``minimize`` なら
            最小値を best として扱う。
        target: 目標値。None の場合は目標到達判定を無効にする。
        target_mode: 目標到達の判定方法。
            ``ge`` は target 以上、``le`` は target 以下、``abs_diff_le`` は
            target との差の絶対値が ``target_tolerance`` 以下。
        target_tolerance: 目標判定の許容幅。``ge`` / ``le`` では余裕幅として、
            ``abs_diff_le`` では絶対誤差の閾値として使う。
        target_patience: 目標到達が何 batch 続いたら停止するか。
        no_improvement_patience: 改善なしが何 batch 続いたら停止するか。
            None の場合は改善停滞による停止を無効にする。
        min_delta: 改善とみなす最小変化量。
        min_completed_trials: この数の COMPLETED trial が溜まるまでは停止判定しない。
        enabled: False の場合は停止判定を無効にする。
    """

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
    """候補生成スケジュールの1区間。

    Args:
        name: 区間名。trial metadata にも保存する。
        num_trials: この step を使う trial 数。複数 step で累積して解決される。
        until_completed: この completed trial 数に到達するまでこの step を使う。
            ``num_trials`` よりも絶対指定に近い。
        q: この step で使う batch size。
        acq_config: この step で使う獲得関数設定。
        opt_config: この step で使う候補点最適化設定。
        data_context: この step で使う DataContext。
        metadata: 任意の補足情報。
    """

    name: str | None = None
    num_trials: int | None = None
    until_completed: int | None = None
    q: int | None = None
    acq_config: AcquisitionConfig | None = None
    opt_config: OptimizeConfig | None = None
    data_context: DataContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationSchedule:
    """completed trial 数に応じて GenerationStep を切り替える設定。"""

    steps: Sequence[GenerationStep]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("GenerationSchedule requires at least one step.")

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


class BochanStudy(_BaseBochanStudy):
    """Early stopping と generation schedule に対応した BochanStudy。"""

    def __init__(
        self,
        *args: Any,
        generation_schedule: GenerationSchedule | Sequence[GenerationStep] | None = None,
        early_stopping_config: EarlyStoppingConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.generation_schedule = _coerce_generation_schedule(generation_schedule)
        self.early_stopping_config = early_stopping_config
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
        acq_config: AcquisitionConfig | None = None,
        opt_config: OptimizeConfig | None = None,
        data_context: DataContext | None = None,
        mark_running: bool = False,
        return_batch: bool = False,
        fit: bool = True,
        generation_schedule: GenerationSchedule | Sequence[GenerationStep] | None = None,
    ) -> Any | CandidateBatch:
        """次候補を生成する。

        明示的な ``q`` / ``acq_config`` / ``opt_config`` / ``data_context`` が
        渡された場合は、それらを schedule より優先します。
        """

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

        result = super().ask(
            q=q,
            acq_config=acq_config,
            opt_config=opt_config,
            data_context=data_context,
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
        acq_config: AcquisitionConfig | None = None,
        opt_config: OptimizeConfig | None = None,
        save_path: str | Path | None = None,
        mark_running: bool = False,
        early_stopping_config: EarlyStoppingConfig | None = None,
        generation_schedule: GenerationSchedule | Sequence[GenerationStep] | None = None,
    ) -> "BochanStudy":
        """Python 関数を評価器として ask/tell ループを自動実行する。

        各 batch の ``tell`` 後に early stopping を判定します。停止した場合は
        ``self.stop_decision`` に理由を保存します。
        """

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
        generation_schedule: GenerationSchedule | Sequence[GenerationStep] | None = None,
    ) -> GenerationStep | None:
        """現在の completed trial 数に対応する generation step を返す。"""

        schedule = _coerce_generation_schedule(generation_schedule) or self.generation_schedule
        if schedule is None:
            return None
        return schedule.resolve(self.n_completed)

    def check_early_stop(
        self,
        *,
        config: EarlyStoppingConfig | None = None,
        trial_ids: Sequence[int] | None = None,
        update: bool = True,
    ) -> StopDecision:
        """early stopping 条件を判定する。

        Args:
            config: 明示的な停止設定。None の場合は study の既定設定を使う。
            trial_ids: 直近 batch の trial ids。None の場合は completed trial 全体を
                直近集合として扱う。
            update: True の場合は patience counter を更新する。
        """

        cfg = config or self.early_stopping_config
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

        recent_trials = self._completed_trials_from_ids(trial_ids) if trial_ids is not None else self.completed_trials()
        recent_values = [_scalar_from_row(trial.y, cfg.output_index) for trial in recent_trials if trial.y is not None]
        all_values = [_scalar_from_row(trial.y, cfg.output_index) for trial in self.completed_trials() if trial.y is not None]
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
            if target_hit:
                target_count += 1
            else:
                target_count = 0
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
            no_improvement_count = int(self._early_stopping_state.get("no_improvement_count") or 0)
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
        snapshot.metadata["early_stopping_config"] = _safe_asdict(self.early_stopping_config)
        snapshot.metadata["early_stopping_state"] = _to_jsonable(self._early_stopping_state)
        snapshot.metadata["stop_decision"] = _safe_asdict(self.stop_decision)
        return snapshot

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        model_config: ModelConfig | None = None,
        acq_config: AcquisitionConfig | None = None,
        opt_config: OptimizeConfig | None = None,
        fit_config: FitConfig | None = None,
        bounds: Any | None = None,
        data_context: DataContext | None = None,
        n_initial_random: int = 0,
        model_registry: Mapping[Any, Any] | None = None,
        acquisition_registry: Mapping[str, Any] | None = None,
        generation_schedule: GenerationSchedule | Sequence[GenerationStep] | None = None,
        early_stopping_config: EarlyStoppingConfig | None = None,
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
            study.next_trial_id = max(study.next_trial_id, max(trial.trial_id for trial in study.trials) + 1)
        state = (raw.get("metadata") or {}).get("early_stopping_state")
        if isinstance(state, Mapping):
            study._early_stopping_state.update(dict(state))
        return study

    def _completed_trials_from_ids(self, trial_ids: Sequence[int] | None) -> list[Trial]:
        if trial_ids is None:
            return self.completed_trials()
        by_id = {trial.trial_id: trial for trial in self.trials}
        trials: list[Trial] = []
        for trial_id in trial_ids:
            trial = by_id.get(int(trial_id))
            if trial is not None and trial.state == TrialState.COMPLETED:
                trials.append(trial)
        return trials

    def _annotate_generation_step(self, trial_ids: Sequence[int], step: GenerationStep) -> None:
        by_id = {trial.trial_id: trial for trial in self.trials}
        for trial_id in trial_ids:
            trial = by_id.get(int(trial_id))
            if trial is None:
                continue
            if step.name is not None:
                trial.metadata["generation_step"] = step.name
            if step.metadata:
                trial.metadata["generation_step_metadata"] = dict(step.metadata)


def _coerce_generation_schedule(
    schedule: GenerationSchedule | Sequence[GenerationStep] | None,
) -> GenerationSchedule | None:
    if schedule is None:
        return None
    if isinstance(schedule, GenerationSchedule):
        return schedule
    return GenerationSchedule(steps=list(schedule))


def _best_value(values: Sequence[float], direction: Direction) -> float:
    if direction == "minimize":
        return min(float(v) for v in values)
    return max(float(v) for v in values)


def _is_improved(value: float, best: float, *, direction: Direction, min_delta: float) -> bool:
    if direction == "minimize":
        return float(value) < float(best) - float(min_delta)
    return float(value) > float(best) + float(min_delta)


def _target_hit(values: Sequence[float], config: EarlyStoppingConfig) -> bool:
    if config.target is None:
        return False
    target = float(config.target)
    tol = float(config.target_tolerance)
    if config.target_mode == "ge":
        return max(float(v) for v in values) >= target + tol
    if config.target_mode == "le":
        return min(float(v) for v in values) <= target - tol
    if config.target_mode == "abs_diff_le":
        return min(abs(float(v) - target) for v in values) <= tol
    raise ValueError(f"Unknown target_mode: {config.target_mode!r}")


def _safe_asdict(value: Any) -> Any:
    if value is None:
        return None
    try:
        return _to_jsonable(asdict(value))
    except Exception:
        return repr(value)
