"""Study-level ask/tell optimization loop for bochan.

このモジュールは、既存の :class:`bochan.api.engine.BayesianOptimizer` を
1 step の候補点生成エンジンとして利用し、その上に実験履歴・pending 管理・
保存/再開・Python 関数の自動評価ループを載せるための軽量 manager を提供します。

設計方針:
    - 既存 API (`ModelConfig`, `AcquisitionConfig`, `OptimizeConfig`,
      `BayesianOptimizer`) をそのまま使う。
    - ask/tell を中核にする。
    - optimize は ask/tell を Python 関数評価用に包む便利関数とする。
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from .configs import (
    AcquisitionConfig,
    CandidateRepairConfig,
    DataContext,
    FitConfig,
    InputTransformConfig,
    ModelConfig,
    MultiOutputConfig,
    ObjectiveConfig,
    OptimizeConfig,
)
from .engine import BayesianOptimizer


TrialStateLike = Literal["CANDIDATE", "RUNNING", "COMPLETED", "FAILED"] | str
StudySuggestMode = Literal["config"] | str


class TrialState(str, Enum):
    """BochanStudy で管理する trial の状態。"""

    CANDIDATE = "CANDIDATE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Trial:
    """1候補点とその評価結果を表す履歴レコード。"""

    trial_id: int
    x: Any
    y: Any | None = None
    state: TrialState = TrialState.CANDIDATE
    acq_value: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON 保存可能な dict に変換する。"""

        return {
            "trial_id": int(self.trial_id),
            "x": _to_jsonable(self.x),
            "y": _to_jsonable(self.y),
            "state": str(self.state.value if isinstance(self.state, TrialState) else self.state),
            "acq_value": _to_jsonable(self.acq_value),
            "metadata": _to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Trial":
        """JSON 由来の dict から Trial を復元する。"""

        return cls(
            trial_id=int(data["trial_id"]),
            x=data.get("x"),
            y=data.get("y"),
            state=TrialState(str(data.get("state", TrialState.CANDIDATE.value))),
            acq_value=data.get("acq_value"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class CandidateBatch:
    """ask() で生成した候補点 batch と trial id の対応。"""

    candidates: Any
    trial_ids: list[int]
    acq_value: Any | None = None
    result: Any | None = None


@dataclass
class StudySnapshot:
    """BochanStudy.save() で保存する JSON スナップショット。"""

    schema_version: int
    next_trial_id: int
    trials: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StudySuggestion:
    """LLM planner が提案した study-level 設定。"""

    mode: str
    plan: dict[str, Any]
    model_config: ModelConfig | None = None
    fit_config: FitConfig | None = None
    acq_config: AcquisitionConfig | None = None
    opt_config: OptimizeConfig | None = None
    warnings: list[Any] = field(default_factory=list)
    reasoning_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON表示・保存向けの dict に変換する。"""

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


class BochanStudy:
    """Optuna / Ax 風に最適化ループを管理する上位 API。

    Args:
        model_config: 既存 API の ModelConfig。
        acq_config: 既定の獲得関数設定。
        opt_config: 既定の候補点最適化設定。
        fit_config: 学習設定。
        bounds: 探索範囲。初期観測がない状態で ask() する場合は必須。
        data_context: 獲得関数に渡す文脈情報。ask() 時に X_baseline,
            Y_baseline, X_pending を補完します。
        n_initial_random: 完了 trial 数がこの値未満の間は bounds からランダム候補を返す。
            0 の場合でも、完了 trial が0件なら初回のみランダム候補を使います。
        model_registry: BayesianOptimizer に渡すモデル registry。
        acquisition_registry: BayesianOptimizer に渡す獲得関数 registry。
        llm_settings: Study 全体で共有する LLM 設定。
        metadata: study に保存する任意メタデータ。

    Notes:
        `save()` は trial 履歴を JSON 保存します。`ModelConfig` 等には callable が含まれる
        場合があるため、継続時は `load(..., model_config=..., acq_config=..., opt_config=...)`
        のように設定を再注入してください。
    """

    def __init__(
        self,
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
        llm_settings: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_config = model_config
        self.acq_config = acq_config
        self.opt_config = opt_config
        self.fit_config = fit_config
        self.bounds = bounds
        self.data_context = data_context or DataContext(bounds=bounds)
        self.n_initial_random = int(n_initial_random)
        self.model_registry = model_registry
        self.acquisition_registry = acquisition_registry
        self.llm_settings = _coerce_llm_settings(llm_settings)
        self.metadata: dict[str, Any] = dict(metadata or {})

        self.trials: list[Trial] = []
        self.next_trial_id: int = 0
        self.optimizer: BayesianOptimizer | None = None
        self.last_candidate_batch: CandidateBatch | None = None
        self.last_suggestion: StudySuggestion | None = None

    # ------------------------------------------------------------------
    # LLM settings / suggestions
    # ------------------------------------------------------------------
    def configure_llm(
        self,
        *,
        goal: Any | None = None,
        llm_config: Any | None = None,
        llm_context: Any | None = None,
        **settings_kwargs: Any,
    ) -> "BochanStudy":
        """Study 全体で共有する LLM 設定を登録する。"""

        from bochan.llm import LLMSettings

        self.llm_settings = LLMSettings(
            goal=goal,
            llm_config=llm_config,
            llm_context=llm_context,
            **settings_kwargs,
        )
        return self

    def suggest(
        self,
        mode: StudySuggestMode = "config",
        *,
        llm_settings: Any | None = None,
        planner_response: Any | None = None,
        apply: bool = False,
    ) -> StudySuggestion:
        """LLMに study-level 設定案を提案させる。

        現時点の対応 mode は ``"config"`` です。完了 trial / pending trial / failed
        trial と現在の設定を要約し、`model_config`, `fit_config`, `acq_config`,
        `opt_config` の候補を返します。
        """

        normalized_mode = str(mode).lower()
        if normalized_mode not in {"config", "model_config", "settings"}:
            raise ValueError("Only suggest(mode='config') is currently supported.")

        from bochan.llm import plan_configs

        settings = _coerce_llm_settings(llm_settings) or self.llm_settings
        settings_kwargs = settings.model_kwargs() if settings is not None else {}
        goal = settings_kwargs.get("goal") or self.metadata.get("goal") or "Suggest bochan study configuration."
        train_X, train_Y = self.completed_data()
        pending_X = self.pending_data()
        plan = plan_configs(
            goal=goal,
            llm_config=settings_kwargs.get("llm_config"),
            llm_context=settings_kwargs.get("llm_context"),
            train_X=train_X,
            train_Y=train_Y,
            bounds=self._optional_bounds(),
            mode="model_config",
            planner_response=planner_response if planner_response is not None else settings_kwargs.get("planner_response"),
            study_summary=self._study_summary(train_X=train_X, train_Y=train_Y, pending_X=pending_X),
            existing_model_config=_safe_config_repr(self.model_config),
            existing_fit_config=_safe_config_repr(self.fit_config),
            existing_acquisition_config=_safe_config_repr(self.acq_config),
            existing_optimize_config=_safe_config_repr(self.opt_config),
        )
        suggestion = _suggestion_from_plan(plan, mode="config")
        self.last_suggestion = suggestion
        if apply:
            self.apply_suggestion(suggestion)
        return suggestion

    def apply_suggestion(
        self,
        suggestion: StudySuggestion | Mapping[str, Any],
        *,
        model_config: bool = True,
        fit_config: bool = True,
        acq_config: bool = True,
        opt_config: bool = True,
    ) -> "BochanStudy":
        """`suggest()` の結果を study の既定 config に反映する。"""

        if not isinstance(suggestion, StudySuggestion):
            suggestion = _suggestion_from_plan(dict(suggestion), mode="config")
        if model_config and suggestion.model_config is not None:
            self.model_config = suggestion.model_config
        if fit_config and suggestion.fit_config is not None:
            self.fit_config = suggestion.fit_config
        if acq_config and suggestion.acq_config is not None:
            self.acq_config = suggestion.acq_config
        if opt_config and suggestion.opt_config is not None:
            self.opt_config = suggestion.opt_config
        self.last_suggestion = suggestion
        return self

    # ------------------------------------------------------------------
    # データ登録
    # ------------------------------------------------------------------
    def add_observations(
        self,
        X: Any,
        Y: Any,
        *,
        metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ) -> "BochanStudy":
        """既存データを COMPLETED trial として追加する。"""

        x_rows = _split_rows(X)
        y_rows = _split_rows_expected(Y, len(x_rows))
        if len(x_rows) != len(y_rows):
            raise ValueError(f"X and Y must have the same number of rows. Got {len(x_rows)} and {len(y_rows)}.")

        metadata_rows = _normalize_metadata_rows(metadata, len(x_rows))
        for x, y, meta in zip(x_rows, y_rows, metadata_rows):
            self.trials.append(
                Trial(
                    trial_id=self._allocate_trial_id(),
                    x=x,
                    y=y,
                    state=TrialState.COMPLETED,
                    metadata=dict(meta),
                )
            )
        return self

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
    ) -> Any | CandidateBatch:
        """次に評価する候補点を生成する。

        `return_batch=True` の場合は `CandidateBatch` を返します。Web アプリや非同期実験で
        trial_id を明示的に扱いたい場合はこちらを推奨します。
        """

        opt_config = self._resolve_opt_config(opt_config=opt_config, q=q)
        q = int(opt_config.q if q is None else q)
        if q <= 0:
            raise ValueError("q must be positive.")

        train_X, train_Y = self.completed_data()
        pending_X = self.pending_data()

        use_random = train_X is None or self.n_completed < max(1, self.n_initial_random)
        if use_random:
            candidates = _sample_random_from_bounds(self._require_bounds(), q=q)
            acq_value = None
            result = None
        else:
            self._check_generation_configs(acq_config=acq_config, opt_config=opt_config)
            context = self._make_data_context(
                data_context=data_context,
                train_X=train_X,
                train_Y=train_Y,
                pending_X=pending_X,
            )
            optimizer = BayesianOptimizer(
                model_config=self.model_config,  # type: ignore[arg-type]
                fit_config=self.fit_config,
                bounds=self.bounds,
                model_registry=self.model_registry,
                acquisition_registry=self.acquisition_registry,
                data_context=context,
                llm_settings=self.llm_settings,
            )
            if fit:
                optimizer.fit(train_X, train_Y)
            else:
                # fit=False はテストや、外部で fit 済み optimizer を差し替える用途を想定。
                optimizer.train_X = train_X
                optimizer.train_Y = train_Y
            result = optimizer.candidate(
                acq_config=acq_config or self.acq_config,  # type: ignore[arg-type]
                opt_config=opt_config,
                data_context=context,
                bounds=self.bounds,
                return_result=True,
            )
            candidates = result.candidates
            acq_value = result.acq_value
            self.optimizer = optimizer

        candidate_rows = _split_rows(candidates)
        if len(candidate_rows) != q:
            # optimize_acqf の返却 shape が q と一致しない場合でも、履歴側は実際の行数を正とする。
            q = len(candidate_rows)

        acq_rows = _split_rows_or_repeat(acq_value, q)
        state = TrialState.RUNNING if mark_running else TrialState.CANDIDATE
        trial_ids: list[int] = []
        for x, one_acq_value in zip(candidate_rows, acq_rows):
            trial_id = self._allocate_trial_id()
            self.trials.append(
                Trial(
                    trial_id=trial_id,
                    x=x,
                    state=state,
                    acq_value=one_acq_value,
                )
            )
            trial_ids.append(trial_id)

        batch = CandidateBatch(candidates=candidates, trial_ids=trial_ids, acq_value=acq_value, result=result)
        self.last_candidate_batch = batch
        return batch if return_batch else candidates

    def tell(
        self,
        candidates_or_trial_ids: Any | None = None,
        values: Any | None = None,
        *,
        trial_ids: Sequence[int] | None = None,
        state: TrialStateLike = TrialState.COMPLETED,
        metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ) -> "BochanStudy":
        """候補点の評価結果を登録する。

        代表的な使い方:
            - `study.tell(candidate_batch, values)`
            - `study.tell(trial_ids=[0, 1], values=Y)`
            - `study.tell(candidates, values)`  # 直前の ask() と同じ順序なら自動対応
        """

        if isinstance(candidates_or_trial_ids, CandidateBatch):
            trial_ids = candidates_or_trial_ids.trial_ids
        elif trial_ids is None and candidates_or_trial_ids is not None:
            if _looks_like_trial_ids(candidates_or_trial_ids):
                trial_ids = [int(v) for v in candidates_or_trial_ids]
            else:
                trial_ids = self._resolve_trial_ids_from_candidates(candidates_or_trial_ids, values)

        if trial_ids is None:
            if self.last_candidate_batch is None:
                raise ValueError("trial_ids are required when there is no previous ask() batch.")
            trial_ids = self.last_candidate_batch.trial_ids

        if values is None:
            raise ValueError("values must be provided.")

        value_rows = _split_rows_expected(values, len(trial_ids))
        if len(value_rows) != len(trial_ids):
            raise ValueError(f"values and trial_ids must have the same length. Got {len(value_rows)} and {len(trial_ids)}.")

        metadata_rows = _normalize_metadata_rows(metadata, len(value_rows))
        target_state = _coerce_state(state)
        by_id = {trial.trial_id: trial for trial in self.trials}
        for trial_id, y, meta in zip(trial_ids, value_rows, metadata_rows):
            if int(trial_id) not in by_id:
                raise KeyError(f"Unknown trial_id: {trial_id}")
            trial = by_id[int(trial_id)]
            trial.y = y
            trial.state = target_state
            trial.metadata.update(dict(meta))
        return self

    def optimize(
        self,
        objective_func: Callable[[Any], Any],
        *,
        n_trials: int,
        q: int = 1,
        acq_config: AcquisitionConfig | None = None,
        opt_config: OptimizeConfig | None = None,
        save_path: str | Path | None = None,
        mark_running: bool = False,
    ) -> "BochanStudy":
        """Python 関数を評価器として ask/tell ループを自動実行する。

        `n_trials` は評価する候補点の総数です。`q > 1` の場合、最後の batch は
        残数に合わせて小さくなります。
        """

        if n_trials <= 0:
            return self
        remaining = int(n_trials)
        while remaining > 0:
            batch_q = min(int(q), remaining)
            batch = self.ask(
                q=batch_q,
                acq_config=acq_config,
                opt_config=opt_config,
                mark_running=mark_running,
                return_batch=True,
            )
            values = objective_func(batch.candidates)
            self.tell(batch, values)
            remaining -= len(batch.trial_ids)
            if save_path is not None:
                self.save(save_path)
        return self

    # ------------------------------------------------------------------
    # 状態管理
    # ------------------------------------------------------------------
    def mark_running(self, trial_ids: Sequence[int]) -> "BochanStudy":
        """指定 trial を RUNNING にする。"""

        return self._set_state(trial_ids, TrialState.RUNNING)

    def mark_failed(
        self,
        trial_ids: Sequence[int],
        *,
        reason: str | None = None,
    ) -> "BochanStudy":
        """指定 trial を FAILED にする。"""

        self._set_state(trial_ids, TrialState.FAILED)
        if reason:
            by_id = {trial.trial_id: trial for trial in self.trials}
            for trial_id in trial_ids:
                by_id[int(trial_id)].metadata["failure_reason"] = reason
        return self

    def _set_state(self, trial_ids: Sequence[int], state: TrialState) -> "BochanStudy":
        by_id = {trial.trial_id: trial for trial in self.trials}
        for trial_id in trial_ids:
            if int(trial_id) not in by_id:
                raise KeyError(f"Unknown trial_id: {trial_id}")
            by_id[int(trial_id)].state = state
        return self

    @property
    def n_completed(self) -> int:
        """COMPLETED trial 数。"""

        return sum(trial.state == TrialState.COMPLETED for trial in self.trials)

    @property
    def n_pending(self) -> int:
        """未完了 trial 数。"""

        return sum(trial.state in {TrialState.CANDIDATE, TrialState.RUNNING} for trial in self.trials)

    def completed_trials(self) -> list[Trial]:
        """COMPLETED trial を返す。"""

        return [trial for trial in self.trials if trial.state == TrialState.COMPLETED]

    def pending_trials(self) -> list[Trial]:
        """CANDIDATE / RUNNING trial を返す。"""

        return [trial for trial in self.trials if trial.state in {TrialState.CANDIDATE, TrialState.RUNNING}]

    def completed_data(self) -> tuple[Any | None, Any | None]:
        """COMPLETED trial から train_X / train_Y を構築する。"""

        trials = self.completed_trials()
        if not trials:
            return None, None
        return _stack_rows([trial.x for trial in trials]), _stack_rows([trial.y for trial in trials])

    def pending_data(self) -> Any | None:
        """未完了 trial の X_pending を構築する。"""

        trials = self.pending_trials()
        if not trials:
            return None
        return _stack_rows([trial.x for trial in trials])

    def trials_dataframe(self) -> Any:
        """履歴を pandas.DataFrame として返す。pandas が無い場合は list[dict] を返す。"""

        rows = [trial.to_dict() for trial in self.trials]
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except Exception:
            return rows

    def best_trials(
        self,
        *,
        top_k: int = 1,
        output_index: int = 0,
        direction: Literal["maximize", "minimize"] = "maximize",
    ) -> list[Trial]:
        """単一出力を基準に上位 trial を返す。"""

        completed = [trial for trial in self.completed_trials() if trial.y is not None]
        reverse = direction == "maximize"
        return sorted(completed, key=lambda trial: _scalar_from_row(trial.y, output_index), reverse=reverse)[: int(top_k)]

    # ------------------------------------------------------------------
    # 保存 / 復元
    # ------------------------------------------------------------------
    def to_snapshot(self) -> StudySnapshot:
        """JSON 保存用 snapshot を作る。"""

        meta = dict(self.metadata)
        meta.setdefault("model_config", _safe_config_repr(self.model_config))
        meta.setdefault("acq_config", _safe_config_repr(self.acq_config))
        meta.setdefault("opt_config", _safe_config_repr(self.opt_config))
        if self.llm_settings is not None:
            meta.setdefault("llm_settings", _safe_llm_settings_repr(self.llm_settings))
        if self.last_suggestion is not None:
            meta.setdefault("last_suggestion", self.last_suggestion.to_dict())
        return StudySnapshot(
            schema_version=1,
            next_trial_id=int(self.next_trial_id),
            trials=[trial.to_dict() for trial in self.trials],
            metadata=meta,
        )

    def save(self, path: str | Path) -> "BochanStudy":
        """trial 履歴を JSON 保存する。"""

        snapshot = self.to_snapshot()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
        return self

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
        llm_settings: Any | None = None,
    ) -> "BochanStudy":
        """save() した JSON から study を復元する。"""

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
            llm_settings=llm_settings,
            metadata=raw.get("metadata") or {},
        )
        study.next_trial_id = int(raw.get("next_trial_id", 0))
        study.trials = [Trial.from_dict(item) for item in raw.get("trials", [])]
        if study.trials:
            study.next_trial_id = max(study.next_trial_id, max(trial.trial_id for trial in study.trials) + 1)
        return study

    # ------------------------------------------------------------------
    # 内部 helper
    # ------------------------------------------------------------------
    def _allocate_trial_id(self) -> int:
        trial_id = self.next_trial_id
        self.next_trial_id += 1
        return trial_id

    def _optional_bounds(self) -> Any | None:
        if self.bounds is not None:
            return self.bounds
        if self.data_context is not None and self.data_context.bounds is not None:
            return self.data_context.bounds
        return None

    def _require_bounds(self) -> Any:
        bounds = self._optional_bounds()
        if bounds is not None:
            return bounds
        raise RuntimeError("bounds are required for initial random candidate generation.")

    def _resolve_opt_config(self, *, opt_config: OptimizeConfig | None, q: int | None) -> OptimizeConfig:
        config = opt_config or self.opt_config
        if config is None:
            if q is None:
                q = 1
            return OptimizeConfig(q=int(q))
        if q is None or int(config.q) == int(q):
            return config
        return replace(config, q=int(q))

    def _check_generation_configs(
        self,
        *,
        acq_config: AcquisitionConfig | None,
        opt_config: OptimizeConfig | None,
    ) -> None:
        if self.model_config is None:
            raise RuntimeError("model_config is required after initial random trials.")
        if acq_config is None and self.acq_config is None:
            raise RuntimeError("acq_config is required after initial random trials.")
        if opt_config is None and self.opt_config is None:
            raise RuntimeError("opt_config is required after initial random trials.")

    def _make_data_context(
        self,
        *,
        data_context: DataContext | None,
        train_X: Any,
        train_Y: Any,
        pending_X: Any | None,
    ) -> DataContext:
        base = data_context or self.data_context or DataContext(bounds=self.bounds)
        context = replace(base)
        if context.bounds is None:
            context.bounds = self.bounds
        if context.X_baseline is None:
            context.X_baseline = train_X
        if context.Y_baseline is None:
            context.Y_baseline = train_Y
        if context.mc_points is None:
            context.mc_points = train_X
        if context.X_pending is None:
            context.X_pending = pending_X
        return context

    def _study_summary(self, *, train_X: Any | None, train_Y: Any | None, pending_X: Any | None) -> dict[str, Any]:
        completed = self.completed_trials()
        pending = self.pending_trials()
        failed = [trial for trial in self.trials if trial.state == TrialState.FAILED]
        return {
            "n_trials": len(self.trials),
            "n_completed": len(completed),
            "n_pending": len(pending),
            "n_failed": len(failed),
            "completed_X_shape": _shape_of(train_X),
            "completed_Y_shape": _shape_of(train_Y),
            "pending_X_shape": _shape_of(pending_X),
            "recent_completed": [trial.to_dict() for trial in completed[-5:]],
            "recent_pending": [trial.to_dict() for trial in pending[-5:]],
            "recent_failed": [trial.to_dict() for trial in failed[-5:]],
            "metadata": _to_jsonable(self.metadata),
        }

    def _resolve_trial_ids_from_candidates(self, candidates: Any, values: Any | None) -> list[int]:
        candidate_rows = _split_rows(candidates)
        if self.last_candidate_batch is not None and len(self.last_candidate_batch.trial_ids) == len(candidate_rows):
            return list(self.last_candidate_batch.trial_ids)
        value_len = 0 if values is None else len(_split_rows(values))
        pending = self.pending_trials()
        if len(pending) == len(candidate_rows) == value_len:
            return [trial.trial_id for trial in pending]
        matched: list[int] = []
        for row in candidate_rows:
            trial = self._find_pending_by_x(row)
            matched.append(trial.trial_id)
        return matched

    def _find_pending_by_x(self, row: Any) -> Trial:
        for trial in self.pending_trials():
            if _row_equal(trial.x, row):
                return trial
        raise ValueError("Could not match candidates to pending trials. Pass trial_ids explicitly.")


# ----------------------------------------------------------------------
# Suggestion/config coercion helpers
# ----------------------------------------------------------------------
def _coerce_llm_settings(value: Any | None) -> Any | None:
    if value is None:
        return None
    from bochan.llm.configs import coerce_llm_settings

    return coerce_llm_settings(value)


def _suggestion_from_plan(plan: Mapping[str, Any], *, mode: str) -> StudySuggestion:
    plan_dict = dict(plan)
    return StudySuggestion(
        mode=mode,
        plan=plan_dict,
        model_config=_coerce_model_config(plan_dict.get("model_config")),
        fit_config=_coerce_fit_config(plan_dict.get("fit_config")),
        acq_config=_coerce_acq_config(plan_dict.get("acquisition_config") or plan_dict.get("acq_config")),
        opt_config=_coerce_opt_config(plan_dict.get("optimize_config") or plan_dict.get("opt_config")),
        warnings=list(plan_dict.get("warnings") or []),
        reasoning_summary=str(plan_dict.get("reasoning_summary") or ""),
    )


def _coerce_model_config(value: Any) -> ModelConfig | None:
    if value is None or isinstance(value, ModelConfig):
        return value
    data = dict(value)
    if isinstance(data.get("input_transform_config"), Mapping):
        data["input_transform_config"] = InputTransformConfig(**dict(data["input_transform_config"]))
    if isinstance(data.get("multi_output_config"), Mapping):
        data["multi_output_config"] = MultiOutputConfig(**dict(data["multi_output_config"]))
    return ModelConfig(**data)


def _coerce_fit_config(value: Any) -> FitConfig | None:
    if value is None or isinstance(value, FitConfig):
        return value
    return FitConfig(**dict(value))


def _coerce_acq_config(value: Any) -> AcquisitionConfig | None:
    if value is None or isinstance(value, AcquisitionConfig):
        return value
    data = dict(value)
    if isinstance(data.get("objective_config"), Mapping):
        data["objective_config"] = ObjectiveConfig(**dict(data["objective_config"]))
    return AcquisitionConfig(**data)


def _coerce_opt_config(value: Any) -> OptimizeConfig | None:
    if value is None or isinstance(value, OptimizeConfig):
        return value
    data = dict(value)
    if isinstance(data.get("repair_config"), Mapping):
        data["repair_config"] = CandidateRepairConfig(**dict(data["repair_config"]))
    return OptimizeConfig(**data)


# ----------------------------------------------------------------------
# 汎用 data helper
# ----------------------------------------------------------------------
def _coerce_state(state: TrialStateLike) -> TrialState:
    if isinstance(state, TrialState):
        return state
    return TrialState(str(state))


def _looks_like_trial_ids(value: Any) -> bool:
    if isinstance(value, CandidateBatch):
        return True
    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Sequence):
        return False
    if len(value) == 0:
        return True
    return all(isinstance(v, int) for v in value)


def _normalize_metadata_rows(
    metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    n: int,
) -> list[Mapping[str, Any]]:
    if metadata is None:
        return [{} for _ in range(n)]
    if isinstance(metadata, Mapping):
        return [metadata for _ in range(n)]
    if len(metadata) != n:
        raise ValueError(f"metadata length must be {n}. Got {len(metadata)}.")
    return list(metadata)


def _shape_of(value: Any) -> list[int] | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            return [len(value)]
        except Exception:
            return None
    return [int(v) for v in shape]


def _split_rows(data: Any) -> list[Any]:
    """Tensor / ndarray / DataFrame / list を行ごとの list に分解する。"""

    if data is None:
        return []

    try:
        import torch

        if isinstance(data, torch.Tensor):
            if data.ndim == 0:
                return [data.reshape(1, 1)]
            if data.ndim == 1:
                return [data]
            return [data[i : i + 1] for i in range(data.shape[0])]
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(data, np.ndarray):
            if data.ndim == 0:
                return [data.reshape(1, 1)]
            if data.ndim == 1:
                return [data]
            return [data[i : i + 1] for i in range(data.shape[0])]
    except Exception:
        pass

    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return [data.iloc[[i]].copy() for i in range(len(data))]
        if isinstance(data, pd.Series):
            return [data.to_frame().T]
    except Exception:
        pass

    if isinstance(data, Mapping):
        return [dict(data)]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return list(data)
    return [data]


def _split_rows_expected(data: Any, expected: int) -> list[Any]:
    """期待行数を考慮して data を行分解する。"""

    if data is None:
        return []

    try:
        import torch

        if isinstance(data, torch.Tensor):
            if data.ndim == 1 and expected > 1 and int(data.shape[0]) == int(expected):
                return [data[i : i + 1] for i in range(expected)]
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(data, np.ndarray):
            if data.ndim == 1 and expected > 1 and int(data.shape[0]) == int(expected):
                return [data[i : i + 1] for i in range(expected)]
    except Exception:
        pass

    return _split_rows(data)


def _split_rows_or_repeat(data: Any, n: int) -> list[Any]:
    if data is None:
        return [None for _ in range(n)]
    rows = _split_rows_expected(data, n)
    if len(rows) == n:
        return rows
    if len(rows) == 1:
        return rows * n
    return [data for _ in range(n)]


def _stack_rows(rows: Sequence[Any]) -> Any | None:
    """行 list を元の data family に近い形式で結合する。"""

    rows = [row for row in rows if row is not None]
    if not rows:
        return None
    first = rows[0]

    try:
        import torch

        if isinstance(first, torch.Tensor):
            tensors = [_ensure_2d_tensor(row) for row in rows]
            return torch.cat(tensors, dim=0)
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(first, np.ndarray):
            arrays = [_ensure_2d_array(row) for row in rows]
            return np.concatenate(arrays, axis=0)
    except Exception:
        pass

    try:
        import pandas as pd

        if isinstance(first, (pd.DataFrame, pd.Series)):
            frames = [row if isinstance(row, pd.DataFrame) else row.to_frame().T for row in rows]
            return pd.concat(frames, axis=0, ignore_index=True)
    except Exception:
        pass

    if isinstance(first, Mapping):
        try:
            import pandas as pd

            return pd.DataFrame(list(rows))
        except Exception:
            return list(rows)

    return list(rows)


def _ensure_2d_tensor(value: Any) -> Any:
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.double)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor


def _ensure_2d_array(value: Any) -> Any:
    import numpy as np

    arr = value if isinstance(value, np.ndarray) else np.asarray(value)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _sample_random_from_bounds(bounds: Any, *, q: int) -> Any:
    """bounds から一様ランダム候補を生成する。"""

    try:
        import torch

        if isinstance(bounds, torch.Tensor):
            if bounds.shape[0] != 2:
                raise ValueError("bounds must have shape 2 x d.")
            lower, upper = bounds[0], bounds[1]
            rand = torch.rand((q, lower.numel()), dtype=bounds.dtype, device=bounds.device)
            return lower + (upper - lower) * rand
    except Exception:
        pass

    try:
        import numpy as np

        arr = np.asarray(bounds)
        if arr.ndim == 2 and arr.shape[0] == 2:
            lower, upper = arr[0], arr[1]
            return lower + (upper - lower) * np.random.random((q, lower.size))
    except Exception:
        pass

    raise TypeError("Random initial candidates currently require torch.Tensor or numpy bounds with shape 2 x d.")


def _row_equal(a: Any, b: Any) -> bool:
    try:
        import torch

        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return bool(torch.allclose(_ensure_2d_tensor(a), _ensure_2d_tensor(b)))
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return bool(np.allclose(_ensure_2d_array(a), _ensure_2d_array(b)))
    except Exception:
        pass

    try:
        import pandas as pd

        if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
            return bool(a.reset_index(drop=True).equals(b.reset_index(drop=True)))
    except Exception:
        pass

    return a == b


def _scalar_from_row(row: Any, output_index: int) -> float:
    try:
        import torch

        if isinstance(row, torch.Tensor):
            flat = row.detach().reshape(-1)
            return float(flat[int(output_index)].item())
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(row, np.ndarray):
            return float(np.asarray(row).reshape(-1)[int(output_index)])
    except Exception:
        pass

    try:
        import pandas as pd

        if isinstance(row, (pd.DataFrame, pd.Series)):
            values = row.to_numpy().reshape(-1)
            return float(values[int(output_index)])
    except Exception:
        pass

    if isinstance(row, Mapping):
        values = list(row.values())
        return float(values[int(output_index)])
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        return float(row[int(output_index)])
    return float(row)


def _to_jsonable(value: Any) -> Any:
    """torch / numpy / pandas / dataclass を JSON 互換値へ変換する。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_jsonable(v) for v in value]

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, pd.Series):
            return value.to_dict()
    except Exception:
        pass

    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable(asdict(value))
    return repr(value)


def _safe_config_repr(config: Any) -> Any:
    if config is None:
        return None
    try:
        return _to_jsonable(asdict(config))
    except Exception:
        return repr(config)


def _safe_llm_settings_repr(settings: Any) -> Any:
    if settings is None:
        return None
    if hasattr(settings, "safe_dict"):
        return _to_jsonable(settings.safe_dict())
    return _safe_config_repr(settings)
