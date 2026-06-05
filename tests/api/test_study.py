import torch

from bochan.api import (
    AcquisitionConfig,
    BochanStudy,
    CandidateBatch,
    EarlyStoppingConfig,
    GenerationSchedule,
    GenerationStep,
    OptimizeConfig,
    TrialState,
)


def test_study_ask_tell_with_random_initial_candidates():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    study = BochanStudy(bounds=bounds, n_initial_random=10)

    batch = study.ask(q=2, return_batch=True)

    assert isinstance(batch, CandidateBatch)
    assert batch.candidates.shape == (2, 2)
    assert len(batch.trial_ids) == 2
    assert study.n_pending == 2

    values = batch.candidates.sum(dim=-1)
    study.tell(batch, values)

    train_X, train_Y = study.completed_data()
    assert train_X.shape == (2, 2)
    assert train_Y.shape == (2, 1)
    assert study.n_completed == 2
    assert study.n_pending == 0


def test_study_optimize_wraps_ask_tell_loop():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    study = BochanStudy(bounds=bounds, n_initial_random=10)

    study.optimize(lambda X: X.sum(dim=-1), n_trials=5, q=2)

    train_X, train_Y = study.completed_data()
    assert study.n_completed == 5
    assert train_X.shape == (5, 2)
    assert train_Y.shape == (5, 1)


def test_study_save_and_load_trial_history(tmp_path):
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    study = BochanStudy(bounds=bounds, n_initial_random=10)
    batch = study.ask(q=1, return_batch=True)
    study.tell(batch, torch.tensor([1.0], dtype=torch.double))

    path = tmp_path / "study.json"
    study.save(path)

    loaded = BochanStudy.load(path, bounds=bounds, n_initial_random=10)

    assert loaded.n_completed == 1
    assert loaded.trials[0].state == TrialState.COMPLETED
    train_X, train_Y = loaded.completed_data()
    # JSON から復元した値は list ベースになる。履歴の継続確認を主眼にする。
    assert train_X is not None
    assert train_Y is not None


def test_study_early_stopping_by_target_reached():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    study = BochanStudy(
        bounds=bounds,
        n_initial_random=10,
        early_stopping_config=EarlyStoppingConfig(
            target=0.5,
            target_mode="ge",
            target_patience=1,
            direction="maximize",
        ),
    )

    study.optimize(lambda X: torch.ones(X.shape[0], dtype=torch.double), n_trials=10, q=2)

    assert study.stop_decision is not None
    assert study.stop_decision.should_stop
    assert study.stop_decision.reason == "target_reached"
    assert study.n_completed == 2


def test_study_early_stopping_by_no_improvement():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    study = BochanStudy(
        bounds=bounds,
        n_initial_random=10,
        early_stopping_config=EarlyStoppingConfig(
            no_improvement_patience=2,
            min_delta=0.01,
            direction="maximize",
        ),
    )

    study.optimize(lambda X: torch.zeros(X.shape[0], dtype=torch.double), n_trials=10, q=1)

    assert study.stop_decision is not None
    assert study.stop_decision.should_stop
    assert study.stop_decision.reason == "no_improvement"
    # 1回目は best 更新、2回連続で改善なしになった時点で停止する。
    assert study.n_completed == 3


def test_study_generation_schedule_changes_q_and_metadata():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    schedule = GenerationSchedule(
        steps=[
            GenerationStep(
                name="explore",
                num_trials=4,
                q=3,
                acq_config=AcquisitionConfig(name="UCB", acqf_kwargs={"beta": 4.0}),
                opt_config=OptimizeConfig(q=3),
            ),
            GenerationStep(
                name="exploit",
                q=1,
                acq_config=AcquisitionConfig(name="EI"),
                opt_config=OptimizeConfig(q=1),
            ),
        ]
    )
    study = BochanStudy(bounds=bounds, n_initial_random=10, generation_schedule=schedule)

    first = study.ask(return_batch=True)
    study.tell(first, torch.zeros(len(first.trial_ids), dtype=torch.double))
    second = study.ask(return_batch=True)

    assert len(first.trial_ids) == 3
    assert len(second.trial_ids) == 1
    assert all(study.trials[i].metadata["generation_step"] == "explore" for i in first.trial_ids)
    assert all(study.trials[i].metadata["generation_step"] == "exploit" for i in second.trial_ids)
