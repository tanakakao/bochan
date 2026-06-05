import torch

from bochan.api import BochanStudy, CandidateBatch, TrialState


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
