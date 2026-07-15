from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from bochan.tabular import TabularBayesianOptimizer


class _PredictingBO:
    def __init__(self, *, model, task_type: str, mean: torch.Tensor, variance: torch.Tensor) -> None:
        self.model = model
        self.bundle = SimpleNamespace(
            model=model,
            task_type=task_type,
            metadata={},
        )
        self._mean = mean
        self._variance = variance

    def predict(self, X, *, return_result=False, posterior_kwargs=None, **kwargs):
        assert return_result is True
        return SimpleNamespace(
            posterior=SimpleNamespace(mean=self._mean, variance=self._variance),
            mean=self._mean,
            variance=self._variance,
            task_type=self.bundle.task_type,
            prediction_space="outcome",
            variance_kind="posterior",
        )


def _optimizer(
    *,
    model,
    task_type: str,
    target_names: list[str],
    mean: torch.Tensor,
    variance: torch.Tensor | None = None,
    inverse_target_category_maps=None,
) -> TabularBayesianOptimizer:
    optimizer = object.__new__(TabularBayesianOptimizer)
    optimizer.dataset = SimpleNamespace(
        target_names=target_names,
        feature_names=["x"],
        inverse_target_category_maps=inverse_target_category_maps or {},
    )
    optimizer.model_config = SimpleNamespace(
        task_type=task_type,
        multi_output_config=None,
    )
    optimizer.bo = _PredictingBO(
        model=model,
        task_type=task_type,
        mean=mean,
        variance=torch.zeros_like(mean) if variance is None else variance,
    )
    return optimizer


def test_hybrid_predict_decodes_ordinal_string_labels() -> None:
    class Model:
        task_types = ["regression", "ordinal"]

        def class_probs_list(self, X, output_indices=None, **kwargs):
            assert output_indices == [1]
            return [
                torch.tensor(
                    [
                        [0.80, 0.15, 0.05],
                        [0.05, 0.20, 0.75],
                    ],
                    dtype=X.dtype,
                )
            ]

    optimizer = _optimizer(
        model=Model(),
        task_type="hybrid",
        target_names=["property", "y_ord_str"],
        mean=torch.tensor([[0.2, 0.1], [0.8, 1.9]], dtype=torch.double),
        inverse_target_category_maps={"y_ord_str": {0: "a", 1: "b", 2: "c"}},
    )

    prediction = optimizer.predict(torch.zeros(2, 1, dtype=torch.double))

    assert prediction["y_ord_str_predicted_class_index"].tolist() == [0, 2]
    assert prediction["y_ord_str_predicted_label"].tolist() == ["a", "c"]
    assert prediction["y_ord_str_predicted_probability"].tolist() == pytest.approx([0.8, 0.75])
    assert "property_predicted_label" not in prediction.columns


def test_multiclass_predict_returns_decoded_label_and_confidence() -> None:
    class Model:
        def class_probs(self, X):
            return torch.tensor(
                [
                    [0.10, 0.70, 0.20],
                    [0.60, 0.30, 0.10],
                ],
                dtype=X.dtype,
            )

    optimizer = _optimizer(
        model=Model(),
        task_type="multiclass",
        target_names=["phase"],
        mean=torch.tensor(
            [
                [0.10, 0.70, 0.20],
                [0.60, 0.30, 0.10],
            ],
            dtype=torch.double,
        ),
        inverse_target_category_maps={"phase": {0: "alpha", 1: "beta", 2: "gamma"}},
    )

    prediction = optimizer.predict(torch.zeros(2, 1, dtype=torch.double))

    assert prediction["phase_predicted_class_index"].tolist() == [1, 0]
    assert prediction["phase_predicted_label"].tolist() == ["beta", "alpha"]
    assert prediction["phase_predicted_probability"].tolist() == pytest.approx([0.7, 0.6])


def test_binary_predict_respects_custom_threshold() -> None:
    class Model:
        def probability_posterior(self, X, **kwargs):
            p1 = torch.tensor([[0.40], [0.80]], dtype=X.dtype)
            return SimpleNamespace(mean=p1, variance=p1 * (1.0 - p1))

    optimizer = _optimizer(
        model=Model(),
        task_type="binary",
        target_names=["defect"],
        mean=torch.tensor([[0.40], [0.80]], dtype=torch.double),
        inverse_target_category_maps={"defect": {0: "ok", 1: "ng"}},
    )

    prediction = optimizer.predict(
        torch.zeros(2, 1, dtype=torch.double),
        binary_threshold=0.7,
    )

    assert prediction["defect_predicted_class_index"].tolist() == [0, 1]
    assert prediction["defect_predicted_label"].tolist() == ["ok", "ng"]
    assert prediction["defect_predicted_probability"].tolist() == pytest.approx([0.6, 0.8])


def test_labels_return_type_returns_only_classification_columns() -> None:
    class Model:
        def class_probs(self, X):
            return torch.tensor([[0.2, 0.8]], dtype=X.dtype)

    optimizer = _optimizer(
        model=Model(),
        task_type="ordinal",
        target_names=["rank"],
        mean=torch.tensor([[0.8]], dtype=torch.double),
    )

    prediction = optimizer.predict(
        torch.zeros(1, 1, dtype=torch.double),
        return_type="labels",
    )

    assert list(prediction.columns) == [
        "rank_predicted_class_index",
        "rank_predicted_label",
        "rank_predicted_probability",
    ]
    assert prediction.iloc[0].to_dict() == {
        "rank_predicted_class_index": 1,
        "rank_predicted_label": 1,
        "rank_predicted_probability": pytest.approx(0.8),
    }


def test_regression_predict_keeps_existing_columns() -> None:
    optimizer = _optimizer(
        model=SimpleNamespace(),
        task_type="regression",
        target_names=["property"],
        mean=torch.tensor([[0.25], [0.75]], dtype=torch.double),
    )

    prediction = optimizer.predict(torch.zeros(2, 1, dtype=torch.double))

    assert list(prediction.columns) == ["property_mean", "property_variance"]


def test_dataframe_index_is_preserved_for_prediction_labels() -> None:
    class Model:
        def class_probs(self, X):
            return torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=X.dtype)

    optimizer = _optimizer(
        model=Model(),
        task_type="multiclass",
        target_names=["phase"],
        mean=torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.double),
    )
    optimizer.data_config = SimpleNamespace(
        target_cols=None,
        input_cols=["x"],
        categorical_cols=[],
        target_categorical_cols=None,
        bounds=None,
        dtype=None,
        device=None,
        dropna=True,
        missing_strategy=None,
        continuous_impute_strategy="mean",
        categorical_impute_strategy="mode",
        impute_targets=False,
        impute_random_state=None,
        impute_max_iter=10,
        multiple_impute_sample_posterior=False,
        encode_categories=True,
        category_maps=None,
        target_category_maps=None,
        return_original_categories=True,
    )
    data = pd.DataFrame({"x": [0.0, 1.0]}, index=["row-a", "row-b"])

    prediction = optimizer.predict(data)

    assert prediction.index.tolist() == ["row-a", "row-b"]
    assert prediction["phase_predicted_label"].tolist() == [0, 1]
