from __future__ import annotations

from typing import Optional, Sequence, Any

import torch
from torch import Tensor
from torch.nn import ModuleList

from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior

from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal


class MultiOutputOrdinalModel(Model):
    """
    independent な single-output OrdinalGPModel 群を
    1 つの multi-output model として扱うラッパ。

    Notes:
        - posterior(X) は latent f の joint posterior を返す。
        - class_probs / predict_class / expected_utility も提供する。
        - この wrapper の posterior(X) は raw-space の X を受け取る。
          そのため、この wrapper の train_inputs も raw-space として公開する。
        - 各 submodel 内部では input_transform が適用される。
    """

    def __init__(
        self,
        *models: Model,
        validate_same_train_inputs: bool = True,
    ) -> None:
        super().__init__()

        if len(models) == 0:
            raise ValueError("At least one submodel must be provided.")

        self.models = ModuleList(models)

        for i, model in enumerate(self.models):
            if getattr(model, "num_outputs", None) != 1:
                raise ValueError(
                    f"Submodel {i} must be single-output, "
                    f"got num_outputs={getattr(model, 'num_outputs', None)}."
                )

        first_cat_dims = list(getattr(self.models[0], "cat_dims", []))
        same_cat_dims = all(
            list(getattr(m, "cat_dims", [])) == first_cat_dims
            for m in self.models
        )
        self.cat_dims = first_cat_dims if same_cat_dims else []

        if validate_same_train_inputs:
            self._validate_same_train_inputs()

    # ---------------------------------------------------------------------
    # Basic BoTorch-style properties
    # ---------------------------------------------------------------------
    @property
    def num_outputs(self) -> int:
        return len(self.models)

    @property
    def batch_shape(self) -> torch.Size:
        batch_shape = getattr(self.models[0], "batch_shape", torch.Size())

        for model in self.models[1:]:
            model_batch_shape = getattr(model, "batch_shape", torch.Size())
            if model_batch_shape != batch_shape:
                raise NotImplementedError(
                    "All submodels must have the same batch_shape. "
                    f"Got {batch_shape} and {model_batch_shape}."
                )

        return batch_shape

    # ---------------------------------------------------------------------
    # train_inputs / train_targets handling
    # ---------------------------------------------------------------------
    @staticmethod
    def _get_submodel_train_input_raw(model: Model) -> Tensor:
        """
        Get raw-space training input from a submodel.

        Priority:
            1. model.train_input_raw
            2. model.train_inputs_raw[0]
            3. model.train_X
            4. model.train_inputs[0]
        """
        if hasattr(model, "train_input_raw"):
            return model.train_input_raw

        if hasattr(model, "train_inputs_raw"):
            train_inputs_raw = model.train_inputs_raw
            if not isinstance(train_inputs_raw, tuple):
                raise TypeError(
                    "model.train_inputs_raw must be a tuple. "
                    f"Got {type(train_inputs_raw).__name__}."
                )
            return train_inputs_raw[0]

        if hasattr(model, "train_X"):
            return model.train_X

        if hasattr(model, "train_inputs"):
            train_inputs = model.train_inputs
            if not isinstance(train_inputs, tuple):
                raise TypeError(
                    "model.train_inputs must be a tuple. "
                    f"Got {type(train_inputs).__name__}."
                )
            return train_inputs[0]

        raise AttributeError(
            "Submodel does not have train_input_raw, train_inputs_raw, "
            "train_X, or train_inputs."
        )

    @staticmethod
    def _get_submodel_train_input(model: Model) -> Tensor:
        """
        Get transformed training input from a submodel.

        Priority:
            1. model.train_input
            2. model.train_inputs[0]
            3. raw-space fallback
        """
        if hasattr(model, "train_input"):
            return model.train_input

        if hasattr(model, "train_inputs"):
            train_inputs = model.train_inputs
            if not isinstance(train_inputs, tuple):
                raise TypeError(
                    "model.train_inputs must be a tuple. "
                    f"Got {type(train_inputs).__name__}."
                )
            return train_inputs[0]

        return MultiOutputOrdinalModel._get_submodel_train_input_raw(model)

    @staticmethod
    def _get_submodel_train_targets(model: Model) -> Tensor:
        """
        Get training targets from a submodel.

        Priority:
            1. model.train_targets
            2. model.train_Y
        """
        if hasattr(model, "train_targets"):
            return model.train_targets

        if hasattr(model, "train_Y"):
            return model.train_Y

        raise AttributeError("Submodel does not have train_targets or train_Y.")

    def _validate_same_train_inputs(self) -> None:
        """
        Validate that all submodels share the same raw-space train inputs.

        MultiOutputOrdinalModel concatenates targets across outputs, so the rows
        of each submodel's training data must correspond to the same X points.
        """
        ref_X = self._get_submodel_train_input_raw(self.models[0])

        for i, model in enumerate(self.models[1:], start=1):
            X_i = self._get_submodel_train_input_raw(model)

            if X_i.shape != ref_X.shape:
                raise ValueError(
                    "All submodels must have the same raw train input shape. "
                    f"Submodel 0 has {tuple(ref_X.shape)}, "
                    f"but submodel {i} has {tuple(X_i.shape)}."
                )

            if not torch.allclose(X_i, ref_X):
                raise ValueError(
                    "All submodels must have the same raw train inputs. "
                    f"Submodel {i} has different train inputs from submodel 0."
                )

    @property
    def train_input_raw(self) -> Tensor:
        """
        Raw-space training inputs.

        This is the X-space accepted by this wrapper's posterior(X).
        """
        return self._get_submodel_train_input_raw(self.models[0])

    @property
    def train_inputs_raw(self) -> tuple[Tensor]:
        """
        Raw-space training inputs as a BoTorch-style tuple.
        """
        return (self.train_input_raw,)

    @property
    def train_input(self) -> Tensor:
        """
        Training input exposed by this wrapper.

        Important:
            This wrapper's posterior(X) accepts raw-space X, so train_input is
            intentionally raw-space.

            Each submodel may still have transformed train_inputs internally.
        """
        return self.train_input_raw

    @property
    def train_inputs(self) -> tuple[Tensor]:
        """
        BoTorch-style training inputs.

        Returns:
            tuple[Tensor]:
                A tuple containing raw-space train_X.

        Notes:
            This must be a tuple, not a Tensor.
        """
        return (self.train_input,)

    @property
    def transformed_train_inputs_list(self) -> list[Tensor]:
        """
        Transformed training inputs of each submodel.

        This is mainly useful for debugging. For acquisition functions and
        baseline construction, prefer this wrapper's train_inputs.
        """
        return [
            self._get_submodel_train_input(model)
            for model in self.models
        ]

    @property
    def train_targets_list(self) -> list[Tensor]:
        """
        Training targets of each submodel.

        Returns:
            list[Tensor]:
                Each element is usually shape [n] or [n, 1].
        """
        return [
            self._get_submodel_train_targets(model)
            for model in self.models
        ]

    @property
    def train_targets(self) -> Tensor:
        """
        Multi-output training targets.

        Returns:
            Tensor:
                Shape [n, m], where m = num_outputs.
        """
        ys = []

        for model in self.models:
            y = self._get_submodel_train_targets(model)

            if y.ndim == 1:
                y = y.unsqueeze(-1)
            elif y.ndim == 2 and y.shape[-1] == 1:
                pass
            else:
                raise ValueError(
                    "Each submodel target must be [n] or [n, 1]. "
                    f"Got shape={tuple(y.shape)}."
                )

            ys.append(y)

        return torch.cat(ys, dim=-1)

    # ---------------------------------------------------------------------
    # Backward-compatible aliases
    # ---------------------------------------------------------------------
    @property
    def train_X(self) -> Tensor:
        """
        Backward-compatible alias.

        Deprecated:
            Use train_input_raw or train_inputs_raw[0] instead.
        """
        return self.train_input_raw

    @property
    def raw_train_X(self) -> Tensor:
        """
        Backward-compatible alias.

        Deprecated:
            Use train_input_raw instead.
        """
        return self.train_input_raw

    @property
    def train_Y(self) -> Tensor:
        """
        Backward-compatible alias.

        Deprecated:
            Use train_targets instead.
        """
        return self.train_targets

    @property
    def num_classes_list(self) -> list[int]:
        return [int(m.num_classes) for m in self.models]

    # ---------------------------------------------------------------------
    # Output handling
    # ---------------------------------------------------------------------
    def _normalize_output_indices(
        self,
        output_indices: Optional[Sequence[int]],
    ) -> list[int]:
        if output_indices is None:
            return list(range(self.num_outputs))

        idcs = [int(i) for i in output_indices]

        for i in idcs:
            if i < 0 or i >= self.num_outputs:
                raise IndexError(
                    f"output index {i} is out of range for "
                    f"num_outputs={self.num_outputs}."
                )

        return idcs

    # ---------------------------------------------------------------------
    # Posterior
    # ---------------------------------------------------------------------
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        if observation_noise is not False:
            raise NotImplementedError(
                "MultiOutputOrdinalModel does not support observation_noise."
            )

        idcs = self._normalize_output_indices(output_indices)

        mvns: list[MultivariateNormal] = []

        for i in idcs:
            post_i = self.models[i].posterior(
                X=X,
                output_indices=None,
                observation_noise=False,
                posterior_transform=None,
                **kwargs,
            )

            dist_i = post_i.distribution

            if not isinstance(dist_i, MultivariateNormal):
                raise TypeError(
                    "Expected MultivariateNormal from submodel posterior, "
                    f"got {type(dist_i).__name__}."
                )

            mvns.append(dist_i)

        joint = MultitaskMultivariateNormal.from_independent_mvns(mvns)
        posterior = GPyTorchPosterior(joint)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)

        return posterior

    def subset_output(self, idcs: list[int]) -> Model:
        submodels = [self.models[i] for i in idcs]

        if len(submodels) == 1:
            return submodels[0]

        return self.__class__(*submodels)

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        **kwargs: Any,
    ) -> "MultiOutputOrdinalModel":
        if Y.shape[-1] != self.num_outputs:
            raise ValueError(
                f"Expected Y.shape[-1] == {self.num_outputs}, "
                f"got {Y.shape[-1]}."
            )

        fantasy_models = []

        for i, model in enumerate(self.models):
            Y_i = Y[..., i : i + 1]
            fantasy_i = model.condition_on_observations(
                X=X,
                Y=Y_i,
                **kwargs,
            )
            fantasy_models.append(fantasy_i)

        return self.__class__(*fantasy_models)

    # ---------------------------------------------------------------------
    # Ordinal prediction utilities
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def class_probs_list(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> list[Tensor]:
        """
        各出力のクラス確率を list で返す。

        Returns:
            list[Tensor]:
                各要素の shape は [..., q, C_j]
        """
        idcs = self._normalize_output_indices(output_indices)

        probs_list = []

        for i in idcs:
            probs_i = self.models[i].class_probs(X, **kwargs)
            probs_list.append(probs_i)

        return probs_list

    @torch.no_grad()
    def class_probs(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        全出力でクラス数が同じ場合にのみ stack して返す。

        Returns:
            Tensor:
                [..., q, m, C]
        """
        probs_list = self.class_probs_list(
            X=X,
            output_indices=output_indices,
            **kwargs,
        )

        num_classes = [p.shape[-1] for p in probs_list]

        if len(set(num_classes)) != 1:
            raise ValueError(
                "Outputs have different numbers of classes, so class_probs() "
                "cannot stack them into a single tensor. "
                "Use class_probs_list() or padded_class_probs() instead. "
                f"Got num_classes={num_classes}."
            )

        return torch.stack(probs_list, dim=-2)

    @torch.no_grad()
    def padded_class_probs(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        pad_value: float = 0.0,
        **kwargs: Any,
    ) -> Tensor:
        """
        出力ごとにクラス数が異なっても、C_max に padding して返す。

        Returns:
            Tensor:
                [..., q, m, C_max]
        """
        probs_list = self.class_probs_list(
            X=X,
            output_indices=output_indices,
            **kwargs,
        )

        max_c = max(p.shape[-1] for p in probs_list)

        padded = []

        for p in probs_list:
            if p.shape[-1] == max_c:
                padded.append(p)
                continue

            pad_shape = list(p.shape[:-1]) + [max_c - p.shape[-1]]

            pad_tensor = torch.full(
                pad_shape,
                fill_value=pad_value,
                device=p.device,
                dtype=p.dtype,
            )

            padded.append(torch.cat([p, pad_tensor], dim=-1))

        return torch.stack(padded, dim=-2)

    @torch.no_grad()
    def predict_class(
        self,
        X: Tensor,
        output_indices: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        各出力の予測クラスをまとめて返す。

        Returns:
            Tensor:
                [..., q, m]
        """
        idcs = self._normalize_output_indices(output_indices)

        preds = []

        for i in idcs:
            pred_i = self.models[i].predict_class(X, **kwargs)

            if pred_i.ndim == X.ndim - 1:
                pred_i = pred_i.unsqueeze(-1)

            preds.append(pred_i)

        return torch.cat(preds, dim=-1)

    @torch.no_grad()
    def expected_utility(
        self,
        X: Tensor,
        utility_values_list: Sequence[Sequence[float] | Tensor],
    ) -> Tensor:
        """
        各出力の expected utility をまとめて返す。

        Returns:
            Tensor:
                [..., q, m]
        """
        if len(utility_values_list) != self.num_outputs:
            raise ValueError(
                f"utility_values_list length {len(utility_values_list)} "
                f"must match num_outputs={self.num_outputs}."
            )

        outs = []

        for j, model in enumerate(self.models):
            utilities_j = torch.as_tensor(
                utility_values_list[j],
                device=X.device,
                dtype=X.dtype,
            )

            out_j = model.expected_utility(X, utilities_j)

            if out_j.ndim == X.ndim - 1:
                out_j = out_j.unsqueeze(-1)

            outs.append(out_j)

        return torch.cat(outs, dim=-1)