from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
from torch import Tensor

from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import InputTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, ProductKernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

from bochan.fit.ordinal import fit_ordinal_gp
from bochan.likelihoods.ordinal import OrdinalLogitLikelihood


def _normalize_dims(cat_dims: Sequence[int], d: int) -> list[int]:
    dims: list[int] = []
    for idx in cat_dims:
        j = idx if idx >= 0 else d + idx
        if j < 0 or j >= d:
            raise ValueError(f"Invalid categorical dim {idx} for input dim {d}.")
        dims.append(int(j))
    return sorted(set(dims))


def _get_cont_dims(d: int, cat_dims: Sequence[int]) -> list[int]:
    cat_set = set(_normalize_dims(cat_dims, d))
    return [i for i in range(d) if i not in cat_set]


def _make_cont_kernel(
    cont_dims: Sequence[int],
    kernel_name: str = "matern52",
) -> Optional[Kernel]:
    cont_dims = list(cont_dims)
    if len(cont_dims) == 0:
        return None

    if kernel_name.lower() == "rbf":
        return ScaleKernel(
            RBFKernel(
                ard_num_dims=len(cont_dims),
                active_dims=tuple(cont_dims),
            )
        )

    if kernel_name.lower() == "matern52":
        return ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=len(cont_dims),
                active_dims=tuple(cont_dims),
            )
        )

    raise ValueError(f"Unknown continuous kernel: {kernel_name}")


def _make_cat_kernel(cat_dims: Sequence[int]) -> Optional[Kernel]:
    cat_dims = list(cat_dims)
    if len(cat_dims) == 0:
        return None
    return ScaleKernel(CategoricalKernel(active_dims=tuple(cat_dims)))


def build_mixed_ordinal_kernel(
    d: int,
    cat_dims: Sequence[int],
    cont_kernel_name: str = "matern52",
) -> Kernel:
    """Build a mixed kernel in the spirit of MixedSingleTaskGP."""
    cat_dims = _normalize_dims(cat_dims, d)
    cont_dims = _get_cont_dims(d, cat_dims)

    if len(cat_dims) == 0:
        kernel = _make_cont_kernel(cont_dims, cont_kernel_name)
        if kernel is None:
            raise ValueError("Failed to build continuous kernel.")
        return kernel

    if len(cont_dims) == 0:
        kernel = _make_cat_kernel(cat_dims)
        if kernel is None:
            raise ValueError("Failed to build categorical kernel.")
        return kernel

    cont_kernel_1 = _make_cont_kernel(cont_dims, cont_kernel_name)
    cont_kernel_2 = _make_cont_kernel(cont_dims, cont_kernel_name)
    cat_kernel_1 = _make_cat_kernel(cat_dims)
    cat_kernel_2 = _make_cat_kernel(cat_dims)

    if None in (cont_kernel_1, cont_kernel_2, cat_kernel_1, cat_kernel_2):
        raise RuntimeError("Failed to build mixed ordinal kernel.")

    return cont_kernel_1 + cat_kernel_1 + ProductKernel(
        cont_kernel_2,
        cat_kernel_2,
    )


def _prepare_input_transform(
    input_transform: Optional[InputTransform],
    ref_X: Tensor,
) -> Optional[InputTransform]:
    if input_transform is None:
        return None

    input_transform = copy.deepcopy(input_transform)
    return input_transform.to(device=ref_X.device, dtype=ref_X.dtype)


def _expand_raw_X_to_match_transformed_q(
    X: Tensor,
    X_tf: Tensor,
) -> Tensor:
    """
    InputPerturbation 後の X_tf と比較できるように raw X の q 次元を展開する。

    想定:
        X.shape    = (*batch, q, d)
        X_tf.shape = (*batch, q_like, d)

    通常:
        q_like = q

    InputPerturbation:
        q_like = q * n_w

    Returns:
        Tensor:
            X_tf と同じ q_like を持つ raw-space X。
            カテゴリ列の保持チェックに使う。
    """
    if X.shape == X_tf.shape:
        return X

    if X.ndim < 2 or X_tf.ndim < 2:
        return X

    if X.shape[-1] != X_tf.shape[-1]:
        return X

    # 同じ batch shape で q だけが q*n_w に展開されている場合。
    if X.shape[:-2] == X_tf.shape[:-2]:
        q = X.shape[-2]
        q_like = X_tf.shape[-2]

        if q_like == q:
            return X

        if q > 0 and q_like % q == 0:
            n_w = q_like // q
            return X.repeat_interleave(n_w, dim=-2)

    # 念のため、全要素数が一致する場合は reshape。
    if X.numel() == X_tf.numel():
        return X.reshape_as(X_tf)

    return X


def _check_categorical_columns_unchanged(
    X: Tensor,
    X_tf: Tensor,
    cat_dims: Optional[Sequence[int]],
) -> None:
    """
    mixed model 用に、input_transform がカテゴリ列を変更していないか確認する。

    InputPerturbation では q -> q*n_w に展開されるため、
    raw X 側も q 次元を repeat してから比較する。
    """
    if cat_dims is None or len(cat_dims) == 0:
        return

    cat_idx = [int(i) for i in cat_dims]
    X_cmp = _expand_raw_X_to_match_transformed_q(X, X_tf)

    if X_cmp.shape[:-1] != X_tf.shape[:-1]:
        raise RuntimeError(
            "Could not align raw X with transformed X for categorical column check. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}, "
            f"X_cmp.shape={tuple(X_cmp.shape)}. "
            "This usually means input_transform changed the batch/q shape in a "
            "non-repeatable way."
        )

    if not torch.allclose(X_tf[..., cat_idx], X_cmp[..., cat_idx]):
        raise ValueError(
            "input_transform must not modify categorical columns. "
            "For mixed models, transform only continuous columns. "
            f"X_cat.shape={tuple(X_cmp[..., cat_idx].shape)}, "
            f"X_tf_cat.shape={tuple(X_tf[..., cat_idx].shape)}."
        )


def _transform_tensor(
    X: Tensor,
    input_transform: Optional[InputTransform],
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    """
    posterior / acquisition 評価用の input transform。

    Notes:
        InputPerturbation の eval mode では q -> q*n_w に展開され得る。
        mixed model のカテゴリ列チェックでは raw X 側も q*n_w に展開して比較する。
    """
    X = X.contiguous()

    if input_transform is None:
        return X

    X_tf = input_transform(X).contiguous()
    _check_categorical_columns_unchanged(X=X, X_tf=X_tf, cat_dims=cat_dims)
    return X_tf


def _transform_tensor_for_training(
    X: Tensor,
    input_transform: Optional[InputTransform],
    cat_dims: Optional[Sequence[int]] = None,
    *,
    name: str = "input_transform",
) -> Tensor:
    """
    学習用 train_X / inducing_points 用の input transform。

    Classification 系と同じ規約で、
        input_transform.train()
        X_tf = input_transform(X)
        input_transform.eval()
    の順にする。

    これにより、InputPerturbation は学習時には通常 q*n_w 展開されず、
    posterior / acquisition 評価時だけ展開される。
    """
    X = X.contiguous()

    if input_transform is None:
        return X

    if hasattr(input_transform, "train"):
        input_transform.train()

    X_tf = input_transform(X).contiguous()

    if hasattr(input_transform, "eval"):
        input_transform.eval()

    if X_tf.shape[-2] != X.shape[-2]:
        raise RuntimeError(
            f"{name} expanded training inputs. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}. "
            "This will not match ordinal train_Y. "
            "For InputPerturbation, ensure transform_on_train=False."
        )

    _check_categorical_columns_unchanged(X=X, X_tf=X_tf, cat_dims=cat_dims)
    return X_tf


def _canonicalize_inducing_points(
    inducing_points: Tensor,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    inducing_points = torch.as_tensor(
        inducing_points,
        device=device,
        dtype=dtype,
    )

    if inducing_points.ndim != 2:
        raise ValueError(
            f"inducing_points must be [m, d], "
            f"got shape={tuple(inducing_points.shape)}"
        )

    if inducing_points.shape[-1] != d:
        raise ValueError(
            f"inducing_points feature dim mismatch: "
            f"expected {d}, got {inducing_points.shape[-1]}"
        )

    return inducing_points.contiguous()


def _infer_num_classes_from_train_Y(train_Y: Tensor) -> int:
    """Infer the number of ordinal classes from train_Y."""
    unique_y = torch.unique(train_Y).sort().values

    if unique_y.numel() < 3:
        raise ValueError(
            "Ordinal GP requires at least 3 classes. "
            f"Got labels {unique_y.detach().cpu().tolist()}."
        )

    expected = torch.arange(
        unique_y.numel(),
        device=unique_y.device,
        dtype=unique_y.dtype,
    )

    if not torch.equal(unique_y, expected):
        raise ValueError(
            "train_Y must be ordinal labels encoded as consecutive integers "
            "starting at 0 when num_classes is inferred. "
            f"Got labels {unique_y.detach().cpu().tolist()}."
        )

    return int(unique_y.numel())


class _OrdinalLatentGP(ApproximateGP):
    """Single-output variational GP for the latent score f(x)."""

    def __init__(
        self,
        train_X: Tensor,
        inducing_points_num: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError("train_X must be shape [N, d].")

        n, d = train_X.shape

        if inducing_points is None:
            m = min(int(inducing_points_num), n)
            perm = torch.randperm(n, device=train_X.device)[:m]
            inducing_points = train_X[perm].clone()

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2]
        )

        variational_strategy = VariationalStrategy(
            model=self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )

        super().__init__(variational_strategy)

        self.mean_module = ConstantMean() if mean_module is None else mean_module
        self.covar_module = (
            ScaleKernel(RBFKernel(ard_num_dims=d))
            if covar_module is None
            else covar_module
        )

    def forward(self, X: Tensor) -> MultivariateNormal:
        return MultivariateNormal(
            self.mean_module(X),
            self.covar_module(X),
        )


class _MixedOrdinalLatentGP(ApproximateGP):
    """Single-output variational GP for mixed continuous/categorical inputs."""

    def __init__(
        self,
        train_X: Tensor,
        cat_dims: Sequence[int],
        inducing_points_num: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        cont_kernel_name: str = "matern52",
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError("train_X must be shape [N, d].")

        n, d = train_X.shape

        if inducing_points is None:
            m = min(int(inducing_points_num), n)
            perm = torch.randperm(n, device=train_X.device)[:m]
            inducing_points = train_X[perm].clone()

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2]
        )

        variational_strategy = VariationalStrategy(
            model=self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )

        super().__init__(variational_strategy)

        self.mean_module = ConstantMean() if mean_module is None else mean_module
        self.covar_module = (
            build_mixed_ordinal_kernel(
                d=d,
                cat_dims=cat_dims,
                cont_kernel_name=cont_kernel_name,
            )
            if covar_module is None
            else covar_module
        )

    def forward(self, X: Tensor) -> MultivariateNormal:
        return MultivariateNormal(
            self.mean_module(X),
            self.covar_module(X),
        )


class _BaseOrdinalGPModel(ApproximateGPyTorchModel):
    """Common utilities for ordinal GP wrappers."""

    @staticmethod
    def _canonicalize_train_X(X: Tensor) -> Tensor:
        X = torch.as_tensor(X)

        if X.ndim != 2:
            raise ValueError(
                f"train_X must be [N, d], got shape={tuple(X.shape)}"
            )

        return X.contiguous()

    @staticmethod
    def _canonicalize_train_Y(
        Y: Tensor,
        n: int,
        device: torch.device,
    ) -> Tensor:
        Y = torch.as_tensor(Y, device=device)

        if Y.ndim == 2 and Y.shape[-1] == 1:
            Y = Y.squeeze(-1)
        elif Y.ndim != 1:
            raise ValueError(
                f"train_Y must be [N] or [N, 1], got shape={tuple(Y.shape)}"
            )

        if Y.shape[0] != n:
            raise ValueError(
                f"train_Y length mismatch: expected {n}, got {Y.shape[0]}"
            )

        return Y.long().contiguous()

    def _set_transformed_inputs(self) -> None:
        """
        BoTorch Model.eval() が呼ぶ transformed input 自動更新を無効化する。

        この ordinal wrapper では、
            raw X -> input_transform -> latent GP
        の対応を train_inputs / train_inputs_raw で明示的に管理する。

        BoTorch 標準の _set_transformed_inputs() は
            self.input_transform.preprocess_transform(self.train_inputs[0])
            self.set_train_data(...)
        を実行しようとするが、ここで train_inputs[0] はすでに transformed
        space の入力であり、InputPerturbation 付きでは eval mode の transform が
        n_w 展開を起こして train_targets と不整合になる可能性がある。

        したがって eval() 時の自動変換は no-op にする。
        """
        return None

    def set_train_data(
        self,
        inputs: Optional[Tensor | tuple[Tensor, ...]] = None,
        targets: Optional[Tensor] = None,
        strict: bool = True,
    ) -> None:
        """
        明示的に training data を差し替えるための helper。

        Args:
            inputs:
                raw-space X を想定する。内部で input_transform を training mode で適用する。
            targets:
                ordinal label。
            strict:
                GPyTorch 互換のための引数。ここでは使用しない。
        """
        _ = strict

        if inputs is not None:
            X_raw = inputs[0] if isinstance(inputs, tuple) else inputs
            X_raw = self._canonicalize_observation_X(X_raw)

            cat_dims = getattr(self, "cat_dims", None)
            X_tf = _transform_tensor_for_training(
                X_raw,
                input_transform=getattr(self, "input_transform", None),
                cat_dims=cat_dims,
                name=f"{self.__class__.__name__}.input_transform",
            )

            self.train_inputs = (X_tf,)
            self.train_inputs_raw = (X_raw,)

            # latent model 側も同期する。
            self.model.train_inputs = (X_tf,)

        if targets is not None:
            n = self.train_inputs_raw[0].shape[-2]
            y = self._canonicalize_train_Y(
                targets,
                n=n,
                device=self.train_inputs_raw[0].device,
            )
            self.train_targets = y

            try:
                self.model.train_targets = y
            except Exception:
                pass

    @property
    def train_input(self) -> Tensor:
        """Transformed training inputs used by the latent GP."""
        return self.train_inputs[0]

    @property
    def train_input_raw(self) -> Tensor:
        """Raw-space training inputs before input_transform."""
        if hasattr(self, "train_inputs_raw"):
            return self.train_inputs_raw[0]
        return self.train_inputs[0]

    @property
    def train_X(self) -> Tensor:
        """
        Backward-compatible alias.

        Deprecated:
            Use train_input_raw or train_inputs_raw[0] instead.
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
    def inducing_points_original(self) -> Tensor:
        """
        Backward-compatible alias.

        Deprecated:
            Use inducing_points_raw instead.
        """
        return self.inducing_points_raw

    @property
    def ordinal_likelihood(self) -> OrdinalLogitLikelihood:
        return self.likelihood

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    def _canonicalize_posterior_X(self, X: Tensor) -> Tensor:
        train_X_raw = self.train_input_raw

        X = torch.as_tensor(
            X,
            device=train_X_raw.device,
            dtype=train_X_raw.dtype,
        )

        if X.ndim == 1:
            if train_X_raw.shape[-1] != 1:
                raise ValueError(
                    "1D X can only be used when input dim is 1, "
                    f"got d={train_X_raw.shape[-1]}."
                )
            X = X.unsqueeze(-1)

        if X.ndim < 2:
            raise ValueError(
                f"X for posterior must have at least 2 dims, "
                f"got shape={tuple(X.shape)}"
            )

        if X.shape[-1] != train_X_raw.shape[-1]:
            raise ValueError(
                f"X feature dim mismatch: expected {train_X_raw.shape[-1]}, "
                f"got {X.shape[-1]}"
            )

        return X.contiguous()

    def _canonicalize_observation_X(self, X: Tensor) -> Tensor:
        X = self._canonicalize_posterior_X(X)

        if X.ndim != 2:
            raise ValueError(
                f"Observation X must be [n, d], got shape={tuple(X.shape)}"
            )

        return X

    def _canonicalize_new_Y(self, Y: Tensor, n: int) -> Tensor:
        Y = torch.as_tensor(Y, device=self.train_targets.device)

        if Y.ndim == 0:
            Y = Y.view(1)
        elif Y.ndim == 2 and Y.shape[-1] == 1:
            Y = Y.squeeze(-1)
        elif Y.ndim != 1:
            raise ValueError(
                f"Y must be [n] or [n, 1], got shape={tuple(Y.shape)}"
            )

        if Y.shape[0] != n:
            raise ValueError(
                f"Y length mismatch: expected {n}, got {Y.shape[0]}"
            )

        return Y.long().contiguous()

    def transform_inputs(self, X: Tensor) -> Tensor:
        return _transform_tensor(
            X=X,
            input_transform=getattr(self, "input_transform", None),
        )

    def forward(self, X: Tensor) -> MultivariateNormal:
        X = self._canonicalize_posterior_X(X)
        X_tf = self.transform_inputs(X)
        return self.model(X_tf)

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs,
    ):
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} is single-output; "
                "output_indices is not used."
            )

        if observation_noise is not False:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support observation_noise. "
                "posterior() returns the latent f posterior only."
            )

        X = self._canonicalize_posterior_X(X)
        posterior = GPyTorchPosterior(distribution=self(X))

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)

        return posterior

    def class_probs_from_posterior(self, posterior) -> Tensor:
        return self.ordinal_likelihood.marginal_class_probs(
            posterior.distribution
        )

    def class_probs(self, X: Tensor) -> Tensor:
        return self.class_probs_from_posterior(self.posterior(X))

    @torch.no_grad()
    def predict_class(self, X: Tensor) -> Tensor:
        return self.class_probs(X).argmax(dim=-1)

    @torch.no_grad()
    def expected_utility(self, X: Tensor, utilities: Tensor) -> Tensor:
        posterior = self.posterior(X)
        return self.ordinal_likelihood.marginal_expected_utility(
            posterior.distribution,
            utilities,
        )


class OrdinalGPModel(_BaseOrdinalGPModel):
    """BoTorch-compatible ordinal GP model for continuous inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        num_classes: Optional[int] = None,
        inducing_points_num: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        input_transform: Optional[InputTransform] = None,
        eps: float = 1e-8,
        init_gap: float = 1.0,
        fix_first_cutpoint: bool = True,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
    ) -> None:
        raw_train_X = self._canonicalize_train_X(train_X)
        train_Y = self._canonicalize_train_Y(
            train_Y,
            raw_train_X.shape[-2],
            raw_train_X.device,
        )

        if num_classes is None:
            num_classes = _infer_num_classes_from_train_Y(train_Y)
        else:
            num_classes = int(num_classes)

        input_transform = _prepare_input_transform(
            input_transform,
            raw_train_X,
        )

        if inducing_points is None:
            m = min(int(inducing_points_num), raw_train_X.shape[-2])
            perm = torch.randperm(raw_train_X.shape[-2], device=raw_train_X.device)[:m]
            raw_inducing_points = raw_train_X[perm].clone()
        else:
            raw_inducing_points = _canonicalize_inducing_points(
                inducing_points,
                d=raw_train_X.shape[-1],
                device=raw_train_X.device,
                dtype=raw_train_X.dtype,
            )

        train_X_tf = _transform_tensor_for_training(
            raw_train_X,
            input_transform=input_transform,
            name="OrdinalGPModel.input_transform",
        )
        inducing_points_tf = _transform_tensor_for_training(
            raw_inducing_points,
            input_transform=input_transform,
            name="OrdinalGPModel.input_transform",
        )

        latent_model = _OrdinalLatentGP(
            train_X=train_X_tf,
            inducing_points_num=inducing_points_num,
            inducing_points=inducing_points_tf,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
        )

        likelihood = OrdinalLogitLikelihood(
            num_classes=num_classes,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
        )

        super().__init__(
            model=latent_model,
            likelihood=likelihood,
            num_outputs=1,
        )

        self.input_transform = input_transform

        # BoTorch-style training data attributes.
        #
        # train_inputs:
        #     Transformed inputs actually used by the latent GP.
        #
        # train_inputs_raw:
        #     Raw inputs in the original search / observation space.
        #
        # train_targets:
        #     Ordinal labels.
        self.train_inputs = (train_X_tf,)
        self.train_inputs_raw = (raw_train_X,)
        self.train_targets = train_Y

        self.model.train_inputs = self.train_inputs
        self.model.train_targets = self.train_targets
        
        # Inducing points are also stored in both raw and transformed spaces.
        self.inducing_points_raw = raw_inducing_points
        self.inducing_points = inducing_points_tf

        self.num_classes = int(num_classes)
        self.inducing_points_num = int(inducing_points_num)
        self.learn_inducing_locations = bool(learn_inducing_locations)

        self.eps = float(eps)
        self.init_gap = float(init_gap)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)

        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        self.to(device=raw_train_X.device, dtype=raw_train_X.dtype)

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        refit: bool = True,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> "OrdinalGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("noise is not supported for OrdinalGPModel.")

        X = self._canonicalize_observation_X(X)
        Y = self._canonicalize_new_Y(Y, n=X.shape[-2])

        new_train_X = torch.cat([self.train_input_raw, X], dim=-2)
        new_train_Y = torch.cat([self.train_targets, Y], dim=0)

        new_model = self.__class__(
            train_X=new_train_X,
            train_Y=new_train_Y,
            num_classes=self.num_classes,
            inducing_points_num=self.model.variational_strategy.inducing_points.shape[-2],
            inducing_points=self.inducing_points_raw.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
            input_transform=copy.deepcopy(self.input_transform),
            eps=self.eps,
            init_gap=self.init_gap,
            fix_first_cutpoint=self.fix_first_cutpoint,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
        )

        new_model.load_state_dict(
            copy.deepcopy(self.state_dict()),
            strict=True,
        )

        if refit:
            steps = self.conditioning_steps if num_steps is None else int(num_steps)
            refit_lr = self.conditioning_lr if lr is None else float(lr)
            refit_bs = (
                self.conditioning_batch_size
                if batch_size is None
                else batch_size
            )

            fit_ordinal_gp(
                new_model,
                num_epochs=steps,
                lr=refit_lr,
                batch_size=refit_bs,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()

        return new_model


class OrdinalMixedGPModel(_BaseOrdinalGPModel):
    """BoTorch-compatible ordinal GP model for mixed inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        num_classes: Optional[int] = None,
        category_counts: Optional[dict[int, int]] = None,
        category_values: Optional[dict[int, Sequence[int | float]]] = None,
        cont_kernel: str = "matern52",
        inducing_points_num: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        input_transform: Optional[InputTransform] = None,
        eps: float = 1e-8,
        init_gap: float = 1.0,
        fix_first_cutpoint: bool = True,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
    ) -> None:
        raw_train_X = self._canonicalize_train_X(train_X)
        train_Y = self._canonicalize_train_Y(
            train_Y,
            raw_train_X.shape[-2],
            raw_train_X.device,
        )

        if num_classes is None:
            num_classes = _infer_num_classes_from_train_Y(train_Y)
        else:
            num_classes = int(num_classes)

        self.cat_dims = _normalize_dims(cat_dims, raw_train_X.shape[-1])
        self.cont_dims = _get_cont_dims(raw_train_X.shape[-1], self.cat_dims)

        self.category_values = self._resolve_category_values(
            X=raw_train_X,
            cat_dims=self.cat_dims,
            category_counts=category_counts,
            category_values=category_values,
        )
        self.category_counts = {
            j: int(v.numel()) for j, v in self.category_values.items()
        }

        self._validate_categorical_values(
            X=raw_train_X,
            cat_dims=self.cat_dims,
            category_values=self.category_values,
        )

        input_transform = _prepare_input_transform(
            input_transform,
            raw_train_X,
        )

        if inducing_points is None:
            m = min(int(inducing_points_num), raw_train_X.shape[-2])
            perm = torch.randperm(raw_train_X.shape[-2], device=raw_train_X.device)[:m]
            raw_inducing_points = raw_train_X[perm].clone()
        else:
            raw_inducing_points = _canonicalize_inducing_points(
                inducing_points,
                d=raw_train_X.shape[-1],
                device=raw_train_X.device,
                dtype=raw_train_X.dtype,
            )

        self._validate_categorical_values(
            X=raw_inducing_points,
            cat_dims=self.cat_dims,
            category_values=self.category_values,
        )

        train_X_tf = _transform_tensor_for_training(
            raw_train_X,
            input_transform=input_transform,
            cat_dims=self.cat_dims,
            name="OrdinalMixedGPModel.input_transform",
        )
        inducing_points_tf = _transform_tensor_for_training(
            raw_inducing_points,
            input_transform=input_transform,
            cat_dims=self.cat_dims,
            name="OrdinalMixedGPModel.input_transform",
        )

        latent_model = _MixedOrdinalLatentGP(
            train_X=train_X_tf,
            cat_dims=self.cat_dims,
            inducing_points_num=inducing_points_num,
            inducing_points=inducing_points_tf,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
            cont_kernel_name=cont_kernel,
        )

        likelihood = OrdinalLogitLikelihood(
            num_classes=num_classes,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
        )

        super().__init__(
            model=latent_model,
            likelihood=likelihood,
            num_outputs=1,
        )

        self.input_transform = input_transform

        # BoTorch-style training data attributes.
        self.train_inputs = (train_X_tf,)
        self.train_inputs_raw = (raw_train_X,)
        self.train_targets = train_Y
        
        self.model.train_inputs = self.train_inputs
        self.model.train_targets = self.train_targets
        
        # Inducing points are stored in both raw and transformed spaces.
        self.inducing_points_raw = raw_inducing_points
        self.inducing_points = inducing_points_tf

        self.num_classes = int(num_classes)
        self.cont_kernel = str(cont_kernel)
        self.inducing_points_num = int(inducing_points_num)
        self.learn_inducing_locations = bool(learn_inducing_locations)

        self.eps = float(eps)
        self.init_gap = float(init_gap)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)

        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        self.to(device=raw_train_X.device, dtype=raw_train_X.dtype)

    @staticmethod
    def _resolve_category_values(
        X: Tensor,
        cat_dims: Sequence[int],
        category_counts: Optional[dict[int, int]] = None,
        category_values: Optional[dict[int, Sequence[int | float]]] = None,
    ) -> dict[int, Tensor]:
        """
        Resolve allowed values for each categorical column.

        Priority:
            1. category_values
            2. category_counts -> 0, 1, ..., K-1
            3. inferred unique values from train_X
        """
        d = X.shape[-1]
        dims = _normalize_dims(cat_dims, d)

        count_dict = (
            {}
            if category_counts is None
            else {int(k): int(v) for k, v in category_counts.items()}
        )

        value_dict = (
            {}
            if category_values is None
            else {
                int(k): torch.as_tensor(
                    v,
                    device=X.device,
                    dtype=X.dtype,
                ).flatten()
                for k, v in category_values.items()
            }
        )

        resolved: dict[int, Tensor] = {}

        for j in dims:
            vals = X[:, j]

            if not torch.allclose(vals, vals.round()):
                raise ValueError(
                    f"Categorical column {j} must be integer-coded values."
                )

            if j in value_dict:
                allowed = torch.unique(value_dict[j].round()).sort().values
            elif j in count_dict:
                allowed = torch.arange(
                    count_dict[j],
                    device=X.device,
                    dtype=X.dtype,
                )
            else:
                allowed = torch.unique(vals.round()).sort().values

            resolved[j] = allowed

        return resolved

    @staticmethod
    def _validate_categorical_values(
        X: Tensor,
        cat_dims: Sequence[int],
        category_values: Optional[dict[int, Tensor]] = None,
    ) -> None:
        """
        Validate that categorical columns are integer-coded and optionally in the
        allowed-value set.
        """
        d = X.shape[-1]
        dims = _normalize_dims(cat_dims, d)

        value_dict = (
            {}
            if category_values is None
            else {
                int(k): torch.as_tensor(
                    v,
                    device=X.device,
                    dtype=X.dtype,
                ).flatten()
                for k, v in category_values.items()
            }
        )

        for j in dims:
            vals = X[:, j]

            if not torch.allclose(vals, vals.round()):
                raise ValueError(
                    f"Categorical column {j} must be integer-coded values."
                )

            if j in value_dict:
                allowed = torch.unique(value_dict[j].round()).sort().values
                is_allowed = (
                    vals.round().unsqueeze(-1) == allowed.unsqueeze(0)
                ).any(dim=-1)

                if not bool(is_allowed.all()):
                    bad = torch.unique(vals[~is_allowed]).detach().cpu().tolist()
                    raise ValueError(
                        f"Categorical column {j} contains unseen values {bad}. "
                        f"Allowed values are {allowed.detach().cpu().tolist()}."
                    )

    def transform_inputs(self, X: Tensor) -> Tensor:
        return _transform_tensor(
            X=X,
            input_transform=getattr(self, "input_transform", None),
            cat_dims=self.cat_dims,
        )

    def _canonicalize_observation_X(self, X: Tensor) -> Tensor:
        X = self._canonicalize_posterior_X(X)

        if X.ndim != 2:
            raise ValueError(
                f"Observation X must be [n, d], got shape={tuple(X.shape)}"
            )

        self._validate_categorical_values(
            X=X,
            cat_dims=self.cat_dims,
            category_values=self.category_values,
        )

        return X

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        refit: bool = True,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> "OrdinalMixedGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError(
                "noise is not supported for OrdinalMixedGPModel."
            )

        X = self._canonicalize_observation_X(X)
        Y = self._canonicalize_new_Y(Y, n=X.shape[-2])

        new_train_X = torch.cat([self.train_input_raw, X], dim=-2)
        new_train_Y = torch.cat([self.train_targets, Y], dim=0)

        new_model = self.__class__(
            train_X=new_train_X,
            train_Y=new_train_Y,
            num_classes=self.num_classes,
            cat_dims=self.cat_dims,
            category_counts=copy.deepcopy(self.category_counts),
            category_values={
                j: v.detach().cpu().tolist()
                for j, v in self.category_values.items()
            },
            cont_kernel=self.cont_kernel,
            inducing_points_num=self.model.variational_strategy.inducing_points.shape[-2],
            inducing_points=self.inducing_points_raw.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
            input_transform=copy.deepcopy(self.input_transform),
            eps=self.eps,
            init_gap=self.init_gap,
            fix_first_cutpoint=self.fix_first_cutpoint,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
        )

        new_model.load_state_dict(
            copy.deepcopy(self.state_dict()),
            strict=True,
        )

        if refit:
            steps = self.conditioning_steps if num_steps is None else int(num_steps)
            refit_lr = self.conditioning_lr if lr is None else float(lr)
            refit_bs = (
                self.conditioning_batch_size
                if batch_size is None
                else batch_size
            )

            fit_ordinal_gp(
                new_model,
                num_epochs=steps,
                lr=refit_lr,
                batch_size=refit_bs,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()

        return new_model