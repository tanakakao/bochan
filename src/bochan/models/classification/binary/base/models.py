from copy import deepcopy
from typing import Callable, Optional, Union, Sequence, Tuple, List
import torch
from torch import Tensor
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, RBFKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior

from ._latent_models import _LatentBinarySVGP, _LatentMixedBinarySVGP
from bochan.posteriors.bernoulli import SimpleBernoulliPosterior, get_sampler_for_simple_bernoulli
from bochan.kernels.categorical_kernel import categorical_kernel


# ============================================================
# utils
# ============================================================

def _normalize_indices(indices: Sequence[int], d: int) -> List[int]:
    """負のindexを正規化し、重複を除いて昇順化する。"""
    out = []
    for idx in indices:
        idx = int(idx)
        if idx < 0:
            idx = d + idx
        if not 0 <= idx < d:
            raise IndexError(f"Index {idx} is out of bounds for dimension {d}.")
        out.append(idx)
    return sorted(set(out))

def _to_device_dtype_transform(
    input_transform: Optional[InputTransform],
    train_X: Tensor,
) -> Optional[InputTransform]:
    """input_transform を train_X に合わせて移す。"""
    if input_transform is None:
        return None
    if hasattr(input_transform, "to"):
        input_transform = input_transform.to(train_X)
    return input_transform

def _select_inducing_points(
    X: Tensor,
    num_inducing_points: int,
    inducing_points: Optional[Tensor] = None,
) -> Tensor:
    """
    誘導点を選択する。

    明示指定があればそれを使い、なければ train_X からランダム抽出。
    """
    if inducing_points is not None:
        return inducing_points

    m = min(num_inducing_points, X.shape[-2])
    if X.shape[-2] > m:
        idx = torch.randperm(X.shape[-2], device=X.device)[:m]
        return X[idx].clone()
    return X.clone()

def _prepare_train_Yvar(
    train_Yvar: Optional[Tensor],
    *,
    train_X: Tensor,
    train_Y: Tensor,
) -> Optional[Tensor]:
    """
    train_Yvar を [N] or [N,1] から受け取り、[N] に正規化する。
    分類版では probability scale 上の既知観測ノイズとして扱う。
    """
    if train_Yvar is None:
        return None

    train_Yvar = train_Yvar.to(device=train_X.device, dtype=train_X.dtype)

    if train_Yvar.ndim == 2 and train_Yvar.shape[-1] == 1:
        train_Yvar = train_Yvar.squeeze(-1)

    if train_Yvar.ndim != 1:
        raise ValueError(
            f"train_Yvar must have shape [N] or [N,1], but got {tuple(train_Yvar.shape)}."
        )

    if train_Yvar.shape[0] != train_X.shape[-2]:
        raise ValueError(
            f"train_Yvar.shape[0] ({train_Yvar.shape[0]}) must match "
            f"train_X.shape[-2] ({train_X.shape[-2]})."
        )

    return train_Yvar.clamp_min(0.0)

def _expand_observation_noise_tensor(
    obs_noise: Tensor,
    *,
    X: Tensor,
) -> Tensor:
    """
    observation_noise を [..., q, 1] へ揃える。
    """
    obs_noise = obs_noise.to(device=X.device, dtype=X.dtype)

    if obs_noise.ndim == X.ndim - 1:
        obs_noise = obs_noise.unsqueeze(-1)

    expected_shape = X.shape[:-1] + (1,)
    if obs_noise.shape != expected_shape:
        try:
            obs_noise = obs_noise.expand(expected_shape)
        except RuntimeError as e:
            raise ValueError(
                f"observation_noise must be broadcastable to {expected_shape}, "
                f"but got {tuple(obs_noise.shape)}."
            ) from e

    return obs_noise.clamp_min(0.0)


def _prepare_binary_conditioning_data(
    X: Tensor,
    Y: Tensor,
    Yvar: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    classification 用の condition_on_observations 入力を整形する。

    対応:
        X:    (q, d) または (b, q, d)
        Y:    (q,), (q, 1), (b, q), (b, q, 1)

    非対応:
        fantasize / KG 用の fantasy sample dim 付き
        例: Y.shape = (n_fantasies, b, q, 1)

    Returns:
        X_flat:    (n_new, d)
        Y_flat:    (n_new,)
        Yvar_flat: (n_new,) or None
    """
    if isinstance(X, tuple):
        X = X[0]

    if X.dim() < 2:
        raise ValueError("X must have shape (q, d) or (b, q, d).")

    expected_y_shape = X.shape[:-1]

    if Y.dim() == X.dim() and Y.shape[-1] == 1:
        Y = Y.squeeze(-1)

    if Y.shape != expected_y_shape:
        raise NotImplementedError(
            "This classification condition_on_observations only supports "
            "non-fantasy observations with Y.shape == X.shape[:-1] "
            "(or trailing singleton output dim). "
            f"Got X.shape={tuple(X.shape)}, Y.shape={tuple(Y.shape)}."
        )

    if Yvar is not None:
        if Yvar.dim() == X.dim() and Yvar.shape[-1] == 1:
            Yvar = Yvar.squeeze(-1)
        if Yvar.shape != expected_y_shape:
            raise ValueError(
                "Yvar must match X.shape[:-1] (or have trailing singleton dim). "
                f"Got X.shape={tuple(X.shape)}, Yvar.shape={tuple(Yvar.shape)}."
            )

    X_flat = X.reshape(-1, X.shape[-1])
    Y_flat = Y.reshape(-1).to(dtype=X.dtype)

    Yvar_flat = None
    if Yvar is not None:
        Yvar_flat = Yvar.reshape(-1).to(dtype=X.dtype)

    return X_flat, Y_flat, Yvar_flat


def _concat_optional_noise(
    old_Y: Tensor,
    old_Yvar: Optional[Tensor],
    new_Y: Tensor,
    new_Yvar: Optional[Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[Tensor]:
    """
    old/new の train_Yvar を連結する。
    どちらか一方だけある場合は、ない側を 0 で埋める。
    """
    if old_Yvar is None and new_Yvar is None:
        return None

    if old_Yvar is None:
        old_Yvar = torch.zeros_like(old_Y, dtype=dtype, device=device)
    else:
        old_Yvar = old_Yvar.reshape(-1).to(dtype=dtype, device=device)

    if new_Yvar is None:
        new_Yvar = torch.zeros_like(new_Y, dtype=dtype, device=device)
    else:
        new_Yvar = new_Yvar.reshape(-1).to(dtype=dtype, device=device)

    return torch.cat([old_Yvar, new_Yvar], dim=0)

# ============================================================
# BoTorch-facing models
# ============================================================

class BinaryClassificationGPModel(ApproximateGPyTorchModel, FantasizeMixin):
    """
    SingleTaskGP 風の binary classification model。

    - 内部の latent GP は ApproximateGP
    - 外側は ApproximateGPyTorchModel
    - posterior() は Bernoulli probability の posterior を返す
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,   # 追加
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        if train_Y.ndim > 1 and train_Y.shape[-1] == 1:
            train_Y = train_Y.squeeze(-1)
        train_Y = train_Y.to(dtype=train_X.dtype)

        train_Yvar = _prepare_train_Yvar(
            train_Yvar,
            train_X=train_X,
            train_Y=train_Y,
        )

        self.train_inputs = (train_X,)
        self.train_targets = train_Y
        self.train_Yvar = train_Yvar
        self.train_inputs_raw = (train_X,)
        self._train_targets = train_Y

        input_transform = _to_device_dtype_transform(input_transform, train_X)

        if input_transform is not None:
            input_transform.train()
            with torch.no_grad():
                transformed_train_X = input_transform(train_X).detach().clone()
            input_transform.eval()
        else:
            transformed_train_X = train_X.detach().clone()

        inducing_points = _select_inducing_points(
            transformed_train_X,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )

        latent_model = _LatentBinarySVGP(
            inducing_points=inducing_points,
            train_inputs=transformed_train_X,
            train_targets=train_Y,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=covar_module,
            learn_inducing_locations=learn_inducing_locations,
        )

        likelihood = likelihood or BernoulliLikelihood()

        super().__init__(
            model=latent_model,
            likelihood=likelihood,
            num_outputs=1,
        )

        self.input_transform = input_transform
        self.to(train_X)

    def forward(self, X: Tensor) -> MultivariateNormal:
        if isinstance(X, tuple):
            X = X[0]
        if self.training:
            X = self.transform_inputs(X)
        return self.model(X)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> SimpleBernoulliPosterior:
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output_indices."
            )

        self.eval()
        self.likelihood.eval()

        if isinstance(X, tuple):
            X = X[0]

        X_tf = self.transform_inputs(X) if self.input_transform is not None else X
        latent_dist = self.model(X_tf)
        pred_dist = self.likelihood(latent_dist)

        p = pred_dist.mean
        var = pred_dist.variance
        if p.ndim == X.ndim - 1:
            p = p.unsqueeze(-1)
            var = var.unsqueeze(-1)

        obs_noise = None

        # BoTorch 風: Tensor が来たらそれを直接使う
        if torch.is_tensor(observation_noise):
            obs_noise = _expand_observation_noise_tensor(observation_noise, X=X)

        # True のときは model 側に既知ノイズがあれば加える
        elif observation_noise:
            noise_model = getattr(self, "noise_model", None)

            if noise_model is not None:
                noise_in = (
                    X_tf
                    if getattr(self, "noise_model_uses_transformed_inputs", True)
                    else X
                )
                obs_noise = noise_model.posterior(noise_in).mean
                obs_noise = _expand_observation_noise_tensor(obs_noise, X=X)

            elif self.train_Yvar is not None:
                # 学習点ごとの既知ノイズは test-time では直接は使えないので、
                # まずは global mean を既定値として使う
                default_noise = self.train_Yvar.mean()
                obs_noise = torch.full(
                    X.shape[:-1] + (1,),
                    fill_value=float(default_noise.item()),
                    device=X.device,
                    dtype=X.dtype,
                )

        if obs_noise is not None:
            var = var + obs_noise

        posterior = SimpleBernoulliPosterior(mean=p, variance=var)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)

        return posterior

    def latent_posterior(self, X: Tensor) -> GPyTorchPosterior:
        # optimize_acqf で X に関する勾配が必要なので no_grad は付けない
        self.eval()
        self.model.eval()
        self.likelihood.eval()

        X_eval = X
        input_tf = getattr(self, "input_transform", None)
        if input_tf is not None:
            X_eval = input_tf(X_eval)

        # ここは likelihood を通さず、latent f の分布を返す
        latent_dist = self.model(X_eval)
        return GPyTorchPosterior(latent_dist)
    
    def set_train_data(
        self,
        inputs: Optional[Union[Tensor, tuple[Tensor, ...]]] = None,
        targets: Optional[Tensor] = None,
        strict: bool = True,
    ) -> None:
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            self.train_inputs = inputs
        if targets is not None:
            if targets.ndim > 1 and targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
            self.train_targets = targets
            self._train_targets = targets

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs,
    ) -> "GPClassificationModel":
        """
        観測 (X, Y) を追加した新しい分類モデルを返す。

        注意:
            - これは厳密な closed-form conditioning ではなく、
              既存データ + 新規データで新しいモデルを再構成する近似実装です。
            - fantasy sample dim 付きの Y には未対応です。
        """
        X_new, Y_new, Yvar_new = _prepare_binary_conditioning_data(X, Y, noise)

        train_X_old = self.train_inputs_raw[0]
        train_Y_old = self.train_targets
        if train_Y_old.ndim > 1 and train_Y_old.shape[-1] == 1:
            train_Y_old = train_Y_old.squeeze(-1)

        train_X_full = torch.cat(
            [train_X_old, X_new.to(dtype=train_X_old.dtype, device=train_X_old.device)],
            dim=0,
        )
        train_Y_full = torch.cat(
            [train_Y_old, Y_new.to(dtype=train_Y_old.dtype, device=train_Y_old.device)],
            dim=0,
        )

        train_Yvar_full = _concat_optional_noise(
            old_Y=train_Y_old,
            old_Yvar=self.train_Yvar,
            new_Y=Y_new,
            new_Yvar=Yvar_new,
            dtype=train_X_old.dtype,
            device=train_X_old.device,
        )

        inducing_points = self.model.variational_strategy.inducing_points.detach().clone()

        new_model = self.__class__(
            train_X=train_X_full,
            train_Y=train_Y_full,
            train_Yvar=train_Yvar_full,
            likelihood=deepcopy(self.likelihood),
            input_transform=deepcopy(self.input_transform),
            mean_module=deepcopy(self.model.mean_module),
            covar_module=deepcopy(self.model.covar_module),
            num_inducing_points=inducing_points.shape[-2],
            inducing_points=inducing_points,
            learn_inducing_locations=getattr(
                self.model.variational_strategy,
                "learn_inducing_locations",
                True,
            ),
        )

        # 学習済みハイパーパラメータ・変分パラメータを引き継ぐ
        new_model.load_state_dict(self.state_dict(), strict=False)
        new_model.eval()
        return new_model
    
    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])


class BinaryClassificationMixedGPModel(ApproximateGPyTorchModel, FantasizeMixin):
    """
    mixed input（連続 + カテゴリ）対応の binary classification model。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Optional[Tensor] = None,   # 追加
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        cont_kernel_factory: Optional[
            Callable[[torch.Size, int, Optional[List[int]]], Kernel]
        ] = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        if len(cat_dims) == 0:
            raise ValueError("cat_dims must be non-empty for GPClassificationMixedModel.")

        if train_Y.ndim > 1 and train_Y.shape[-1] == 1:
            train_Y = train_Y.squeeze(-1)
        train_Y = train_Y.to(dtype=train_X.dtype)

        train_Yvar = _prepare_train_Yvar(
            train_Yvar,
            train_X=train_X,
            train_Y=train_Y,
        )

        self.cat_dims = _normalize_indices(cat_dims, d=train_X.shape[-1])
        self._ignore_X_dims_scaling_check = list(self.cat_dims)

        self.train_inputs = (train_X,)
        self.train_targets = train_Y
        self.train_Yvar = train_Yvar
        self.train_inputs_raw = (train_X,)
        self._train_targets = train_Y

        input_transform = _to_device_dtype_transform(input_transform, train_X)

        if input_transform is not None:
            input_transform.train()
            with torch.no_grad():
                transformed_train_X = input_transform(train_X).detach().clone()
            input_transform.eval()
        else:
            transformed_train_X = train_X.detach().clone()

        inducing_points = _select_inducing_points(
            transformed_train_X,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )

        latent_model = _LatentMixedBinarySVGP(
            inducing_points=inducing_points,
            cat_dims=self.cat_dims,
            train_inputs=transformed_train_X,
            train_targets=train_Y,
            mean_module=mean_module,
            covar_module=covar_module,
            cont_kernel_factory=cont_kernel_factory,
            learn_inducing_locations=learn_inducing_locations,
        )

        likelihood = likelihood or BernoulliLikelihood()

        super().__init__(
            model=latent_model,
            likelihood=likelihood,
            num_outputs=1,
        )

        self.input_transform = input_transform
        self.to(train_X)

    def forward(self, X: Tensor) -> MultivariateNormal:
        if isinstance(X, tuple):
            X = X[0]
        if self.training:
            X = self.transform_inputs(X)
        return self.model(X)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> SimpleBernoulliPosterior:
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output_indices."
            )

        self.eval()
        self.likelihood.eval()

        if isinstance(X, tuple):
            X = X[0]

        X_tf = self.transform_inputs(X) if self.input_transform is not None else X
        latent_dist = self.model(X_tf)
        pred_dist = self.likelihood(latent_dist)

        p = pred_dist.mean
        var = pred_dist.variance
        if p.ndim == X.ndim - 1:
            p = p.unsqueeze(-1)
            var = var.unsqueeze(-1)

        obs_noise = None

        if torch.is_tensor(observation_noise):
            obs_noise = _expand_observation_noise_tensor(observation_noise, X=X)

        elif observation_noise:
            noise_model = getattr(self, "noise_model", None)

            if noise_model is not None:
                noise_in = (
                    X_tf
                    if getattr(self, "noise_model_uses_transformed_inputs", True)
                    else X
                )
                obs_noise = noise_model.posterior(noise_in).mean
                obs_noise = _expand_observation_noise_tensor(obs_noise, X=X)

            elif self.train_Yvar is not None:
                default_noise = self.train_Yvar.mean()
                obs_noise = torch.full(
                    X.shape[:-1] + (1,),
                    fill_value=float(default_noise.item()),
                    device=X.device,
                    dtype=X.dtype,
                )

        if obs_noise is not None:
            var = var + obs_noise

        posterior = SimpleBernoulliPosterior(mean=p, variance=var)

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)

        return posterior

    def latent_posterior(self, X: Tensor) -> GPyTorchPosterior:
        # optimize_acqf で X に関する勾配が必要なので no_grad は付けない
        self.eval()
        self.model.eval()
        self.likelihood.eval()

        X_eval = X
        input_tf = getattr(self, "input_transform", None)
        if input_tf is not None:
            X_eval = input_tf(X_eval)

        # ここは likelihood を通さず、latent f の分布を返す
        latent_dist = self.model(X_eval)
        return GPyTorchPosterior(latent_dist)
    
    def set_train_data(
        self,
        inputs: Optional[Union[Tensor, tuple[Tensor, ...]]] = None,
        targets: Optional[Tensor] = None,
        strict: bool = True,
    ) -> None:
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            self.train_inputs = inputs
        if targets is not None:
            if targets.ndim > 1 and targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
            self.train_targets = targets
            self._train_targets = targets

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs,
    ) -> "GPClassificationMixedModel":
        """
        mixed classification model 用の condition_on_observations。

        注意:
            - これは厳密な closed-form conditioning ではなく、
              新しいモデルを再構成する近似実装です。
            - fantasy sample dim 付きの Y には未対応です。
        """
        X_new, Y_new, Yvar_new = _prepare_binary_conditioning_data(X, Y, noise)

        train_X_old = self.train_inputs_raw[0]
        train_Y_old = self.train_targets
        if train_Y_old.ndim > 1 and train_Y_old.shape[-1] == 1:
            train_Y_old = train_Y_old.squeeze(-1)

        train_X_full = torch.cat(
            [train_X_old, X_new.to(dtype=train_X_old.dtype, device=train_X_old.device)],
            dim=0,
        )
        train_Y_full = torch.cat(
            [train_Y_old, Y_new.to(dtype=train_Y_old.dtype, device=train_Y_old.device)],
            dim=0,
        )

        train_Yvar_full = _concat_optional_noise(
            old_Y=train_Y_old,
            old_Yvar=self.train_Yvar,
            new_Y=Y_new,
            new_Yvar=Yvar_new,
            dtype=train_X_old.dtype,
            device=train_X_old.device,
        )

        inducing_points = self.model.variational_strategy.inducing_points.detach().clone()

        new_model = self.__class__(
            train_X=train_X_full,
            train_Y=train_Y_full,
            cat_dims=list(self.cat_dims),
            train_Yvar=train_Yvar_full,
            likelihood=deepcopy(self.likelihood),
            input_transform=deepcopy(self.input_transform),
            mean_module=deepcopy(self.model.mean_module),
            covar_module=deepcopy(self.model.covar_module),
            cont_kernel_factory=None,
            num_inducing_points=inducing_points.shape[-2],
            inducing_points=inducing_points,
            learn_inducing_locations=getattr(
                self.model.variational_strategy,
                "learn_inducing_locations",
                True,
            ),
        )

        new_model.load_state_dict(self.state_dict(), strict=False)
        new_model.eval()
        return new_model
    
    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])