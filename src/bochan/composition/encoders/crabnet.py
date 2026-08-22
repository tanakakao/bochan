"""CrabNet-backed material encoder.

The third-party CrabNet package is imported only when this adapter constructs
an upstream encoder.  Importing :mod:`bochan.composition` therefore does not
require Bochan's ``materials`` optional dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor, nn

from .base import MaterialEncoder

Checkpoint = str | PathLike[str] | Mapping[str, object]
Initialization = Literal["random", "injected", "checkpoint"]

_CHECKPOINT_STATE_KEYS = ("weights", "state_dict", "model_state_dict")
_ENCODER_PREFIXES = (
    "module.model.encoder.",
    "model.encoder.",
    "module.encoder.",
    "encoder.",
    "module.",
)
_MATERIALS_INSTALL_HINT = (
    "CrabNet support requires Bochan's optional 'materials' dependency. "
    'Install it with `pip install "bochan[materials]"`.'
)


def _upstream_encoder_class() -> type[nn.Module]:
    """Return the upstream CrabNet encoder class after a lazy import."""

    try:
        module = import_module("crabnet.kingcrab")
    except ImportError as error:
        raise ImportError(_MATERIALS_INSTALL_HINT) from error

    encoder_class = getattr(module, "Encoder", None)
    if not isinstance(encoder_class, type) or not issubclass(encoder_class, nn.Module):
        raise RuntimeError(
            "The installed CrabNet package does not expose crabnet.kingcrab.Encoder as a torch.nn.Module."
        )
    return cast(type[nn.Module], encoder_class)


def _positive_int(name: str, value: int) -> int:
    """Validate and return a positive integer configuration value."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class CrabNetEncoder(MaterialEncoder):
    """Encode elemental compositions with CrabNet's transformer encoder.

    Args:
        encoder: Optional injected encoder with the upstream call signature
            ``encoder(element_ids, fractions)``.  Injection supports custom or
            already-created CrabNet encoders without importing the optional
            dependency.
        checkpoint: Optional path or checkpoint mapping.  Upstream CrabNet
            checkpoints containing a ``weights`` mapping, common
            ``state_dict`` wrappers, direct encoder state dictionaries, and
            :class:`CrabNetEncoder` state dictionaries are supported.
        output_dim: Encoder output width.  It is inferred from ``d_model`` or
            ``output_dim`` on an injected encoder when omitted.
        d_model: Upstream transformer feature width.
        num_layers: Number of upstream transformer layers.
        num_heads: Number of upstream attention heads.
        dim_feedforward: Upstream transformer feed-forward width.
        dropout: Upstream transformer dropout probability.
        pe_resolution: Resolution of the linear fractional encoder.
        ple_resolution: Resolution of the logarithmic fractional encoder.
        elem_prop: CrabNet elemental-property table name.
        strict_checkpoint: Require every encoder state entry when loading a
            checkpoint.

    ``checkpoint=None`` is intentionally explicit: a constructed upstream
    encoder is randomly initialized, while an injected encoder retains its
    caller-provided initialization.  Neither case is described as pretrained.
    Process variables are deliberately excluded from this composition-only
    encoder and should be combined through a material/process fusion module.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        checkpoint: Checkpoint | None = None,
        output_dim: int | None = None,
        d_model: int = 512,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pe_resolution: int = 5000,
        ple_resolution: int = 5000,
        elem_prop: str = "mat2vec",
        strict_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        inferred_output_dim: int | None
        initialization: Initialization

        if encoder is None:
            d_model = _positive_int("d_model", d_model)
            num_layers = _positive_int("num_layers", num_layers)
            num_heads = _positive_int("num_heads", num_heads)
            dim_feedforward = _positive_int("dim_feedforward", dim_feedforward)
            pe_resolution = _positive_int("pe_resolution", pe_resolution)
            ple_resolution = _positive_int("ple_resolution", ple_resolution)
            if d_model % 2:
                raise ValueError("d_model must be even for CrabNet fractional encodings.")
            if d_model % num_heads:
                raise ValueError("d_model must be divisible by num_heads.")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in the interval [0, 1).")
            if not elem_prop:
                raise ValueError("elem_prop must be a non-empty string.")

            encoder_class = _upstream_encoder_class()
            encoder = encoder_class(
                d_model=d_model,
                N=num_layers,
                heads=num_heads,
                extend_features=None,
                fractional=True,
                attention=True,
                compute_device=None,
                pe_resolution=pe_resolution,
                ple_resolution=ple_resolution,
                elem_prop=elem_prop,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            inferred_output_dim = d_model
            initialization = "random"
        else:
            if not isinstance(encoder, nn.Module):
                raise TypeError("encoder must be a torch.nn.Module.")
            inferred_output_dim = self._infer_output_dim(encoder)
            initialization = "injected"

        if output_dim is not None:
            output_dim = _positive_int("output_dim", output_dim)
            if inferred_output_dim is not None and output_dim != inferred_output_dim:
                raise ValueError(
                    "output_dim does not match the encoder's declared feature width: "
                    f"{output_dim} != {inferred_output_dim}."
                )
            inferred_output_dim = output_dim
        if inferred_output_dim is None:
            raise ValueError("output_dim is required when an injected encoder exposes neither d_model nor output_dim.")

        self.encoder = encoder
        self._output_dim = inferred_output_dim
        self._initialization: Initialization = initialization
        self._checkpoint_path: str | None = None

        if checkpoint is not None:
            self.load_checkpoint(checkpoint, strict=strict_checkpoint)

    @staticmethod
    def _infer_output_dim(encoder: nn.Module) -> int | None:
        for attribute in ("d_model", "output_dim"):
            value = getattr(encoder, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

    @property
    def output_dim(self) -> int:
        """Return the width of the pooled material representation."""

        return self._output_dim

    @property
    def initialization(self) -> Initialization:
        """Return whether weights are random, injected, or checkpoint-loaded."""

        return self._initialization

    @property
    def checkpoint_path(self) -> str | None:
        """Return the loaded checkpoint path, if loading used a path."""

        return self._checkpoint_path

    def _floating_reference(self) -> Tensor | None:
        for value in (*self.encoder.parameters(), *self.encoder.buffers()):
            if value.is_floating_point():
                return value
        return None

    def _validate_inputs(self, element_ids: Tensor, fractions: Tensor) -> None:
        if not torch.is_tensor(element_ids):
            raise TypeError("element_ids must be a Tensor.")
        if not torch.is_tensor(fractions):
            raise TypeError("fractions must be a Tensor.")
        if element_ids.shape != fractions.shape:
            raise ValueError("element_ids and fractions must have identical shapes.")
        if element_ids.ndim == 0:
            raise ValueError("element_ids and fractions must include an element dimension.")
        if element_ids.numel() == 0 or element_ids.shape[-1] == 0:
            raise ValueError("Each composition must provide at least one element slot.")
        if element_ids.dtype != torch.long:
            raise TypeError("element_ids must have dtype torch.long.")
        if not fractions.is_floating_point():
            raise TypeError("fractions must have a floating-point dtype.")
        if element_ids.device != fractions.device:
            raise ValueError("element_ids and fractions must be on the same device.")

        reference = self._floating_reference()
        if reference is not None:
            if fractions.device != reference.device:
                raise ValueError(
                    "fractions must be on the same device as the encoder. "
                    "Move the adapter and inputs with `.to(device=...)`."
                )
            if fractions.dtype != reference.dtype:
                raise ValueError(
                    "fractions must have the same dtype as the encoder. "
                    "Move the adapter and inputs with `.to(dtype=...)`."
                )

        if not torch.isfinite(fractions).all():
            raise ValueError("fractions must contain only finite values.")
        if (element_ids < 0).any():
            raise ValueError("element_ids must be non-negative; zero is reserved for padding.")
        if (fractions < 0).any():
            raise ValueError("fractions must be non-negative.")

        active = element_ids.ne(0)
        if not active.any(dim=-1).all():
            raise ValueError("Each composition must contain at least one non-padding element.")
        if (fractions.masked_select(~active) != 0).any():
            raise ValueError("Padding slots with element_id zero must have fraction zero.")
        if (fractions.masked_select(active) <= 0).any():
            raise ValueError("Non-padding elements must have positive fractions.")

        row_sums = fractions.sum(dim=-1)
        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=1e-4,
            atol=1e-6,
        ):
            raise ValueError("Fractions for each composition must sum to one.")

        embedder = getattr(self.encoder, "embed", None)
        embedding = getattr(embedder, "cbfv", None)
        if isinstance(embedding, nn.Embedding):
            maximum = int(element_ids.max().item())
            if maximum >= embedding.num_embeddings:
                raise ValueError(
                    "element_ids contains an atomic number outside the installed "
                    f"CrabNet table: {maximum} >= {embedding.num_embeddings}."
                )

    def element_embeddings(self, element_ids: Tensor, fractions: Tensor) -> Tensor:
        """Return contextualized per-element embeddings.

        The tensors have shape ``(..., n_elements)`` and the result has shape
        ``(..., n_elements, output_dim)``.  Atomic number zero denotes padding.
        Leading batch and q-batch dimensions are preserved.
        """

        self._validate_inputs(element_ids, fractions)
        leading_shape = element_ids.shape[:-1]
        n_elements = element_ids.shape[-1]
        flat_ids = element_ids.reshape(-1, n_elements)
        flat_fractions = fractions.reshape(-1, n_elements)

        embeddings = self.encoder(flat_ids, flat_fractions)
        if not torch.is_tensor(embeddings):
            raise TypeError("The CrabNet encoder must return a Tensor.")
        expected_shape = (flat_ids.shape[0], n_elements, self.output_dim)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"The CrabNet encoder returned an unexpected shape: {tuple(embeddings.shape)} != {expected_shape}."
            )
        if embeddings.device != fractions.device or embeddings.dtype != fractions.dtype:
            raise ValueError("The CrabNet encoder output must match the input fractions' device and dtype.")
        return embeddings.reshape(*leading_shape, n_elements, self.output_dim)

    def forward(self, element_ids: Tensor, fractions: Tensor) -> Tensor:
        """Return a padding-aware mean of contextualized element embeddings."""

        embeddings = self.element_embeddings(element_ids, fractions)
        active = element_ids.ne(0).unsqueeze(-1)
        masked_embeddings = embeddings.masked_fill(~active, 0)
        counts = active.sum(dim=-2).to(dtype=embeddings.dtype)
        return masked_embeddings.sum(dim=-2) / counts

    def load_checkpoint(self, checkpoint: Checkpoint, *, strict: bool = True) -> None:
        """Load encoder weights from an upstream or adapter checkpoint.

        Path checkpoints are loaded on CPU with ``weights_only=True``.  Only
        tensors belonging to the wrapped encoder are accepted; CrabNet output
        heads and scaler metadata are deliberately ignored.
        """

        checkpoint_path: str | None = None
        if isinstance(checkpoint, (str, PathLike)):
            path = Path(checkpoint)
            if not path.is_file():
                raise FileNotFoundError(f"CrabNet checkpoint does not exist: {path}")
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint_path = str(path)
        elif isinstance(checkpoint, Mapping):
            loaded = checkpoint
        else:
            raise TypeError("checkpoint must be a path or mapping.")

        if not isinstance(loaded, Mapping):
            raise TypeError("CrabNet checkpoint must contain a mapping.")
        state: Mapping[object, object] = loaded
        for key in _CHECKPOINT_STATE_KEYS:
            candidate = loaded.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                break

        target_keys = set(self.encoder.state_dict())
        encoder_state: dict[str, Tensor] = {}
        for raw_key, value in state.items():
            if not isinstance(raw_key, str) or not torch.is_tensor(value):
                continue
            candidates = [raw_key]
            candidates.extend(
                raw_key.removeprefix(prefix) for prefix in _ENCODER_PREFIXES if raw_key.startswith(prefix)
            )
            matched_key = next((key for key in candidates if key in target_keys), None)
            if matched_key is None:
                continue
            if matched_key in encoder_state:
                raise ValueError(f"Checkpoint contains duplicate encoder key: {matched_key}")
            encoder_state[matched_key] = value

        if not encoder_state:
            raise ValueError("Checkpoint contains no weights matching the CrabNet encoder.")

        self.encoder.load_state_dict(encoder_state, strict=strict)
        self._initialization = "checkpoint"
        self._checkpoint_path = checkpoint_path


__all__ = ["CrabNetEncoder"]
