"""ALIGNN-backed crystal-structure encoder.

The optional ALIGNN package is imported only when this adapter constructs an
upstream model. Injected ``torch.nn.Module`` instances keep the core Bochan
package importable without ALIGNN or DGL installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from .base import MaterialEncoder

Checkpoint = str | PathLike[str] | Mapping[str, object]
Initialization = Literal["random", "injected", "checkpoint"]

_CHECKPOINT_STATE_KEYS = ("model", "weights", "state_dict", "model_state_dict")
_ENCODER_PREFIXES = (
    "module.model.",
    "model.",
    "module.encoder.",
    "encoder.",
    "module.",
)
_ALIGNN_INSTALL_HINT = (
    "ALIGNN support requires the optional upstream ALIGNN stack. "
    "Install a compatible `alignn` package and its DGL runtime, or inject an "
    "already-created ALIGNN-compatible torch.nn.Module."
)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _upstream_alignn(config: Mapping[str, object] | None) -> nn.Module:
    """Construct the upstream scalar-property ALIGNN model lazily."""

    try:
        module = import_module("alignn.models.alignn")
    except Exception as error:
        raise ImportError(_ALIGNN_INSTALL_HINT) from error

    model_class = getattr(module, "ALIGNN", None)
    config_class = getattr(module, "ALIGNNConfig", None)
    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise RuntimeError("The installed ALIGNN package does not expose ALIGNN as torch.nn.Module.")
    if not isinstance(config_class, type):
        raise RuntimeError("The installed ALIGNN package does not expose ALIGNNConfig.")

    config_data = dict(config or {})
    config_data.setdefault("name", "alignn")
    config_data.setdefault("output_features", 1)
    config_data.setdefault("classification", False)
    config_data.setdefault("extra_features", 0)
    if bool(config_data.get("classification", False)):
        raise ValueError("ALIGNNEncoder currently supports regression encoders only.")
    if int(config_data.get("extra_features", 0)) != 0:
        raise ValueError(
            "ALIGNNEncoder keeps process variables outside the upstream model; "
            "ALIGNN config.extra_features must therefore be zero."
        )

    try:
        upstream_config = config_class(**config_data)
        return cast(nn.Module, model_class(upstream_config))
    except Exception as error:
        raise ImportError(_ALIGNN_INSTALL_HINT) from error


def _module_output_dim(encoder: nn.Module) -> int | None:
    for attribute in ("output_dim", "hidden_features"):
        value = getattr(encoder, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value

    config = getattr(encoder, "config", None)
    value = getattr(config, "hidden_features", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _floating_reference(module: nn.Module) -> Tensor | None:
    for value in (*module.parameters(), *module.buffers()):
        if value.is_floating_point():
            return value
    return None


def _move_structure_input(value: Any, *, reference: Tensor | None) -> Any:
    """Move tensor or DGL-like graph inputs to the encoder device and dtype."""

    if value is None or reference is None:
        return value
    if torch.is_tensor(value):
        dtype = reference.dtype if value.is_floating_point() else value.dtype
        return value.to(device=reference.device, dtype=dtype)
    if isinstance(value, tuple):
        return tuple(_move_structure_input(item, reference=reference) for item in value)
    if isinstance(value, list):
        return [_move_structure_input(item, reference=reference) for item in value]

    prepared = value
    local_var = getattr(prepared, "local_var", None)
    if callable(local_var):
        prepared = local_var()
    move = getattr(prepared, "to", None)
    if callable(move):
        prepared = move(reference.device)

    for store_name in ("ndata", "edata"):
        store = getattr(prepared, store_name, None)
        if store is None or not hasattr(store, "items"):
            continue
        for key, tensor in list(store.items()):
            if not torch.is_tensor(tensor):
                continue
            dtype = reference.dtype if tensor.is_floating_point() else tensor.dtype
            store[key] = tensor.to(device=reference.device, dtype=dtype)
    return prepared


class ALIGNNEncoder(MaterialEncoder):
    """Encode one crystal structure into ALIGNN's pooled graph representation.

    Standard upstream ALIGNN predicts a property by applying ``fc`` after the
    graph-level average-pooling module ``readout``. This adapter captures that
    pooled tensor and exposes it as the material representation used by Bochan.

    For test doubles or custom encoders without ``readout``, ``forward`` may
    directly return a one-dimensional latent tensor whose width equals
    ``output_dim``.

    Args:
        encoder: Optional injected upstream or ALIGNN-compatible module.
        checkpoint: Optional path or checkpoint mapping.
        config: Optional upstream ALIGNNConfig values when constructing ALIGNN.
        output_dim: Explicit pooled feature width for custom encoders.
        strict_checkpoint: Require a complete encoder checkpoint state.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        checkpoint: Checkpoint | None = None,
        config: Mapping[str, object] | None = None,
        output_dim: int | None = None,
        strict_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if encoder is None:
            encoder = _upstream_alignn(config)
            initialization: Initialization = "random"
        else:
            if not isinstance(encoder, nn.Module):
                raise TypeError("encoder must be a torch.nn.Module.")
            if config is not None:
                raise ValueError("config is valid only when ALIGNNEncoder constructs the upstream model.")
            initialization = "injected"

        inferred_output_dim = _module_output_dim(encoder)
        if output_dim is not None:
            output_dim = _positive_int("output_dim", output_dim)
            if inferred_output_dim is not None and output_dim != inferred_output_dim:
                raise ValueError(
                    "output_dim does not match the encoder's declared pooled feature width: "
                    f"{output_dim} != {inferred_output_dim}."
                )
            inferred_output_dim = output_dim
        if inferred_output_dim is None:
            raise ValueError(
                "output_dim is required when the injected ALIGNN encoder exposes neither "
                "output_dim, hidden_features, nor config.hidden_features."
            )

        upstream_config = getattr(encoder, "config", None)
        if bool(getattr(upstream_config, "classification", False)):
            raise ValueError("ALIGNNEncoder currently supports regression encoders only.")
        if int(getattr(upstream_config, "extra_features", 0) or 0) != 0:
            raise ValueError(
                "ALIGNNEncoder keeps process variables outside the upstream model; "
                "encoder.config.extra_features must therefore be zero."
            )

        self.encoder = encoder
        self._output_dim = inferred_output_dim
        self._initialization: Initialization = initialization
        self._checkpoint_path: str | None = None

        if checkpoint is not None:
            self.load_checkpoint(checkpoint, strict=strict_checkpoint)

    @property
    def output_dim(self) -> int:
        """Return the pooled crystal-representation width."""

        return self._output_dim

    @property
    def initialization(self) -> Initialization:
        """Return whether weights are random, injected, or checkpoint-loaded."""

        return self._initialization

    @property
    def checkpoint_path(self) -> str | None:
        """Return the checkpoint path when loading used a path."""

        return self._checkpoint_path

    def _normalize_graph_contract(self, structure: Any) -> Any:
        """Adapt current upstream two-graph builders to ALIGNN's three-item call."""

        config = getattr(self.encoder, "config", None)
        alignn_layers = int(getattr(config, "alignn_layers", 0) or 0)
        if alignn_layers > 0 and isinstance(structure, tuple) and len(structure) == 2:
            return (*structure, None)
        return structure

    def forward(self, structure: Any) -> Tensor:
        """Return one pooled ALIGNN representation for one crystal structure."""

        reference = _floating_reference(self.encoder)
        prepared = _move_structure_input(structure, reference=reference)
        prepared = self._normalize_graph_contract(prepared)

        readout = getattr(self.encoder, "readout", None)
        if isinstance(readout, nn.Module):
            captured: list[Tensor] = []

            def _capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                if torch.is_tensor(output):
                    captured.append(output)

            handle = readout.register_forward_hook(_capture)
            try:
                _ = self.encoder(prepared)
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError(
                    "ALIGNN readout hook must capture exactly one pooled representation; "
                    f"captured {len(captured)}."
                )
            features = captured[0]
        else:
            features = self.encoder(prepared)

        if not torch.is_tensor(features):
            raise TypeError("ALIGNN encoder must produce a Tensor representation.")
        if features.ndim == 2 and features.shape[0] == 1:
            features = features.squeeze(0)
        if features.ndim != 1 or features.shape[0] != self.output_dim:
            raise ValueError(
                "ALIGNNEncoder expects one pooled vector per structure: "
                f"got {tuple(features.shape)}, expected ({self.output_dim},)."
            )
        if reference is not None and (features.device != reference.device or features.dtype != reference.dtype):
            raise ValueError("ALIGNN pooled features must match the encoder device and dtype.")
        if not torch.isfinite(features).all():
            raise FloatingPointError("ALIGNN pooled representation contains non-finite values.")
        return features

    def load_checkpoint(self, checkpoint: Checkpoint, *, strict: bool = True) -> None:
        """Load an upstream ALIGNN or adapter checkpoint into the wrapped model."""

        checkpoint_path: str | None = None
        if isinstance(checkpoint, (str, PathLike)):
            path = Path(checkpoint)
            if not path.is_file():
                raise FileNotFoundError(f"ALIGNN checkpoint does not exist: {path}")
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint_path = str(path)
        elif isinstance(checkpoint, Mapping):
            loaded = checkpoint
        else:
            raise TypeError("checkpoint must be a path or mapping.")

        if not isinstance(loaded, Mapping):
            raise TypeError("ALIGNN checkpoint must contain a mapping.")
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
                raise ValueError(f"Checkpoint contains duplicate ALIGNN key: {matched_key}")
            encoder_state[matched_key] = value

        if not encoder_state:
            raise ValueError("Checkpoint contains no weights matching the ALIGNN encoder.")

        self.encoder.load_state_dict(encoder_state, strict=strict)
        self._initialization = "checkpoint"
        self._checkpoint_path = checkpoint_path


__all__ = ["ALIGNNEncoder"]
