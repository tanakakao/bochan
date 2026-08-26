"""Pure-PyTorch ALIGNN-backed crystal-structure encoder.

The optional ALIGNN package is imported only when this adapter constructs an
upstream model. Bochan's canonical upstream path uses
``ALIGNNAtomWisePure`` / ``TorchGraph`` and therefore does not require DGL.
Injected encoders remain supported for tests and custom integrations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
_ENCODER_PREFIXES = ("module.model.", "model.", "module.encoder.", "encoder.", "module.")
_MATERIALS_INSTALL_HINT = (
    "ALIGNN support requires alignn with its pure-PyTorch model and graph builder. "
    "Install alignn==2026.8.11 or a compatible newer release. DGL is not required."
)
_PURE_MODEL_NAME = "alignn_atomwise_pure"


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _default_pure_config() -> dict[str, object]:
    """Return the current upstream scalar-property pure ALIGNN defaults."""

    return {
        "name": _PURE_MODEL_NAME,
        "alignn_layers": 4,
        "gcn_layers": 4,
        "atom_input_features": 92,
        "edge_input_features": 80,
        "triplet_input_features": 40,
        "embedding_features": 64,
        "hidden_features": 256,
        "output_features": 1,
        "calculate_gradient": False,
        "gradwise_weight": 0.0,
        "energy_mult_natoms": False,
    }


def _upstream_alignn_classes() -> tuple[type[nn.Module], type[Any]]:
    """Return upstream pure-PyTorch ALIGNN model/config classes lazily."""

    try:
        module = import_module("alignn.models.alignn_atomwise_pure")
    except ImportError as error:
        raise ImportError(_MATERIALS_INSTALL_HINT) from error

    model_class = getattr(module, "ALIGNNAtomWisePure", None)
    config_class = getattr(module, "ALIGNNAtomWisePureConfig", None)
    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise RuntimeError(
            "The installed ALIGNN package does not expose "
            "alignn.models.alignn_atomwise_pure.ALIGNNAtomWisePure."
        )
    if not isinstance(config_class, type):
        raise RuntimeError(
            "The installed ALIGNN package does not expose ALIGNNAtomWisePureConfig."
        )
    return cast(type[nn.Module], model_class), config_class


def _prepare_graph_object(value: Any, *, device: torch.device, dtype: torch.dtype) -> Any:
    """Move tensors / TorchGraph-like graph data to the encoder device/dtype."""

    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_prepare_graph_object(item, device=device, dtype=dtype) for item in value)
    if isinstance(value, list):
        return [_prepare_graph_object(item, device=device, dtype=dtype) for item in value]
    if isinstance(value, dict):
        return {
            key: _prepare_graph_object(item, device=device, dtype=dtype)
            for key, item in value.items()
        }

    move = getattr(value, "to", None)
    moved = move(device) if callable(move) else value
    for frame_name in ("ndata", "edata"):
        frame = getattr(moved, frame_name, None)
        if frame is None:
            continue
        try:
            keys = list(frame.keys())
        except AttributeError:
            continue
        for key in keys:
            tensor = frame[key]
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                frame[key] = tensor.to(device=device, dtype=dtype)
            elif torch.is_tensor(tensor):
                frame[key] = tensor.to(device=device)
    return moved


def _bond_cosines(r_ij: Tensor, r_jk: Tensor, eps: float = 1e-12) -> Tensor:
    """Return ALIGNN bond-angle cosines using the pure-Torch convention."""

    numerator = -(r_ij * r_jk).sum(dim=-1)
    denominator = r_ij.norm(dim=-1).clamp_min(eps) * r_jk.norm(dim=-1).clamp_min(eps)
    return (numerator / denominator).clamp(-1.0, 1.0)


def _pure_cutoff_values(encoder: nn.Module, bondlength: Tensor) -> Tensor:
    """Apply upstream pure ALIGNN's optional smooth cutoff transformation."""

    try:
        module = import_module("alignn.models.alignn_atomwise_pure")
    except ImportError as error:
        raise ImportError(_MATERIALS_INSTALL_HINT) from error
    cutoff = getattr(module, "cutoff_function_based_edges", None)
    if not callable(cutoff):
        raise RuntimeError("The installed pure ALIGNN module does not expose cutoff_function_based_edges.")
    config = encoder.config
    return cutoff(
        bondlength,
        inner_cutoff=float(config.inner_cutoff),
        exponent=int(config.exponent),
    )


class ALIGNNEncoder(MaterialEncoder):
    """Encode crystal ``TorchGraph`` objects into pooled ALIGNN representations.

    The canonical upstream encoder is ``ALIGNNAtomWisePure``. Bochan executes
    its representation backbone directly and returns the mean-pooled hidden
    state immediately before the scalar property head. This keeps ALIGNN-GP and
    ALIGNN-DKL entirely on PyTorch tensors while preserving the normal ALIGNN
    pair-distance and line-graph angle message passing.

    An injected encoder may expose ``encode(graph)`` or
    ``forward_features(graph)``. Otherwise its normal ``forward`` output is
    treated as the representation.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        checkpoint: Checkpoint | None = None,
        output_dim: int | None = None,
        config: Mapping[str, object] | object | None = None,
        strict_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        inferred_output_dim: int | None
        initialization: Initialization

        if encoder is None:
            model_class, config_class = _upstream_alignn_classes()
            if config is None:
                resolved_config = config_class(**_default_pure_config())
            elif isinstance(config, Mapping):
                config_kwargs = dict(config)
                config_kwargs.setdefault("name", _PURE_MODEL_NAME)
                if config_kwargs.get("name") != _PURE_MODEL_NAME:
                    raise ValueError(
                        "Bochan's canonical ALIGNN path is pure PyTorch; "
                        f"model config name must be {_PURE_MODEL_NAME!r}."
                    )
                resolved_config = config_class(**config_kwargs)
            elif isinstance(config, config_class):
                resolved_config = config
            else:
                raise TypeError("config must be an ALIGNNAtomWisePureConfig instance or mapping.")

            if int(getattr(resolved_config, "extra_features", 0)) != 0:
                raise NotImplementedError("ALIGNNEncoder does not support upstream extra_features yet.")
            encoder = model_class(resolved_config)
            inferred_output_dim = self._infer_output_dim(encoder)
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
            raise ValueError(
                "output_dim is required when an injected ALIGNN encoder exposes neither "
                "output_dim nor config.hidden_features."
            )

        self.encoder = encoder
        self._output_dim = inferred_output_dim
        self._initialization: Initialization = initialization
        self._checkpoint_path: str | None = None

        if checkpoint is not None:
            self.load_checkpoint(checkpoint, strict=strict_checkpoint)

    @staticmethod
    def _infer_output_dim(encoder: nn.Module) -> int | None:
        value = getattr(encoder, "output_dim", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        config = getattr(encoder, "config", None)
        value = getattr(config, "hidden_features", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        value = getattr(encoder, "hidden_features", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return None

    @property
    def output_dim(self) -> int:
        """Return the pooled structure representation width."""

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

    def backbone_modules(self) -> tuple[nn.Module, ...]:
        """Return modules that constitute the representation backbone."""

        required = ("atom_embedding", "edge_embedding", "alignn_layers", "gcn_layers")
        if all(hasattr(self.encoder, name) for name in required):
            modules: list[nn.Module] = []
            for name in (
                "atom_embedding",
                "edge_embedding",
                "angle_embedding",
                "alignn_layers",
                "gcn_layers",
            ):
                module = getattr(self.encoder, name, None)
                if isinstance(module, nn.Module):
                    modules.append(module)
            if modules:
                return tuple(modules)
        return (self.encoder,)

    def graph_conv_layers(self) -> tuple[nn.Module, ...]:
        """Return ordered ALIGNN/GCN blocks for partial fine tuning."""

        layers: list[nn.Module] = []
        for name in ("alignn_layers", "gcn_layers"):
            module = getattr(self.encoder, name, None)
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                layers.extend(module)
        return tuple(layers)

    def _is_pure_upstream_layout(self) -> bool:
        required = (
            "atom_embedding",
            "edge_embedding",
            "angle_embedding",
            "alignn_layers",
            "gcn_layers",
            "config",
        )
        return all(hasattr(self.encoder, name) for name in required)

    def _encode_pure_backbone(self, graph: Any) -> Tensor:
        """Execute the pure ALIGNN backbone and stop before the property head."""

        if isinstance(graph, (tuple, list)):
            if len(graph) < 2:
                raise ValueError("Pure ALIGNN graph input requires (graph, line_graph).")
            g, lg = graph[0], graph[1]
        else:
            raise TypeError("Pure ALIGNN graph input must be a (graph, line_graph) pair.")

        for name in ("src", "dst", "num_nodes", "ndata", "edata"):
            if not hasattr(g, name):
                raise TypeError(f"Pure ALIGNN graph is missing {name!r}.")
        for name in ("src", "dst", "ndata", "edata"):
            if not hasattr(lg, name):
                raise TypeError(f"Pure ALIGNN line graph is missing {name!r}.")

        r = g.edata["r"]
        bondlength = torch.norm(r, dim=1)
        x = self.encoder.atom_embedding(g.ndata["atom_features"])

        config = self.encoder.config
        if bool(getattr(config, "use_cutoff_function", False)):
            cutoff_values = _pure_cutoff_values(self.encoder, bondlength)
            if bool(getattr(config, "multiply_cutoff", False)):
                y = self.encoder.edge_embedding(bondlength) * cutoff_values.unsqueeze(-1)
            else:
                y = self.encoder.edge_embedding(cutoff_values)
        else:
            y = self.encoder.edge_embedding(bondlength)

        alignn_layers = self.encoder.alignn_layers
        if len(alignn_layers) > 0:
            h = _bond_cosines(r[lg.src], r[lg.dst])
            z = self.encoder.angle_embedding(h)
            for layer in alignn_layers:
                x, y, z = layer(g, lg, x, y, z)

        for layer in self.encoder.gcn_layers:
            x, y = layer(g, x, y)

        if x.ndim != 2 or x.shape[0] != int(g.num_nodes):
            raise ValueError("Pure ALIGNN backbone must return one hidden vector per atom.")
        return x.mean(dim=0)

    def _encode_one(self, graph: Any) -> Tensor:
        encode = getattr(self.encoder, "encode", None)
        if callable(encode):
            result = encode(graph)
        else:
            forward_features = getattr(self.encoder, "forward_features", None)
            if callable(forward_features):
                result = forward_features(graph)
            elif self._is_pure_upstream_layout():
                result = self._encode_pure_backbone(graph)
            else:
                result = self.encoder(graph)

        if not torch.is_tensor(result):
            raise TypeError("The ALIGNN encoder must return a Tensor representation.")
        if result.ndim == 2 and result.shape[0] == 1:
            result = result.squeeze(0)
        if result.ndim != 1 or result.shape[0] != self.output_dim:
            raise ValueError(
                "Each ALIGNN structure representation must have shape "
                f"[{self.output_dim}], got {tuple(result.shape)}."
            )
        return result

    def forward(self, graphs: Sequence[Any]) -> Tensor:
        """Return one pooled structure representation per graph pair."""

        if not isinstance(graphs, Sequence) or isinstance(graphs, (str, bytes)):
            raise TypeError("graphs must be a sequence of graph objects.")
        if not graphs:
            raise ValueError("graphs must contain at least one structure.")

        reference = self._floating_reference()
        if reference is None:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()
        else:
            device = reference.device
            dtype = reference.dtype

        encoded = []
        for graph in graphs:
            prepared = _prepare_graph_object(graph, device=device, dtype=dtype)
            representation = self._encode_one(prepared)
            if representation.device != device or representation.dtype != dtype:
                raise ValueError("ALIGNN encoder output must match the encoder device and dtype.")
            if not torch.isfinite(representation).all():
                raise FloatingPointError("ALIGNN encoder produced non-finite structure features.")
            encoded.append(representation)
        return torch.stack(encoded, dim=0)

    def load_checkpoint(self, checkpoint: Checkpoint, *, strict: bool = True) -> None:
        """Load pure ALIGNN weights into the wrapped encoder."""

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
                raw_key.removeprefix(prefix)
                for prefix in _ENCODER_PREFIXES
                if raw_key.startswith(prefix)
            )
            matched_key = next((key for key in candidates if key in target_keys), None)
            if matched_key is None:
                continue
            if matched_key in encoder_state:
                raise ValueError(f"Checkpoint contains duplicate ALIGNN encoder key: {matched_key}")
            encoder_state[matched_key] = value

        if not encoder_state:
            raise ValueError(
                "Checkpoint contains no weights matching the pure-PyTorch ALIGNN encoder. "
                "Legacy DGL ALIGNN checkpoints are not loaded implicitly."
            )

        self.encoder.load_state_dict(encoder_state, strict=strict)
        self._initialization = "checkpoint"
        self._checkpoint_path = checkpoint_path


__all__ = ["ALIGNNEncoder"]
