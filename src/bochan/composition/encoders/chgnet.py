"""CHGNet-backed crystal-structure representation encoder.

The upstream CHGNet dependency is imported lazily. Bochan uses CHGNet's
public ``return_crystal_feas=True`` forward path so crystal representations
remain PyTorch tensors and stay differentiable for later DKL integration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import Tensor, nn

from .base import MaterialEncoder

if TYPE_CHECKING:
    from bochan.structure.adapter import StructureAdapter

Checkpoint = str | PathLike[str] | Mapping[str, object]
Initialization = Literal["pretrained", "injected", "checkpoint"]

_CHGNET_INSTALL_HINT = (
    "CHGNet support requires chgnet>=0.4.2,<0.5. "
    "Install bochan[materials] or install chgnet directly."
)
_SUPPORTED_MODEL_NAMES = {"0.2.0", "0.3.0", "r2scan"}
_FLOAT_GRAPH_FIELDS = ("atom_frac_coord", "neighbor_image", "lattice")


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _upstream_chgnet_class() -> type[nn.Module]:
    """Return upstream ``CHGNet`` after a lazy dependency check."""

    try:
        module = import_module("chgnet.model.model")
    except ImportError as error:
        raise ImportError(_CHGNET_INSTALL_HINT) from error

    model_class = getattr(module, "CHGNet", None)
    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise RuntimeError("The installed chgnet package does not expose chgnet.model.model.CHGNet.")
    return cast(type[nn.Module], model_class)


def _structure_adapter_class() -> type[Any]:
    """Return StructureAdapter lazily to avoid composition/structure import cycles."""

    module = import_module("bochan.structure.adapter")
    adapter_class = getattr(module, "StructureAdapter", None)
    if not isinstance(adapter_class, type):
        raise RuntimeError("bochan.structure.adapter.StructureAdapter is unavailable.")
    return adapter_class


def _is_chgnet_graph(value: Any) -> bool:
    cls = type(value)
    return cls.__name__ == "CrystalGraph" and cls.__module__.startswith("chgnet.graph.")


def _extract_model_dict(checkpoint: Mapping[str, object]) -> Mapping[str, object]:
    """Extract the dictionary accepted by ``CHGNet.from_dict``."""

    model = checkpoint.get("model")
    if isinstance(model, Mapping) and "state_dict" in model and "model_args" in model:
        return model
    if "state_dict" in checkpoint and "model_args" in checkpoint:
        return checkpoint
    raise ValueError(
        "CHGNet checkpoint mapping must contain state_dict/model_args directly "
        "or under the 'model' key."
    )


class CHGNetEncoder(MaterialEncoder):
    """Encode crystal structures with CHGNet's official crystal feature.

    Inputs may be common in-memory crystal structures accepted by
    :meth:`bochan.structure.StructureAdapter.to_pymatgen` or pre-built CHGNet
    ``CrystalGraph`` objects. For normal structures, the graph converter owned
    by the wrapped CHGNet model is used, keeping graph cutoffs consistent with
    the selected pretrained checkpoint.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        model_name: str = "0.3.0",
        checkpoint: Checkpoint | None = None,
        output_dim: int | None = None,
        strict_checkpoint: bool = True,
        adapter: StructureAdapter | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if encoder is None and checkpoint is None and model_name not in _SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"model_name must be one of {sorted(_SUPPORTED_MODEL_NAMES)}, got {model_name!r}."
            )

        initialization: Initialization
        checkpoint_path: str | None = None

        if encoder is None:
            model_class = _upstream_chgnet_class()
            if checkpoint is None:
                load = getattr(model_class, "load", None)
                if not callable(load):
                    raise RuntimeError("The installed CHGNet class does not expose CHGNet.load().")
                encoder = load(model_name=model_name, use_device="cpu", verbose=False)
                initialization = "pretrained"
            elif isinstance(checkpoint, (str, PathLike)):
                path = Path(checkpoint)
                if not path.is_file():
                    raise FileNotFoundError(f"CHGNet checkpoint does not exist: {path}")
                from_file = getattr(model_class, "from_file", None)
                if not callable(from_file):
                    raise RuntimeError("The installed CHGNet class does not expose CHGNet.from_file().")
                encoder = from_file(str(path))
                initialization = "checkpoint"
                checkpoint_path = str(path)
            elif isinstance(checkpoint, Mapping):
                from_dict = getattr(model_class, "from_dict", None)
                if not callable(from_dict):
                    raise RuntimeError("The installed CHGNet class does not expose CHGNet.from_dict().")
                encoder = from_dict(dict(_extract_model_dict(checkpoint)))
                initialization = "checkpoint"
            else:
                raise TypeError("checkpoint must be a path or mapping.")
        else:
            if not isinstance(encoder, nn.Module):
                raise TypeError("encoder must be a torch.nn.Module.")
            initialization = "injected"
            if checkpoint is not None:
                self._load_injected_checkpoint(encoder, checkpoint, strict=strict_checkpoint)
                initialization = "checkpoint"
                if isinstance(checkpoint, (str, PathLike)):
                    checkpoint_path = str(Path(checkpoint))

        inferred_output_dim = self._infer_output_dim(encoder)
        if output_dim is not None:
            output_dim = _positive_int("output_dim", output_dim)
            if inferred_output_dim is not None and output_dim != inferred_output_dim:
                raise ValueError(
                    "output_dim does not match the encoder's declared crystal feature width: "
                    f"{output_dim} != {inferred_output_dim}."
                )
            inferred_output_dim = output_dim
        if inferred_output_dim is None:
            raise ValueError(
                "output_dim is required when the injected CHGNet-compatible encoder "
                "does not expose output_dim or atom_fea_dim."
            )

        adapter_class = _structure_adapter_class()
        if adapter is not None and not isinstance(adapter, adapter_class):
            raise TypeError("adapter must be a StructureAdapter.")

        self.encoder = encoder
        self._output_dim = inferred_output_dim
        self._model_name = model_name
        self._initialization: Initialization = initialization
        self._checkpoint_path = checkpoint_path
        self.adapter = adapter or adapter_class()

    @staticmethod
    def _load_injected_checkpoint(
        encoder: nn.Module,
        checkpoint: Checkpoint,
        *,
        strict: bool,
    ) -> None:
        if isinstance(checkpoint, (str, PathLike)):
            path = Path(checkpoint)
            if not path.is_file():
                raise FileNotFoundError(f"CHGNet checkpoint does not exist: {path}")
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        elif isinstance(checkpoint, Mapping):
            loaded = checkpoint
        else:
            raise TypeError("checkpoint must be a path or mapping.")
        if not isinstance(loaded, Mapping):
            raise TypeError("Injected CHGNet checkpoint must deserialize to a mapping.")
        state: object = loaded
        for key in ("state_dict", "model_state_dict", "weights"):
            candidate = loaded.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                break
        if not isinstance(state, Mapping):
            raise TypeError("Injected CHGNet checkpoint does not contain a state dict.")
        encoder.load_state_dict(dict(state), strict=strict)

    @staticmethod
    def _infer_output_dim(encoder: nn.Module) -> int | None:
        value = getattr(encoder, "output_dim", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value

        atom_fea_dim = getattr(encoder, "atom_fea_dim", None)
        if not isinstance(atom_fea_dim, int) or isinstance(atom_fea_dim, bool) or atom_fea_dim <= 0:
            return None
        if bool(getattr(encoder, "mlp_first", True)):
            return atom_fea_dim

        read_out_type = getattr(encoder, "read_out_type", "ave")
        if read_out_type != "attn":
            return atom_fea_dim
        model_args = getattr(encoder, "model_args", {})
        num_heads = model_args.get("num_heads", 3) if isinstance(model_args, Mapping) else 3
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
            return None
        return atom_fea_dim * num_heads

    @property
    def output_dim(self) -> int:
        """Return CHGNet crystal feature width."""

        return self._output_dim

    @property
    def model_name(self) -> str:
        """Return the requested upstream pretrained model name."""

        return self._model_name

    @property
    def initialization(self) -> Initialization:
        """Return whether the model is pretrained, injected, or checkpoint-loaded."""

        return self._initialization

    @property
    def checkpoint_path(self) -> str | None:
        """Return the local checkpoint path, when applicable."""

        return self._checkpoint_path

    def _floating_reference(self) -> Tensor | None:
        for value in (*self.encoder.parameters(), *self.encoder.buffers()):
            if value.is_floating_point():
                return value
        return None

    def backbone_modules(self) -> tuple[nn.Module, ...]:
        """Return modules contributing to CHGNet ``crystal_fea``."""

        names = (
            "atom_embedding",
            "bond_basis_expansion",
            "bond_embedding",
            "bond_weights_ag",
            "bond_weights_bg",
            "angle_basis_expansion",
            "angle_embedding",
            "atom_conv_layers",
            "bond_conv_layers",
            "angle_layers",
            "readout_norm",
            "pooling",
        )
        modules = [getattr(self.encoder, name, None) for name in names]
        resolved = tuple(module for module in modules if isinstance(module, nn.Module))
        return resolved or (self.encoder,)

    def _graph_converter(self):
        converter = getattr(self.encoder, "graph_converter", None)
        if not callable(converter):
            raise RuntimeError(
                "CHGNet structure conversion requires encoder.graph_converter to be callable."
            )
        return converter

    def _prepare_graph(
        self,
        value: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any:
        if _is_chgnet_graph(value):
            graph = value
        else:
            structure = self.adapter.to_pymatgen(value)
            graph = self._graph_converter()(structure)

        move = getattr(graph, "to", None)
        if callable(move):
            graph = move(str(device))
        for name in _FLOAT_GRAPH_FIELDS:
            tensor = getattr(graph, name, None)
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                setattr(graph, name, tensor.to(device=device, dtype=dtype))
        return graph

    def forward(self, structures: Sequence[Any]) -> Tensor:
        """Return one differentiable CHGNet crystal feature per structure."""

        if isinstance(structures, (str, bytes)) or not isinstance(structures, Sequence):
            raise TypeError("structures must be a non-empty sequence.")
        if not structures:
            raise ValueError("structures must contain at least one structure.")

        reference = self._floating_reference()
        if reference is None:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()
        else:
            device = reference.device
            dtype = reference.dtype

        graphs = [
            self._prepare_graph(structure, device=device, dtype=dtype)
            for structure in structures
        ]
        prediction = self.encoder(
            graphs,
            task="e",
            return_crystal_feas=True,
        )
        if not isinstance(prediction, Mapping):
            raise TypeError("CHGNet forward must return a mapping containing 'crystal_fea'.")
        crystal_fea = prediction.get("crystal_fea")
        if not torch.is_tensor(crystal_fea):
            raise TypeError("CHGNet forward did not return Tensor 'crystal_fea'.")
        expected = (len(structures), self.output_dim)
        if tuple(crystal_fea.shape) != expected:
            raise ValueError(
                "CHGNet crystal_fea must have shape "
                f"{expected}, got {tuple(crystal_fea.shape)}."
            )
        if crystal_fea.device != device or crystal_fea.dtype != dtype:
            raise ValueError("CHGNet crystal_fea must match the encoder device and dtype.")
        if not torch.isfinite(crystal_fea).all():
            raise FloatingPointError("CHGNet encoder produced non-finite crystal features.")
        return crystal_fea


__all__ = ["CHGNetEncoder"]
