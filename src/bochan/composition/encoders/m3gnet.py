"""MatGL M3GNet-backed crystal-structure representation encoder."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import Tensor, nn

from .base import MaterialEncoder

if TYPE_CHECKING:
    from bochan.structure.adapter import StructureAdapter

Initialization = Literal["pretrained", "injected"]

_MATGL_INSTALL_HINT = (
    "M3GNet support requires matgl>=4.0.3,<5. "
    "Install bochan[materials] or install matgl directly."
)
_DEFAULT_MODEL_NAME = "M3GNet-PES-MatPES-PBE-2025.2"


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _matgl_api() -> tuple[Callable[..., Any], type[Any]]:
    """Return MatGL model loader and pymatgen graph converter lazily."""

    try:
        matgl_module = import_module("matgl")
        pymatgen_module = import_module("matgl.ext.pymatgen")
    except ImportError as error:
        raise ImportError(_MATGL_INSTALL_HINT) from error

    load_model = getattr(matgl_module, "load_model", None)
    converter_class = getattr(pymatgen_module, "Structure2Graph", None)
    if not callable(load_model):
        raise RuntimeError("The installed matgl package does not expose matgl.load_model().")
    if not isinstance(converter_class, type):
        raise RuntimeError("The installed matgl package does not expose Structure2Graph.")
    return load_model, converter_class


def _structure_adapter_class() -> type[Any]:
    module = import_module("bochan.structure.adapter")
    adapter_class = getattr(module, "StructureAdapter", None)
    if not isinstance(adapter_class, type):
        raise RuntimeError("bochan.structure.adapter.StructureAdapter is unavailable.")
    return adapter_class


def _unwrap_pretrained_model(loaded: Any) -> nn.Module:
    """Return the bare MatGL M3GNet model from a loaded model or Potential."""

    model = getattr(loaded, "model", loaded)
    if not isinstance(model, nn.Module):
        raise TypeError("matgl.load_model() did not return a torch model or Potential.")
    if type(model).__name__ != "M3GNet" or not type(model).__module__.startswith("matgl.models"):
        raise ValueError(
            "M3GNetEncoder requires a MatGL M3GNet pretrained model; "
            f"loaded {type(model).__module__}.{type(model).__name__}."
        )
    return model


def _first_linear_in_features(module: nn.Module | None) -> int | None:
    if module is None:
        return None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return int(child.in_features)
    return None


class M3GNetEncoder(MaterialEncoder):
    """Encode periodic crystal structures with a MatGL M3GNet readout vector.

    The encoder deliberately calls the bare M3GNet model directly instead of
    ``predict_structure``. MatGL's convenience prediction method detaches the
    final property prediction, whereas bochan needs the differentiable graph
    readout for later DKL integration.

    Common in-memory structure objects are normalized through
    :class:`bochan.structure.StructureAdapter`, then converted with MatGL's
    ``Structure2Graph`` using the exact pretrained model's ``element_types`` and
    cutoff. The original M3GNet property head is not used as a representation.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        model_name: str = _DEFAULT_MODEL_NAME,
        output_dim: int | None = None,
        adapter: StructureAdapter | None = None,
        graph_converter: Any | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")

        initialization: Initialization
        if encoder is None:
            load_model, _ = _matgl_api()
            encoder = _unwrap_pretrained_model(load_model(model_name))
            initialization = "pretrained"
        else:
            if not isinstance(encoder, nn.Module):
                raise TypeError("encoder must be a torch.nn.Module.")
            initialization = "injected"

        if not bool(getattr(encoder, "is_intensive", True)):
            raise ValueError(
                "M3GNetEncoder requires an intensive M3GNet readout; "
                "extensive atomic readouts do not have a fixed crystal width."
            )

        inferred_output_dim = self._infer_output_dim(encoder)
        if output_dim is not None:
            output_dim = _positive_int("output_dim", output_dim)
            if inferred_output_dim is not None and output_dim != inferred_output_dim:
                raise ValueError(
                    "output_dim does not match the encoder readout width: "
                    f"{output_dim} != {inferred_output_dim}."
                )
            inferred_output_dim = output_dim
        if inferred_output_dim is None:
            raise ValueError(
                "output_dim is required when the injected M3GNet-compatible encoder "
                "does not expose a discoverable readout width."
            )

        adapter_class = _structure_adapter_class()
        if adapter is not None and not isinstance(adapter, adapter_class):
            raise TypeError("adapter must be a StructureAdapter.")

        self.encoder = encoder
        self._output_dim = inferred_output_dim
        self._model_name = model_name
        self._initialization: Initialization = initialization
        self.adapter = adapter or adapter_class()
        self.graph_converter = graph_converter or self._build_graph_converter()

        reference = self._floating_reference()
        output_reference = torch.empty(0) if reference is None else reference.new_empty(0)
        self.register_buffer("_output_reference", output_reference, persistent=False)

    @staticmethod
    def _infer_output_dim(encoder: nn.Module) -> int | None:
        value = getattr(encoder, "output_dim", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value

        final_layer = getattr(encoder, "final_layer", None)
        if isinstance(final_layer, nn.Module):
            width = _first_linear_in_features(final_layer)
            if width is not None and width > 0:
                return width

        init_args = getattr(encoder, "_init_args", None)
        if not isinstance(init_args, Mapping):
            return None
        include_state = bool(init_args.get("include_state", False))
        state_dim = int(init_args.get("dim_state_embedding", 0)) if include_state else 0
        readout_type = str(init_args.get("readout_type", "weighted_atom"))
        field = str(init_args.get("field", "node_feat"))
        if readout_type == "weighted_atom":
            units = init_args.get("units", 64)
            return int(units) + state_dim if isinstance(units, int) else None
        feature_key = "dim_node_embedding" if field == "node_feat" else "dim_edge_embedding"
        input_width = init_args.get(feature_key, 64)
        if not isinstance(input_width, int):
            return None
        if readout_type == "set2set":
            return 2 * input_width + state_dim
        if readout_type == "reduce_atom":
            return input_width + state_dim
        return None

    @property
    def output_dim(self) -> int:
        """Return the fixed M3GNet graph-readout width."""

        return self._output_dim

    @property
    def model_name(self) -> str:
        """Return the requested MatGL pretrained model identifier."""

        return self._model_name

    @property
    def initialization(self) -> Initialization:
        """Return whether the wrapped encoder was pretrained or injected."""

        return self._initialization

    def _floating_reference(self) -> Tensor | None:
        for value in (*self.encoder.parameters(), *self.encoder.buffers()):
            if value.is_floating_point():
                return value
        return None

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> M3GNetEncoder:
        """Move with the parent while preserving the M3GNet native dtype.

        MatGL graph construction is float32-oriented for the supported 4.x
        pretrained M3GNet potentials. The wrapper lets the output-reference
        buffer follow an outer model's dtype/device but restores the M3GNet
        backbone to the dtype in which it was loaded. Only the exported readout
        crosses that dtype boundary, preserving future DKL autograd.
        """

        reference = self._floating_reference()
        native_dtype = reference.dtype if reference is not None else None
        module = super()._apply(fn, recurse=recurse)
        if native_dtype is not None:
            self.encoder.to(dtype=native_dtype)
        return cast(M3GNetEncoder, module)

    def backbone_modules(self) -> tuple[nn.Module, ...]:
        """Return modules that contribute to the M3GNet readout representation."""

        names = (
            "bond_expansion",
            "embedding",
            "basis_expansion",
            "three_body_interactions",
            "graph_layers",
            "readout",
        )
        modules = [getattr(self.encoder, name, None) for name in names]
        resolved = tuple(module for module in modules if isinstance(module, nn.Module))
        return resolved or (self.encoder,)

    def _build_graph_converter(self) -> Any:
        _, converter_class = _matgl_api()
        element_types = getattr(self.encoder, "element_types", None)
        cutoff = getattr(self.encoder, "cutoff", None)
        if not isinstance(element_types, Sequence) or isinstance(element_types, (str, bytes)):
            raise ValueError(
                "M3GNet graph conversion requires encoder.element_types from the pretrained model."
            )
        if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool) or cutoff <= 0:
            raise ValueError("M3GNet graph conversion requires a positive encoder.cutoff.")
        return converter_class(element_types=tuple(element_types), cutoff=float(cutoff))

    def _prepare_graph(self, structure: Any) -> tuple[Any, Tensor | None]:
        pmg_structure = self.adapter.to_pymatgen(structure)
        get_graph = getattr(self.graph_converter, "get_graph", None)
        if not callable(get_graph):
            raise TypeError("graph_converter must expose get_graph(structure).")
        converted = get_graph(pmg_structure)
        if not isinstance(converted, tuple) or len(converted) != 3:
            raise TypeError("M3GNet graph_converter.get_graph() must return (graph, lattice, state_attr).")
        graph, lattice, state_attr = converted

        reference = self._floating_reference()
        device = torch.device("cpu") if reference is None else reference.device
        dtype = torch.get_default_dtype() if reference is None else reference.dtype

        move = getattr(graph, "to", None)
        if callable(move):
            graph = move(device)
        lattice_tensor = torch.as_tensor(lattice, device=device, dtype=dtype)
        if lattice_tensor.shape == (3, 3):
            lattice_tensor = lattice_tensor.unsqueeze(0)
        if lattice_tensor.shape != (1, 3, 3):
            raise ValueError(
                "M3GNet graph lattice must have shape [1, 3, 3], "
                f"got {tuple(lattice_tensor.shape)}."
            )

        pbc_offset = getattr(graph, "pbc_offset", None)
        frac_coords = getattr(graph, "frac_coords", None)
        if not torch.is_tensor(pbc_offset) or not torch.is_tensor(frac_coords):
            raise TypeError("M3GNet graph must expose Tensor pbc_offset and frac_coords fields.")
        pbc_offset = pbc_offset.to(device=device, dtype=dtype)
        frac_coords = frac_coords.to(device=device, dtype=dtype)
        graph.pbc_offset = pbc_offset
        graph.frac_coords = frac_coords
        graph.pbc_offshift = torch.matmul(pbc_offset, lattice_tensor[0])
        graph.pos = frac_coords @ lattice_tensor[0]

        state_tensor: Tensor | None = None
        if bool(getattr(self.encoder, "include_state", False)):
            state_tensor = torch.as_tensor(state_attr, device=device, dtype=dtype)
        return graph, state_tensor

    def _readout_one(self, structure: Any) -> Tensor:
        graph, state_attr = self._prepare_graph(structure)
        self.encoder(g=graph, state_attr=state_attr)
        feature_dict = getattr(self.encoder, "feature_dict", None)
        if not isinstance(feature_dict, Mapping):
            raise RuntimeError("M3GNet forward did not populate encoder.feature_dict.")
        readout = feature_dict.get("readout")
        if not torch.is_tensor(readout):
            raise TypeError("M3GNet feature_dict['readout'] must be a Tensor.")
        if readout.ndim == 1:
            readout = readout.unsqueeze(0)
        expected = (1, self.output_dim)
        if tuple(readout.shape) != expected:
            raise ValueError(
                "M3GNet readout must contain one fixed-width vector per structure: "
                f"{tuple(readout.shape)} != {expected}."
            )
        return readout

    def forward(self, structures: Sequence[Any]) -> Tensor:
        """Return one differentiable M3GNet readout vector per structure."""

        if isinstance(structures, (str, bytes)) or not isinstance(structures, Sequence):
            raise TypeError("structures must be a non-empty sequence.")
        if not structures:
            raise ValueError("structures must contain at least one structure.")

        readouts = [self._readout_one(structure) for structure in structures]
        features = torch.cat(readouts, dim=0)
        expected = (len(structures), self.output_dim)
        if tuple(features.shape) != expected:
            raise ValueError(
                f"M3GNet features must have shape {expected}, got {tuple(features.shape)}."
            )
        if not torch.isfinite(features).all():
            raise FloatingPointError("M3GNet encoder produced non-finite readout features.")
        return features.to(
            device=self._output_reference.device,
            dtype=self._output_reference.dtype,
        )


__all__ = ["M3GNetEncoder"]
