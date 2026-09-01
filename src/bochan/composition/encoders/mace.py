"""MACE-backed invariant crystal-structure representation encoder."""

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
Pooling = Literal["mean", "sum"]
BatchBuilder = Callable[[Any], Mapping[str, Any]]

_MACE_INSTALL_HINT = (
    "MACE support requires mace-torch>=0.3.16,<0.4. "
    "Install bochan[materials] or install mace-torch directly."
)
_DEFAULT_MODEL_NAME = "medium-mpa-0"


def _mace_module(name: str) -> Any:
    try:
        return import_module(name)
    except ImportError as error:
        raise ImportError(_MACE_INSTALL_HINT) from error


def _structure_adapter_class() -> type[Any]:
    module = import_module("bochan.structure.adapter")
    adapter_class = getattr(module, "StructureAdapter", None)
    if not isinstance(adapter_class, type):
        raise RuntimeError("bochan.structure.adapter.StructureAdapter is unavailable.")
    return adapter_class


def _scalar_int(name: str, value: Any) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"MACE encoder.{name} must be scalar.")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"MACE encoder.{name} must be integer-valued.")
    resolved = int(value)
    if resolved <= 0 or float(value) != float(resolved):
        raise ValueError(f"MACE encoder.{name} must be a positive integer.")
    return resolved


def _scalar_float(name: str, value: Any) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"MACE encoder.{name} must be scalar.")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"MACE encoder.{name} must be numeric.")
    resolved = float(value)
    if not resolved > 0.0:
        raise ValueError(f"MACE encoder.{name} must be positive.")
    return resolved


def _load_pretrained_model(model_name: str) -> nn.Module:
    module = _mace_module("mace.calculators.foundations_models")
    loader = getattr(module, "mace_mp", None)
    if not callable(loader):
        raise RuntimeError("The installed mace-torch package does not expose mace_mp().")
    model = loader(model=model_name, device="cpu", return_raw_model=True)
    if not isinstance(model, nn.Module):
        raise TypeError("mace_mp(return_raw_model=True) did not return torch.nn.Module.")
    return model


def _available_heads(model: nn.Module) -> tuple[str, ...]:
    raw_heads = getattr(model, "heads", None)
    if raw_heads is None:
        return ("Default",)
    if isinstance(raw_heads, str):
        heads = (raw_heads,)
    elif isinstance(raw_heads, Sequence):
        heads = tuple(str(head) for head in raw_heads)
    else:
        raise TypeError("MACE encoder.heads must be a string sequence when provided.")
    if not heads or any(not head for head in heads):
        raise ValueError("MACE encoder.heads must contain non-empty names.")
    return heads


def _resolve_head(model: nn.Module, requested: str | None) -> tuple[tuple[str, ...], str]:
    heads = _available_heads(model)
    if requested is not None:
        if not isinstance(requested, str) or not requested:
            raise ValueError("head must be a non-empty string when provided.")
        if requested not in heads:
            raise ValueError(f"MACE head {requested!r} is not available; choose one of {list(heads)!r}.")
        return heads, requested
    if len(heads) == 1:
        return heads, heads[0]
    defaults = [head for head in heads if head.lower() == "default"]
    if len(defaults) == 1:
        return heads, defaults[0]
    raise ValueError(
        "MACE model exposes multiple heads and no unique 'Default' head; pass head explicitly."
    )


def _descriptor_metadata(model: nn.Module, num_layers: int) -> tuple[int, int, int, int]:
    products = getattr(model, "products", None)
    if not isinstance(products, (nn.ModuleList, list, tuple)) or not products:
        raise ValueError("MACE representation extraction requires encoder.products.")
    first_product = products[0]
    linear = getattr(first_product, "linear", None)
    irreps_out = getattr(linear, "irreps_out", None)
    if irreps_out is None:
        raise ValueError("MACE representation extraction requires products[0].linear.irreps_out.")

    o3_module = _mace_module("e3nn.o3")
    irreps_class = getattr(o3_module, "Irreps", None)
    if irreps_class is None:
        raise RuntimeError("e3nn.o3.Irreps is unavailable.")
    irreps = irreps_class(str(irreps_out))
    l_max = int(irreps.lmax)
    denominator = (l_max + 1) ** 2
    if denominator <= 0 or int(irreps.dim) % denominator != 0:
        raise ValueError("MACE product irreps do not expose a regular invariant feature layout.")
    invariant_features = int(irreps.dim) // denominator
    if invariant_features <= 0:
        raise ValueError("MACE invariant descriptor width must be positive.")

    num_interactions = _scalar_int("num_interactions", getattr(model, "num_interactions", None))
    if num_layers == -1:
        resolved_layers = num_interactions
    elif isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers must be -1 or a positive integer.")
    elif num_layers > num_interactions:
        raise ValueError(
            "num_layers exceeds the number of MACE interactions: "
            f"{num_layers} > {num_interactions}."
        )
    else:
        resolved_layers = num_layers
    return resolved_layers, num_interactions, invariant_features, l_max


class MACEEncoder(MaterialEncoder):
    """Encode periodic crystals with differentiable invariant MACE descriptors.

    The implementation follows MACE 0.3.16's descriptor contract without using
    ``MACECalculator.get_descriptors`` because that convenience API detaches and
    converts its result to NumPy. bochan calls the raw torch model directly,
    extracts the scalar ``l=0`` channels from ``output['node_feats']`` with
    MACE's own ``extract_invariant`` helper, and pools those per-atom features to
    one fixed-width crystal vector. The original energy readouts are not part of
    the exported representation.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        model_name: str = _DEFAULT_MODEL_NAME,
        num_layers: int = -1,
        pooling: Pooling = "mean",
        head: str | None = None,
        adapter: StructureAdapter | None = None,
        batch_builder: BatchBuilder | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if pooling not in {"mean", "sum"}:
            raise ValueError("pooling must be 'mean' or 'sum'.")

        initialization: Initialization
        if encoder is None:
            encoder = _load_pretrained_model(model_name)
            initialization = "pretrained"
        else:
            if not isinstance(encoder, nn.Module):
                raise TypeError("encoder must be a torch.nn.Module.")
            initialization = "injected"

        resolved_layers, num_interactions, invariant_features, l_max = _descriptor_metadata(
            encoder,
            num_layers,
        )
        available_heads, resolved_head = _resolve_head(encoder, head)
        atomic_numbers = getattr(encoder, "atomic_numbers", None)
        if torch.is_tensor(atomic_numbers):
            atomic_numbers = atomic_numbers.detach().cpu().tolist()
        if not isinstance(atomic_numbers, Sequence) or isinstance(atomic_numbers, (str, bytes)):
            raise ValueError("MACE representation extraction requires encoder.atomic_numbers.")
        resolved_atomic_numbers = tuple(int(number) for number in atomic_numbers)
        if not resolved_atomic_numbers or any(number <= 0 for number in resolved_atomic_numbers):
            raise ValueError("MACE encoder.atomic_numbers must contain positive atomic numbers.")
        cutoff = _scalar_float("r_max", getattr(encoder, "r_max", None))

        adapter_class = _structure_adapter_class()
        if adapter is not None and not isinstance(adapter, adapter_class):
            raise TypeError("adapter must be a StructureAdapter.")
        if batch_builder is not None and not callable(batch_builder):
            raise TypeError("batch_builder must be callable when provided.")

        self.encoder = encoder
        self._model_name = model_name
        self._initialization: Initialization = initialization
        self._num_layers = resolved_layers
        self._num_interactions = num_interactions
        self._num_invariant_features = invariant_features
        self._l_max = l_max
        self._output_dim = resolved_layers * invariant_features
        self._pooling: Pooling = pooling
        self._available_heads = available_heads
        self._head = resolved_head
        self._atomic_numbers = resolved_atomic_numbers
        self._cutoff = cutoff
        self.adapter = adapter if adapter is not None else adapter_class()
        self.batch_builder = batch_builder

        reference = self._floating_reference()
        output_reference = torch.empty(0) if reference is None else reference.new_empty(0)
        self.register_buffer("_output_reference", output_reference, persistent=False)

    @property
    def output_dim(self) -> int:
        """Return the invariant pooled MACE representation width."""

        return self._output_dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def initialization(self) -> Initialization:
        return self._initialization

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_interactions(self) -> int:
        return self._num_interactions

    @property
    def pooling(self) -> Pooling:
        return self._pooling

    @property
    def available_heads(self) -> tuple[str, ...]:
        return self._available_heads

    @property
    def head(self) -> str:
        return self._head

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        return self._atomic_numbers

    @property
    def cutoff(self) -> float:
        return self._cutoff

    def _floating_reference(self) -> Tensor | None:
        for value in (*self.encoder.parameters(), *self.encoder.buffers()):
            if value.is_floating_point():
                return value
        return None

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> MACEEncoder:
        """Move with the parent while preserving the pretrained MACE native dtype."""

        reference = self._floating_reference()
        native_dtype = reference.dtype if reference is not None else None
        module = super()._apply(fn, recurse=recurse)
        if native_dtype is not None:
            self.encoder.to(dtype=native_dtype)
        return cast(MACEEncoder, module)

    def backbone_modules(self) -> tuple[nn.Module, ...]:
        """Return modules contributing to MACE node descriptors, excluding energy heads."""

        names = (
            "node_embedding",
            "radial_embedding",
            "spherical_harmonics",
            "interactions",
            "products",
        )
        modules = [getattr(self.encoder, name, None) for name in names]
        resolved = tuple(module for module in modules if isinstance(module, nn.Module))
        return resolved or (self.encoder,)

    def _model_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        reference = self._floating_reference()
        if reference is None:
            return torch.get_default_dtype(), torch.device("cpu")
        if reference.dtype not in {torch.float32, torch.float64}:
            raise TypeError(f"MACE floating dtype must be float32 or float64, got {reference.dtype}.")
        return reference.dtype, reference.device

    def _default_batch(self, structure: Any) -> Mapping[str, Any]:
        atoms = self.adapter.to_ase(structure)
        data_module = _mace_module("mace.data")
        tools_module = _mace_module("mace.tools")
        torch_tools_module = _mace_module("mace.tools.torch_tools")

        config_from_atoms = getattr(data_module, "config_from_atoms", None)
        atomic_data_class = getattr(data_module, "AtomicData", None)
        key_specification_class = getattr(data_module, "KeySpecification", None)
        atomic_number_table_class = getattr(tools_module, "AtomicNumberTable", None)
        torch_geometric = getattr(tools_module, "torch_geometric", None)
        default_dtype = getattr(torch_tools_module, "default_dtype", None)
        if not callable(config_from_atoms) or not isinstance(atomic_data_class, type):
            raise RuntimeError("The installed mace-torch data API is incomplete.")
        if not isinstance(key_specification_class, type) or not isinstance(atomic_number_table_class, type):
            raise RuntimeError("The installed mace-torch structure metadata API is incomplete.")
        batch_class = getattr(torch_geometric, "Batch", None)
        if not isinstance(batch_class, type) or not callable(default_dtype):
            raise RuntimeError("The installed mace-torch batching API is incomplete.")

        dtype, device = self._model_dtype_device()
        dtype_name = "float64" if dtype == torch.float64 else "float32"
        z_table = atomic_number_table_class(list(self.atomic_numbers))
        key_specification = key_specification_class.from_defaults()
        with default_dtype(dtype_name):
            config = config_from_atoms(
                atoms,
                key_specification=key_specification,
                head_name=self.head,
            )
            graph = atomic_data_class.from_config(
                config,
                z_table=z_table,
                cutoff=self.cutoff,
                heads=list(self.available_heads),
            )
        batch = batch_class.from_data_list([graph]).to(device)
        for key in batch.keys:
            value = batch[key]
            if torch.is_tensor(value) and value.is_floating_point():
                batch[key] = value.to(device=device, dtype=dtype)
        return batch.to_dict()

    def _build_batch(self, structure: Any) -> Mapping[str, Any]:
        batch = self._default_batch(structure) if self.batch_builder is None else self.batch_builder(structure)
        if not isinstance(batch, Mapping):
            raise TypeError("MACE batch_builder must return a mapping accepted by the raw model.")
        return batch

    def _invariant_node_features(self, node_features: Tensor) -> Tensor:
        if node_features.ndim != 2:
            raise ValueError(
                "MACE output['node_feats'] must have shape [n_atoms, descriptor_dim]."
            )
        if node_features.shape[0] <= 0:
            raise ValueError("MACE output['node_feats'] must contain at least one atom.")
        utils_module = _mace_module("mace.modules.utils")
        extract_invariant = getattr(utils_module, "extract_invariant", None)
        if not callable(extract_invariant):
            raise RuntimeError("The installed mace-torch package does not expose extract_invariant().")
        invariant = extract_invariant(
            node_features,
            num_layers=self.num_layers,
            num_features=self._num_invariant_features,
            l_max=self._l_max,
        )
        if not torch.is_tensor(invariant):
            raise TypeError("MACE extract_invariant() must return a Tensor.")
        expected_width = self.output_dim
        if invariant.ndim != 2 or invariant.shape[1] != expected_width:
            raise ValueError(
                "MACE invariant descriptor width does not match encoder metadata: "
                f"{tuple(invariant.shape)} vs (*, {expected_width})."
            )
        return invariant

    def _representation_one(self, structure: Any) -> Tensor:
        output = self.encoder(self._build_batch(structure))
        if not isinstance(output, Mapping):
            raise TypeError("Raw MACE forward must return a mapping.")
        node_features = output.get("node_feats")
        if not torch.is_tensor(node_features):
            raise TypeError("Raw MACE forward output['node_feats'] must be a Tensor.")
        invariant = self._invariant_node_features(node_features)
        if self.pooling == "mean":
            return invariant.mean(dim=0, keepdim=True)
        return invariant.sum(dim=0, keepdim=True)

    def forward(self, structures: Sequence[Any]) -> Tensor:
        """Return one differentiable invariant MACE representation per crystal."""

        if isinstance(structures, (str, bytes)) or not isinstance(structures, Sequence):
            raise TypeError("structures must be a non-empty sequence.")
        if not structures:
            raise ValueError("structures must contain at least one structure.")

        representations = [self._representation_one(structure) for structure in structures]
        features = torch.cat(representations, dim=0)
        expected = (len(structures), self.output_dim)
        if tuple(features.shape) != expected:
            raise ValueError(f"MACE features must have shape {expected}, got {tuple(features.shape)}.")
        if not torch.isfinite(features).all():
            raise FloatingPointError("MACE encoder produced non-finite representation features.")
        return features.to(
            device=self._output_reference.device,
            dtype=self._output_reference.dtype,
        )


__all__ = ["MACEEncoder"]
