"""Roost-backed composition encoder and pure-Torch graph construction.

The optional Aviary package is imported only when this adapter constructs an
upstream Roost backbone. Importing :mod:`bochan.composition` therefore remains
independent of Bochan's ``materials`` optional dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from math import prod
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor, nn

from .base import MaterialEncoder

Checkpoint = str | PathLike[str] | Mapping[str, object]
Initialization = Literal["random", "injected", "checkpoint"]

_CHECKPOINT_STATE_KEYS = ("state_dict", "model_state", "model_state_dict", "weights", "model")
_ENCODER_PREFIXES = (
    "module.material_encoder.encoder.",
    "material_encoder.encoder.",
    "module.model.encoder.",
    "model.encoder.",
    "module.encoder.",
    "encoder.",
    "module.model.",
    "model.",
    "module.",
)
_MATERIALS_INSTALL_HINT = "Roost support requires the optional Roost/materials dependency."


@dataclass(frozen=True)
class RoostGraph:
    """Batched graph tensors consumed by an Aviary Roost descriptor network.

    ``leading_shape`` records the batch and q-batch dimensions flattened by
    :func:`build_roost_graph`. The remaining tensors follow Aviary's current
    ``DescriptorNetwork`` input contract.
    """

    elem_weights: Tensor
    elem_fea: Tensor
    self_idx: Tensor
    nbr_idx: Tensor
    cry_elem_idx: Tensor
    leading_shape: torch.Size

    @property
    def num_materials(self) -> int:
        """Return the number of flattened compositions in the graph batch."""

        return prod(self.leading_shape)

    def model_inputs(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return tensors in the order expected by Aviary's Roost backbone."""

        return (
            self.elem_weights,
            self.elem_fea,
            self.self_idx,
            self.nbr_idx,
            self.cry_elem_idx,
        )


def _validate_composition_inputs(element_ids: Tensor, fractions: Tensor) -> None:
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
    if not torch.isfinite(fractions).all():
        raise ValueError("fractions must contain only finite values.")
    if (element_ids < 0).any():
        raise ValueError("element_ids must be non-negative; zero is reserved for padding.")
    if (fractions < 0).any():
        raise ValueError("fractions must be non-negative.")
    if (fractions.masked_select(element_ids.eq(0)) != 0).any():
        raise ValueError("Padding slots with element_id zero must have fraction zero.")

    active = element_ids.ne(0) & fractions.gt(0)
    if not active.any(dim=-1).all():
        raise ValueError("Each composition must contain at least one positive-fraction element.")
    row_sums = fractions.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-4, atol=1e-6):
        raise ValueError("Fractions for each composition must sum to one.")


def build_roost_graph(element_ids: Tensor, fractions: Tensor) -> RoostGraph:
    """Build an Aviary-compatible batched Roost graph using only Torch.

    Args:
        element_ids: Atomic numbers with shape ``(..., n_elements)``. Zero is
            padding. Positive atomic numbers paired with an exact zero fraction
            are also omitted, allowing closed-simplex boundary compositions.
        fractions: Non-negative normalized fractions with the same shape,
            device, and leading dimensions as ``element_ids``.

    Returns:
        A graph containing positive-fraction element nodes, all directed
        within-composition element pairs (including self-pairs), and the
        node-to-composition mapping expected by Aviary Roost.
    """

    _validate_composition_inputs(element_ids, fractions)
    leading_shape = element_ids.shape[:-1]
    n_elements = element_ids.shape[-1]
    flat_ids = element_ids.reshape(-1, n_elements)
    flat_fractions = fractions.reshape(-1, n_elements)
    active = flat_ids.ne(0) & flat_fractions.gt(0)

    counts = active.sum(dim=-1)
    offsets = counts.cumsum(dim=0) - counts
    local_node_idx = active.cumsum(dim=-1) - 1
    node_idx = offsets.unsqueeze(-1) + local_node_idx
    pair_mask = active.unsqueeze(-1) & active.unsqueeze(-2)

    self_idx = node_idx.unsqueeze(-1).expand(-1, n_elements, n_elements)[pair_mask]
    nbr_idx = node_idx.unsqueeze(-2).expand(-1, n_elements, n_elements)[pair_mask]
    cry_elem_idx = torch.repeat_interleave(
        torch.arange(flat_ids.shape[0], device=element_ids.device),
        counts,
    )

    return RoostGraph(
        elem_weights=flat_fractions[active].unsqueeze(-1),
        elem_fea=flat_ids[active],
        self_idx=self_idx,
        nbr_idx=nbr_idx,
        cry_elem_idx=cry_elem_idx,
        leading_shape=leading_shape,
    )


class _AviaryRoostBackbone(nn.Module):
    """Aviary element embedding plus Roost descriptor without prediction heads."""

    def __init__(self, elem_embedding: nn.Module, material_nn: nn.Module, output_dim: int) -> None:
        super().__init__()
        self.elem_embedding = elem_embedding
        self.material_nn = material_nn
        self.output_dim = output_dim

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: Tensor,
        nbr_idx: Tensor,
        cry_elem_idx: Tensor,
    ) -> Tensor:
        embedded_elements = self.elem_embedding(elem_fea)
        return self.material_nn(
            elem_weights,
            embedded_elements,
            self_idx,
            nbr_idx,
            cry_elem_idx,
        )


def _upstream_roost_components() -> tuple[type[nn.Module], Callable[[str], nn.Module]]:
    """Return current Aviary Roost components after lazy imports."""

    try:
        model_module = import_module("aviary.roost.model")
        utils_module = import_module("aviary.utils")
    except ImportError as error:
        raise ImportError(_MATERIALS_INSTALL_HINT) from error

    descriptor_class = getattr(model_module, "DescriptorNetwork", None)
    embedding_factory = getattr(utils_module, "get_element_embedding", None)
    if not isinstance(descriptor_class, type) or not issubclass(descriptor_class, nn.Module):
        raise RuntimeError("The installed Aviary package does not expose aviary.roost.model.DescriptorNetwork.")
    if not callable(embedding_factory):
        raise RuntimeError("The installed Aviary package does not expose aviary.utils.get_element_embedding.")
    return cast(type[nn.Module], descriptor_class), cast(Callable[[str], nn.Module], embedding_factory)


def _positive_int(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be a positive integer.")
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return value


def _hidden_dims(name: str, values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of positive integers.")
    resolved = tuple(values)
    for value in resolved:
        _positive_int(name, value)
    return resolved


class RoostEncoder(MaterialEncoder):
    """Encode padded elemental compositions with Aviary's Roost descriptor.

    Args:
        encoder: Optional injected graph encoder with Aviary's five-tensor Roost
            call signature. Injection avoids importing the optional dependency.
        checkpoint: Optional upstream Aviary or adapter checkpoint.
        output_dim: Representation width. Required when an injected encoder
            does not expose a positive integer ``output_dim``.
        elem_embedding: Aviary element-embedding name or file path.
        elem_fea_len: Roost node and pooled representation width.
        n_graph: Number of message-passing layers.
        elem_heads: Number of message attention heads.
        elem_gate: Hidden widths of element-level gate networks.
        elem_msg: Hidden widths of element-level message networks.
        cry_heads: Number of composition pooling heads.
        cry_gate: Hidden widths of composition-level gate networks.
        cry_msg: Hidden widths of composition-level message networks.
        strict_checkpoint: Require every backbone state entry when loading.

    The constructed adapter owns only Aviary's element embedding and
    ``DescriptorNetwork``. Roost task heads are deliberately excluded. Exact
    zero fractions are removed before message passing so weighted attention
    remains finite on closed-simplex boundaries.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        checkpoint: Checkpoint | None = None,
        output_dim: int | None = None,
        elem_embedding: str = "matscholar200",
        elem_fea_len: int = 64,
        n_graph: int = 3,
        elem_heads: int = 3,
        elem_gate: Sequence[int] = (256,),
        elem_msg: Sequence[int] = (256,),
        cry_heads: int = 3,
        cry_gate: Sequence[int] = (256,),
        cry_msg: Sequence[int] = (256,),
        strict_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        inferred_output_dim: int | None
        initialization: Initialization

        if encoder is None:
            if not isinstance(elem_embedding, str) or not elem_embedding:
                raise ValueError("elem_embedding must be a non-empty string.")
            elem_fea_len = _positive_int("elem_fea_len", elem_fea_len, minimum=2)
            n_graph = _positive_int("n_graph", n_graph)
            elem_heads = _positive_int("elem_heads", elem_heads)
            cry_heads = _positive_int("cry_heads", cry_heads)
            elem_gate = _hidden_dims("elem_gate", elem_gate)
            elem_msg = _hidden_dims("elem_msg", elem_msg)
            cry_gate = _hidden_dims("cry_gate", cry_gate)
            cry_msg = _hidden_dims("cry_msg", cry_msg)

            descriptor_class, embedding_factory = _upstream_roost_components()
            upstream_embedding = embedding_factory(elem_embedding)
            if not isinstance(upstream_embedding, nn.Module):
                raise RuntimeError("Aviary get_element_embedding must return a torch.nn.Module.")
            raw_weight = getattr(upstream_embedding, "weight", None)
            if not torch.is_tensor(raw_weight):
                raise RuntimeError("Aviary's element embedding must expose a rank-two weight tensor.")
            weight = cast(Tensor, raw_weight)
            if weight.ndim != 2:
                raise RuntimeError("Aviary's element embedding must expose a rank-two weight tensor.")
            material_nn = descriptor_class(
                elem_emb_len=int(weight.shape[1]),
                elem_fea_len=elem_fea_len,
                n_graph=n_graph,
                elem_heads=elem_heads,
                elem_gate=elem_gate,
                elem_msg=elem_msg,
                cry_heads=cry_heads,
                cry_gate=cry_gate,
                cry_msg=cry_msg,
            )
            encoder = _AviaryRoostBackbone(upstream_embedding, material_nn, elem_fea_len)
            inferred_output_dim = elem_fea_len
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
            raise ValueError("output_dim is required when an injected Roost encoder does not expose output_dim.")

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
        return None

    @property
    def output_dim(self) -> int:
        """Return the pooled Roost representation width."""

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

    def _validate_encoder_inputs(self, element_ids: Tensor, fractions: Tensor) -> None:
        _validate_composition_inputs(element_ids, fractions)
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

        embedding = getattr(self.encoder, "elem_embedding", None)
        if isinstance(embedding, nn.Embedding):
            maximum = int(element_ids.max().item())
            if maximum >= embedding.num_embeddings:
                raise ValueError(
                    "element_ids contains an atomic number outside the installed "
                    f"Aviary table: {maximum} >= {embedding.num_embeddings}."
                )

    def forward(self, element_ids: Tensor, fractions: Tensor) -> Tensor:
        """Return Roost composition embeddings with leading dimensions intact."""

        self._validate_encoder_inputs(element_ids, fractions)
        graph = build_roost_graph(element_ids, fractions)
        embeddings = self.encoder(*graph.model_inputs())
        if not torch.is_tensor(embeddings):
            raise TypeError("The Roost encoder must return a Tensor.")
        expected_shape = (graph.num_materials, self.output_dim)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"The Roost encoder returned an unexpected shape: {tuple(embeddings.shape)} != {expected_shape}."
            )
        if embeddings.device != fractions.device or embeddings.dtype != fractions.dtype:
            raise ValueError("The Roost encoder output must match the input fractions' device and dtype.")
        if not torch.isfinite(embeddings).all():
            raise FloatingPointError("The Roost encoder produced non-finite features.")
        return embeddings.reshape((*graph.leading_shape, self.output_dim))

    def load_checkpoint(self, checkpoint: Checkpoint, *, strict: bool = True) -> None:
        """Load Roost backbone weights from an Aviary or adapter checkpoint."""

        checkpoint_path: str | None = None
        if isinstance(checkpoint, (str, PathLike)):
            path = Path(checkpoint)
            if not path.is_file():
                raise FileNotFoundError(f"Roost checkpoint does not exist: {path}")
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint_path = str(path)
        elif isinstance(checkpoint, Mapping):
            loaded = checkpoint
        else:
            raise TypeError("checkpoint must be a path or mapping.")

        if not isinstance(loaded, Mapping):
            raise TypeError("Roost checkpoint must contain a mapping.")
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
                raise ValueError(f"Checkpoint contains duplicate Roost encoder key: {matched_key}")
            encoder_state[matched_key] = value

        if not encoder_state:
            raise ValueError("Checkpoint contains no weights matching the Roost encoder.")

        self.encoder.load_state_dict(encoder_state, strict=strict)
        self._initialization = "checkpoint"
        self._checkpoint_path = checkpoint_path


__all__ = ["RoostEncoder", "RoostGraph", "build_roost_graph"]
