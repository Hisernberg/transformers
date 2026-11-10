# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Context parallelism: each rank of the `cp` mesh holds a contiguous slice of every sequence.

A context parallel plan pairs two styles:

- `"cp_split_sequence"` on the decoder layers, which run on this rank's slice. The model built the attention mask and
  the position embeddings for the whole sequence before the first layer.
- `"cp_attention"` on their attention modules, whose attention function runs through a [`CPImplementation`] that
  gives it the whole sequence back. [`UlyssesCP`] all-to-alls the query, key and value, so that every rank holds the
  whole sequence for `num_heads / cp_size` heads, and all-to-alls the output back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils.generic import GeneralInterface
from ..utils.import_utils import is_torch_available, is_torch_distributed_available
from .parallel_utils import ALL_PARALLEL_STYLES


if TYPE_CHECKING:
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh

if is_torch_available():
    import torch

if is_torch_distributed_available():
    import torch.distributed as dist
    from torch.distributed._functional_collectives import all_to_all_single_autograd


def split_sequence(tensor: torch.Tensor, dim: int, cp_mesh: DeviceMesh) -> torch.Tensor:
    """This rank's contiguous slice of `tensor` along its sequence dimension `dim`."""
    cp_size = cp_mesh.size()
    if tensor.shape[dim] % cp_size:
        raise ValueError(
            f"Context parallelism splits every sequence across the {cp_size} ranks of the `cp` mesh, so the sequence "
            f"length must be divisible by `cp_size`, got {tensor.shape[dim]}. Pad the sequences to a multiple of "
            "`cp_size`."
        )
    return tensor.chunk(cp_size, dim=dim)[cp_mesh.get_local_rank()].contiguous()


def all_to_all(tensor: torch.Tensor, scatter_dim: int, gather_dim: int, group) -> torch.Tensor:
    """Split `tensor` along `scatter_dim` across the ranks of `group`, concatenate what they send back along `gather_dim`."""
    chunks = torch.stack(tensor.chunk(dist.get_world_size(group), dim=scatter_dim))
    chunks = all_to_all_single_autograd(chunks, None, None, group)
    return torch.cat(chunks.unbind(), dim=gather_dim)


class CPImplementation:
    """
    Runs the attention function of a `"cp_attention"` module over the whole sequence, when each rank of `cp_mesh`
    holds a slice of it. Register a subclass with `CPImplementationInterface.register(name, cls)` to select it with
    `cp_implementation=name`.
    """

    def __init__(self, cp_mesh: DeviceMesh):
        self.cp_mesh = cp_mesh

    def check(self, module: nn.Module) -> None:
        """Raise a `ValueError` if this implementation cannot run the attention of `module` (or of a whole model)."""

    def __call__(self, attention_function, module, query, key, value, attention_mask, **kwargs):
        raise NotImplementedError


class UlyssesCP(CPImplementation):
    """All-to-all to the whole sequence with `num_heads / cp_size` heads, run the attention, all-to-all back."""

    def check(self, module: nn.Module) -> None:
        from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

        attn_implementation = module.config._attn_implementation
        if attn_implementation not in ALL_ATTENTION_FUNCTIONS or attn_implementation == "flex_attention":
            raise ValueError(
                f"Ulysses context parallelism routes the attention calls of {module.__class__.__name__} through "
                f"`ALL_ATTENTION_FUNCTIONS`, but it uses `attn_implementation={attn_implementation}`, which takes no "
                "attention mask of the whole sequence. Load the model with `sdpa`, `flash_attention_2`, "
                "`flash_attention_3` or a hub flash kernel, or disable context parallelism."
            )

        config = module.config.get_text_config()
        num_kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        cp_size = self.cp_mesh.size()
        if config.num_attention_heads % cp_size or num_kv_heads % cp_size:
            raise ValueError(
                f"Ulysses context parallelism splits the attention heads across the {cp_size} ranks of the `cp` mesh, "
                f"so `cp_size` must divide both the number of attention heads ({config.num_attention_heads}) and the "
                f"number of key/value heads ({num_kv_heads}) of {module.__class__.__name__}. Use another `cp_size`, or "
                "disable context parallelism."
            )

    def __call__(self, attention_function, module, query, key, value, attention_mask, **kwargs):
        group = self.cp_mesh.get_group()
        # [batch, heads, seq / cp_size, head_dim] -> [batch, heads / cp_size, seq, head_dim]
        query, key, value = (all_to_all(x, scatter_dim=1, gather_dim=2, group=group) for x in (query, key, value))
        if kwargs.get("s_aux") is not None:  # attention sinks: one logit per head
            kwargs["s_aux"] = kwargs["s_aux"].chunk(self.cp_mesh.size())[self.cp_mesh.get_local_rank()]
        output, attn_weights = attention_function(module, query, key, value, attention_mask, **kwargs)
        # [batch, seq, heads / cp_size, head_dim] -> [batch, seq / cp_size, heads, head_dim]
        return all_to_all(output, scatter_dim=1, gather_dim=2, group=group), attn_weights


class CPImplementationInterface(GeneralInterface):
    """Registry of the `CPImplementation`s `DistributedConfig.cp_implementation` names."""

    _global_mapping = {"ulysses": UlyssesCP}


ALL_CP_IMPLEMENTATIONS: CPImplementationInterface = CPImplementationInterface()


def wrap_attention_function(attn_implementation: str) -> None:
    """
    Wrap the `ALL_ATTENTION_FUNCTIONS` entry of `attn_implementation`, so that the modules `"cp_attention"` gave a
    `cp_implementation` run it through it. A no-op if the entry is wrapped already: call it again after registering
    another function under that key, as loading a hub kernel does.
    """
    from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

    attention_function = ALL_ATTENTION_FUNCTIONS[attn_implementation]
    if getattr(attention_function, "is_context_parallel", False):
        return

    def context_parallel_attention(module, *args, **kwargs):
        if hasattr(module, "cp_implementation"):
            return module.cp_implementation(attention_function, module, *args, **kwargs)
        return attention_function(module, *args, **kwargs)

    context_parallel_attention.is_context_parallel = True
    ALL_ATTENTION_FUNCTIONS.register(attn_implementation, context_parallel_attention)


class SplitSequenceParallel:
    """
    Decoder layer: run on this rank's slice of the sequence. The first layer of the plan cuts `hidden_states`, and every
    layer cuts the `position_embeddings` the model computed for the whole sequence. The attention mask and
    `position_ids` stay those of the whole sequence, which `"cp_attention"` gives the attention function.
    """

    def install_forward(self, module: nn.Module, cp_mesh: DeviceMesh, *, split_hidden_states: bool = False):
        forward = module.forward

        def cp_forward(hidden_states, *args, **kwargs):
            if split_hidden_states:
                hidden_states = split_sequence(hidden_states, 1, cp_mesh)
            position_embeddings = kwargs.get("position_embeddings")
            if isinstance(position_embeddings, tuple):
                kwargs["position_embeddings"] = tuple(split_sequence(x, -2, cp_mesh) for x in position_embeddings)
            elif position_embeddings is not None:
                kwargs["position_embeddings"] = split_sequence(position_embeddings, -2, cp_mesh)
            return forward(hidden_states, *args, **kwargs)

        module.forward = cp_forward
        return module


class AttentionContextParallel:
    """Attention module: run its attention function through the `CPImplementation` that `cp_implementation` names."""

    def install_forward(self, module: nn.Module, cp_mesh: DeviceMesh, *, cp_implementation: str = "ulysses"):
        if cp_implementation not in ALL_CP_IMPLEMENTATIONS:
            raise ValueError(
                f"`cp_implementation={cp_implementation!r}` is not registered: use one of "
                f"{ALL_CP_IMPLEMENTATIONS.valid_keys()}, or register a `CPImplementation` with "
                "`CPImplementationInterface.register`."
            )
        implementation = ALL_CP_IMPLEMENTATIONS[cp_implementation](cp_mesh)
        implementation.check(module)
        module.cp_implementation = implementation
        wrap_attention_function(module.config._attn_implementation)
        return module


if is_torch_distributed_available():
    ALL_PARALLEL_STYLES.register("cp_split_sequence", SplitSequenceParallel())
    ALL_PARALLEL_STYLES.register("cp_attention", AttentionContextParallel())
