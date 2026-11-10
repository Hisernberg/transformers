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

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from ..loss.loss_utils import LOSS_MAPPING, ForCausalLMLoss
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
    """
    All-to-all to the whole sequence with `1 / cp_size` of the heads this rank holds, run the attention, all-to-all
    back. Under tensor parallelism this rank holds `num_heads / tp_size` heads.
    """

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
        distributed_config = getattr(module.config, "distributed_config", None)
        tp_size = 1 if distributed_config is None else distributed_config.tp_size
        if config.num_attention_heads % (cp_size * tp_size) or num_kv_heads % (cp_size * tp_size):
            raise ValueError(
                f"Ulysses context parallelism splits the attention heads across the {cp_size} ranks of the `cp` mesh "
                f"and the {tp_size} ranks of the `tp` mesh, so `cp_size * tp_size` must divide both the number of "
                f"attention heads ({config.num_attention_heads}) and the number of key/value heads ({num_kv_heads}) "
                f"of {module.__class__.__name__}. Use smaller `cp_size` or `tp_size`, or disable context parallelism."
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


# Layer types that only mix tokens through softmax attention over the whole sequence. Every other one (recurrent,
# convolutional, sparse-indexer, ...) mixes tokens outside attention and would silently be cut at the slice boundaries.
SOFTMAX_ATTENTION_LAYER_TYPES = ("full_attention", "sliding_attention", "chunked_attention")


def check_context_parallelism(model: nn.Module) -> None:
    """Raise a `ValueError` if context parallelism would silently change the logits or the loss of `model`."""
    if not model._supports_attention_backend:
        raise ValueError(
            f"Context parallelism routes attention through `ALL_ATTENTION_FUNCTIONS`, but {model.__class__.__name__} "
            "does not declare that all its attention layers go through it with the model's kwargs "
            "(`_supports_attention_backend`), so they could run on this rank's slice only. Use another model, or "
            "disable context parallelism."
        )

    if model._is_stateful:
        raise ValueError(
            f"Context parallelism does not support {model.__class__.__name__}: its recurrent layers carry a state "
            "along the sequence, so they would run on this rank's slice only and be silently cut at the slice "
            "boundaries. Use another model, or disable context parallelism."
        )

    layer_types = set(getattr(model.config.get_text_config(), "layer_types", None) or ())
    if unsupported := layer_types - set(SOFTMAX_ATTENTION_LAYER_TYPES):
        raise ValueError(
            f"Context parallelism does not support the {sorted(unsupported)} layers of {model.__class__.__name__}: "
            "they mix tokens outside softmax attention, so they would run on this rank's slice only and be silently "
            "cut at the slice boundaries. Use another model, or disable context parallelism."
        )

    # `loss_type=None` (base models, and heads whose name maps to no loss) falls back to the causal LM loss
    if model.loss_type is not None and LOSS_MAPPING.get(model.loss_type) is not ForCausalLMLoss:
        raise ValueError(
            f"Context parallelism only supports the causal language modeling loss, but {model.__class__.__name__} "
            f"uses `loss_type={model.loss_type}`. Its loss would be computed from this rank's slice of the shifted "
            "labels while the logits stay sharded. Use a `*ForCausalLM` model, or disable context parallelism."
        )


# The two styles a context parallel plan pairs up: the decoder layers keep a slice, their attention sees everything.
ALL_CP_STYLES = ("cp_split_sequence", "cp_attention")


def pad_sequence(tensor: torch.Tensor, pad_len: int, value) -> torch.Tensor:
    """`tensor` with `pad_len` copies of `value` appended along its sequence dimension 1."""
    padding = tensor.new_full((tensor.shape[0], pad_len, *tensor.shape[2:]), value)
    return torch.cat([tensor, padding], dim=1)


def context_parallel_pre_forward(module: nn.Module, args: tuple, kwargs: dict, *, cp_mesh: DeviceMesh):
    """
    Forward pre-hook of the model: refuse a cache, which context parallelism does not split, and pad every sequence at
    the end to a multiple of `cp_size`, before the model builds its masks from them.
    """
    if kwargs.get("past_key_values") is not None:
        raise ValueError(
            "Context parallelism only supports training: `past_key_values` are not allowed. Run the model on full "
            "sequences without a cache, or disable context parallelism."
        )
    kwargs["use_cache"] = False
    # a hub kernel loaded since may have registered another function under the attention key
    wrap_attention_function(module.config._attn_implementation)

    tokens = args[0] if args else kwargs.get("input_ids")
    if tokens is None:
        tokens = kwargs["inputs_embeds"]
    pad_len = -tokens.shape[1] % cp_mesh.size()
    if pad_len == 0:
        return args, kwargs

    pad_token_id = getattr(module.config, "pad_token_id", None) or 0
    if args:
        args = (pad_sequence(args[0], pad_len, pad_token_id),) + args[1:]
    padding_values = {
        "input_ids": pad_token_id,
        "inputs_embeds": 0,
        "attention_mask": 0,
        "labels": -100,
        "shift_labels": -100,
    }
    for name, value in padding_values.items():
        if kwargs.get(name) is not None:
            kwargs[name] = pad_sequence(kwargs[name], pad_len, value)
    if kwargs.get("position_ids") is not None:
        # the padding continues the positions of the last document of every row
        position_ids = kwargs["position_ids"]
        steps = torch.arange(1, pad_len + 1, device=position_ids.device)
        kwargs["position_ids"] = torch.cat([position_ids, position_ids[..., -1:] + steps], dim=-1)
    return args, kwargs


def context_parallel_loss(
    loss_function: Callable,
    cp_mesh: DeviceMesh,
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    The loss of the whole sequences, which every rank reports, from this rank's slice of the logits and its slice of
    the shifted labels. Its gradient stays this rank's, multiplied by `cp_size`, so that FSDP2's average over the cp
    ranks gives the gradient of the full-sequence loss.
    """
    shift_labels = kwargs.pop("shift_labels", None)
    if shift_labels is None:
        shift_labels = torch.nn.functional.pad(labels[..., 1:], (0, 1), value=-100)
    if num_items_in_batch is None:
        num_items_in_batch = (shift_labels != -100).sum()
    kwargs["shift_labels"] = split_sequence(shift_labels, 1, cp_mesh)
    loss = loss_function(logits, labels, vocab_size, num_items_in_batch=num_items_in_batch, **kwargs)

    total = loss.detach().clone()
    dist.all_reduce(total, group=cp_mesh.get_group())
    return total + (loss - loss.detach()) * cp_mesh.size()


def apply_context_parallelism(
    model: nn.Module, cp_mesh: DeviceMesh, cp_plan: dict[str, str], cp_implementation: str = "ulysses"
) -> nn.Module:
    """
    Let the ranks of `cp_mesh` split every sequence: the plan's `"cp_split_sequence"` decoder layers run on this rank's
    slice, and its `"cp_attention"` modules run their attention through `cp_implementation`. A forward pre-hook refuses
    a cache and pads the sequences to a multiple of `cp_size`, and `model.loss_function` returns the loss of the whole
    sequences.
    """
    check_context_parallelism(model)
    modules = dict(model.named_modules())
    # the first decoder layer still gets the whole sequence: it is the one that cuts it
    first_layer = next(name for name, style in cp_plan.items() if style == "cp_split_sequence")
    for name, style in cp_plan.items():
        if style == "cp_split_sequence":
            ALL_PARALLEL_STYLES[style].install_forward(modules[name], cp_mesh, split_hidden_states=name == first_layer)
        else:
            ALL_PARALLEL_STYLES[style].install_forward(modules[name], cp_mesh, cp_implementation=cp_implementation)

    model.register_forward_pre_hook(partial(context_parallel_pre_forward, cp_mesh=cp_mesh), with_kwargs=True)
    model.loss_function = partial(context_parallel_loss, model.loss_function, cp_mesh)
    return model
