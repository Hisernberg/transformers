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
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from parameterized import parameterized
from torch.distributed.device_mesh import init_device_mesh

from transformers import LlamaConfig, MistralConfig
from transformers.distributed.context_parallel import (
    AttentionContextParallel,
    CPImplementation,
    CPImplementationInterface,
    SplitSequenceParallel,
    UlyssesCP,
    all_to_all,
)
from transformers.distributed.parallel_utils import ALL_PARALLEL_STYLES
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.testing_utils import is_fsdp_test

from ..distributed_testing_utils import run_distributed


class _MeshStub:
    """A `cp` mesh that only has to report its size and the caller's rank."""

    def __init__(self, size, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


class _StandInAttention(torch.nn.Module):
    """The attributes the attention functions read from a model's attention module."""

    def __init__(self, attn_implementation="sdpa", num_key_value_groups=1):
        super().__init__()
        self.config = LlamaConfig(hidden_size=16, num_attention_heads=4, attn_implementation=attn_implementation)
        self.is_causal = True
        self.num_key_value_groups = num_key_value_groups
        self.layer_idx = 0


class _StandInLayer(torch.nn.Module):
    """A decoder layer stand-in that records the inputs its forward receives."""

    def __init__(self):
        super().__init__()
        self.seen = {}

    def forward(self, hidden_states, attention_mask=None, position_ids=None, position_embeddings=None):
        self.seen = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
        }
        return hidden_states


class _SpyImplementation(CPImplementation):
    """Records the modules it is asked to run, then runs them normally."""

    def __init__(self, cp_mesh):
        super().__init__(cp_mesh)
        self.modules_seen = []

    def __call__(self, attention_function, module, query, key, value, attention_mask, **kwargs):
        self.modules_seen.append(module)
        return attention_function(module, query, key, value, attention_mask, **kwargs)


def _same_on_every_rank(*shape, seed=0):
    """A tensor every rank agrees on, from which each rank takes its own slice."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _init_cp_mesh(device_type):
    return init_device_mesh(device_type, (dist.get_world_size(),))


def _test_all_to_all(rank, cp_size):
    mesh = _init_cp_mesh("cpu")
    group = mesh.get_group()
    batch, heads, seq, head_dim = 2, 4, 8, 3
    full = _same_on_every_rank(batch, heads, seq, head_dim)

    local = full.chunk(cp_size, dim=2)[rank].contiguous()
    gathered = all_to_all(local, scatter_dim=1, gather_dim=2, group=group)
    assert tuple(gathered.shape) == (batch, heads // cp_size, seq, head_dim)
    torch.testing.assert_close(gathered, full.reshape(batch, cp_size, heads // cp_size, seq, head_dim)[:, rank])

    # around the attention call the output is in [batch, seq, heads, head_dim] layout, with the heads split by rank
    full_output = full.transpose(1, 2)
    heads_chunked = full_output.reshape(batch, seq, cp_size, heads // cp_size, head_dim)[:, :, rank].contiguous()
    gathered_output = all_to_all(heads_chunked, scatter_dim=1, gather_dim=2, group=group)
    assert tuple(gathered_output.shape) == (batch, seq // cp_size, heads, head_dim)
    torch.testing.assert_close(gathered_output, full_output.chunk(cp_size, dim=1)[rank])

    back = all_to_all(gathered.transpose(1, 2), scatter_dim=1, gather_dim=2, group=group)
    torch.testing.assert_close(back, local.transpose(1, 2))


def _test_all_to_all_gradients(rank, cp_size):
    mesh = _init_cp_mesh("cpu")
    group = mesh.get_group()
    batch, heads, seq, head_dim = 2, 4, 8, 3
    full = _same_on_every_rank(batch, heads, seq, head_dim)
    # each rank backpropagates its own upstream gradient through its own piece of the gathered sequence
    upstreams = [
        _same_on_every_rank(batch, heads // cp_size, seq, head_dim, seed=1 + other) for other in range(cp_size)
    ]

    local = full.chunk(cp_size, dim=2)[rank].clone().requires_grad_(True)
    all_to_all(local, scatter_dim=1, gather_dim=2, group=group).backward(upstreams[rank])

    reference = full.clone().requires_grad_(True)
    head_chunks = reference.reshape(batch, cp_size, heads // cp_size, seq, head_dim)
    (head_chunks * torch.stack(upstreams).permute(1, 0, 2, 3, 4)).sum().backward()
    torch.testing.assert_close(local.grad, reference.grad.chunk(cp_size, dim=2)[rank])


def _full_sequence_mask(config, batch, seq, hidden_size, mask_name):
    """The attention mask a model builds over the whole sequence."""
    inputs = torch.zeros(batch, seq, hidden_size)
    if mask_name == "causal":
        causal = torch.ones(seq, seq, dtype=torch.bool).tril()
        return causal.view(1, 1, seq, seq).expand(batch, 1, seq, seq)

    # two packed documents
    position_ids = torch.cat([torch.arange(seq // 2), torch.arange(seq // 2)]).expand(batch, -1)
    mask_function = create_sliding_window_causal_mask if mask_name == "sliding" else create_causal_mask
    return mask_function(config, inputs, None, None, position_ids=position_ids)


def _test_ulysses_attention(rank, heads, kv_heads, mask_name, cp_size):
    mesh = _init_cp_mesh("cpu")
    batch, seq, head_dim, hidden_size = 2, 8, 4, 16
    config = MistralConfig(
        hidden_size=hidden_size,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        sliding_window=4,
        attn_implementation="sdpa",
    )
    mask = _full_sequence_mask(config, batch, seq, hidden_size, mask_name)

    query = _same_on_every_rank(batch, heads, seq, head_dim, seed=0).requires_grad_(True)
    key = _same_on_every_rank(batch, kv_heads, seq, head_dim, seed=1).requires_grad_(True)
    value = _same_on_every_rank(batch, kv_heads, seq, head_dim, seed=2).requires_grad_(True)
    upstream = _same_on_every_rank(batch, seq, heads, head_dim, seed=3)
    module = _StandInAttention(num_key_value_groups=heads // kv_heads)

    # this rank holds a contiguous slice of the sequence with every head, and the mask of the whole sequence
    local = [t.detach().chunk(cp_size, dim=2)[rank].clone().requires_grad_(True) for t in (query, key, value)]
    output, attn_weights = UlyssesCP(mesh)(sdpa_attention_forward, module, *local, mask)
    assert attn_weights is None
    (output * upstream.chunk(cp_size, dim=1)[rank]).sum().backward()

    reference, _ = sdpa_attention_forward(module, query, key, value, mask)
    (reference * upstream).sum().backward()

    torch.testing.assert_close(output, reference.chunk(cp_size, dim=1)[rank])
    for local_tensor, full_tensor in zip(local, (query, key, value)):
        torch.testing.assert_close(local_tensor.grad, full_tensor.grad.chunk(cp_size, dim=2)[rank])


def _test_attention_kwargs(rank, cp_size):
    """The mask, the positions and the other kwargs the model built for the whole sequence are passed unchanged."""
    mesh = _init_cp_mesh("cpu")
    batch, heads, seq, head_dim = 1, 4, 8, 2
    local = [torch.randn(batch, heads, seq // cp_size, head_dim) for _ in range(3)]
    mask = torch.ones(batch, 1, seq, seq, dtype=torch.bool).tril()
    # attention sinks of the full head list, and boundaries a collator computed over the whole sequence
    s_aux = torch.randn(heads)
    position_ids = torch.cat([torch.arange(seq // 2), torch.arange(seq // 2)]).unsqueeze(0)
    cu_seq_lens = torch.tensor([0, seq // 2, seq])
    seen = {}

    def recording_attention(module, query, key, value, attention_mask, **kwargs):
        seen.update(kwargs, attention_mask=attention_mask, qkv_shape=tuple(query.shape))
        return torch.zeros(batch, seq, heads // cp_size, head_dim), None

    output, _ = UlyssesCP(mesh)(
        recording_attention,
        _StandInAttention(),
        *local,
        mask,
        s_aux=s_aux,
        position_ids=position_ids,
        cu_seq_lens_q=cu_seq_lens,
        max_length_q=seq // 2,
        dropout=0.0,
    )

    assert seen["qkv_shape"] == (batch, heads // cp_size, seq, head_dim)
    assert seen["attention_mask"] is mask
    assert seen["dropout"] == 0.0
    torch.testing.assert_close(seen["position_ids"], position_ids)
    torch.testing.assert_close(seen["cu_seq_lens_q"], cu_seq_lens)
    assert seen["max_length_q"] == seq // 2
    torch.testing.assert_close(seen["s_aux"], s_aux.chunk(cp_size)[rank])
    assert tuple(output.shape) == (batch, seq // cp_size, heads, head_dim)


@is_fsdp_test
class TestUlyssesDistributed(unittest.TestCase):
    """Two CPU ranks, gloo: the collectives, and the attention against plain attention on the full sequence."""

    def test_all_to_all(self):
        run_distributed(world_size=2, backend="gloo")(_test_all_to_all)(2)

    def test_all_to_all_gradients(self):
        run_distributed(world_size=2, backend="gloo")(_test_all_to_all_gradients)(2)

    @parameterized.expand(
        [
            (4, 4, "causal"),
            (4, 4, "packed"),
            (4, 4, "sliding"),
            (4, 2, "causal"),
            (4, 2, "packed"),
            (4, 2, "sliding"),
        ]
    )
    def test_attention_matches_full_attention(self, heads, kv_heads, mask_name):
        run_distributed(world_size=2, backend="gloo")(_test_ulysses_attention)(heads, kv_heads, mask_name, 2)

    def test_attention_kwargs_are_passed_through(self):
        run_distributed(world_size=2, backend="gloo")(_test_attention_kwargs)(2)


class TestContextParallelAttentionRouting(unittest.TestCase):
    def setUp(self):
        self.original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.addCleanup(ALL_ATTENTION_FUNCTIONS.register, "sdpa", self.original)
        patcher = mock.patch.dict(CPImplementationInterface._global_mapping, {"spy": _SpyImplementation})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_installed_modules_are_routed(self):
        routed, plain = _StandInAttention(), _StandInAttention()
        AttentionContextParallel().install_forward(routed, _MeshStub(2), cp_implementation="spy")
        attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
        query, key, value = (torch.randn(2, 4, 8, 4) for _ in range(3))

        output, _ = attention(routed, query, key, value, None)
        self.assertEqual(routed.cp_implementation.modules_seen, [routed])
        torch.testing.assert_close(output, self.original(routed, query, key, value, None)[0])
        # a module the style was not installed on keeps plain attention
        torch.testing.assert_close(
            attention(plain, query, key, value, None)[0], self.original(plain, query, key, value, None)[0]
        )

        # installing again keeps the wrapper, and a function registered since under the key (a hub kernel, for one) is
        # wrapped in turn
        AttentionContextParallel().install_forward(_StandInAttention(), _MeshStub(2), cp_implementation="spy")
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], attention)
        replacement = mock.Mock(wraps=self.original)
        ALL_ATTENTION_FUNCTIONS.register("sdpa", replacement)
        AttentionContextParallel().install_forward(_StandInAttention(), _MeshStub(2), cp_implementation="spy")
        ALL_ATTENTION_FUNCTIONS["sdpa"](routed, query, key, value, None)
        self.assertEqual(replacement.call_count, 1)
        self.assertEqual(routed.cp_implementation.modules_seen, [routed, routed])

    def test_unknown_implementation(self):
        with self.assertRaisesRegex(ValueError, "`cp_implementation='missing'` is not registered"):
            AttentionContextParallel().install_forward(_StandInAttention(), _MeshStub(2), cp_implementation="missing")


class TestSplitSequenceParallel(unittest.TestCase):
    def test_first_layer_cuts_hidden_states(self):
        mesh = _MeshStub(size=2, rank=1)
        first, second = _StandInLayer(), _StandInLayer()
        SplitSequenceParallel().install_forward(first, mesh, split_hidden_states=True)
        SplitSequenceParallel().install_forward(second, mesh)

        hidden_states = torch.randn(2, 8, 4, generator=torch.Generator().manual_seed(0))
        attention_mask = torch.ones(2, 1, 8, 8, dtype=torch.bool).tril()
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
        cos, sin = (torch.randn(2, 8, 2, generator=torch.Generator().manual_seed(seed)) for seed in (1, 2))

        local = hidden_states.chunk(2, dim=1)[1]
        output = first(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=(cos, sin)
        )
        torch.testing.assert_close(first.seen["hidden_states"], local)
        torch.testing.assert_close(first.seen["position_embeddings"][0], cos.chunk(2, dim=-2)[1])
        torch.testing.assert_close(first.seen["position_embeddings"][1], sin.chunk(2, dim=-2)[1])
        torch.testing.assert_close(first.seen["attention_mask"], attention_mask)
        torch.testing.assert_close(first.seen["position_ids"], position_ids)

        # a later layer gets this rank's slice already, and cuts the rotary embeddings the model computed for all of it
        second(output, position_embeddings=cos)
        torch.testing.assert_close(second.seen["hidden_states"], local)
        torch.testing.assert_close(second.seen["position_embeddings"], cos.chunk(2, dim=-2)[1])

    def test_refuses_a_sequence_that_is_not_divisible(self):
        mesh = _MeshStub(size=2)
        layer = _StandInLayer()
        SplitSequenceParallel().install_forward(layer, mesh, split_hidden_states=True)
        with self.assertRaisesRegex(ValueError, "must be divisible by `cp_size`"):
            layer(torch.randn(1, 7, 4), position_embeddings=torch.randn(1, 7, 2))

        # the same check on the rotary embeddings of a later layer
        layer = _StandInLayer()
        SplitSequenceParallel().install_forward(layer, mesh)
        with self.assertRaisesRegex(ValueError, "must be divisible by `cp_size`"):
            layer(torch.randn(1, 4, 4), position_embeddings=torch.randn(1, 7, 2))


class TestUlyssesCheck(unittest.TestCase):
    def _check(self, config, cp_size=2):
        model = torch.nn.Module()
        model.config = config
        UlyssesCP(_MeshStub(cp_size)).check(model)

    def test_refuses_attention_it_cannot_route(self):
        for attn_implementation in ("eager", "flex_attention"):
            with self.subTest(attn_implementation), self.assertRaisesRegex(ValueError, "takes no attention mask"):
                self._check(LlamaConfig(attn_implementation=attn_implementation))

    def test_refuses_indivisible_heads(self):
        # 6 attention heads over 4 ranks
        config = LlamaConfig(hidden_size=48, num_attention_heads=6, num_key_value_heads=6, attn_implementation="sdpa")
        with self.assertRaisesRegex(ValueError, "must divide both the number of attention heads"):
            self._check(config, cp_size=4)
        # 8 attention heads but only 2 key/value heads over 4 ranks
        config = LlamaConfig(hidden_size=32, num_attention_heads=8, num_key_value_heads=2, attn_implementation="sdpa")
        with self.assertRaisesRegex(ValueError, "must divide both the number of attention heads"):
            self._check(config, cp_size=4)

    def test_accepts_supported_config(self):
        self._check(LlamaConfig(attn_implementation="sdpa", num_attention_heads=8, num_key_value_heads=4))
        # the mask the model builds over the whole sequence already has the sliding window in it
        self._check(MistralConfig(attn_implementation="sdpa", sliding_window=5))
        # flash kernels build the sliding window themselves
        self._check(MistralConfig(attn_implementation="flash_attention_2", sliding_window=5))


class TestContextParallelStyles(unittest.TestCase):
    def test_styles_are_registered(self):
        self.assertIsInstance(ALL_PARALLEL_STYLES["cp_split_sequence"], SplitSequenceParallel)
        self.assertIsInstance(ALL_PARALLEL_STYLES["cp_attention"], AttentionContextParallel)
