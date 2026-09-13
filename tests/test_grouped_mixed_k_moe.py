"""CPU tests for grouped per-K-triple fused dispatch on heterogeneous packed-K MoE layers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3

HIDDEN, INTER = 16, 8


def _linear(k_words: int, fill: float) -> SimpleNamespace:
    return SimpleNamespace(
        trellis=torch.zeros(1, 1, k_words, dtype=torch.int16),
        suh=torch.full((HIDDEN,), fill, dtype=torch.float16),
        svh=torch.ones(INTER, dtype=torch.float16),
        mcg=False,
        mul1=True,
    )


def _inners(triples):
    return [
        {"gate": _linear(16 * kg, e + 1.0), "up": _linear(16 * ku, e + 1.0), "down": _linear(16 * kd, e + 1.0)}
        for e, (kg, ku, kd) in enumerate(triples)
    ]


class _FakeExllama:
    """exl3_moe double: expert e scales its rows by e + 1 and checks the launch K."""

    def __init__(self, inners):
        self.calls = []
        self.by_gate_suh = {int(p["gate"].suh.data_ptr()): e for e, p in enumerate(inners)}
        self.triples = {
            e: tuple(int(p[w].trellis.shape[-1]) // 16 for w in ("gate", "up", "down"))
            for e, p in enumerate(inners)
        }

    def exl3_moe_max_concurrency(self, index):
        return 1

    def exl3_moe(self, xh, out, count, token_sorted, weight_sorted, t0, t1, t2, t3, act, kg, ku, kd,
                 gate_t, gate_suh, gate_svh, up_t, up_suh, up_svh, down_t, down_suh, down_svh, *rest):
        n = int(count.numel()) - 1
        offsets = [0] + torch.cumsum(count[:n], 0).tolist()
        experts = []
        for j in range(n):
            e = self.by_gate_suh[int(gate_suh[j])]
            assert self.triples[e] == (kg, ku, kd)
            lo, hi = offsets[j], offsets[j + 1]
            if hi == lo:
                continue
            experts.append(e)
            rows = token_sorted[lo:hi]
            out.index_add_(0, rows, xh[rows].float() * (e + 1) * weight_sorted[lo:hi].float().unsqueeze(-1))
        self.calls.append(((kg, ku, kd), experts))


@pytest.fixture()
def grouped(monkeypatch: pytest.MonkeyPatch):
    inners = _inners([(2, 2, 2), (1, 1, 2), (2, 2, 2), (1, 2, 3)])
    fake = _FakeExllama(inners)
    monkeypatch.setattr(exl3, "load_exllamav3_ext", lambda: fake)
    monkeypatch.setitem(sys.modules, "exllamav3_ext", fake)
    monkeypatch.setattr(exl3, "_FUSED_TEMP_CACHE", {})
    monkeypatch.setattr(exl3, "get_moe_kernel_backend", lambda: "exllamav3")
    monkeypatch.setattr(exl3, "FAT_EXPERT_THRESHOLD", 1 << 30)
    layer = SimpleNamespace(
        w13_suh=torch.zeros(1),
        _exl3_hidden_size=HIDDEN,
        _exl3_intermediate_local=INTER,
        _exl3_codebook_flags=(False, True) * 3,
        _exl3_ptrs=None,
        _exl3_inners=inners,
        expert_map=None,
    )
    exl3.build_exl3_grouped_fused_state(layer, inners)
    return layer, inners, fake


def test_grouped_state_partitions_experts_by_physical_k_triple(grouped) -> None:
    layer, inners, _ = grouped
    groups = layer._exl3_k_groups
    assert [(g["k"], g["members"]) for g in groups] == [((1, 1, 2), [1]), ((1, 2, 3), [3]), ((2, 2, 2), [0, 2])]
    assert groups[-1]["local_to_group"].tolist() == [0, 2, 1, 2, 2]
    assert [g["base"] for g in groups] == [0, 2, 4]
    dispatch = layer._exl3_k_dispatch
    assert dispatch["flat_key"].tolist() == [4, 0, 5, 2, 7] and dispatch["total"] == 7
    assert dispatch["slot_valid"].tolist() == [1, 0, 1, 0, 1, 1, 0]
    assert groups[-1]["ptrs"]["down_svh"].tolist() == [
        inners[0]["down"].svh.data_ptr(),
        inners[2]["down"].svh.data_ptr(),
    ]
    assert layer._exl3_k == -1 and layer._exl3_ptrs is None
    assert tuple(layer._exl3_fused_temps[2].shape) == (1, exl3.TEMP_ROWS_FUSED, INTER)


def test_grouped_launches_match_per_route_reference(grouped) -> None:
    layer, inners, fake = grouped
    torch.manual_seed(0)
    x = torch.randn(6, HIDDEN)
    ids = torch.tensor([[0, 1], [2, 3], [0, 2], [1, -1], [3, 0], [2, 2]])
    weights = torch.rand(6, 2)
    out = exl3.apply_exl3_fused_moe(x, ids, weights, layer, inners, None)
    expect = torch.zeros(6, HIDDEN)
    xh = x.half().float()
    for t in range(6):
        for k in range(2):
            e = int(ids[t, k])
            if e >= 0:
                expect[t] += xh[t] * (e + 1) * weights[t, k].half().float()
    torch.testing.assert_close(out, expect, rtol=1e-3, atol=1e-3)
    assert sorted(k for k, _ in fake.calls) == [(1, 1, 2), (1, 2, 3), (2, 2, 2)]


def test_groups_without_routes_are_not_launched(grouped) -> None:
    layer, inners, fake = grouped
    exl3.apply_exl3_fused_moe(torch.randn(3, HIDDEN), torch.tensor([[0], [2], [0]]), torch.ones(3, 1), layer,
                              inners, None)
    assert fake.calls == [((2, 2, 2), [0, 2])]


def test_apply_experts_treats_grouped_state_as_fused(grouped, monkeypatch: pytest.MonkeyPatch) -> None:
    layer, _, fake = grouped
    monkeypatch.setenv("EXL3_FUSED_MOE", "1")
    out = exl3.apply_exl3_experts(torch.randn(1, 4, HIDDEN), torch.tensor([[1, 3]] * 4), torch.ones(4, 2), layer)
    assert layer._exl3_last_apply == "fused"
    assert tuple(out.shape) == (4, HIDDEN)
    assert sorted(k for k, _ in fake.calls) == [(1, 1, 2), (1, 2, 3)]


def test_mixed_k_load_builds_grouped_state_when_fused_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from test_mixed_k_exl3 import _load_expert, _make_method_layer, _stub_linear
    except ImportError:  # pragma: no cover - package-style test discovery
        from tests.test_mixed_k_exl3 import _load_expert, _make_method_layer, _stub_linear

    monkeypatch.setenv("VLLM_EXL3_TRELLIS_ARENA", "0")
    monkeypatch.setenv("VLLM_EXL3_ARENA_PRESCAN", "0")
    monkeypatch.setenv("EXL3_FUSED_MOE", "1")
    monkeypatch.setattr(exl3, "make_linear_exl3", _stub_linear)
    monkeypatch.setattr(exl3, "_FUSED_TEMP_CACHE", {})
    monkeypatch.setattr(exl3, "get_moe_kernel_backend", lambda: "exllamav3")
    fake = SimpleNamespace(exl3_moe=lambda *args: None, exl3_moe_max_concurrency=lambda index: 1)
    monkeypatch.setattr(exl3, "load_exllamav3_ext", lambda: fake)
    method, layer = _make_method_layer(n_experts=3, bits=2)
    _load_expert(method, layer, 0, 2, 2, 2)
    _load_expert(method, layer, 1, 1, 1, 2)
    _load_expert(method, layer, 2, 2, 2, 2)
    method.process_weights_after_loading(layer)
    assert layer._exl3_mixed_k is True
    assert [(g["k"], g["members"]) for g in layer._exl3_k_groups] == [((1, 1, 2), [1]), ((2, 2, 2), [0, 2])]
    assert layer._exl3_k == -1


def test_mixed_k_load_keeps_python_loop_when_fused_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from test_mixed_k_exl3 import _load_expert, _make_method_layer, _stub_linear
    except ImportError:  # pragma: no cover - package-style test discovery
        from tests.test_mixed_k_exl3 import _load_expert, _make_method_layer, _stub_linear

    monkeypatch.setenv("VLLM_EXL3_TRELLIS_ARENA", "0")
    monkeypatch.setenv("VLLM_EXL3_ARENA_PRESCAN", "0")
    monkeypatch.setenv("EXL3_FUSED_MOE", "0")
    monkeypatch.setattr(exl3, "make_linear_exl3", _stub_linear)
    method, layer = _make_method_layer(n_experts=2, bits=2)
    _load_expert(method, layer, 0, 1, 1, 1)
    _load_expert(method, layer, 1, 2, 2, 2)
    method.process_weights_after_loading(layer)
    assert layer._exl3_mixed_k is True
    assert layer._exl3_k_groups is None and layer._exl3_ptrs is None
