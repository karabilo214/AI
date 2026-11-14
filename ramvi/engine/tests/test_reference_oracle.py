"""Эталонный контракт: ручные случаи, независимое квантование, tiny torch model."""

from typing import Any

import pytest
from ramvi_engine.validation.comparison import compare_logits, compare_tokens


def test_logit_tolerance_does_not_hide_argmax_change() -> None:
    result = compare_logits([1.0, 1.001], [1.001, 1.0], atol=0.01, rtol=0.0)
    assert result["logits_within_tolerance"]
    assert not result["argmax_matches"] and not result["passed"]
    assert result["actual_argmax"] == 1 and result["expected_argmax"] == 0
    assert compare_logits([2.0, 2.0], [2.0, 2.0], atol=0.0, rtol=0.0)["actual_argmax"] == 0


def test_reference_rejects_lazy_constant_and_missing_suffix() -> None:
    assert not compare_logits([0.0, 0.0, 0.0], [-2.0, 1.0, 7.0], atol=0.001, rtol=0.0)["passed"]
    assert not compare_tokens([4], [4, 9])["passed"]
    assert compare_tokens([4], [4, 9])["first_mismatch"] == 1
    assert compare_tokens([100003, 29, 100032], [100003, 29, 100032])["passed"]
    with pytest.raises(ValueError):
        compare_logits([float("nan")], [1.0], atol=0.0, rtol=0.0)
    with pytest.raises(ValueError):
        compare_logits([], [], atol=0.0, rtol=0.0)


def test_independent_quant_decoder_manual_blocks() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from ramvi_engine.validation.oracle import decode_quant

    # Q4: d=2,dmin=1; every scale=3,min=1; low nibble=4,high=5 -> 23,29.
    q4 = np.zeros(144, np.uint8)
    q4[:4] = np.array([2.0, 1.0], np.float16).view(np.uint8)
    q4[4:8] = 3
    q4[8:12] = 1
    q4[12:16] = 3 | (1 << 4)
    q4[16:] = 4 | (5 << 4)
    decoded = decode_quant(q4, "Q4_K", 256).reshape(8, 32)
    for row in range(8):
        np.testing.assert_array_equal(decoded[row], 23.0 if row % 2 == 0 else 29.0)
    # Q6: low=0,high=2 -> q=32 -> signed0. Then high=3 -> signed16, scale=-2,d=.5 -> -16.
    q6 = np.zeros(210, np.uint8)
    q6[128:192] = 170
    q6[192:208] = np.array([-2] * 16, np.int8).view(np.uint8)
    q6[208:] = np.array([0.5], np.float16).view(np.uint8)
    np.testing.assert_array_equal(decode_quant(q6, "Q6_K", 256), 0.0)
    q6[128:192] = 255
    np.testing.assert_array_equal(decode_quant(q6, "Q6_K", 256), -16.0)


def test_quant_decoder_matches_reader_on_fresh_blocks() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from ramvi_engine.formats.gguf import dequant
    from ramvi_engine.validation.oracle import decode_quant

    rng = np.random.default_rng(72819)
    for kind, size in [("Q4_K", 144), ("Q6_K", 210)]:
        for _ in range(3):
            raw = rng.integers(0, 256, size * 7, dtype=np.uint8).reshape(7, size)
            if kind == "Q4_K":
                raw[:, :4] = np.array([0.125, 0.25], np.float16).view(np.uint8)
            else:
                raw[:, 208:] = np.array([0.125], np.float16).view(np.uint8)
            np.testing.assert_array_equal(
                decode_quant(raw.reshape(-1), kind, 7 * 256),
                dequant(raw.reshape(-1), kind, 7 * 256),  # type: ignore[no-untyped-call]
            )


def test_tiny_full_forward_is_deterministic_and_causal() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from ramvi_engine.validation.oracle import TorchReference

    # Construct a tiny fully specified network, not a trained/model-quality claim.
    oracle = object.__new__(TorchReference)
    oracle.device, oracle.native_kv, oracle.gguf = "cpu", False, False
    oracle.arch, oracle.L, oracle.D, oracle.H, oracle.KV, oracle.HD = "qwen3", 1, 4, 2, 1, 2
    oracle.eps, oracle.theta, oracle.F, oracle.V, oracle.E, oracle.K = 1e-6, 10000.0, 8, 7, 0, 0
    oracle.head_name = "lm_head.weight"
    generator = torch.Generator().manual_seed(99531)
    weights = {
        "model.embed_tokens.weight": torch.randn(7, 4, generator=generator),
        "lm_head.weight": torch.randn(7, 4, generator=generator),
        "model.norm.weight": torch.ones(4),
    }
    sizes = {
        "q": (4, 4),
        "k": (2, 4),
        "v": (2, 4),
        "o": (4, 4),
        "gate": (8, 4),
        "up": (8, 4),
        "down": (4, 8),
    }
    for name, key in oracle._names(0).items():
        weights[key] = (
            torch.randn(*sizes[name], generator=generator) * 0.1
            if name in sizes
            else torch.ones(2 if name in ("qn", "kn") else 4)
        )

    def weight(name: str, begin: int = 0, end: int | None = None) -> Any:
        value = weights[name]
        return value[begin:end] if value.ndim > 1 else value

    oracle.weight = weight  # type: ignore[method-assign]
    first, _ = oracle.hidden([1, 2, 3])
    second, _ = oracle.hidden([1, 2, 3, 4])
    torch.testing.assert_close(first, second[:3], atol=1e-6, rtol=1e-6)
    # Bulk teacher forcing must equal separate causal prefixes at every position.
    for tokens in ([1, 2, 3, 4], [6, 1, 0, 5, 2]):
        bulk = oracle.logits(list(tokens), positions=list(range(len(tokens))))
        for position in range(len(tokens)):
            torch.testing.assert_close(
                bulk[position],
                oracle.logits(list(tokens[: position + 1]))[0],
                atol=1e-6,
                rtol=1e-6,
            )
    for _ in range(3):
        torch.testing.assert_close(
            oracle.logits([1, 2, 3]), oracle.project(first[-1:]), atol=0.0, rtol=0.0
        )


def test_hf_native_projection_moves_hidden_to_reference_device() -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from ramvi_engine.validation.native import native_logits

    class Hidden:
        def __init__(self, device: str = "cuda") -> None:
            self.device = device

        def __getitem__(self, key: Any) -> "Hidden":
            return self

        def detach(self) -> "Hidden":
            return self

        def float(self) -> "Hidden":
            return self

        def to(self, device: str) -> "Hidden":
            return Hidden(device)

    class Reference:
        device = "cpu"

        def project(self, hidden: Hidden) -> Any:
            assert hidden.device == self.device
            return torch.tensor([[3.0, 1.0]])

    engine = SimpleNamespace(hb=Hidden(), process=lambda *args, **kwargs: [0], ctx=[])
    logits, selected = native_logits(engine, Reference(), [1])
    assert selected == 0 and logits.tolist() == [3.0, 1.0]


def test_gguf_projection_bounds_decode_working_set() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from types import SimpleNamespace

    from ramvi_engine.validation.oracle import TorchReference

    reference = object.__new__(TorchReference)
    reference.gguf = True
    generator = torch.Generator().manual_seed(662041)
    matrix = torch.randn(2053, 37, generator=generator)
    inputs = torch.randn(5, 37, generator=generator)
    reference.g = SimpleNamespace(raw=lambda name: (None, 2053, 37, "F32"))

    def bounded_weight(name: str, begin: int = 0, end: int | None = None) -> Any:
        assert end is not None and end - begin <= 1024
        return matrix[begin:end]

    reference.weight = bounded_weight  # type: ignore[method-assign]
    expected = inputs @ matrix.T
    for _ in range(2):
        torch.testing.assert_close(reference.linear(inputs, "weight"), expected)
    with pytest.raises(ValueError):
        reference.linear(inputs, "weight", chunk=0)
