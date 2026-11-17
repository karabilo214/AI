"""4.4 MOD-03: заголовок GGUF без весов, копии заголовков и статус модели по плану движка."""

import json
import struct
from pathlib import Path
from typing import Any

import pytest
from ramvi_engine.planning.resources import estimate_gguf
from ramvi_service import compat, gguf_meta
from ramvi_service.catalog.headers import FIRST, HeaderStore, parse_prefix, read_file_header
from ramvi_service.hardware import GPUInfo
from ramvi_service.model_fit import ModelFit, model_key, plan
from test_compat import LAPTOP

D, H, KV, HD, FF, LAYERS, VOCAB = 256, 4, 2, 64, 512, 3, 1024
Q4_K, Q6_K, F32, F16, Q8_0 = 12, 14, 0, 1, 8
BLOCK = {Q4_K: (144, 256), Q6_K: (210, 256), F32: (4, 1), F16: (2, 1), Q8_0: (34, 32)}


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def tiny_qwen3(path: Path, matrix: int = Q4_K, pad_tokens: int = 0) -> Path:
    """Настоящий маленький GGUF Qwen3 (веса — нули): его читают и движок, и gguf_meta."""
    kv = [
        _s("general.architecture") + struct.pack("<I", 8) + _s("qwen3"),
        _s("general.size_label") + struct.pack("<I", 8) + _s("0.01B"),
        _s("general.file_type") + struct.pack("<II", 4, 15),
        _s("tokenizer.ggml.tokens")
        + struct.pack("<IIQ", 9, 8, pad_tokens)
        + b"".join(_s(f"t{i}") for i in range(pad_tokens)),
    ]
    for key, value in (
        ("block_count", LAYERS),
        ("embedding_length", D),
        ("attention.head_count", H),
        ("attention.head_count_kv", KV),
        ("attention.key_length", HD),
        ("attention.value_length", HD),
        ("feed_forward_length", FF),
        ("context_length", 4096),
    ):
        kv.append(_s(f"qwen3.{key}") + struct.pack("<II", 4, value))
    tensors: list[tuple[str, tuple[int, ...], int]] = [
        ("token_embd.weight", (D, VOCAB), Q6_K),
        ("output_norm.weight", (D,), F32),
        ("output.weight", (D, VOCAB), Q6_K),
    ]
    for i in range(LAYERS):
        tensors += [
            (f"blk.{i}.attn_norm.weight", (D,), F32),
            (f"blk.{i}.ffn_norm.weight", (D,), F32),
            (f"blk.{i}.attn_q_norm.weight", (HD,), F32),
            (f"blk.{i}.attn_k_norm.weight", (HD,), F32),
            (f"blk.{i}.attn_q.weight", (D, H * HD), matrix),
            (f"blk.{i}.attn_k.weight", (D, KV * HD), matrix),
            (f"blk.{i}.attn_v.weight", (D, KV * HD), matrix),
            (f"blk.{i}.attn_output.weight", (H * HD, D), matrix),
            (f"blk.{i}.ffn_gate.weight", (D, FF), matrix),
            (f"blk.{i}.ffn_up.weight", (D, FF), matrix),
            (f"blk.{i}.ffn_down.weight", (FF, D), Q6_K if matrix == Q4_K else matrix),
        ]
    infos, offset = [], 0
    for name, dims, kind in tensors:
        infos.append(
            _s(name)
            + struct.pack("<I", len(dims))
            + struct.pack(f"<{len(dims)}Q", *dims)
            + struct.pack("<IQ", kind, offset)
        )
        size = dims[0] // BLOCK[kind][1] * BLOCK[kind][0] * (dims[1] if len(dims) > 1 else 1)
        offset += (size + 31) // 32 * 32
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(kv) + b"".join(infos)
    head += b"\0" * (-len(head) % 32)
    path.write_bytes(head + b"\0" * offset)
    return path


# --- gguf_meta -------------------------------------------------------------------------


def test_header_matches_engine_parser_and_estimate(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    from ramvi_engine.formats.gguf import GGUF

    path = tiny_qwen3(tmp_path / "m.gguf")
    engine = GGUF(str(path))  # type: ignore[no-untyped-call]
    light = read_file_header(path)

    assert {n: (tuple(d), t) for n, (d, t, _) in engine.tensors.items()} == light.tensors
    assert all(light.meta[k] == v for k, v in engine.meta.items() if not isinstance(v, list))
    for route in (["gpu"] * 3, ["gpu", "stream", "cpu"], ["gpu", "hyb", "hyb"], ["cpu"] * 3):
        for native in (False, True):
            ours = estimate_gguf(light, route, 2048, native_f16=native).to_dict()
            assert ours == estimate_gguf(engine, route, 2048, native_f16=native).to_dict()
    engine.close()  # type: ignore[no-untyped-call]


def test_truncated_header_reports_how_much_is_needed(tmp_path: Path) -> None:
    data = tiny_qwen3(tmp_path / "m.gguf").read_bytes()
    full = gguf_meta.parse(data)
    with pytest.raises(gguf_meta.Truncated) as short:
        gguf_meta.parse(data[:100])
    assert short.value.need > 100
    assert gguf_meta.GGUFMeta.from_json(json.loads(json.dumps(full.json()))).tensors == full.tensors


def test_bad_headers_are_rejected() -> None:
    with pytest.raises(gguf_meta.NotGGUF):
        gguf_meta.parse(b"GGML" + b"\0" * 100)
    with pytest.raises(gguf_meta.NotGGUF):
        gguf_meta.parse(b"GGUF" + struct.pack("<IQQ", 9, 0, 0))
    huge = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + _s("k") + struct.pack("<IIQ", 9, 8, 10**9)
    with pytest.raises(gguf_meta.NotGGUF):
        gguf_meta.parse(huge)


def test_unknown_tensor_type_cannot_be_estimated() -> None:
    meta = gguf_meta.GGUFMeta({}, {"x": ((256, 4), 99)})
    with pytest.raises(NotImplementedError):
        meta.raw("x")


# --- копии заголовков ------------------------------------------------------------------


def test_prefix_grows_until_header_fits(tmp_path: Path) -> None:
    data = tiny_qwen3(tmp_path / "m.gguf", pad_tokens=1_100_000).read_bytes()  # заголовок > 8 МиБ
    sizes: list[int] = []

    def read(size: int) -> bytes:
        sizes.append(size)
        return data[:size]

    header = parse_prefix(read)
    assert sizes[0] == FIRST and len(sizes) == 2 and sizes[1] > FIRST
    assert header.meta["general.architecture"] == "qwen3"


def test_store_caches_by_sha_and_never_rereads(tmp_path: Path) -> None:
    data = tiny_qwen3(tmp_path / "m.gguf").read_bytes()
    calls: list[tuple[str, int]] = []

    def read(url: str, size: int) -> bytes:
        calls.append((url, size))
        return data[:size]

    store = HeaderStore(tmp_path / "headers", read)
    sha = "sha256:" + "a" * 64
    assert store.cached(sha) is None
    first = store.fetch("https://example.org/m.gguf", sha, len(data))
    again = HeaderStore(tmp_path / "headers", read).fetch("https://example.org/m.gguf", sha)
    assert len(calls) == 1 and calls[0][1] <= len(data)  # не больше файла
    assert again.tensors == first.tensors
    with pytest.raises(gguf_meta.NotGGUF):
        store.fetch("https://example.org/x", "md5:1")


def test_not_a_gguf_is_not_cached(tmp_path: Path) -> None:
    store = HeaderStore(tmp_path / "headers", lambda url, size: b"<html>error</html>")
    with pytest.raises(gguf_meta.NotGGUF):
        store.fetch("https://example.org/x", "sha256:" + "b" * 64, 100)
    assert (
        not list((tmp_path / "headers").glob("*.json")) if (tmp_path / "headers").exists() else True
    )


# --- статус по плану движка ------------------------------------------------------------


def header(tmp_path: Path, **kw: Any) -> gguf_meta.GGUFMeta:
    return read_file_header(tiny_qwen3(tmp_path / "m.gguf", **kw))


def fit(profile: Any = LAPTOP) -> ModelFit:
    return ModelFit(HeaderStore(Path("unused")), lambda: profile, native_f16=lambda: True)


def test_small_model_in_vram_is_recommended(tmp_path: Path) -> None:
    verdict = fit().verdict(header(tmp_path), "tiny", 10**6, downloaded=False)
    assert verdict["status"] == compat.RECOMMENDED
    assert verdict["plan"]["gpu"] == LAYERS and verdict["plan"]["off_gpu_bytes"] == 0
    assert verdict["memory"]["vram"] > 0


def test_partial_vram_is_slow_not_unsuitable(tmp_path: Path) -> None:
    meta = header(tmp_path)
    _, estimate = plan(meta, LAPTOP)
    full_mb = estimate.required.vram // (1024 * 1024)
    seen = []
    for vram_mb in range(full_mb, 0, -1):  # карта всё меньше: от «всё в VRAM» до «ничего»
        gpu = GPUInfo("Tiny NVIDIA", "NVIDIA", vram_mb, vram_mb, "1")
        small = LAPTOP.__class__(**{**LAPTOP.__dict__, "gpus": (gpu,)})
        verdict = fit(small).verdict(meta, "tiny", 10**6, downloaded=False)
        layout = verdict.get("plan")
        if layout is not None and 0 < layout["gpu"] < LAYERS:
            seen.append(vram_mb)
            assert verdict["status"] == compat.SLOW, verdict  # часть на CPU — не отказ
            assert layout["off_gpu_bytes"] > 0
    assert seen, "ни при каком размере карты не нашлось частичного плана"


def test_no_nvidia_and_unsupported_types(tmp_path: Path) -> None:
    no_gpu = LAPTOP.__class__(**{**LAPTOP.__dict__, "gpus": ()})
    assert fit(no_gpu).verdict(header(tmp_path), "t", 1, False)["status"] == compat.UNSUPPORTED
    q8 = read_file_header(tiny_qwen3(tmp_path / "q8.gguf", matrix=Q8_0))
    verdict = fit().verdict(q8, "q8", 1, False)
    assert verdict["status"] == compat.UNSUPPORTED
    assert verdict["reasons"][0] == {"code": "compat.reason.format", "params": {"format": "Q8_0"}}


def test_missing_header_is_unknown_not_a_guess() -> None:
    entry = {"checksum": "sha256:" + "c" * 64, "name": "x", "bytes": 1}
    verdict = fit().for_entry(entry)
    assert verdict == {
        "status": "compat.unknown",
        "reasons": [{"code": "compat.reason.no_header", "params": {}}],
    }


def test_model_key_for_measurements(tmp_path: Path) -> None:
    assert model_key(header(tmp_path)) == "qwen3-0.01b-q4_k_m"


def test_installed_file_is_assessed_from_its_header(tmp_path: Path) -> None:
    path = tiny_qwen3(tmp_path / "m.gguf")
    model_fit = fit()
    assert model_fit.for_file(path, "m")["status"] == compat.RECOMMENDED
    assert model_fit.for_file(tmp_path / "missing.gguf", "x")["status"] == "compat.unknown"
    (tmp_path / "junk.gguf").write_bytes(b"junk")
    assert model_fit.for_file(tmp_path / "junk.gguf", "j")["status"] == "compat.unknown"


def test_api_statuses_for_catalog_and_installed_models(tmp_path: Path) -> None:
    from test_download_manager import App, FileServer, entry

    files = FileServer()
    data = tiny_qwen3(tmp_path / "src.gguf").read_bytes()
    files.files["/tiny-Q4_K_M.gguf"] = data
    app = App(tmp_path, [entry(files.base, "/tiny-Q4_K_M.gguf", data)], (0.05,), headers=True)
    try:
        # App уже обновил каталог (и прочитал заголовок) до запуска сервиса.
        requests_before = len(files.ranges)
        saved = tmp_path / "data" / "catalog" / "catalog.json"
        written = saved.stat().st_mtime_ns
        listed = app.client.ok("GET", "/catalog")["entries"]
        assert len(files.ranges) == requests_before  # GET /catalog в сеть не ходит
        assert saved.stat().st_mtime_ns == written  # и каталог не обновляет
        (item,) = listed
        assert item["compat"]["status"] in compat.ORDER  # посчитан (на этом ПК — свой статус)
        assert files.ranges and files.ranges[0] is not None  # заголовок — запросом Range

        did = app.client.ok("POST", "/downloads", {"entry_id": item["id"]})["download"]["id"]
        assert app.until(did, "done", "failed")["state"] == "done"
        (model,) = app.client.ok("GET", "/models")["models"]
        assert model["compat"]["status"] == item["compat"]["status"]
    finally:
        app.stop()
        files.close()


def test_catalog_entry_without_header_is_unknown_until_refresh(tmp_path: Path) -> None:
    from ramvi_service.catalog import Catalog
    from test_download_manager import Provider, entry

    data = tiny_qwen3(tmp_path / "src.gguf").read_bytes()
    reads: list[int] = []

    def read(url: str, size: int) -> bytes:
        reads.append(size)
        return data[:size]

    catalog = Catalog(
        tmp_path,
        [Provider([entry("https://example.org", "/tiny-Q4_K_M.gguf", data)])],
        headers=HeaderStore(tmp_path / "h", read),
    )
    fit_ = ModelFit(catalog.headers, lambda: LAPTOP, native_f16=lambda: True)
    catalog.path.parent.mkdir(parents=True)
    assert catalog.view()["entries"] == []
    view = catalog.refresh()
    assert len(reads) == 1
    assert fit_.for_entry(view["entries"][0])["status"] == compat.RECOMMENDED
    catalog.refresh()
    assert len(reads) == 1  # заголовок с тем же SHA-256 повторно не читается


def test_speed_is_an_estimate_from_the_plan(tmp_path: Path) -> None:
    verdict = fit().verdict(header(tmp_path), "tiny", 10**6, downloaded=False)
    assert verdict["speed"]["code"] == "compat.speed_estimate"
    low, high = float(verdict["speed"]["params"]["low"]), float(verdict["speed"]["params"]["high"])
    assert 0 < low <= high


def test_traffic_follows_the_plan_and_moe_reads_only_chosen_experts(tmp_path: Path) -> None:
    from ramvi_service.model_fit import traffic

    meta = header(tmp_path)
    layer = sum(meta.raw(n)[0].size for n in meta.tensors if n.startswith("blk.0."))
    head = meta.raw("output.weight")[0].size
    t = traffic(meta, ["gpu", "stream", "hyb"], paging=7)
    assert (t.vram, t.bus, t.cpu, t.disk, t.layers, t.moe) == (
        layer + head,
        layer,
        layer,
        7,
        3,
        False,
    )

    moe = gguf_meta.GGUFMeta(
        {
            "general.architecture": "qwen3moe",
            "qwen3moe.expert_used_count": 8,
            "qwen3moe.expert_count": 128,
        },
        {
            "blk.0.ffn_up_exps.weight": ((256, 128, 128), Q4_K),
            "blk.0.attn_q.weight": ((256, 256), Q4_K),
        },
    )
    experts, attention = (
        moe.raw("blk.0.ffn_up_exps.weight")[0].size,
        moe.raw("blk.0.attn_q.weight")[0].size,
    )
    t = traffic(moe, ["cpu"])
    assert t.moe and t.cpu == attention + experts * 8 // 128
