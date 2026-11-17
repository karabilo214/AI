from dataclasses import replace
from typing import Any

from ramvi_engine.planning.resources import MemoryUsage, ResourceEstimate
from ramvi_service import compat
from ramvi_service.hardware import GPUInfo, HardwareProfile
from ramvi_service.i18n import Msg, load_catalog, render

RU, EN = load_catalog("ru"), load_catalog("en")


def ru(message: Msg | str) -> str:
    """Текст сообщения так, как его покажет оболочка на русском (3.14)."""
    return render(message, RU)


GB = 1024**3

# Как тестовый ноутбук: RTX 5060 8 ГБ + встроенная Radeon, 16 ГБ RAM.
LAPTOP = HardwareProfile(
    os="Windows 11",
    arch="AMD64",
    cpu_name="Ryzen 7 260",
    isa=("avx2",),
    cores_physical=8,
    cores_logical=16,
    ram_total_mb=15652,
    ram_available_mb=7654,
    disk_path="C:\\",
    disk_total_mb=486272,
    disk_free_mb=185345,
    gpus=(GPUInfo("RTX 5060", "NVIDIA", 8151, 7899, "616.92"), GPUInfo("Radeon 780M", "AMD")),
)


def est(vram_gb: float, ram_gb: float = 0.0, mapped_gb: float = 0.0) -> ResourceEstimate:
    return ResourceEstimate(
        weights=MemoryUsage(int(vram_gb * GB), int(ram_gb * GB), int(mapped_gb * GB))
    )


def cand(name: str, estimate: ResourceEstimate, size_gb: float = 5, **kw: Any) -> compat.Candidate:
    return compat.Candidate(name, "gguf", int(size_gb * GB), estimate, **kw)


def test_small_model_recommended() -> None:
    assert compat.assess(LAPTOP, cand("4B", est(4))).status == compat.RECOMMENDED


def test_14b_on_8gb_gpu_is_slow_not_unsuitable() -> None:
    v = compat.assess(LAPTOP, cand("14B", est(6.5, ram_gb=2, mapped_gb=1), size_gb=9))
    assert v.status == compat.SLOW and v.reasons[0].code == "compat.reason.on_cpu"
    assert ru(v.reasons[0]) == "3072 МБ весов не помещаются в VRAM и считаются на CPU"
    assert render(v.reasons[0], EN) == "3072 MB of weights do not fit in VRAM and run on the CPU"


def test_low_available_ram_is_note_not_failure() -> None:
    v = compat.assess(LAPTOP, cand("14B", est(6.5, ram_gb=6, mapped_gb=3)))
    assert v.status == compat.SLOW and any("закройте" in ru(r) for r in v.reasons)


def test_not_enough_ram_disk_format_gpu() -> None:
    assert compat.assess(LAPTOP, cand("70B", est(7, ram_gb=30))).status == compat.NO_RAM
    assert compat.assess(LAPTOP, cand("huge", est(4), size_gb=200)).status == compat.NO_DISK
    assert (
        compat.assess(LAPTOP, cand("huge", est(4), size_gb=200, downloaded=True)).status
        == compat.RECOMMENDED
    )
    assert (
        compat.assess(LAPTOP, replace(cand("x", est(4)), file_format="onnx")).status
        == compat.UNSUPPORTED
    )
    assert (
        compat.assess(LAPTOP, cand("x", est(4), architecture_supported=False)).status
        == compat.UNSUPPORTED
    )
    no_nvidia = replace(LAPTOP, gpus=(GPUInfo("Radeon 780M", "AMD"),))
    assert compat.assess(no_nvidia, cand("4B", est(4))).status == compat.UNSUPPORTED


def test_file_bigger_than_ram_is_slow_with_paging_not_no_ram() -> None:
    """Веса из файла дочитываются с диска: qwen3-8b-f16 работал на 16 ГБ RAM (2,35 ток/с)."""
    v = compat.assess(LAPTOP, cand("8B F16", est(6, ram_gb=0.5, mapped_gb=16), size_gb=16))
    assert v.status == compat.SLOW
    paging = next(r for r in v.reasons if r.code == "compat.reason.paging")
    assert paging.params["mb"] == (int(0.5 * GB) + 16 * GB) // compat.MIB - 15652
    assert not any(r.code == "compat.reason.ram_busy" for r in v.reasons)
    # Собственная RAM движка больше всей памяти — по-прежнему «Не хватает RAM».
    assert compat.assess(LAPTOP, cand("x", est(6, ram_gb=16))).status == compat.NO_RAM


def test_layers_in_vram_are_recommended_even_with_file_embeddings() -> None:
    """Входные эмбеддинги движок всегда читает из файла — это не «часть модели на CPU»."""
    whole = cand("8B", est(6, mapped_gb=0.3), off_gpu_bytes=0)
    assert compat.assess(LAPTOP, whole).status == compat.RECOMMENDED
    part = cand("14B", est(7.5, ram_gb=1.5, mapped_gb=1.6), off_gpu_bytes=1400 * compat.MIB)
    v = compat.assess(LAPTOP, part)
    assert v.status == compat.SLOW and v.reasons[0].params["mb"] == 1400


def test_plan_over_vram_flagged() -> None:
    assert compat.assess(LAPTOP, cand("bad plan", est(12))).status == compat.NO_VRAM


def test_filter_order_and_size() -> None:
    cands = [
        cand("70B", est(7, ram_gb=30)),
        cand("14B", est(6.5, ram_gb=2), size_gb=9),
        cand("4B", est(4), size_gb=3),
    ]
    assert [v.name for v in compat.filter_models(LAPTOP, cands)] == ["4B", "14B", "70B"]
    assert [v.name for v in compat.filter_models(LAPTOP, cands, statuses=[compat.SLOW])] == ["14B"]
    assert [v.name for v in compat.filter_models(LAPTOP, cands, max_file_bytes=4 * GB)] == ["4B"]


def test_speed_only_from_measurements() -> None:
    v = compat.assess(LAPTOP, cand("Qwen3-14B", est(6.5, ram_gb=2), model_id="qwen3-14b-q4_k_m"))
    assert v.status == compat.SLOW and ru(v.speed) == "примерно 7.8–9.1 токенов/с"
    assert ru(compat.assess(LAPTOP, cand("4B", est(4), model_id="qwen3-4b")).speed) == "нет замера"
    other_gpu = replace(LAPTOP, gpus=(GPUInfo("RTX 4090", "NVIDIA", 24564),))
    assert (
        ru(
            compat.assess(
                other_gpu, cand("14B", est(6.5, ram_gb=2), model_id="qwen3-14b-q4_k_m")
            ).speed
        )
        == "нет замера"
    )
    own = cand("4B", est(4), measured_tok_s=(50, 60))
    assert ru(compat.assess(LAPTOP, own).speed) == "примерно 50–60 токенов/с"
    assert (
        ru(compat.assess(LAPTOP, cand("70B", est(7, ram_gb=30), measured_tok_s=(2, 4))).speed)
        == "нет замера"
    )


def test_estimate_shown_when_no_measurement_and_measurement_wins() -> None:
    estimated = cand("8B", est(6), off_gpu_bytes=0, estimated_tok_s=(52.0, 62.0))
    speed = compat.assess(LAPTOP, estimated).speed
    assert speed.code == "compat.speed_estimate" and ru(speed) == "оценка ~52–62 токенов/с"
    measured = cand(
        "14B", est(7.5, ram_gb=1.5), model_id="qwen3-14b-q4_k_m", estimated_tok_s=(3.9, 7.7)
    )
    assert ru(compat.assess(LAPTOP, measured).speed) == "примерно 7.8–9.1 токенов/с"
