import json
import socket

import pytest

from ramvi_service import hardware


def test_parse_nvidia_smi():
    gpus = hardware.parse_nvidia_smi(
        "NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7900, 560.94\n\n"
    )
    assert gpus == (
        hardware.GPUInfo("NVIDIA GeForce RTX 4060 Laptop GPU", "NVIDIA", 8188, 7900, "560.94"),
    )


def test_parse_meminfo():
    text = "MemTotal:       32768000 kB\nMemFree: 100 kB\nMemAvailable:   16384000 kB\n"
    assert hardware.parse_meminfo(text) == (32000, 16000)


def test_parse_cpuinfo_counts_physical_cores():
    block = "processor\t: {p}\nmodel name\t: Ryzen 7\nflags\t: sse4_2 avx avx2 fma\nphysical id\t: 0\ncore id\t: {c}\n\n"
    text = "".join(block.format(p=p, c=p // 2) for p in range(4))
    name, flags, physical = hardware.parse_cpuinfo(text)
    assert name == "Ryzen 7"
    assert hardware._isa_from_flags(flags) == ("sse4_2", "avx", "avx2", "fma")
    assert physical == 2


def test_collect_real_machine_no_network(monkeypatch, tmp_path):
    def blocked(*args, **kwargs):
        raise AssertionError("профиль железа не должен открывать сеть")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    profile = hardware.collect(tmp_path)
    data = profile.to_dict()
    json.dumps(data)
    assert data["cores_logical"] >= 1
    assert data["ram_total_mb"] and data["ram_total_mb"] > 0
    assert data["ram_available_mb"] is None or 0 <= data["ram_available_mb"] <= data["ram_total_mb"]
    assert data["disk_free_mb"] is not None and data["disk_free_mb"] <= data["disk_total_mb"]
    for gpu in profile.gpus:
        assert gpu.name


def test_missing_nvidia_smi_gives_no_gpus(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    assert hardware._nvidia_gpus() == ()


def test_parse_wmi_and_merge():
    text = ('{"cpu":[{"NumberOfCores":8}],"gpu":[{"Name":"AMD Radeon 780M Graphics","DriverVersion":"32.0"},'
            '{"Name":"NVIDIA GeForce RTX 5060 Laptop GPU","DriverVersion":"32.0.15"}]}')
    cores, gpus = hardware.parse_wmi_json(text)
    assert cores == 8 and [g.vendor for g in gpus] == ["AMD", "NVIDIA"]
    nvidia = hardware.parse_nvidia_smi("NVIDIA GeForce RTX 5060 Laptop GPU, 8151, 7899, 616.92")
    merged = hardware._merge_gpus(nvidia, gpus)
    assert [(g.vendor, g.vram_total_mb) for g in merged] == [("NVIDIA", 8151), ("AMD", None)]
    assert hardware._merge_gpus((), gpus) == gpus


def test_parse_system_profiler():
    text = '{"SPDisplaysDataType":[{"_name":"kHW_AppleM1ProItem","sppci_model":"Apple M1 Pro"}]}'
    assert hardware.parse_system_profiler(text) == (hardware.GPUInfo("Apple M1 Pro", "Apple"),)


if __name__ == "__main__":
    pytest.main([__file__])
