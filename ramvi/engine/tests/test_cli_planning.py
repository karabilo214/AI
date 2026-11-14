"""CLI planner flags and exact isolated-process argument roundtrip."""

from argparse import Namespace

import pytest
from ramvi_engine.cli import _planning_config
from ramvi_engine.engine_process import EngineProcess


def options(**overrides: object) -> Namespace:
    values = dict(
        model="test",
        draft=None,
        maxt=2048,
        no_cpu=False,
        resident=None,
        tokenizer="Qwen3-8B",
        reserve_gib=None,
        max_batch=None,
        plan_mode="auto",
        draft_policy="auto",
        prefill="auto",
        no_paging=False,
        load_attempts=4,
    )
    return Namespace(**(values | overrides))


def test_default_auto_and_explicit_legacy() -> None:
    config = _planning_config(options())
    assert config.planning is not None and config.planning.mode == "auto"
    assert _planning_config(options(plan_mode="legacy")).planning is None
    with pytest.raises(ValueError):
        _planning_config(options(plan_mode="manual"))
    with pytest.raises(ValueError):
        _planning_config(options(plan_mode="legacy", max_batch=2))


def test_manual_flags_reach_child_without_secret_argv() -> None:
    config = _planning_config(
        options(
            plan_mode="manual",
            resident=3,
            reserve_gib=0.5,
            max_batch=4,
            draft="draft",
            draft_policy="on",
            prefill="blocked",
            no_paging=True,
        )
    )
    child = EngineProcess(config)
    child._port = 12345
    args = child._arguments()
    for flag, value in [
        ("--resident", "3"),
        ("--plan-mode", "manual"),
        ("--reserve-gib", "0.5"),
        ("--max-batch", "4"),
        ("--draft-policy", "on"),
        ("--prefill", "blocked"),
    ]:
        assert args[args.index(flag) + 1] == value
    assert "--no-paging" in args
    assert "--api-key" not in args and child._key not in args


@pytest.mark.parametrize("reserve", [-1, float("nan"), float("inf")])
def test_invalid_reserve_rejected_before_loading(reserve: float) -> None:
    with pytest.raises(ValueError):
        _planning_config(options(reserve_gib=reserve))
