"""Проверка2.7: fp32 oracle, точные IDs/logits и исходный проблемный 2+2 prompt.

Запускается явно в выбранной среде; использует локальные веса без скачивания.
"""

import argparse
import gc
import hashlib
import json
from importlib import import_module
from pathlib import Path
from typing import Any

CASES = [
    ("original", "Reply with just the number: 2 + 2 =", "4"),
    ("echo_control", "Reply with exactly this digit: 4", "4"),
    ("fresh", "Reply with just the number: 137 * 29 =", "3973"),
    ("outside", "Reply with just the number: 100003 + 29 =", "100032"),
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(reference: Any) -> dict[str, Any]:
    if reference.gguf:
        path = Path(reference.g.path)
        # Big quantized files are identified by full content, not a filename claim.
        return {
            "format": "gguf",
            "architecture": reference.arch,
            "weights_sha256": _sha(path),
            "weights_bytes": path.stat().st_size,
            "metadata": {
                key: reference.g.meta.get(key)
                for key in ("general.name", "general.architecture", "tokenizer.chat_template")
            },
        }
    folder = Path(reference.ck.dir)
    return {
        "format": "safetensors",
        "architecture": reference.arch,
        "config": reference.ck.cfg,
        "files": {
            p.name: {"sha256": _sha(p), "bytes": p.stat().st_size}
            for p in sorted(folder.glob("*.safetensors"))
        },
    }


def transformers_logits(model: str, ids: list[int], device: str) -> tuple[Any, dict[str, Any]]:
    torch = import_module("torch")
    from ramvi_engine.formats.model import find_model

    AutoModelForCausalLM = import_module("transformers").AutoModelForCausalLM

    reference = (
        AutoModelForCausalLM.from_pretrained(
            find_model(model),  # type: ignore[no-untyped-call]
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    try:
        with torch.no_grad():
            result = (
                reference(torch.tensor([ids], device=device), use_cache=False)
                .logits[0, -1]
                .float()
                .cpu()
            )
        return result, {"class": type(reference).__name__, "dtype": "float32", "attention": "eager"}
    finally:
        del reference
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="Qwen3-8B")
    parser.add_argument("--resident", type=int)
    parser.add_argument(
        "--thinking", action="store_true", help="use original tokenizer thinking-mode template"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--legacy-template",
        action="store_true",
        help="explicitly reproduce tokenizer template before model policy",
    )
    parser.add_argument("--case", choices=[name for name, _, _ in CASES], action="append")
    parser.add_argument("--steps", type=int, default=2, help="teacher-forced positions per case")
    parser.add_argument(
        "--atol", type=float, default=0.02, help="configured tolerance, not measured error"
    )
    parser.add_argument("--rtol", type=float, default=0.002)
    parser.add_argument(
        "--trace", action="store_true", help="per-layer last hidden diagnosis on first mismatch"
    )
    parser.add_argument(
        "--transformers", action="store_true", help="independent HF eager-fp32 implementation"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 16:
        parser.error("steps от1до16")
    torch = import_module("torch")
    from ramvi_engine.formats.tokenizer_check import fingerprint
    from ramvi_engine.model_loader import ModelConfig, create_session
    from ramvi_engine.validation.comparison import compare_logits, compare_tokens
    from ramvi_engine.validation.native import native_logits, native_trace
    from ramvi_engine.validation.oracle import TorchReference

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    session = create_session(
        ModelConfig(args.model, maxt=512, resident=args.resident, tokenizer=args.tokenizer)
    )
    reference = TorchReference(args.model, device=args.device)
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": False,
        "passed": False,
        "weights": identity(reference),
        "native_kv_rounding": reference.gguf,
        "thinking": args.thinking,
        "torch": torch.__version__,
        "cases": [],
    }

    def save() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        session.load()
        tokenizer = session.tokenizer
        if args.legacy_template:
            original = getattr(tokenizer, "ramvi_legacy_chat_template", None)
            if not isinstance(original, str):
                raise ValueError("legacy template is not available")
            session._get_bundle().tokenizer.chat_template = original
        report["legacy_template"] = args.legacy_template
        report["thinking_required"] = bool(getattr(session, "thinking_required", False))
        report["template_source"] = getattr(session, "chat_template_source", "tokenizer")
        report["native_plan"] = list(session.place)
        report["context_tokens"] = session.maxt
        report["native_device"] = str(session.pos.device)
        report["tokenizer_sha256"] = fingerprint(session._get_bundle().tokenizer)  # type: ignore[no-untyped-call]
        for label, question, expected_answer in CASES:
            if args.case and label not in args.case:
                continue
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.thinking,
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            native_tokens: list[int] = []
            ref_tokens: list[int] = []
            positions: list[dict[str, Any]] = []
            report["active_case"] = {"name": label, "input_ids": ids, "positions": positions}
            save()
            tf_row = None
            if args.transformers:
                if reference.gguf:
                    raise ValueError("Transformers oracle requires HF safetensors")
                tf_row, tf_metadata = transformers_logits(args.model, ids, args.device)
            proposed: list[int] | None = None
            bulk_logits = None
            if reference.gguf:
                session._get_bundle().engine.ctx = []
                proposed = list(session.generate(ids, args.steps))
                if not proposed:
                    raise RuntimeError("native returned no prefix for oracle")
                # One causal forward checks every proposed prefix. If all argmax
                # checks pass, induction proves the same greedy reference sequence.
                bulk_logits = reference.logits(
                    ids + proposed[:-1],
                    positions=list(range(len(ids) - 1, len(ids) + len(proposed) - 1)),
                ).cpu()
                report["reference_conditioning"] = "causal_native_prefix_teacher_forcing"
            for step in range(len(proposed) if proposed is not None else args.steps):
                prefix = ids + (proposed[:step] if proposed is not None else ref_tokens)
                expected = (
                    bulk_logits[step]
                    if bulk_logits is not None
                    else reference.logits(prefix)[0].cpu()
                )
                actual, selected = native_logits(session._get_bundle().engine, reference, prefix)
                row = compare_logits(
                    actual.tolist(), expected.tolist(), atol=args.atol, rtol=args.rtol
                )
                row["native_selected"] = selected
                row["native_selection_matches_full_projection"] = selected == row["actual_argmax"]
                row["passed"] &= row["native_selection_matches_full_projection"]
                row["step"] = step
                if proposed is not None:
                    row["proposed_token"] = proposed[step]
                    row["passed"] &= proposed[step] == row["expected_argmax"]
                row["top_reference"] = [
                    {
                        "token_id": int(i),
                        "logit": float(expected[i]),
                        "text": tokenizer.decode([int(i)]),
                    }
                    for i in torch.topk(expected, 5).indices.tolist()
                ]
                if step == 0 and tf_row is not None:
                    row["transformers"] = {
                        "metadata": tf_metadata,
                        "vs_torch_oracle": compare_logits(
                            expected.tolist(), tf_row.tolist(), atol=args.atol, rtol=args.rtol
                        ),
                        "vs_native": compare_logits(
                            actual.tolist(), tf_row.tolist(), atol=args.atol, rtol=args.rtol
                        ),
                    }
                if step == 0 and args.trace and not row["passed"]:
                    _, _, native_history = native_trace(
                        session._get_bundle().engine, reference, prefix
                    )
                    _, expected_history = reference.hidden(prefix, trace=True)
                    row["layer_trace"] = [
                        {
                            "layer": i,
                            "max_abs_error": float((native_history[i] - value).abs().max()),
                            "rms_error": float((native_history[i] - value).square().mean().sqrt()),
                        }
                        for i, value in enumerate(expected_history)
                        if i in native_history
                    ]
                    row["trace_layers_captured"] = len(native_history)
                if step == 0 and tf_row is not None:
                    row["passed"] &= row["transformers"]["vs_torch_oracle"]["passed"]
                    row["passed"] &= row["transformers"]["vs_native"]["passed"]
                positions.append(row)
                save()
                native_tokens.append(selected)
                ref_tokens.append(row["expected_argmax"])
                if ref_tokens[-1] in session.ck.eos:
                    break
            session._get_bundle().engine.ctx = []
            free_native_tokens = list(session.generate(ids, args.steps))
            case = {
                "name": label,
                "question": question,
                "expected_math_answer": expected_answer,
                "rendered_prompt": rendered,
                "input_ids": ids,
                "token_count": len(ids),
                "input_tokens": tokenizer.convert_ids_to_tokens(ids),
                "native_teacher_forced_tokens": native_tokens,
                "native_prefix_tokens": proposed,
                "native_tokens": free_native_tokens,
                "reference_tokens": ref_tokens,
                "native_text": tokenizer.decode(free_native_tokens, skip_special_tokens=True),
                "reference_text": tokenizer.decode(ref_tokens, skip_special_tokens=True),
                "token_comparison": compare_tokens(free_native_tokens, ref_tokens),
                "positions": positions,
            }
            report["cases"].append(case)
            report.pop("active_case", None)
            save()
            print(
                json.dumps(
                    {
                        "case": label,
                        "native": native_tokens,
                        "reference": ref_tokens,
                        "max_abs_errors": [r["max_abs_error"] for r in positions],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        report["complete"] = True
        report["passed"] = all(
            c["token_comparison"]["passed"] and all(p["passed"] for p in c["positions"])
            for c in report["cases"]
        )
        save()
    finally:
        reference.close()
        session.close()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
