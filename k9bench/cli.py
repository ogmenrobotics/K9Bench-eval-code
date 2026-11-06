"""Command line interface: ``k9bench <command>``.

Thin dispatcher over the runner scripts. It resolves which runner to invoke
from the model registry and builds the argument list; the runners themselves
are unchanged and can still be called directly.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .registry import all_specs, get


def repo_root() -> Path:
    """Repository root, i.e. the directory holding run_eval.sh.

    Works for an editable install (`pip install -e .`), which is how the suite
    is meant to be used. Override with K9BENCH_ROOT if the package was copied
    somewhere else.
    """
    env = os.environ.get("K9BENCH_ROOT")
    if env:
        return Path(env).resolve()
    root = Path(__file__).resolve().parents[1]
    if not (root / "run_eval.sh").exists():
        raise SystemExit(
            f"Cannot locate the K9Bench repository (looked in {root}).\n"
            "Install with `pip install -e .` from the clone, or set K9BENCH_ROOT.")
    return root


def cmd_models(args) -> int:
    specs = all_specs()
    if args.kind:
        specs = [s for s in specs if s.kind == args.kind]
    print(f"{'name':<24} {'kind':<6} {'family':<10} {'protocols':<18} notes")
    print("-" * 96)
    for s in specs:
        flag = "" if s.available else "[unavailable] "
        note = flag + s.note
        print(f"{s.name:<24} {s.kind:<6} {s.family:<10} "
              f"{','.join(s.protocols):<18} {note}")
    print(f"\n{len(specs)} models. Add one by editing k9bench/models/ "
          "(see docs/ADDING_A_MODEL.md)")
    return 0


def submit_slurm(spec, args, root: Path) -> int:
    """Hand off to run_eval.sh, which owns the SLURM path.

    The registry already knows the model's family, GPU count and whether it is a
    thinking variant, so the launcher's environment variables are derived rather
    than retyped.
    """
    if spec.kind == "api":
        raise SystemExit(
            f"--shards submits SLURM jobs for locally-loaded models; '{spec.name}' "
            "is API-served and is rate-limited rather than GPU-bound, so it runs "
            "in a single process. Drop --shards.")

    env = os.environ.copy()
    exported = {
        # sbatch runs a copy of the script from SLURM's spool dir, so the
        # launcher cannot infer the repo root from its own path.
        "K9BENCH_ROOT": str(root),
        "MODEL_NAME": spec.model_id,
        "MODEL_FAMILY": spec.family,
        "PROMPT_STYLE": args.protocol,
        "THINKING_MODE": "1" if spec.thinking else "0",
        "GPUS": str(spec.gpus),
        "MAX_FRAMES": str(args.max_frames),
        "VIDEO_DIR": args.video_dir,
        "NUM_SHARDS": str(args.shards),
    }
    # run_eval.sh defaults to a 10-sample smoke test. --shards means "the real
    # run", so default to the full dataset unless the user asked otherwise.
    exported["MAX_SAMPLES"] = str(args.max_samples if args.max_samples is not None else -1)
    if args.out_dir:
        exported["OUT_DIR"] = args.out_dir
    env.update(exported)

    # NUM_SHARDS=1 means "one SLURM job doing everything", so it must be
    # submitted with sbatch; running it with bash would put inference on the
    # login node. NUM_SHARDS>1 is the launcher, which runs locally and submits.
    launcher = ["sbatch"] if args.shards == 1 else ["bash"]
    cmd = launcher + [str(root / "run_eval.sh")]

    print("+ " + " ".join(f"{k}={v}" for k, v in exported.items()) + " " + " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=root, env=env)


def cmd_eval(args) -> int:
    spec = get(args.model)
    root = repo_root()

    if not spec.available:
        raise SystemExit(f"'{spec.name}' cannot be run by this suite yet. {spec.note}")
    if args.protocol not in spec.protocols:
        raise SystemExit(
            f"'{spec.name}' does not support --protocol {args.protocol} "
            f"(supported: {', '.join(spec.protocols)})")

    if args.shards is not None:
        return submit_slurm(spec, args, root)

    runner = root / spec.runner(args.protocol)

    if spec.kind == "api":
        config = args.config or spec.config
        if not config:
            raise SystemExit(f"'{spec.name}' needs --config")
        cmd = [sys.executable, str(runner), "--config", str(root / config)]
    else:
        out = args.output_file or (
            f"eval_results/{spec.name}_mf{args.max_frames}"
            f"{'_mcq' if args.protocol == 'mcq' else ''}/raw.json")
        cmd = [sys.executable, str(runner),
               "--model_name_or_path", spec.model_id,
               "--bf16", "--torch_dtype", "bfloat16", "--trust_remote_code",
               "--max_frames", str(args.max_frames),
               "--video_dir", args.video_dir,
               "--output_file", out]
        if args.auto_download:
            cmd.append("--auto_download")
        if spec.thinking:
            cmd.append("--thinking_mode")
        if args.max_samples is not None:
            cmd += ["--max_samples", str(args.max_samples)]
        cmd += spec.extra_args

    cmd += args.extra

    print("+ " + " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=root)


def cmd_score(args) -> int:
    root = repo_root()
    if args.protocol == "mcq":
        cmd = [sys.executable, str(root / "evaluate_results_mcq.py"),
               "--input_file", args.input_file, "--output_file", args.output_file]
        print("+ " + " ".join(cmd))
        return 0 if args.dry_run else subprocess.call(cmd, cwd=root)

    sim = args.sim_file or args.output_file.replace(".json", "_sim.json")
    steps = [
        [sys.executable, str(root / "evaluate_results.py"), "--mode", "calculate",
         "--input_file", args.input_file, "--output_file", sim],
        [sys.executable, str(root / "evaluate_results.py"), "--mode", "evaluate",
         "--input_file", sim, "--output_file", args.output_file,
         "--threshold", str(args.threshold)],
    ]
    for cmd in steps:
        print("+ " + " ".join(cmd))
        if not args.dry_run:
            rc = subprocess.call(cmd, cwd=root)
            if rc != 0:
                return rc
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="k9bench", description="K9Bench evaluation suite")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="list registered models")
    p.add_argument("--kind", choices=["local", "api"])
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("eval", help="run inference for one model")
    p.add_argument("--model", required=True)
    p.add_argument("--protocol", default="openended", choices=["openended", "mcq"])
    p.add_argument("--shards", type=int, default=None, metavar="N",
                   help="submit to SLURM via run_eval.sh instead of running here: "
                        "1 = a single job doing download+inference+scoring, "
                        ">1 = that many parallel inference shards, merged and scored. "
                        "Omit to run in this process (no SLURM needed).")
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--video_dir", default="k9bench_videos")
    p.add_argument("--out_dir", default=None,
                   help="with --shards: results root (default eval_results)")
    p.add_argument("--output_file", default=None)
    p.add_argument("--config", default=None, help="API models: override the config file")
    p.add_argument("--auto_download", action="store_true")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="print the command without running it")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="arguments passed through to the runner")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("score", help="score a raw result file")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--protocol", default="openended", choices=["openended", "mcq"])
    p.add_argument("--sim_file", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.set_defaults(func=cmd_score)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
