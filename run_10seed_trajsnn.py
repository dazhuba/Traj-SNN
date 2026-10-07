"""Resume-safe 10-seed Traj-SNN, ablation, and causal-logic evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-icassp" / "Scripts" / "python.exe"
STUDY = ROOT / "external_baselines" / "icassp_study"
DATA = STUDY / "data"
RESULTS = STUDY / "results"
PHYSICS = RESULTS / "modern_physics_rules"
FULL = RESULTS / "phy_ssm_10seed"
NEURAL = STUDY / "ablation_multiscale_moderntcn"
ABLATION = STUDY / "ablation"
CAUSAL = STUDY / "causal_logic_10seed"


def run_command(command: list[str], metric: Path, force: bool) -> str:
    if metric.is_file() and not force:
        return "reused"
    metric.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    with (metric.parent / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, check=True,
        )
    if not metric.is_file():
        raise RuntimeError(f"missing expected output: {metric}")
    return "completed"


def run_phase(name: str, jobs, worker, workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, *job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            status = future.result()
            print(f"[{name} {index}/{len(jobs)}] {status}: {job}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--windows", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    def physics(window: int, seed: int) -> str:
        metric = PHYSICS / f"{window}s" / f"seed_{seed}" / "metrics.json"
        command = [
            str(PYTHON), "-m", "research_intent.train_modern_physics_residual",
            "--data-dir", str(DATA), "--modern-root", str(RESULTS / "moderntcn"),
            "--output-root", str(PHYSICS), "--window", str(window),
            "--seed", str(seed), "--cpu-threads", "1",
        ]
        return run_command(command, metric, args.force)

    all_jobs = [(window, seed) for window in args.windows for seed in args.seeds]
    run_phase("physics", all_jobs, physics, args.workers)

    def multires(window: int, seed: int, mode: str, output: Path) -> str:
        metric = output / f"{window}s" / f"seed_{seed}" / "metrics.json"
        command = [
            str(PYTHON), "-m", "research_intent.evaluate_modern_inwindow_multiscale",
            "--data-dir", str(DATA), "--modern-root", str(RESULTS / "moderntcn"),
            "--prior-root", str(PHYSICS), "--output-root", str(output),
            "--window", str(window), "--seed", str(seed), "--mode", mode,
            "--cpu-threads", "1",
        ]
        return run_command(command, metric, args.force)

    run_phase(
        "full-multires", all_jobs,
        lambda window, seed: multires(window, seed, "full", FULL), args.workers,
    )
    formal_jobs = [
        (window, seed) for window in args.windows if window in (8, 16, 32)
        for seed in args.seeds
    ]
    run_phase(
        "neural-multires", formal_jobs,
        lambda window, seed: multires(window, seed, "neural", NEURAL), args.workers,
    )

    def ablation(window: int, seed: int) -> str:
        metric = ABLATION / f"{window}s" / f"seed_{seed}" / "ablation_metrics.json"
        command = [
            str(PYTHON), "-m", "research_intent.evaluate_modern_residual_ablation",
            "--data-dir", str(DATA), "--modern-root", str(RESULTS / "moderntcn"),
            "--residual-root", str(PHYSICS), "--output-root", str(ABLATION),
            "--window", str(window), "--seed", str(seed), "--cpu-threads", "1",
        ]
        return run_command(command, metric, args.force)

    run_phase("residual-ablation", all_jobs, ablation, args.workers)

    def causal(window: int, seed: int) -> str:
        metric = CAUSAL / f"{window}s" / f"seed_{seed}" / "metrics.json"
        command = [
            str(PYTHON), "-m", "research_intent.evaluate_modern_causal_memory",
            "--data-dir", str(DATA), "--modern-root", str(RESULTS / "moderntcn"),
            "--prior-root", str(PHYSICS), "--multires-root", str(FULL),
            "--output-root", str(CAUSAL),
            "--window", str(window), "--seed", str(seed), "--profile", "robust",
            "--minimum-val-improvement", "0.003", "--cpu-threads", "1",
        ]
        return run_command(command, metric, args.force)

    run_phase("causal-logic", formal_jobs, causal, args.workers)


if __name__ == "__main__":
    main()
