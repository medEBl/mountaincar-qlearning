"""Run several configurations x several seeds in parallel and compare them properly.

Outputs (in results/multiseed/):
  - <config>_s<seed>.csv     reward per episode for each run
  - summary_runs.csv         one line per run
  - summary.csv              mean and std over seeds, per configuration
  - compare_curves.png       learning curves: mean over seeds +/- std band
"""
import contextlib
import io
import os
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train import evaluate, train

EPISODES = 20000
SEEDS = [0, 1, 2]
COMMON = dict(gamma=0.99, eps_start=0.1, eps_end=0.01, decay_frac=0.5)
CONFIGS = {
    "bins12":            dict(bins=12, alpha=0.1, alpha_end=None),
    "bins20":            dict(bins=20, alpha=0.1, alpha_end=None),
    "bins40":            dict(bins=40, alpha=0.1, alpha_end=None),
    "bins20_alphadecay": dict(bins=20, alpha=0.1, alpha_end=0.01),
}
OUT = os.path.join("results", "multiseed")


def run_one(job):
    """Train + evaluate one (config, seed). Training prints are silenced."""
    name, seed = job
    cfg = {**COMMON, **CONFIGS[name]}
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        env, disc, agent, returns = train(cfg["bins"], EPISODES, cfg["alpha"], cfg["gamma"],
                                          seed, cfg["eps_start"], cfg["eps_end"],
                                          cfg["decay_frac"], cfg["alpha_end"])
        g_mean, g_std = evaluate(env, disc, agent, seed=seed)
    pd.DataFrame({"episode": range(len(returns)), "reward": returns}).to_csv(
        os.path.join(OUT, f"{name}_s{seed}.csv"), index=False)
    first_goal = next((i for i, r in enumerate(returns) if r > -200), None)
    print(f"done: {name:18s} seed={seed} greedy={g_mean:7.1f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return dict(config=name, seed=seed, first_goal=first_goal,
                train_last100=float(np.mean(returns[-100:])),
                greedy_mean=float(g_mean), greedy_std=float(g_std),
                minutes=(time.time() - t0) / 60, returns=returns)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(name, s) for name in CONFIGS for s in SEEDS]
    print(f"Running {len(jobs)} runs ({len(CONFIGS)} configs x {len(SEEDS)} seeds), "
          f"{EPISODES} episodes each...")
    with Pool(min(len(jobs), os.cpu_count())) as pool:
        results = pool.map(run_one, jobs)

    # tables
    runs = pd.DataFrame([{k: v for k, v in r.items() if k != "returns"} for r in results])
    runs.to_csv(os.path.join(OUT, "summary_runs.csv"), index=False)
    summary = runs.groupby("config")[["first_goal", "train_last100", "greedy_mean"]] \
                  .agg(["mean", "std"]).round(1)
    summary.to_csv(os.path.join(OUT, "summary.csv"))
    print("\n=== Summary (mean and std over seeds) ===")
    print(summary.to_string())

    # learning curves: moving average per run, then mean +/- std over seeds
    plt.figure(figsize=(10, 5))
    for name in CONFIGS:
        curves = np.array([pd.Series(r["returns"]).rolling(100).mean().to_numpy()
                           for r in results if r["config"] == name])
        m, s = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
        x = np.arange(len(m))
        plt.plot(x, m, label=name)
        plt.fill_between(x, m - s, m + s, alpha=0.2)
    plt.axhline(-110, color="red", linestyle="--", label="Solved threshold (-110)")
    plt.xlabel("Episode"); plt.ylabel("Cumulative reward (moving avg 100)")
    plt.title(f"Tabular Q-learning on MountainCar-v0 — mean ± std over {len(SEEDS)} seeds")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "compare_curves.png"), dpi=150)
    print(f"\nSaved {os.path.join(OUT, 'compare_curves.png')}")