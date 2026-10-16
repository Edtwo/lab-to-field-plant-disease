"""Run the full experiment matrix sequentially. Skips runs that already have metrics.json.
A failing run is retried once; if it fails a second time the whole queue stops (no blind retrying).
Usage: python scripts/run_matrix.py [--blocks A,B,C,D,E]
"""
import argparse, os, subprocess, sys, time

PY = sys.executable
MODELS = ["mnv3", "effb0", "dinov2s_lp", "dinov2s_ft"]
SEEDS = [0, 1, 2]


def jobs(blocks):
    J = []
    if "A" in blocks:  # source-only
        for s in SEEDS:
            for m in MODELS:
                J.append((f"{m}_pv_pre_s{s}", [PY, "src/train.py", "--model", m, "--data", "pv", "--seed", str(s)]))
    if "B" in blocks:
        J.append(("clip_b32_zeroshot", [PY, "src/clip_zeroshot.py"]))
    if "C" in blocks:  # PV checkpoint -> field labels
        for s in SEEDS:
            for m in MODELS:
                for d in ["pd_k5", "pd_all"]:
                    J.append((f"{m}_{d}_init_s{s}", [PY, "src/train.py", "--model", m, "--data", d, "--seed", str(s),
                                                     "--init", f"results/runs/{m}_pv_pre_s{s}/model.pt"]))
    if "D" in blocks:  # field-only references
        for s in SEEDS:
            for m in MODELS:
                for d in ["pd_k5", "pd_all"]:
                    J.append((f"{m}_{d}_pre_s{s}", [PY, "src/train.py", "--model", m, "--data", d, "--seed", str(s)]))
    if "S" in blocks:  # post-hoc sensitivity (added after self-review): DINOv2 fine-tune learning rate, PV only
        for s in SEEDS:
            for lr in ["1e-5", "2e-4"]:
                n = f"dinov2s_ft_pv_pre_lr{lr}_s{s}"
                J.append((n, [PY, "src/train.py", "--model", "dinov2s_ft", "--data", "pv", "--seed", str(s), "--lr", lr, "--name", n]))
    if "E" in blocks:
        J.append(("edge", [PY, "src/profile_edge.py"]))
    return J


def done(name):
    if name == "edge":
        return os.path.exists("results/edge/edge_profile.json")
    return os.path.exists(f"results/runs/{name}/metrics.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--blocks", default="A,B,C,D,E")
    a = ap.parse_args()
    os.makedirs("results/logs", exist_ok=True)
    q = jobs(a.blocks.split(","))
    print(f"{len(q)} jobs", flush=True)
    for i, (name, cmd) in enumerate(q):
        if done(name):
            print(f"[{i+1}/{len(q)}] skip {name}", flush=True); continue
        for attempt in (1, 2):
            t = time.time()
            with open(f"results/logs/{name}.log", "w") as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
            print(f"[{i+1}/{len(q)}] {name} attempt {attempt} rc={rc} {time.time()-t:.0f}s", flush=True)
            if rc == 0:
                break
            if attempt == 2:
                print(f"STOP: {name} failed twice; see results/logs/{name}.log", flush=True)
                sys.exit(1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
