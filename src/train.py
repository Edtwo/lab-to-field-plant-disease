"""Train one configuration and evaluate it on PV-test, PD-test and de-duplicated PD-test.

Examples:
  python src/train.py --model mnv3 --data pv --seed 0
  python src/train.py --model mnv3 --data pd_k5 --init results/runs/A_mnv3_pv_s0/model.pt --seed 0
  python src/train.py --model mnv3 --data pd_all --seed 0          # field-only reference

--data: pv | pd_k5 | pd_all.  --init: checkpoint to start from (else ImageNet/DINOv2 weights).
Writes results/runs/<name>/{config.json, metrics.json, log.txt, logits_*.npy, model.pt}.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from common import (ArrDS, LR, MODELS, NUM_CLASSES, build_model, dump, kshot_indices, metrics, seed_all, CACHE)

EPOCHS = {"pv": 5, "pd_k5": 30, "pd_all": 15}


def evaluate(model, split, idx=None, bs=256):
    dl = DataLoader(ArrDS(split, idx), batch_size=bs, num_workers=4, persistent_workers=False)
    model.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in dl:
            out.append(model(x.cuda(non_blocking=True).to(memory_format=torch.channels_last)).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--data", required=True, choices=list(EPOCHS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="override the default per-model learning rate")
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true", help="few steps only, for the smoke test")
    a = ap.parse_args()
    seed_all(a.seed)
    torch.backends.cudnn.benchmark = True
    epochs = a.epochs or EPOCHS[a.data]
    tag = "init" if a.init else "pre"
    name = a.name or f"{a.model}_{a.data}_{tag}_s{a.seed}"
    out = os.path.join("results", "runs" if not a.smoke else "smoke", name)
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.txt"), "w")
    def P(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); log.write(msg + "\n"); log.flush()

    if a.data == "pv":
        split, idx = "pv_train", None
    else:
        split = "pd_train"
        y = np.load(os.path.join(CACHE, "pd_train_y.npy"))
        idx = kshot_indices(y, 5, a.seed) if a.data == "pd_k5" else None
    ds = ArrDS(split, idx, train=True)
    steps_per_epoch = max(1, len(ds) // a.bs)
    dl = DataLoader(ds, batch_size=min(a.bs, len(ds)), shuffle=True, drop_last=True, num_workers=8,
                    persistent_workers=True, pin_memory=True,
                    generator=torch.Generator().manual_seed(a.seed))

    model = build_model(a.model, pretrained=a.init is None)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
    model = model.cuda().to(memory_format=torch.channels_last)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = a.lr or LR[a.model]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    total = epochs * steps_per_epoch if not a.smoke else 5
    warm = min(100, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    cfg = dict(vars(a), epochs=epochs, lr=lr, wd=0.05, train_split=split, n_train=len(ds), steps=total,
               timm_model=MODELS[a.model][0], kind=MODELS[a.model][1], torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0))
    dump(os.path.join(out, "config.json"), cfg)
    P("config", json.dumps(cfg))

    t0 = time.time(); step = 0
    model.train()
    if MODELS[a.model][1] == "lp":
        model.eval()  # frozen backbone: keep norm layers in eval mode
    while step < total:
        for x, y in dl:
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last); y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            if step % 50 == 0 or step == total:
                P(f"step {step}/{total} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} t {time.time()-t0:.0f}s")
            if step >= total:
                break
    train_s = time.time() - t0

    res = {"train_seconds": train_s}
    pd_test_y = np.load(os.path.join(CACHE, "pd_test_y.npy"))
    dup = set(np.load(os.path.join(CACHE, "pd_test_dup_idx.npy")).tolist())
    keep = np.array([i for i in range(len(pd_test_y)) if i not in dup])
    for sp in ["pv_test", "pd_test"]:
        y = np.load(os.path.join(CACHE, f"{sp}_y.npy"))
        idx = None
        if a.smoke:
            idx = np.arange(min(256, len(y)))
        lg = evaluate(model, sp, idx)
        yy = y if idx is None else y[idx]
        np.save(os.path.join(out, f"logits_{sp}.npy"), lg)
        res[sp] = metrics(lg, yy)
        if sp == "pd_test" and not a.smoke:
            res["pd_test_dedup"] = metrics(lg[keep], yy[keep])
    res["total_seconds"] = time.time() - t0
    dump(os.path.join(out, "metrics.json"), res)
    P("metrics", json.dumps(res))
    if not a.smoke:
        torch.save(model.state_dict(), os.path.join(out, "model.pt"))


if __name__ == "__main__":
    main()
