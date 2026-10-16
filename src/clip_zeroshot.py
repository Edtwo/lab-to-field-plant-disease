"""CLIP ViT-B/32 (OpenAI weights) zero-shot on the 28-class label space. Deterministic, no training.
Usage: python src/clip_zeroshot.py
"""
import os, sys
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, dump, metrics

TEMPLATES = ["a photo of a {}.", "a close-up photo of a {}.", "a field photo of a {}."]
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def main():
    import open_clip
    torch.manual_seed(0)
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model = model.cuda().eval()
    cm = yaml.safe_load(open("configs/classmap.yaml"))["classes"]
    names = [f"healthy {c['crop']} leaf" if c["condition"] == "healthy" else f"{c['crop']} leaf with {c['condition']}" for c in cm]
    with torch.no_grad():
        W = []
        for n in names:
            e = model.encode_text(tok([t.format(n) for t in TEMPLATES]).cuda()).float()
            e = e / e.norm(dim=-1, keepdim=True)
            W.append(e.mean(0) / e.mean(0).norm())
        W = torch.stack(W)
    out = "results/runs/clip_b32_zeroshot"
    res = {"prompts": [TEMPLATES[0].format(n) for n in names]}
    for sp in ["pv_test", "pd_test"]:
        x = np.load(os.path.join(CACHE, f"{sp}_x.npy"), mmap_mode="r")
        y = np.load(os.path.join(CACHE, f"{sp}_y.npy"))
        lg = []
        with torch.no_grad():
            for i in range(0, len(y), 256):
                b = torch.from_numpy(np.ascontiguousarray(x[i:i + 256])).permute(0, 3, 1, 2).float() / 255
                b = ((b - CLIP_MEAN) / CLIP_STD).cuda()
                f = model.encode_image(b).float(); f = f / f.norm(dim=-1, keepdim=True)
                lg.append((100 * f @ W.T).cpu())
        lg = torch.cat(lg).numpy()
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, f"logits_{sp}.npy"), lg)
        res[sp] = metrics(lg, y)
        if sp == "pd_test":
            dup = set(np.load(os.path.join(CACHE, "pd_test_dup_idx.npy")).tolist())
            keep = np.array([i for i in range(len(y)) if i not in dup])
            res["pd_test_dedup"] = metrics(lg[keep], y[keep])
    dump(os.path.join(out, "config.json"), {"model": "open_clip ViT-B-32 openai", "templates": TEMPLATES,
                                            "open_clip": open_clip.__version__, "seed": 0})
    dump(os.path.join(out, "metrics.json"), res)
    print(res["pv_test"], res["pd_test"])


if __name__ == "__main__":
    main()
