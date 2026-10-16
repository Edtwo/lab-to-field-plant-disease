"""Shared data, model and metric utilities."""
import json, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

CACHE = os.environ.get("HV_CACHE", "data/cache")
NUM_CLASSES = 28
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

MODELS = {
    # key: (timm name, kind)
    "mnv3": ("mobilenetv3_large_100.ra_in1k", "ft"),
    "effb0": ("efficientnet_b0.ra_in1k", "ft"),
    "dinov2s_ft": ("vit_small_patch14_dinov2.lvd142m", "ft"),
    "dinov2s_lp": ("vit_small_patch14_dinov2.lvd142m", "lp"),
}
LR = {"mnv3": 1e-3, "effb0": 1e-3, "dinov2s_ft": 5e-5, "dinov2s_lp": 1e-3}


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_split(name):
    x = np.load(os.path.join(CACHE, f"{name}_x.npy"), mmap_mode="r")
    y = np.load(os.path.join(CACHE, f"{name}_y.npy"))
    return x, y


def kshot_indices(y, k, seed):
    """k images per class (all if fewer), chosen with a seed-specific RNG."""
    rng = np.random.RandomState(1000 + seed)
    idx = []
    for c in range(NUM_CLASSES):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        idx += ci[:k].tolist()
    return np.array(sorted(idx))


class ArrDS(Dataset):
    def __init__(self, split, idx=None, train=False):
        self.split, self.train = split, train
        self.x = None
        self.y = np.load(os.path.join(CACHE, f"{split}_y.npy"))
        self.idx = np.arange(len(self.y)) if idx is None else np.asarray(idx)
        aug = [T.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True), T.RandomHorizontalFlip(),
               T.ColorJitter(0.3, 0.3, 0.3, 0.05)] if train else []
        self.tf = T.Compose(aug + [T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.x is None:  # open lazily inside each worker
            self.x = np.load(os.path.join(CACHE, f"{self.split}_x.npy"), mmap_mode="r")
        j = self.idx[i]
        img = torch.from_numpy(np.ascontiguousarray(self.x[j])).permute(2, 0, 1)
        return self.tf(img), int(self.y[j])


def build_model(key, pretrained=True):
    import timm
    name, kind = MODELS[key]
    kw = {"img_size": 224} if "dinov2" in name else {}
    m = timm.create_model(name, pretrained=pretrained, num_classes=NUM_CLASSES, **kw)
    if kind == "lp":
        for n, p in m.named_parameters():
            p.requires_grad = n.startswith("head")
    return m


def ece(probs, y, bins=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def metrics(logits, y):
    from sklearn.metrics import f1_score
    probs = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), 1).numpy()
    pred = probs.argmax(1)
    return {"top1": float((pred == y).mean() * 100),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=np.unique(y).tolist(), zero_division=0) * 100),
            "ece": ece(probs, y) * 100, "n": int(len(y))}


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
