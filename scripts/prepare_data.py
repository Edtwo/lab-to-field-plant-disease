"""Build cached uint8 arrays (N,224,224,3) for PlantVillage (28 mapped classes) and PlantDoc.

Resize shorter side to 224 then center crop. Also computes a 64-bit average hash per image and
lists PlantDoc test images whose hash is within Hamming distance <=4 of any PlantDoc train image
(near-duplicates), so results can be reported on both the official and the de-duplicated test set.

Usage: python scripts/prepare_data.py --raw data/raw --out data/cache
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import yaml
from PIL import Image

SIZE = 224


def load(path):
    try:
        im = Image.open(path)
        im.draft("RGB", (SIZE * 2, SIZE * 2))
        im = im.convert("RGB")
        w, h = im.size
        s = SIZE / min(w, h)
        im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - SIZE) // 2, (h - SIZE) // 2
        im = im.crop((l, t, l + SIZE, t + SIZE))
        a = np.asarray(im, dtype=np.uint8)
        g = np.asarray(im.convert("L").resize((8, 8), Image.BILINEAR), dtype=np.float32)
        bits = (g > g.mean()).flatten()
        return a, int("".join("1" if b else "0" for b in bits), 2)
    except Exception as e:  # corrupt / unreadable file
        return None, repr(e)


def build(name, items, out):
    paths = [p for p, _ in items]
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        res = list(ex.map(load, paths, chunksize=64))
    keep = [i for i, (a, _) in enumerate(res) if a is not None]
    bad = [(paths[i], res[i][1]) for i in range(len(res)) if res[i][0] is None]
    X = np.lib.format.open_memmap(os.path.join(out, f"{name}_x.npy"), mode="w+", dtype=np.uint8,
                                  shape=(len(keep), SIZE, SIZE, 3))
    for j, i in enumerate(keep):
        X[j] = res[i][0]
    X.flush()
    y = np.array([items[i][1] for i in keep], dtype=np.int64)
    h = np.array([res[i][1] for i in keep], dtype=np.uint64)
    np.save(os.path.join(out, f"{name}_y.npy"), y)
    np.save(os.path.join(out, f"{name}_hash.npy"), h)
    with open(os.path.join(out, f"{name}_files.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(paths[i] for i in keep))
    print(f"{name}: {len(keep)} images, {len(bad)} unreadable", flush=True)
    return {"n": len(keep), "unreadable": bad, "per_class": np.bincount(y, minlength=28).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--classmap", default="configs/classmap.yaml")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cm = yaml.safe_load(open(a.classmap))["classes"]
    pv2i = {c["plantvillage"]: i for i, c in enumerate(cm)}
    pd2i = {c["plantdoc"]: i for i, c in enumerate(cm)}
    stats = {}

    pv_root = os.path.join(a.raw, "plantvillage")  # contains raw/color/...
    for split in ["train", "test"]:
        lines = open(os.path.join(a.raw, "plantvillage_hf", "splits", f"color_{split}.txt"), encoding="utf-8").read().split("\n")
        items = []
        for l in lines:
            l = l.strip()
            if not l:
                continue
            cls = l.split("/")[2]
            if cls in pv2i:
                items.append((os.path.join(pv_root, l), pv2i[cls]))
        stats[f"pv_{split}"] = build(f"pv_{split}", items, a.out)

    pd_root = [d for d in os.listdir(a.raw) if d.lower().startswith("plantdoc-dataset")]
    pd_root = os.path.join(a.raw, pd_root[0])
    for split in ["train", "test"]:
        items = []
        for cls in sorted(os.listdir(os.path.join(pd_root, split))):
            if cls not in pd2i:
                print("WARNING unmapped PlantDoc class:", cls, file=sys.stderr)
                continue
            d = os.path.join(pd_root, split, cls)
            items += [(os.path.join(d, f), pd2i[cls]) for f in sorted(os.listdir(d))]
        stats[f"pd_{split}"] = build(f"pd_{split}", items, a.out)

    # near-duplicate check PlantDoc test vs train (aHash Hamming <= 4)
    htr = np.load(os.path.join(a.out, "pd_train_hash.npy"))
    hte = np.load(os.path.join(a.out, "pd_test_hash.npy"))
    dup = []
    for i, h in enumerate(hte):
        d = np.array([bin(int(h) ^ int(t)).count("1") for t in htr])
        if d.min() <= 4:
            dup.append(i)
    np.save(os.path.join(a.out, "pd_test_dup_idx.npy"), np.array(dup, dtype=np.int64))
    stats["pd_test_near_duplicates_of_train"] = len(dup)
    print("PlantDoc test near-duplicates of train:", len(dup))
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
