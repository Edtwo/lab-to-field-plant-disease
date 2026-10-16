"""Download PlantVillage (HF mohanty/PlantVillage, CC BY-SA 3.0) and PlantDoc (GitHub, CC BY 4.0), extract only what is used.
Usage: python scripts/download_data.py --raw /path/to/raw
"""
import argparse, os, urllib.request, zipfile


def safe_extract(z, dest, members):
    for n in members:
        assert ".." not in n and not n.startswith(("/", "\\")), n
    z.extractall(dest, members=members)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); a = ap.parse_args()
    os.makedirs(a.raw, exist_ok=True)
    from huggingface_hub import snapshot_download
    hf = snapshot_download("mohanty/PlantVillage", repo_type="dataset", local_dir=os.path.join(a.raw, "plantvillage_hf"))
    z = zipfile.ZipFile(os.path.join(hf, "data.zip"))
    safe_extract(z, os.path.join(a.raw, "plantvillage"), [n for n in z.namelist() if n.startswith("raw/color/") and not n.endswith("/")])
    pz = os.path.join(a.raw, "plantdoc.zip")
    if not os.path.exists(pz):
        urllib.request.urlretrieve("https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/refs/heads/master", pz)
    z = zipfile.ZipFile(pz)
    safe_extract(z, a.raw, [n for n in z.namelist() if "/train/" in n or "/test/" in n or n.endswith("LICENSE.txt")])
    print("done")


if __name__ == "__main__":
    main()
