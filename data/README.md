# Data

Raw data and caches live in `data/` and are gitignored. `scripts/download_data.sh` fetches the raw data and `scripts/prepare_data.py` builds the caches.

| Dataset | Source | Size | License | Used for |
|---|---|---|---|---|
| PlantVillage (color) | https://huggingface.co/datasets/mohanty/PlantVillage (`data.zip`, `splits/color_{train,test}.txt`) | 54,305 images (43,596 train / 10,709 test, leaf-grouped); zip 2.2 GB for all 3 variants, of which we use `raw/color` | CC BY-SA 3.0 (dataset card `license: cc-by-sa-3.0`; original PlantVillage release, Hughes & Salathé 2015, arXiv:1511.08060) | Source (lab) domain |
| PlantDoc (classification) | https://github.com/pratikkayal/PlantDoc-Dataset (`train/`, `test/`) | 2,598 images (as reported by the authors), 28 folders | CC BY 4.0 (`LICENSE.txt` in repo) | Target (field) domain |

Attribution: PlantVillage, Mohanty, Hughes & Salathé, Front. Plant Sci. 2016, doi:10.3389/fpls.2016.01419. PlantDoc, Singh et al., CoDS-COMAD 2020, doi:10.1145/3371158.3371196.
Both licenses allow research use with attribution. CC BY-SA additionally requires that redistributed derivatives of the *data* carry the same license. We do not redistribute any images.

## Label space
`configs/classmap.yaml` holds the 28 PlantDoc classes, each mapped one-to-one to a PlantVillage class. All other PlantVillage classes (10) are dropped.

## Known caveats
- PlantDoc has a small test set and some near-duplicate images across its train and test splits (we check this, see `scripts/prepare_data.py`). Corrupt or unreadable files are logged and skipped.
- The PlantVillage leaf-grouped split comes from the dataset maintainers, so images of the same leaf never appear in both train and test.
