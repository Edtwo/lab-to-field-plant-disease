# Plan: Idea 1, "Lab-to-Field on a Budget"

## Claims (falsifiable; each one can come out false and we report it either way)
- **C1 (gap):** Every model trained only on PlantVillage loses at least 30 points of top-1 accuracy going from the PV test set to the PlantDoc test set. The gap is measured across 3 seeds. *False if* any model's mean gap is under 30 points.
- **C2 (pretraining > size):** A frozen DINOv2 ViT-S linear probe trained on PV beats fine-tuned MobileNetV3/EfficientNet-B0 on PlantDoc (0 field labels). *False if* the DINOv2 mean is within one combined std of, or below, the best CNN.
- **C3 (few labels):** Adding 5 field images per class (≈140 images) recovers at least half of the gap between the PV-only model and the model trained with all PlantDoc training labels. *False if* the recovery is under 50% for the majority of models.
- **C4 (edge):** At least one configuration reaches ≥X% PlantDoc accuracy (X = best-model mean − 5) at <15 ms per image on CPU (ONNX Runtime, batch 1, 224 px). *False if* no model meets both conditions.
- **C5 (variance):** Seed-to-seed std on PlantDoc is large enough (≥2 points) that single-run comparisons from prior work cannot be trusted. *False if* the std is under 2 points for all models.

## Experiment matrix
| Block | Models | Train data | Seeds | Runs |
|---|---|---|---|---|
| A. Source-only | MNv3-L, EffNet-B0 (fine-tune), DINOv2-S (linear probe), DINOv2-S (fine-tune) | PV train, the 28 mapped classes | 3 | 12 |
| B. Zero-shot | CLIP ViT-B/32 (prompts "a photo of a {crop} leaf with {disease}") | none | 1 (deterministic) | 1 |
| C. Few-shot field | the 4 models in A, initialized from the PV checkpoint, plus k ∈ {5, all} PlantDoc train images per class | PV + PlantDoc | 3 | 24 |
| D. Field-only ref | the 4 models in A, ImageNet/DINOv2 init, PlantDoc train only | PlantDoc | 3 | 12 |
| E. Edge profile | all 4 models | n/a | n/a | latency/params/memory, GPU and CPU-ONNX |
| F. Ablation (if time) | MNv3-L with background-paste augmentation | PV | 3 | 3 |

Training: 224 px, AdamW, cosine schedule, 5 epochs on PV (about 30k images in the mapped subset) and 30 epochs on few-shot sets, AMP. Images are pre-resized once and cached so the CPU loader doesn't become the bottleneck.
Estimated GPU time: A ≈ 1.2 h, C+D ≈ 0.6 h, B/E ≈ 0.1 h. **Total ≈ 2 h, against a 5 h cap.** Per-run limit: 20 min. A smoke test (1 batch per stage, ≤5 min) runs before everything else.

**Baseline sanity check:** in-domain PV accuracy should be ≥97% (Mohanty et al. 2016 report 99.35% on a random split; the leaf-grouped split is harder). If MobileNetV3 lands below 95% in-domain, stop and debug before building on it.

**Kill / reframe criteria:** If the test-set CIs are so wide that C2/C3 can't be decided, the paper becomes a "variance and protocol" paper centered on C1 and C5. If the in-domain sanity check fails twice, stop and report.

## Schedule (ET)
- Oct 15, evening: environment, data download and cache, smoke test, block A launched in the background.
- Oct 15, night: blocks B–E (unattended, queued).
- Oct 16, morning: `results/summary.md` auto-generated; honest check against the claims; go/no-go on the framing.
- Oct 16, midday: paper in the WACV template (8 pages including refs); figures and tables generated from `results/`.
- Oct 16, afternoon: internal review from three reviewer perspectives, fixes, final compile.
- Oct 16, evening: add author block, final PDF, submit on OpenReview before 23:59 ET.
