# Candidate project ideas (area: lab→field plant disease shift)

Budget: 5 GPU-hours on 1× RTX 5080 16 GB. Deadline Oct 16 2026, 23:59 ET.
Datasets (both verified 2026-10-15):
- **PlantVillage** color, 54,306 images, 38 classes, leaf-grouped 80/20 split. Source: HF `mohanty/PlantVillage` (≈2.2 GB for all configs; color only ≈0.8 GB). License: CC BY-SA 3.0.
- **PlantDoc**, 2,598 field images, 28 class folders with train/test split. Source: github.com/pratikkayal/PlantDoc-Dataset. License: CC BY 4.0.
- All 28 PlantDoc classes map one-to-one onto a PlantVillage class (`configs/classmap.yaml`).

## Idea 1 (recommended): "Lab-to-Field on a Budget": a seeded, edge-aware benchmark
- **Problem:** Reported PlantVillage→PlantDoc numbers range from 25% to 68%. Nobody reports variance, the class mappings are unpublished, and edge cost is ignored.
- **Fit:** reproducibility/benchmarking, domain shift, scarce field labels, edge deployment.
- **Method:** No new architecture. A fixed protocol (released class map, leaf-grouped splits, 3 seeds) that compares:
  - edge CNNs: MobileNetV3-L, EfficientNet-B0
  - a self-supervised ViT-S/14 (DINOv2), frozen linear probe vs. fine-tuning
  - CLIP ViT-B/32 zero-shot
  - each with 0 / 5 / all field labels per class
  - latency and memory measured on the GPU and on CPU (ONNX Runtime)
- **Baselines (code):** timm (github.com/huggingface/pytorch-image-models), open_clip (github.com/mlfoundations/open_clip), DINOv2 (github.com/facebookresearch/dinov2).
- **Metrics:** top-1 and macro-F1 on PlantDoc test (28 classes); in-domain PV accuracy; ECE; latency in ms per image, params, peak memory.
- **GPU-hours:** about 2.5 estimated, plus 1 h of margin.
- **Novelty:** modest. The value is the controlled protocol, the variance numbers and the edge axis.
- **Risk:** The PlantDoc test set is small (~236 images), so the error bars will be wide. We report them honestly and add a bootstrap CI.
- **Feasibility:** 5/5.

## Idea 2: Background matters. Segmentation-guided augmentation for lab→field
- Paste PlantVillage *segmented* leaves onto random field-like backgrounds, train MobileNetV3, and test on PlantDoc.
- About 2 GPU-h. More novel (3/5), but riskier: the gain may be zero, and there's no time to iterate. Feasibility 3/5.
- *Could be folded into Idea 1 as one ablation if time allows.*

## Idea 3: Few-shot field adaptation with frozen foundation features
- DINOv2 / CLIP features plus prototype or linear heads trained on k field shots, comparing source-trained heads and field-only heads.
- About 0.5 GPU-h. Mostly a subset of Idea 1. Feasibility 5/5, novelty 2/5.

**Recommendation:** Idea 1, which already covers Idea 3. Add Idea 2's background-paste augmentation as a single ablation only if the main matrix finishes by Oct 16 noon ET.
