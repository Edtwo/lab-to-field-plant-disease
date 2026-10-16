# Landscape: lab-to-field shift in plant disease recognition (2023–2026)

Every entry was checked against its arXiv abstract page, the arXiv API, a journal page, or CVF/ML Anthology on 2026-10-15. Entries carry their URL; BibTeX is in `refs/verified.bib`.
"Gap" lists what the abstract or page does **not** show. It is not a claim about what the full paper lacks.

## Recent work (2023–2026)

| # | Paper | Contribution (one line) | Open gap relevant to us |
|---|---|---|---|
| 1 | Xu et al., *Plant Disease Recognition Datasets in the Age of Deep Learning* (arXiv 2312.07905, 2023) | Argues benchmark accuracy is an "illusion" because datasets cater to DL, not fields; proposes a dataset taxonomy. | Position paper; no controlled lab→field measurement. |
| 2 | Xu et al., *Embrace Limited and Imperfect Training Datasets* (arXiv 2305.11533, 2023; Front. Plant Sci.) | Taxonomy of challenges from small/imperfect datasets. | No benchmark numbers. |
| 3 | Wang et al., *Self-supervised transformer-based pre-training with General Plant Infection dataset* (arXiv 2407.14911, 2024) | Large pest/disease dataset + contrastive+MIM pretraining. | Abstract does not report an OOD (lab→field) evaluation. |
| 4 | Wei et al., *Benchmarking In-the-wild Multimodal Disease Recognition (PlantWild)* (arXiv 2408.03120, 2024) | 18k+ in-the-wild images, 89 classes; multimodal baseline for supervised/few/zero-shot. | Not a lab→field transfer study; no edge cost. |
| 5 | Wei et al., *Snap and Diagnose* (arXiv 2408.14723, 2024; MMAsia'24) | CLIP-based multimodal retrieval for in-the-wild diagnosis. | Retrieval system; no edge-model comparison. |
| 6 | Wei et al., *PlantSeg* (arXiv 2409.04038, 2024) | 11.4k in-the-wild segmentation images, 115 diseases. | Segmentation; classification shift not addressed. |
| 7 | Yang et al., *From laboratory to field: cross-domain few-shot learning for crop disease identification* (Front. Plant Sci. 15:1434222, 2024) | CD-FSL framework, PlantVillage/others → new field dataset FCDD; 5-way 5-shot up to 83.6%. | Episodic N-way protocol, not full-label-space accuracy; FCDD availability/edge cost unclear. |
| 8 | Agri-FM+ (Al Nahian et al., CVPRW 2025) | SlotCon weights continually adapted on 147K agri images; +1.27% over ImageNet-sup. on 8 det/seg benchmarks. | Detection/segmentation; no disease-classification shift test. |
| 9 | Nguyen Quoc et al., *SCOLD: VLM for leaf disease identification* (arXiv 2505.07019, 2025) | Soft-target contrastive VLM trained on 186k image–caption pairs. | Large model; edge feasibility not shown. |
| 10 | Salman et al., *Plant disease classification in the wild using ViT and MoE* (Front. Plant Sci. 16:1522985, 2025) | ViT+MoE trained on PlantVillage subset, 68% on PlantDoc (ViT-B 48%, EfficientNet 43%). | Single run (no seeds/variance reported in what we read); class-mapping procedure not described; no latency. |
| 11 | Benabbas et al., *Rethinking Plant Disease Diagnosis… ViTs and Zero-Shot Learning* (arXiv 2511.18989, 2025) | CNN vs ViT vs CLIP zero-shot; ViTs generalize better under shift. | Abstract gives no field-test numbers or edge cost. |
| 12 | Al Mamun et al., *StateSpace-SSL* (arXiv 2512.09492, 2025; AAAI AgriAI'26 workshop) | Vision-Mamba SSL with prototype teacher–student, linear-time. | Datasets unnamed in abstract; cross-domain protocol unclear. |
| 13 | Al Mamun et al., *PSMamba* (arXiv 2512.14309, 2025) | Progressive SSL Vision Mamba; multi-scale lesion features; claims gains under domain shift. | Shift protocol not specified in abstract. |
| 14 | Sonavane, *Domain-Specific SSL Pre-training for Agricultural Disease Classification* (arXiv 2601.11612, 2026) | SimCLR on 3k unlabeled agri images gives +4.57%, more than architecture changes; 87.1% in-domain PlantDoc. | In-domain only; no cross-dataset transfer in abstract. |
| 15 | Mooraj et al., *AgriPath* (arXiv 2603.13354, 2026) | CNN vs contrastive VLM vs generative VLM on new 111k lab/field-separated benchmark; CNNs degrade most under shift. | No edge/latency evaluation; code/data "upon publication". |
| 16 | Lundqvist et al., *Does Your VFM Speak Plant?* (arXiv 2604.09920, 2026) | Prompt-axis optimization for open-vocab detectors in cowpea; prompts don't transfer across models. | Detection; shows foundation-model sensitivity in agri. |

## Foundational / baselines (pre-2023, cited for datasets and models)
- Hughes & Salathé 2015, PlantVillage release (arXiv 1511.08060). Mohanty et al. 2016 (Front. Plant Sci., 10.3389/fpls.2016.01419): 99.35% held-out PlantVillage accuracy; the arXiv version reports 31.4% on web images.
- Singh et al. 2020, PlantDoc (arXiv 1911.10317; CoDS-COMAD): 2,598 field images, 13 crops, 17 diseases. Reports that ImageNet+PlantVillage transfer roughly doubles PlantDoc classification accuracy (~29.7% vs ~13.7%, per a secondary source; **to check in the PDF before citing a number**).
- Models: MobileNetV3 (1905.02244), EfficientNet (1905.11946), ConvNeXt (2201.03545), ViT (2010.11929), CLIP (2103.00020), DINOv2 (2304.07193).

## Synthesis: where the gap is
1. Everyone agrees the PlantVillage→field gap is large, but the reported numbers vary widely (25%–68% on PlantDoc). The class mappings differ and are rarely published, and **no one reports seed variance**.
2. Most papers propose a bigger model (ViT+MoE, VLMs, Mamba SSL). Few ask what an **edge-sized** model can do, or what it costs on a CPU or phone.
3. Few papers separate the effects of **pretraining source** (ImageNet-supervised vs. self-supervised DINOv2 vs. CLIP) and **a handful of field labels** under one fixed, released protocol.
4. Leakage hygiene is inconsistent. PlantVillage has multiple images per leaf, and only the leaf-grouped split controls for that.

These points motivate a small, fully reproducible benchmark rather than a new architecture. That is a good fit for a workshop that lists *reproducibility, benchmarking, edge computing* among its topics.
