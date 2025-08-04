# From Gaze to Insight: Bridging Human Visual Attention and Vision-Language Model Explanation for Weakly-Supervised Medical Image Segmentation

> Code and accompanying material for the paper: *From Gaze to Insight: Bridging Human Visual Attention and Vision Language Model Explanation for Weakly-Supervised Medical Image Segmentation*. This repository implements the full weakly-supervised teacher–student segmentation framework that fuses clinician gaze (where experts look) with vision-language explanations (why regions matter) to produce high-quality segmentation with minimal dense annotation. If you use or build upon this work, please cite the paper. fileciteturn0file0

## Abstract

Medical image segmentation usually requires expensive dense annotations. We propose **From Gaze to Insight**, a weakly-supervised framework that synergistically integrates *human visual attention* (clinician gaze) and *vision-language model explanations* within a teacher–student paradigm. The teacher learns robust multi-modal representations from high-confidence sparse gaze-derived pseudomasks combined with textual semantic cues, while the student distills that knowledge under noisier supervision using consistency regularization and a novel **Disagreement-Aware Random Masking (DARM)** mechanism. The dual “where” (gaze) and “why” (textual explanation) signals deliver competitive segmentation accuracy on Kvasir-SEG, NCI-ISBI (prostate MRI), and ISIC (skin lesion) benchmarks, substantially reducing annotation cost and maintaining interpretability. Extensive ablations validate the contributions of cross-modal fusion, confidence-aware consistency, and disagreement-aware masking.

## Key Contributions

- **Dual weak supervision fusion**: Combines sparse clinician gaze (spatial attention) with structured semantic explanations from vision-language models to capture both *where* and *why*.  
- **Teacher–Student knowledge distillation**: Teacher provides stable cross-modal guidance; Student learns under noisy full gaze labels with consistency enforcement.  
- **Disagreement-Aware Random Masking (DARM)**: Identifies high-disagreement regions between teacher and student and stochastically masks them during training to improve robustness.  
- **Strong performance with limited annotation**: Validated on three diverse medical segmentation datasets, narrowing the gap to fully supervised methods.  
- **Retained interpretability**: Combining gaze and textual cues offers insight into model decisions.

## Table of Contents

- [Installation](#installation)  
- [Datasets](#datasets)  
- [Model Weights and Training Logs](#model-weights-and-training-logs)  
- [Paper and Reference](#paper-and-reference)  
- [Quick Start](#quick-start)  
- [Reproducibility Details](#reproducibility-details)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Directory Layout](#directory-layout)  
- [Citation & References](#citation--references)  
- [Contact](#contact)  
- [License](#license)  
- [Acknowledgements](#acknowledgements)

## Installation

Recommended: use `conda` or `virtualenv` to isolate environment.

```bash
conda create -n gaze_insight python=3.10 -y
conda activate gaze_insight
pip install -r requirements.txt
```

**Core dependencies (lock exact versions in `requirements.txt`)**:

- `torch` (CUDA-enabled)  
- `transformers`  
- `opencv-python` / `Pillow`  
- `numpy`, `scipy`  
- Evaluation utilities (Dice, Hausdorff, mIoU, etc.)

## Datasets

Benchmarks used in this work:

1. **Kvasir-SEG** – Endoscopic polyp segmentation.  
2. **NCI-ISBI** – Prostate MRI segmentation (ISBI challenge).  
3. **ISIC** – Dermoscopic skin lesion segmentation (with simulated gaze annotations).  

### Acquisition & Preprocessing

Obtain raw images and ground-truth masks from original sources. Preprocessing includes:

- Image normalization and resizing  
- Constructing sparse gaze-derived pseudomasks  
- Generating / encoding vision-language textual cues  

Scripts live under `./scripts/` (e.g., `preprocess_kvasir.py`, `build_gaze_mask.py`). Configuration templates are in `configs/`.

## Model Weights and Training Logs

All checkpoints (teacher/student per dataset), training logs (loss trajectories, seed records, hyperparameters, ablation variants), and evaluation snapshots are available:  
**Google Drive folder**:  
https://drive.google.com/drive/folders/1khBf2FavCg2LyiN7pNmb55ZPzDvzlTKi?usp=drive_link

### Usage Advice

1. Download matching checkpoint(s) (e.g., `nci_isbi_teacher.pth`, `isic_student_seed3/`).  
2. Use the corresponding config from `configs/`.  
3. To reproduce published numbers, apply the exact random seed and hyperparameters logged.  
4. Inspect alternate fusion/ablation settings from the logs for comparison.

## Paper and Reference

Full paper (preprint, second revision) is available on arXiv:  
**PDF**: https://arxiv.org/pdf/2504.11368 fileciteturn0file0

Refer to it for detailed methodology, architecture diagrams, loss formulations, dataset splits, experimental protocol, quantitative/qualitative results, and ablation studies.

## Quick Start

Example pipeline (replace with actual paths):

```bash
gazesup_kvasir_2_levels.sh

gazesup_prostate_2_levels.sh

gazesup_isic_2_levels.sh
```

Include convenience scripts like `run_kvasir.sh` or `eval_isic.sh` in `examples/`.

## Reproducibility Details

- **Random seeds**: Multiple seeds logged; exact values stored in each experiment folder.  
- **Hyperparameters (typical defaults)**:  
  - Teacher: partial cross-entropy weight for sparse gaze, learnable multi-scale fusion λ.  
  - Student: λ_AFC = 0.1, λ_CWC = 1.0; confidence thresholds τ_pos = 0.8, τ_neg = 0.2.  
  - Optimizer: Adam with initial LR `1e-2`, cosine decay.  
- **DARM**: Disagreement-Aware Random Masking adaptively hides high-disagreement spatial regions during student training to encourage robust feature alignment (implementation in `models/darm.py`).

## Evaluation Metrics

- **Dice Similarity Coefficient**  
- **Mean Intersection over Union (mIoU)**  
- **95% Hausdorff Distance (HD95)**  
- **Average Surface Distance (ASD)**  
- **Annotation Time Efficiency** (comparing gaze-based weak supervision to dense labeling) fileciteturn0file0

## Directory Layout (Suggested)

```
├── configs/                    # Example config files
├── scripts/                    # Preprocessing / utility scripts
├── models/                    # Model definitions (teacher, student, DARM, etc.)
├── datasets/                 # Dataset wrappers / loaders
├── checkpoints/              # Stored model weights
├── outputs/                  # Prediction logs, evaluation, ablation outputs
├── examples/                 # Shell scripts / usage examples
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── LICENSE                 # License
```

## Citation & References

If you use or build upon this work, please cite the main paper:

```bibtex
@article{chen2025gaze,
  title={From Gaze to Insight: Bridging Human Visual Attention and Vision Language Model Explanation for Weakly-Supervised Medical Image Segmentation},
  author={Chen, Jingkun and Duan, Haoran and Zhang, Xiao and Gao, Boyan and Grau, Vicente and Han, Jungong},
  journal={arXiv preprint arXiv:2504.11368},
  year={2025}
}
```


## Contact

- **Corresponding author**: Jungong Han — jghan@tsinghua.edu.cn  
- **Code lead / First author**: Jingkun Chen — jingkun.chen@eng.ox.ac.uk  

## License

Specify the license, e.g.:

```
MIT License
```

Include a matching `LICENSE` file in the repository.

## Acknowledgements

This work was supported in part by relevant funding agencies. Collaborative research between Tsinghua University and the University of Oxford. fileciteturn0file0
