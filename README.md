# From Gaze to Insight: Bridging Human Visual Attention and Vision-Language Model Explanation for Weakly-Supervised Medical Image Segmentation

> Code accompanying the paper: *From Gaze to Insight: Bridging Human Visual Attention and Vision Language Model Explanation for Weakly-Supervised Medical Image Segmentation*. This project implements a teacher–student weakly-supervised segmentation framework that fuses expert gaze (where clinicians look) with vision-language model explanations (why regions matter) to achieve high-quality segmentation under sparse/noisy supervision. If you use this code or build on this work, please cite the paper. :contentReference[oaicite:4]{index=4}

## Key Features

- **Complementary weak signals fusion**: Combines sparse but precise clinician gaze supervision with structured semantic explanations derived from vision-language models.  
- **Teacher–Student architecture**: Teacher learns robust cross-modal representations from high-confidence gaze pseudomasks and textual cues; Student distills this knowledge while handling noisier full gaze labels with consistency regularization and disagreement-aware masking.  
- **Core techniques**: Multi-scale text-vision fusion, confidence-aware feature consistency, and Disagreement-Aware Random Masking (DARM).  
- **Evaluated on three medical benchmarks**: Demonstrates strong weakly-supervised performance on Kvasir-SEG, NCI-ISBI (prostate MRI), and ISIC (skin lesion), with retained interpretability. :contentReference[oaicite:5]{index=5}

## Table of Contents

- [Installation](#installation)  
- [Datasets](#datasets)  
- [Model Weights and Training Logs](#model-weights-and-training-logs)  
- [Quick Start](#quick-start)  
- [Reproducibility Details](#reproducibility-details)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Citation](#citation)  
- [Contact](#contact)  
- [License](#license)  
- [Acknowledgements](#acknowledgements)

## Installation

Create and activate a Python environment (e.g., with conda):

```bash
conda create -n gaze_insight python=3.10 -y
conda activate gaze_insight
pip install -r requirements.txt
