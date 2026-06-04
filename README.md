# GSTransformFL — Benchmark Experiments

Federated learning benchmarks evaluating the impact of Gerchberg-Saxton (GS)
optical transforms on medical image classification in a federated setting.

Built on top of the [GSTransformFL](https://github.com/seha-ay/GSTransformFL)
package — a production-ready NVFlare executor wrapping the GS algorithm.

## Experiments

| Folder | Dataset | Task | Status |
|--------|---------|------|--------|
| [`brain-tumor-mri/`](brain-tumor-mri/) | Brain Tumor MRI | 4-class tumor classification | ✅ Complete |
| [`oct-retinal/`](oct-retinal/) | OCT2017 Retinal | 4-class retinal classification | ✅ Complete |

## Experiment Design

Each experiment runs 4 conditions in a 3-client NVFlare federated simulation:

| Condition | Description |
|-----------|-------------|
| `baseline` | Raw images, no transform |
| `gs_0` | GS transform, maskP=0.0 |
| `gs_20` | GS transform, maskP=0.2 |
| `gs_50` | GS transform, maskP=0.5 |

The test set is transformed with the same maskP as training for fair evaluation.

## Requirements

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (tested on NVIDIA L4 24GB)
- CUDA 11.8+
- NVFlare 2.7.2
- Kaggle account with API key

## Quick Start

See the README inside each experiment folder for dataset-specific instructions.
