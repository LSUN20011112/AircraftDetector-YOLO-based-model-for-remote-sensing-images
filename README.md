# Sequential Transfer Learning for Remote-Sensing Aircraft Detection

This course project implements a single-model, sequential transfer-learning pipeline for horizontal-bounding-box (HBB) aircraft detection in remote-sensing imagery.

```text
COCO-pretrained YOLOv5x
→ HRPlanesv2 fine-tuning
→ HRPlanesv2 validation checkpoint selection
→ HRPlanesv2 held-out test
→ CORS-ADD-HBB fine-tuning
→ CORS internal validation checkpoint selection
→ official CORS val2017 held-out final test
→ qualitative inference on independent Houston airport imagery
```

Only one YOLOv5x architecture is used throughout. Validation selects the best epoch-specific checkpoint at each training stage; the held-out test sets are used only for final evaluation.

## Results

| Evaluation split | Images | Aircraft instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| HRPlanesv2 test | 210 | 2,046 | 0.988 | 0.984 | 0.991 | 0.796 |
| CORS-ADD-HBB held-out test | 1,722 | 7,768 | 0.946 | 0.905 | 0.937 | 0.534 |

Houston inference used one independently sourced 3,000 × 2,500 USDA NAIP image. The image was divided into 30 overlapping tiles. The model produced 97 tile-level detections and 63 detections after global non-maximum suppression. Because complete Houston ground-truth annotations are unavailable, Houston results are reported qualitatively rather than as Precision, Recall, F1, or mAP.

## Dataset policy

- HRPlanesv2 retains its supplied train/validation/test split: 1,490 / 420 / 210 images.
- The official CORS-ADD-HBB `train2017` portion is split group-wise by airport/source into 3,199 training images and 565 internal-validation images using seed 42.
- The official CORS-ADD-HBB `val2017` portion (1,722 images) remains untouched and is mapped to the final held-out test split.
- HBB annotations are used; the optional OBB release is not used.

The large source datasets are intentionally excluded from this repository. Download them from their official sources:

- [HRPlanesv2](https://github.com/dilsadunsal/HRPlanesv2-Data-Set)
- [CORS-ADD](https://github.com/sgtojd/CORS-ADD-Complex-Optical-Remote-Sensing-Aircraft-Detection-Dataset)
- [Original Efficient-YOLO remote-sensing project](https://github.com/RSandAI/Efficient-YOLO-RS-Airplane-Detection)

## Repository structure

```text
notebooks/
  03_Full_Sequential_Transfer_Learning_Pipeline.ipynb
  executed/                         # completed run retained as an experiment record
  modular/                          # optional 00/01/02 workflow
tools/                              # notebook generation utilities
metadata/
  environment.json
  splits/
  data_qa/
results/
  stage1_hrplanesv2/
  stage2_cors_add_hbb/
  evaluation/
  houston/
sample_data/houston/                # public-domain USDA NAIP example and provenance
docs/                               # workflow notes and course-proposal slide
```

## Run in Google Colab

1. Put the full private project folder at `MyDrive/RS_PRO`.
2. Keep the datasets in the layout documented in `docs/COLAB_WORKFLOW.md`.
3. Open `notebooks/03_Full_Sequential_Transfer_Learning_Pipeline.ipynb` in Colab.
4. Select a GPU runtime.
5. Leave `RUN_FULL_PIPELINE = True` and run all cells.

The notebook writes persistent outputs to `MyDrive/RS_PRO/artifacts`. Temporary prepared data under `/content` is rebuilt after a runtime reset. Completed stages are skipped, while interrupted training can resume from `last.pt`.

If GPU memory is insufficient, reduce `BATCH_SIZE_OVERRIDE` from 4 to 2 or 1.

## Model weights

PyTorch checkpoints are excluded because the final `best.pt` files are approximately 173 MB each and periodic epoch checkpoints are approximately 691 MB each. Publish final weights separately through a model-hosting service, Google Drive, GitHub Releases, or Git LFS if redistribution is required.

## Reproducibility

- Random seed: 42
- Input size: 640 × 640
- Stage-1 maximum epochs: 100
- Stage-2 maximum epochs: 100
- Early-stopping patience: 20 epochs
- YOLOv5 source revision and runtime versions: `metadata/environment.json`
- Reproducible split manifests: `metadata/splits/`

## Attribution

The Houston image is USDA National Agriculture Imagery Program imagery distributed through USGS The National Map. See `sample_data/houston/README.md` and the accompanying source JSON for acquisition and bounding-box metadata.

Third-party projects, datasets, and licenses are listed in `THIRD_PARTY_NOTICES.md`.

