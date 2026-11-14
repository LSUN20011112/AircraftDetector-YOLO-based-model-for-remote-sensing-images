# RS_PRO Colab workflow

This folder contains a three-notebook implementation of the finalized sequential transfer-learning pipeline:

```text
COCO-pretrained YOLOv5x
-> HRPlanesv2 fine-tuning
-> HRPlanesv2 validation selects Stage-1 best.pt
-> HRPlanesv2 held-out test
-> CORS-ADD-HBB fine-tuning from Stage-1 best.pt
-> CORS internal validation selects final best.pt
-> official CORS val2017 is reserved as the held-out final test
-> qualitative inference on independently sourced Houston airport imagery
```

## Upload location

Upload the entire `RS_PRO` folder to:

```text
Google Drive/MyDrive/RS_PRO
```

If the folder is stored elsewhere, change `PROJECT_ROOT` near the top of every notebook.

## Notebook order

1. `00_Data_QA_and_Splits.ipynb`
   - Checks both HBB datasets.
   - Preserves the existing HRPlanesv2 train/validation/test split.
   - Splits CORS-ADD-HBB `train2017` into approximately 85% training and 15% internal validation data.
   - Uses airport/source groups rather than randomly mixing individual tiles.
   - Reserves the official CORS-ADD-HBB `val2017` portion as the final test set.
   - Removes invalid zero-area annotations only from a temporary prepared copy.
   - Creates YOLOv5 YAML files and reproducible split manifests.

2. `01_Sequential_Transfer_Learning_YOLOv5x.ipynb`
   - Installs the official Ultralytics YOLOv5 v7.0 implementation.
   - Fine-tunes COCO-pretrained `yolov5x.pt` on HRPlanesv2.
   - Evaluates the Stage-1 `best.pt` on the HRPlanesv2 test split.
   - Fine-tunes the Stage-1 `best.pt` on CORS-ADD-HBB.
   - Evaluates the final `best.pt` once on the held-out CORS test split.
   - Stores weights, training histories, plots, predictions, and evaluation outputs in Google Drive.

3. `02_Houston_External_Inference.ipynb`
   - Reads independently sourced Houston airport imagery.
   - Tiles large images into overlapping 640 x 640 patches.
   - Runs final-model inference.
   - Maps detections back to full-image coordinates.
   - Applies global NMS to remove duplicate detections in overlapping tiles.
   - Saves CSV outputs and full-image visualizations.

## Important runtime behavior

Notebook 00 stages the prepared datasets under:

```text
/content/rs_pro_work
```

This avoids modifying the original datasets and normally trains faster than reading every image directly from Google Drive. Run notebooks 00 and 01 in the same Colab runtime. If Colab resets, rerun notebook 00 before resuming training.

Training results and checkpoints are written to:

```text
MyDrive/RS_PRO/artifacts
```

Therefore, a runtime reset does not delete completed checkpoints.

## Training safety switches

Formal training and evaluation cells do not run automatically. Change the relevant Boolean only when ready:

```python
RUN_STAGE1_TRAINING = True
RUN_HR_TEST = True
RUN_STAGE2_TRAINING = True
RUN_FINAL_CORS_TEST = True
RUN_HOUSTON_INFERENCE = True
```

Run them in that order. Do not enable the final CORS test until Stage 2 training and checkpoint selection are complete.

## CORS-ADD-HBB split policy

The downloaded HBB release contains:

- `train2017`: 3,764 images
- `val2017`: 1,722 images

The workflow uses:

```text
train2017
├── approximately 85% model training
└── approximately 15% internal validation and checkpoint selection

val2017
└── held-out final testing only
```

The original folder name `val2017` is retained by the release, but this project never uses it for checkpoint selection.

## Houston imagery

Place independently sourced images under:

```text
RS_PRO/Houston_Imagery/input
```

Notebook 02 creates a metadata template at:

```text
RS_PRO/Houston_Imagery/metadata.csv
```

Record the provider, source URL, acquisition date, spatial resolution, license, and whether the image has been checked for overlap with CORS-ADD. Because CORS-ADD includes Houston airport imagery, avoid claiming that Houston is an unseen geographic location unless the training data is redesigned accordingly.

Without complete Houston ground-truth annotations, report Houston results only as qualitative external inference. Do not calculate or claim Precision, Recall, F1, or mAP.

## Data preservation

- The notebooks do not edit `HRPlanesv2` or `CORS-ADD-HBB` source files.
- `CORS-ADD-OBB` is not used.
- Invalid annotations are removed only from the temporary `/content/rs_pro_work` copy.
- Split manifests and cleaning reports are saved under `RS_PRO/artifacts`.

