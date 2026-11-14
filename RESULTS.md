# Experiment Results

## Stage 1 — HRPlanesv2

COCO-pretrained YOLOv5x was fine-tuned on the HRPlanesv2 training split. The supplied validation split selected the Stage-1 `best.pt` checkpoint.

Held-out HRPlanesv2 test results:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.988 | 0.984 | 0.991 | 0.796 |

## Stage 2 — CORS-ADD-HBB

The Stage-1 checkpoint initialized fine-tuning on CORS-ADD-HBB. A group-aware internal validation split selected the final checkpoint. Early stopping ended Stage 2 before the 100-epoch maximum.

Official held-out CORS test results:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.946 | 0.905 | 0.937 | 0.534 |

## Houston qualitative inference

- Source images: 1
- Overlapping 640 × 640 tiles: 30
- Raw tile-level detections: 97
- Final detections after global NMS: 63

Houston results are qualitative because complete ground-truth labels were not created for this external image.

