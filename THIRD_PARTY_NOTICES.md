# Third-Party Notices and Data Sources

This repository contains original course-project orchestration code and experiment outputs built on third-party software and datasets. Each upstream project retains its own license and terms.

## Software and reference implementation

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) — used as the training, validation, and inference implementation. Consult the upstream repository for its current license and terms.
- [Efficient YOLO for Remote-Sensing Airplane Detection](https://github.com/RSandAI/Efficient-YOLO-RS-Airplane-Detection) — methodological reference for the course-project pipeline.

## Datasets

- [HRPlanesv2](https://github.com/dilsadunsal/HRPlanesv2-Data-Set) — first-stage aircraft dataset.
- [CORS-ADD](https://github.com/sgtojd/CORS-ADD-Complex-Optical-Remote-Sensing-Aircraft-Detection-Dataset) — second-stage aircraft dataset and held-out evaluation source.

The complete HRPlanesv2 and CORS-ADD datasets are not redistributed in this repository. Users must obtain them from their official sources and comply with the applicable terms.

## Houston external image

- USDA National Agriculture Imagery Program (NAIP), distributed through USGS The National Map Orthoimagery service.
- Acquisition date: 2022-09-14.
- Approximate ground sample distance: 0.6 m.
- Credit: `USGS, USDA, The National Map: Orthoimagery`.

Detailed source metadata is stored under `sample_data/houston/`.

