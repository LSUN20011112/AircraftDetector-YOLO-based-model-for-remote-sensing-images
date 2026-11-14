# Houston imagery input

Place independently sourced Houston airport remote-sensing images in the `input` folder before running `02_Houston_External_Inference.ipynb`.

Complete one row per image in `metadata.csv`. Record the provider, source URL, acquisition date, spatial resolution, license or terms of use, and whether you confirmed that the image does not duplicate or overlap CORS-ADD imagery.

Because CORS-ADD includes George Bush Intercontinental Houston Airport imagery, describe this stage as qualitative inference on independently sourced imagery rather than validation on an unseen airport.

Without complete ground-truth annotations, do not report Precision, Recall, F1, or mAP for these images.

## Included candidate image

`input/Houston_Hobby_USDA_NAIP_2022-09-14.png` is a natural-color crop of William P. Hobby Airport from USDA NAIP imagery served by USGS The National Map. The source acquisition date is September 14, 2022 and the source ground sample distance is 0.6 meter. The image is public domain; credit it as `USGS, USDA, The National Map: Orthoimagery`.

The crop contains commercial aircraft near the terminal, aircraft on taxiways, and smaller aircraft near maintenance facilities. It is intended for qualitative inference only and has no ground-truth bounding boxes.
