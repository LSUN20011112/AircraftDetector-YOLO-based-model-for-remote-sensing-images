# GitHub upload checklist

Upload the **contents** of this `GITHUB_READY` folder to the root of a new GitHub repository.

## Included

- Clean integrated Colab notebook: `notebooks/03_Full_Sequential_Transfer_Learning_Pipeline.ipynb`
- Completed notebook with saved outputs: `notebooks/executed/`
- Optional modular notebooks: `notebooks/modular/`
- Reproducibility metadata and exact split manifests: `metadata/`
- Training, evaluation, and Houston inference figures/tables: `results/`
- One public-domain Houston NAIP sample image: `sample_data/houston/`
- Notebook builders and documentation: `tools/` and `docs/`

## Intentionally excluded

- HRPlanesv2 and CORS-ADD image/label datasets
- YOLOv5 clone and Python environment folders
- `best.pt`, `last.pt`, epoch checkpoints, caches, and TensorBoard events
- Large per-image prediction JSON and generated label folders

The two final checkpoints are about 173 MB each, so they must not be added as ordinary GitHub repository files. If weights need to be shared later, publish them separately with GitHub Releases/LFS or an external model host, and add the download links to `README.md`.

## Before making the repository public

1. Confirm your preferred repository name and visibility.
2. Review `README.md`, `RESULTS.md`, and `THIRD_PARTY_NOTICES.md`.
3. Choose a license only after deciding how you want others to reuse your original code.
4. Drag everything inside `GITHUB_READY` into the repository upload page (not the outer `GITHUB_READY` folder itself), then commit.
