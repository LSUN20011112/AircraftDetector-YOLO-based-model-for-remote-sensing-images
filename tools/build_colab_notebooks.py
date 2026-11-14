from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip() + "\n",
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


def write_notebook(name: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = ROOT / name
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")


notebook_00 = [
    markdown(
        """
        # 00 - Data quality control and reproducible dataset splits

        This notebook prepares the two HBB datasets for the sequential transfer-learning workflow:

        1. Preserve the existing HRPlanesv2 train/validation/test split.
        2. Split the official CORS-ADD-HBB `train2017` portion into approximately 85% training and 15% internal validation data.
        3. Keep the official CORS-ADD-HBB `val2017` portion untouched as the held-out final test set.
        4. Remove malformed or zero-area boxes only from the temporary prepared copy. The original files in Google Drive are never edited.
        5. Create YOLOv5-compatible folders, YAML files, and split manifests.

        Run this notebook in Colab after uploading the entire `RS_PRO` folder to Google Drive. Keep the same Colab runtime for notebook 01 because the prepared datasets are staged under `/content` for faster training.
        """
    ),
    markdown(
        """
        ## Expected Google Drive layout

        ```text
        MyDrive/
        └── RS_PRO/
            ├── HRPlanesv2/
            │   ├── train/
            │   ├── val/
            │   └── test/
            ├── CORS-ADD-HBB/
            │   ├── images/train2017/
            │   ├── images/val2017/
            │   ├── labels/train2017/
            │   └── labels/val2017/
            └── *.ipynb
        ```

        The optional `CORS-ADD-OBB` folder is ignored by this workflow.
        """
    ),
    code(
        """
        from google.colab import drive

        drive.mount('/content/drive')
        """
    ),
    code(
        """
        from pathlib import Path
        import csv
        import json
        import math
        import os
        import random
        import re
        import shutil
        from collections import Counter, defaultdict

        import matplotlib.pyplot as plt
        import pandas as pd
        from PIL import Image, ImageDraw

        # Change only this line if RS_PRO is stored elsewhere in Google Drive.
        PROJECT_ROOT = Path('/content/drive/MyDrive/RS_PRO')

        # Prepared image data is staged on Colab local disk for faster training.
        WORK_ROOT = Path('/content/rs_pro_work')
        ARTIFACT_ROOT = PROJECT_ROOT / 'artifacts'

        RANDOM_SEED = 42
        CORS_INTERNAL_VAL_FRACTION = 0.15

        # copy: uses more local disk but is usually faster during training.
        # symlink: uses less local disk but reads images from Google Drive.
        STAGE_MODE = 'copy'

        # True safely rebuilds only /content/rs_pro_work. It never deletes Drive data.
        REBUILD_PREPARED = True

        IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

        assert PROJECT_ROOT.exists(), f'Project folder not found: {PROJECT_ROOT}'
        assert str(WORK_ROOT).startswith('/content/'), 'WORK_ROOT must remain under /content.'

        if REBUILD_PREPARED and WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)

        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_ROOT / 'splits').mkdir(parents=True, exist_ok=True)
        (ARTIFACT_ROOT / 'data_qa').mkdir(parents=True, exist_ok=True)

        print('PROJECT_ROOT:', PROJECT_ROOT)
        print('WORK_ROOT:', WORK_ROOT)
        print('STAGE_MODE:', STAGE_MODE)
        """
    ),
    markdown("## Verify the source folders"),
    code(
        """
        required_paths = [
            PROJECT_ROOT / 'HRPlanesv2' / 'train',
            PROJECT_ROOT / 'HRPlanesv2' / 'val',
            PROJECT_ROOT / 'HRPlanesv2' / 'test',
            PROJECT_ROOT / 'CORS-ADD-HBB' / 'images' / 'train2017',
            PROJECT_ROOT / 'CORS-ADD-HBB' / 'images' / 'val2017',
            PROJECT_ROOT / 'CORS-ADD-HBB' / 'labels' / 'train2017',
            PROJECT_ROOT / 'CORS-ADD-HBB' / 'labels' / 'val2017',
        ]

        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError('Missing required paths:\\n' + '\\n'.join(missing))

        for path in required_paths:
            print('OK:', path.relative_to(PROJECT_ROOT))
        """
    ),
    markdown(
        """
        ## Helper functions

        CORS-ADD tiles are grouped by the airport/source name encoded before `_640_x_y`. The internal validation split is selected by source group rather than by individual tiles, reducing leakage from neighboring patches.
        """
    ),
    code(
        """
        def list_images(folder: Path) -> list[Path]:
            return sorted(
                path for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )


        def cors_source_group(image_path: Path) -> str:
            match = re.match(r'^(.*)_640_\\d+_\\d+$', image_path.stem)
            return match.group(1) if match else image_path.stem


        def select_group_validation_split(
            images: list[Path],
            fraction: float,
            seed: int,
            trials_per_group_count: int = 2000,
        ) -> tuple[list[Path], list[Path], set[str]]:
            groups = defaultdict(list)
            for image in images:
                groups[cors_source_group(image)].append(image)

            group_names = sorted(groups)
            target_images = round(len(images) * fraction)
            target_group_count = max(1, round(len(group_names) * fraction))
            rng = random.Random(seed)

            best_groups = None
            best_score = None

            lower = max(1, target_group_count - 3)
            upper = min(len(group_names) - 1, target_group_count + 3)

            for group_count in range(lower, upper + 1):
                for _ in range(trials_per_group_count):
                    candidate = set(rng.sample(group_names, group_count))
                    candidate_count = sum(len(groups[name]) for name in candidate)
                    score = (
                        abs(candidate_count - target_images),
                        abs(group_count - target_group_count),
                        sorted(candidate),
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_groups = candidate

            validation_images = [
                image for image in images if cors_source_group(image) in best_groups
            ]
            training_images = [
                image for image in images if cors_source_group(image) not in best_groups
            ]

            return sorted(training_images), sorted(validation_images), set(best_groups)


        def stage_image(source: Path, destination: Path, mode: str) -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                return

            if mode == 'copy':
                shutil.copy2(source, destination)
            elif mode == 'symlink':
                os.symlink(source, destination)
            else:
                raise ValueError("STAGE_MODE must be 'copy' or 'symlink'.")


        def parse_and_clean_label(source: Path, destination: Path) -> list[dict]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            retained = []
            removed = []

            if not source.exists():
                removed.append({
                    'file': str(source),
                    'line': None,
                    'reason': 'missing_label_file',
                    'content': '',
                })
                destination.write_text('', encoding='utf-8')
                return removed

            for line_number, raw_line in enumerate(
                source.read_text(encoding='utf-8', errors='replace').splitlines(), start=1
            ):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                reason = None

                if len(parts) != 5:
                    reason = 'expected_5_columns'
                else:
                    try:
                        class_id = int(float(parts[0]))
                        x_center, y_center, width, height = map(float, parts[1:])
                        values = [x_center, y_center, width, height]

                        if class_id != 0:
                            reason = 'unexpected_class_id'
                        elif not all(math.isfinite(value) for value in values):
                            reason = 'non_finite_coordinate'
                        elif not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                            reason = 'center_outside_normalized_range'
                        elif not (0 < width <= 1 and 0 < height <= 1):
                            reason = 'zero_or_invalid_box_size'
                        elif (
                            x_center - width / 2 < -1e-6
                            or x_center + width / 2 > 1 + 1e-6
                            or y_center - height / 2 < -1e-6
                            or y_center + height / 2 > 1 + 1e-6
                        ):
                            reason = 'box_extends_outside_image'
                    except ValueError:
                        reason = 'non_numeric_label'

                if reason is None:
                    retained.append(line)
                else:
                    removed.append({
                        'file': str(source),
                        'line': line_number,
                        'reason': reason,
                        'content': line,
                    })

            destination.write_text(
                ('\\n'.join(retained) + '\\n') if retained else '',
                encoding='utf-8',
            )
            return removed


        def stage_yolo_split(
            items: list[tuple[Path, Path, str]],
            dataset_root: Path,
            split_name: str,
            stage_mode: str,
        ) -> tuple[list[dict], list[dict]]:
            manifest_rows = []
            removed_rows = []

            for image_source, label_source, source_group in items:
                image_destination = dataset_root / 'images' / split_name / image_source.name
                label_destination = dataset_root / 'labels' / split_name / f'{image_source.stem}.txt'

                stage_image(image_source, image_destination, stage_mode)
                removed_rows.extend(parse_and_clean_label(label_source, label_destination))

                manifest_rows.append({
                    'split': split_name,
                    'source_group': source_group,
                    'source_image': str(image_source),
                    'source_label': str(label_source),
                    'prepared_image': str(image_destination),
                    'prepared_label': str(label_destination),
                })

            return manifest_rows, removed_rows


        def count_label_boxes(label_folder: Path) -> int:
            count = 0
            for label_path in label_folder.glob('*.txt'):
                count += sum(
                    1 for line in label_path.read_text(encoding='utf-8').splitlines()
                    if line.strip()
                )
            return count
        """
    ),
    markdown("## Prepare HRPlanesv2 without changing its official split"),
    code(
        """
        hr_source = PROJECT_ROOT / 'HRPlanesv2'
        hr_prepared = WORK_ROOT / 'hrplanesv2'

        hr_manifest = []
        removed_annotations = []

        for split_name in ['train', 'val', 'test']:
            source_folder = hr_source / split_name
            images = list_images(source_folder)
            items = [
                (image, source_folder / f'{image.stem}.txt', f'hrplanesv2_{split_name}')
                for image in images
            ]
            manifest_rows, removed_rows = stage_yolo_split(
                items, hr_prepared, split_name, STAGE_MODE
            )
            hr_manifest.extend(manifest_rows)
            removed_annotations.extend(removed_rows)
            print(f'HRPlanesv2 {split_name}: {len(images):,} images')
        """
    ),
    markdown("## Create the group-aware CORS-ADD-HBB internal validation split"),
    code(
        """
        cors_source = PROJECT_ROOT / 'CORS-ADD-HBB'
        cors_prepared = WORK_ROOT / 'cors_add_hbb'

        official_train_images = list_images(cors_source / 'images' / 'train2017')
        official_test_images = list_images(cors_source / 'images' / 'val2017')

        cors_train_images, cors_val_images, cors_val_groups = select_group_validation_split(
            official_train_images,
            fraction=CORS_INTERNAL_VAL_FRACTION,
            seed=RANDOM_SEED,
        )

        train_groups = {cors_source_group(image) for image in cors_train_images}
        validation_groups = {cors_source_group(image) for image in cors_val_images}

        assert train_groups.isdisjoint(validation_groups)
        assert len(cors_train_images) + len(cors_val_images) == len(official_train_images)

        print(f'Official train2017: {len(official_train_images):,} images')
        print(f'Internal training: {len(cors_train_images):,} images')
        print(f'Internal validation: {len(cors_val_images):,} images')
        print(f'Validation fraction of train2017: {len(cors_val_images) / len(official_train_images):.2%}')
        print(f'Held-out official val2017/test: {len(official_test_images):,} images')
        print(f'Train source groups: {len(train_groups)}')
        print(f'Validation source groups: {len(validation_groups)}')
        """
    ),
    code(
        """
        cors_manifest = []

        cors_split_sources = {
            'train': (
                cors_train_images,
                cors_source / 'labels' / 'train2017',
            ),
            'val': (
                cors_val_images,
                cors_source / 'labels' / 'train2017',
            ),
            # The folder is named val2017 in the release, but this project reserves it
            # as a held-out final test set and never uses it for checkpoint selection.
            'test': (
                official_test_images,
                cors_source / 'labels' / 'val2017',
            ),
        }

        for split_name, (images, label_folder) in cors_split_sources.items():
            items = [
                (image, label_folder / f'{image.stem}.txt', cors_source_group(image))
                for image in images
            ]
            manifest_rows, removed_rows = stage_yolo_split(
                items, cors_prepared, split_name, STAGE_MODE
            )
            cors_manifest.extend(manifest_rows)
            removed_annotations.extend(removed_rows)
            print(f'CORS-ADD-HBB {split_name}: {len(images):,} images')
        """
    ),
    markdown("## Save reproducible split manifests and the data-cleaning report"),
    code(
        """
        split_root = ARTIFACT_ROOT / 'splits'
        qa_root = ARTIFACT_ROOT / 'data_qa'

        hr_manifest_path = split_root / 'hrplanesv2_manifest.csv'
        cors_manifest_path = split_root / 'cors_add_hbb_manifest.csv'
        cors_groups_path = split_root / 'cors_internal_validation_groups.json'
        removed_path = qa_root / 'removed_annotations.csv'

        pd.DataFrame(hr_manifest).to_csv(hr_manifest_path, index=False)
        pd.DataFrame(cors_manifest).to_csv(cors_manifest_path, index=False)
        pd.DataFrame(
            removed_annotations,
            columns=['file', 'line', 'reason', 'content'],
        ).to_csv(removed_path, index=False)

        cors_groups_path.write_text(
            json.dumps(
                {
                    'random_seed': RANDOM_SEED,
                    'target_validation_fraction': CORS_INTERNAL_VAL_FRACTION,
                    'validation_groups': sorted(cors_val_groups),
                },
                indent=2,
            ),
            encoding='utf-8',
        )

        print('Saved:', hr_manifest_path)
        print('Saved:', cors_manifest_path)
        print('Saved:', cors_groups_path)
        print('Saved:', removed_path)
        print('Removed invalid annotations:', len(removed_annotations))
        """
    ),
    markdown("## Create YOLOv5 dataset YAML files"),
    code(
        """
        config_root = WORK_ROOT / 'configs'
        config_root.mkdir(parents=True, exist_ok=True)

        hr_yaml = config_root / 'hrplanesv2.yaml'
        cors_yaml = config_root / 'cors_add_hbb.yaml'

        hr_yaml.write_text(
            f"path: {hr_prepared}\\n"
            "train: images/train\\n"
            "val: images/val\\n"
            "test: images/test\\n"
            "nc: 1\\n"
            "names: ['aircraft']\\n",
            encoding='utf-8',
        )

        cors_yaml.write_text(
            f"path: {cors_prepared}\\n"
            "train: images/train\\n"
            "val: images/val\\n"
            "test: images/test\\n"
            "nc: 1\\n"
            "names: ['aircraft']\\n",
            encoding='utf-8',
        )

        # Save archival copies in Drive. Training uses the /content versions above.
        shutil.copy2(hr_yaml, ARTIFACT_ROOT / 'splits' / hr_yaml.name)
        shutil.copy2(cors_yaml, ARTIFACT_ROOT / 'splits' / cors_yaml.name)

        print(hr_yaml.read_text())
        print(cors_yaml.read_text())
        """
    ),
    markdown("## Final prepared-data summary"),
    code(
        """
        summary_rows = []
        for dataset_name, dataset_root in [
            ('HRPlanesv2', hr_prepared),
            ('CORS-ADD-HBB', cors_prepared),
        ]:
            for split_name in ['train', 'val', 'test']:
                image_count = len(list_images(dataset_root / 'images' / split_name))
                box_count = count_label_boxes(dataset_root / 'labels' / split_name)
                summary_rows.append({
                    'dataset': dataset_name,
                    'split': split_name,
                    'images': image_count,
                    'valid_boxes': box_count,
                })

        summary = pd.DataFrame(summary_rows)
        display(summary)
        summary.to_csv(ARTIFACT_ROOT / 'data_qa' / 'prepared_dataset_summary.csv', index=False)
        """
    ),
    markdown("## Visual label check"),
    code(
        """
        def draw_yolo_boxes(image_path: Path, label_path: Path, max_boxes: int = 100) -> Image.Image:
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            width, height = image.size

            lines = label_path.read_text(encoding='utf-8').splitlines()
            for line in lines[:max_boxes]:
                class_id, x_center, y_center, box_width, box_height = map(float, line.split())
                x1 = (x_center - box_width / 2) * width
                y1 = (y_center - box_height / 2) * height
                x2 = (x_center + box_width / 2) * width
                y2 = (y_center + box_height / 2) * height
                draw.rectangle((x1, y1, x2, y2), outline='red', width=max(2, width // 1000))

            return image


        samples = [
            ('HRPlanesv2 train', hr_prepared / 'images' / 'train'),
            ('CORS-ADD-HBB train', cors_prepared / 'images' / 'train'),
            ('CORS-ADD-HBB internal val', cors_prepared / 'images' / 'val'),
            ('CORS-ADD-HBB held-out test', cors_prepared / 'images' / 'test'),
        ]

        figure, axes = plt.subplots(2, 2, figsize=(14, 10))
        for axis, (title, image_folder) in zip(axes.ravel(), samples):
            image_path = list_images(image_folder)[0]
            label_path = image_folder.parent.parent / 'labels' / image_folder.name / f'{image_path.stem}.txt'
            checked_image = draw_yolo_boxes(image_path, label_path)
            axis.imshow(checked_image)
            axis.set_title(title)
            axis.axis('off')

        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## Next step

        Keep this Colab runtime active and open `01_Sequential_Transfer_Learning_YOLOv5x.ipynb`. The next notebook expects these files:

        - `/content/rs_pro_work/configs/hrplanesv2.yaml`
        - `/content/rs_pro_work/configs/cors_add_hbb.yaml`

        If the runtime resets, rerun this notebook before resuming training.
        """
    ),
]


notebook_01 = [
    markdown(
        """
        # 01 - Sequential transfer learning with YOLOv5x

        This notebook implements the finalized pipeline:

        ```text
        COCO-pretrained YOLOv5x
        -> HRPlanesv2 fine-tuning
        -> HRPlanesv2 validation selects Stage-1 best.pt
        -> HRPlanesv2 held-out test
        -> CORS-ADD-HBB fine-tuning from Stage-1 best.pt
        -> internal validation selects final best.pt
        -> official val2017, reserved as held-out final test
        ```

        Run notebook 00 first in the same Colab runtime. Formal training cells are guarded by explicit Boolean switches so that opening or running setup cells cannot accidentally start a long GPU job.
        """
    ),
    code(
        """
        from google.colab import drive

        drive.mount('/content/drive')
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import os
        import platform
        import shutil
        import subprocess
        import sys

        import pandas as pd
        import torch

        PROJECT_ROOT = Path('/content/drive/MyDrive/RS_PRO')
        WORK_ROOT = Path('/content/rs_pro_work')
        ARTIFACT_ROOT = PROJECT_ROOT / 'artifacts'
        RUNS_ROOT = ARTIFACT_ROOT / 'runs'
        EVALUATION_ROOT = ARTIFACT_ROOT / 'evaluation'

        YOLOV5_ROOT = Path('/content/yolov5')
        YOLOV5_REF = 'v7.0'

        HR_YAML = WORK_ROOT / 'configs' / 'hrplanesv2.yaml'
        CORS_YAML = WORK_ROOT / 'configs' / 'cors_add_hbb.yaml'

        IMAGE_SIZE = 640
        BATCH_SIZE = 4
        EPOCHS_STAGE1 = 100
        EPOCHS_STAGE2 = 100
        PATIENCE = 20
        WORKERS = 2
        RANDOM_SEED = 42

        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)

        for required in [PROJECT_ROOT, HR_YAML, CORS_YAML]:
            if not required.exists():
                raise FileNotFoundError(
                    f'Missing {required}. Run notebook 00 in this Colab runtime first.'
                )

        print('GPU available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('GPU:', torch.cuda.get_device_name(0))
        print('PyTorch:', torch.__version__)
        print('Python:', sys.version)
        """
    ),
    markdown("## Install the official YOLOv5 implementation"),
    code(
        """
        if not YOLOV5_ROOT.exists():
            subprocess.run(
                [
                    'git', 'clone', '--depth', '1', '--branch', YOLOV5_REF,
                    'https://github.com/ultralytics/yolov5.git',
                    str(YOLOV5_ROOT),
                ],
                check=True,
            )

        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '-r', str(YOLOV5_ROOT / 'requirements.txt')],
            check=True,
        )

        commit = subprocess.run(
            ['git', '-C', str(YOLOV5_ROOT), 'rev-parse', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        environment = {
            'yolov5_ref': YOLOV5_REF,
            'yolov5_commit': commit,
            'python': sys.version,
            'pytorch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'image_size': IMAGE_SIZE,
            'batch_size': BATCH_SIZE,
            'random_seed': RANDOM_SEED,
        }

        (ARTIFACT_ROOT / 'environment.json').write_text(
            json.dumps(environment, indent=2), encoding='utf-8'
        )
        print(json.dumps(environment, indent=2))
        """
    ),
    markdown(
        """
        ## Stage 1 - Fine-tune COCO-pretrained YOLOv5x on HRPlanesv2

        `train.py` evaluates the HRPlanesv2 validation split after every epoch and writes `best.pt` based on validation fitness. The test split is not used by this command.
        """
    ),
    code(
        """
        RUN_STAGE1_TRAINING = False  # Change to True when ready to start formal training.

        stage1_name = 'stage1_hrplanesv2_yolov5x'
        stage1_dir = RUNS_ROOT / stage1_name

        stage1_command = [
            sys.executable,
            str(YOLOV5_ROOT / 'train.py'),
            '--weights', 'yolov5x.pt',
            '--data', str(HR_YAML),
            '--img', str(IMAGE_SIZE),
            '--batch-size', str(BATCH_SIZE),
            '--epochs', str(EPOCHS_STAGE1),
            '--patience', str(PATIENCE),
            '--workers', str(WORKERS),
            '--seed', str(RANDOM_SEED),
            '--save-period', '10',
            '--project', str(RUNS_ROOT),
            '--name', stage1_name,
            '--exist-ok',
        ]

        print(' '.join(stage1_command))
        if RUN_STAGE1_TRAINING:
            subprocess.run(stage1_command, cwd=YOLOV5_ROOT, check=True)
        else:
            print('Stage 1 was not started. Set RUN_STAGE1_TRAINING = True when ready.')
        """
    ),
    markdown("## Inspect Stage-1 training history"),
    code(
        """
        stage1_results = stage1_dir / 'results.csv'
        if stage1_results.exists():
            stage1_history = pd.read_csv(stage1_results)
            display(stage1_history.tail())
        else:
            print('No Stage-1 results yet:', stage1_results)
        """
    ),
    markdown(
        """
        ## Evaluate the selected Stage-1 model on the HRPlanesv2 test split

        Run this only after Stage 1 has completed. The resulting metrics describe the HRPlanesv2 stage and must not be used to select the checkpoint.
        """
    ),
    code(
        """
        RUN_HR_TEST = False  # Change to True only after Stage 1 is complete.

        HR_BEST = stage1_dir / 'weights' / 'best.pt'

        hr_test_command = [
            sys.executable,
            str(YOLOV5_ROOT / 'val.py'),
            '--weights', str(HR_BEST),
            '--data', str(HR_YAML),
            '--task', 'test',
            '--img', str(IMAGE_SIZE),
            '--batch-size', str(BATCH_SIZE),
            '--workers', str(WORKERS),
            '--save-txt',
            '--save-conf',
            '--save-json',
            '--project', str(EVALUATION_ROOT),
            '--name', 'hrplanesv2_stage1_test',
            '--exist-ok',
        ]

        print(' '.join(hr_test_command))
        if RUN_HR_TEST:
            if not HR_BEST.exists():
                raise FileNotFoundError(f'Stage-1 best.pt not found: {HR_BEST}')
            subprocess.run(hr_test_command, cwd=YOLOV5_ROOT, check=True)
        else:
            print('HRPlanesv2 test was not started.')
        """
    ),
    markdown(
        """
        ## Stage 2 - Fine-tune the Stage-1 model on CORS-ADD-HBB

        This stage must initialize from the HRPlanesv2 `best.pt`, not from the original COCO weight. The internal CORS validation split selects the final checkpoint. The official `val2017` images remain untouched during training.
        """
    ),
    code(
        """
        RUN_STAGE2_TRAINING = False  # Change to True after Stage 1 is complete.

        stage2_name = 'stage2_cors_add_hbb_yolov5x'
        stage2_dir = RUNS_ROOT / stage2_name

        stage2_command = [
            sys.executable,
            str(YOLOV5_ROOT / 'train.py'),
            '--weights', str(HR_BEST),
            '--data', str(CORS_YAML),
            '--img', str(IMAGE_SIZE),
            '--batch-size', str(BATCH_SIZE),
            '--epochs', str(EPOCHS_STAGE2),
            '--patience', str(PATIENCE),
            '--workers', str(WORKERS),
            '--seed', str(RANDOM_SEED),
            '--save-period', '10',
            '--project', str(RUNS_ROOT),
            '--name', stage2_name,
            '--exist-ok',
        ]

        print(' '.join(stage2_command))
        if RUN_STAGE2_TRAINING:
            if not HR_BEST.exists():
                raise FileNotFoundError(f'Stage-1 best.pt not found: {HR_BEST}')
            subprocess.run(stage2_command, cwd=YOLOV5_ROOT, check=True)
        else:
            print('Stage 2 was not started. Set RUN_STAGE2_TRAINING = True when ready.')
        """
    ),
    markdown("## Inspect Stage-2 training history"),
    code(
        """
        stage2_results = stage2_dir / 'results.csv'
        if stage2_results.exists():
            stage2_history = pd.read_csv(stage2_results)
            display(stage2_history.tail())
        else:
            print('No Stage-2 results yet:', stage2_results)
        """
    ),
    markdown(
        """
        ## Final held-out CORS-ADD-HBB test

        The CORS YAML maps the original `val2017` release folder to `test`. Run this cell once after the final Stage-2 checkpoint has been selected. Do not use these results to change training settings or choose a different checkpoint.
        """
    ),
    code(
        """
        RUN_FINAL_CORS_TEST = False  # Change to True only after Stage 2 is complete.

        CORS_BEST = stage2_dir / 'weights' / 'best.pt'

        cors_test_command = [
            sys.executable,
            str(YOLOV5_ROOT / 'val.py'),
            '--weights', str(CORS_BEST),
            '--data', str(CORS_YAML),
            '--task', 'test',
            '--img', str(IMAGE_SIZE),
            '--batch-size', str(BATCH_SIZE),
            '--workers', str(WORKERS),
            '--save-txt',
            '--save-conf',
            '--save-json',
            '--project', str(EVALUATION_ROOT),
            '--name', 'cors_add_hbb_final_test',
            '--exist-ok',
        ]

        print(' '.join(cors_test_command))
        if RUN_FINAL_CORS_TEST:
            if not CORS_BEST.exists():
                raise FileNotFoundError(f'Final best.pt not found: {CORS_BEST}')
            subprocess.run(cors_test_command, cwd=YOLOV5_ROOT, check=True)
        else:
            print('Final CORS-ADD-HBB test was not started.')
        """
    ),
    markdown("## Resume interrupted training"),
    code(
        """
        # Example only. Set the path to the relevant last.pt and change RUN_RESUME to True.
        RUN_RESUME = False
        RESUME_CHECKPOINT = stage2_dir / 'weights' / 'last.pt'

        resume_command = [
            sys.executable,
            str(YOLOV5_ROOT / 'train.py'),
            '--resume', str(RESUME_CHECKPOINT),
        ]

        print(' '.join(resume_command))
        if RUN_RESUME:
            if not RESUME_CHECKPOINT.exists():
                raise FileNotFoundError(RESUME_CHECKPOINT)
            subprocess.run(resume_command, cwd=YOLOV5_ROOT, check=True)
        """
    ),
    markdown(
        """
        ## Outputs

        Important outputs remain in Google Drive:

        - `RS_PRO/artifacts/runs/stage1_hrplanesv2_yolov5x/weights/best.pt`
        - `RS_PRO/artifacts/runs/stage2_cors_add_hbb_yolov5x/weights/best.pt`
        - `RS_PRO/artifacts/evaluation/hrplanesv2_stage1_test/`
        - `RS_PRO/artifacts/evaluation/cors_add_hbb_final_test/`

        Continue with notebook 02 after the final Stage-2 model is available.
        """
    ),
]


notebook_02 = [
    markdown(
        """
        # 02 - Qualitative external inference on Houston airport imagery

        This notebook applies the final CORS-ADD-HBB model to independently sourced Houston airport imagery. It tiles large remote-sensing images, runs YOLOv5x inference, maps detections back to original image coordinates, removes duplicates in overlapping tiles, and saves full-image visualizations.

        Without complete Houston ground-truth annotations, these results are qualitative and must not be reported as Precision, Recall, F1, or mAP.
        """
    ),
    code(
        """
        from google.colab import drive

        drive.mount('/content/drive')
        """
    ),
    code(
        """
        from pathlib import Path
        import csv
        import json
        import math
        import re
        import shutil
        import subprocess
        import sys

        import matplotlib.pyplot as plt
        import pandas as pd
        import torch
        from PIL import Image, ImageDraw, ImageFont
        from torchvision.ops import nms

        PROJECT_ROOT = Path('/content/drive/MyDrive/RS_PRO')
        ARTIFACT_ROOT = PROJECT_ROOT / 'artifacts'
        YOLOV5_ROOT = Path('/content/yolov5')

        FINAL_WEIGHTS = (
            ARTIFACT_ROOT
            / 'runs'
            / 'stage2_cors_add_hbb_yolov5x'
            / 'weights'
            / 'best.pt'
        )

        HOUSTON_INPUT = PROJECT_ROOT / 'Houston_Imagery' / 'input'
        HOUSTON_OUTPUT = ARTIFACT_ROOT / 'houston_external_inference'

        LOCAL_TILE_ROOT = Path('/content/houston_tiles')
        LOCAL_DETECTION_ROOT = Path('/content/houston_detection')

        TILE_SIZE = 640
        TILE_OVERLAP = 0.25
        CONFIDENCE_THRESHOLD = 0.25
        DETECTION_IOU_THRESHOLD = 0.45
        GLOBAL_NMS_IOU_THRESHOLD = 0.40

        HOUSTON_INPUT.mkdir(parents=True, exist_ok=True)
        HOUSTON_OUTPUT.mkdir(parents=True, exist_ok=True)

        print('Place independently sourced Houston images in:', HOUSTON_INPUT)
        print('Final weights:', FINAL_WEIGHTS)
        """
    ),
    markdown(
        """
        ## Record image provenance

        Before inference, create or complete `Houston_Imagery/metadata.csv` with one row per source image. Record the provider, acquisition date, spatial resolution, airport, and confirmation that the image does not duplicate or overlap CORS-ADD imagery.
        """
    ),
    code(
        """
        metadata_path = PROJECT_ROOT / 'Houston_Imagery' / 'metadata.csv'

        if not metadata_path.exists():
            pd.DataFrame(
                columns=[
                    'filename',
                    'airport',
                    'source_or_provider',
                    'source_url',
                    'acquisition_date',
                    'spatial_resolution',
                    'license_or_terms',
                    'non_overlap_with_cors_add_confirmed',
                    'notes',
                ]
            ).to_csv(metadata_path, index=False)
            print('Created metadata template:', metadata_path)
        else:
            display(pd.read_csv(metadata_path))
        """
    ),
    markdown("## Install YOLOv5 if this is a new Colab runtime"),
    code(
        """
        YOLOV5_REF = 'v7.0'

        if not YOLOV5_ROOT.exists():
            subprocess.run(
                [
                    'git', 'clone', '--depth', '1', '--branch', YOLOV5_REF,
                    'https://github.com/ultralytics/yolov5.git',
                    str(YOLOV5_ROOT),
                ],
                check=True,
            )

        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '-r', str(YOLOV5_ROOT / 'requirements.txt')],
            check=True,
        )

        if not FINAL_WEIGHTS.exists():
            raise FileNotFoundError(
                f'Final Stage-2 weights are missing: {FINAL_WEIGHTS}. Run notebook 01 first.'
            )
        """
    ),
    markdown("## Tile the large Houston images"),
    code(
        """
        IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


        def slugify(value: str) -> str:
            value = re.sub(r'[^A-Za-z0-9_-]+', '_', value)
            return value.strip('_') or 'image'


        def axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
            if length <= tile_size:
                return [0]
            positions = list(range(0, length - tile_size + 1, stride))
            final_position = length - tile_size
            if positions[-1] != final_position:
                positions.append(final_position)
            return positions


        def tile_images(
            source_paths: list[Path],
            tile_root: Path,
            tile_size: int,
            overlap: float,
        ) -> pd.DataFrame:
            if tile_root.exists():
                shutil.rmtree(tile_root)
            tile_root.mkdir(parents=True, exist_ok=True)

            stride = max(1, round(tile_size * (1 - overlap)))
            manifest_rows = []

            for source_index, source_path in enumerate(source_paths):
                with Image.open(source_path) as source_image:
                    source_image = source_image.convert('RGB')
                    width, height = source_image.size

                    x_positions = axis_positions(width, tile_size, stride)
                    y_positions = axis_positions(height, tile_size, stride)

                    for y0 in y_positions:
                        for x0 in x_positions:
                            x1 = min(x0 + tile_size, width)
                            y1 = min(y0 + tile_size, height)
                            tile = source_image.crop((x0, y0, x1, y1))

                            tile_name = (
                                f'{source_index:03d}_{slugify(source_path.stem)}'
                                f'__x{x0}__y{y0}.png'
                            )
                            tile_path = tile_root / tile_name
                            tile.save(tile_path)

                            manifest_rows.append({
                                'tile_name': tile_name,
                                'tile_path': str(tile_path),
                                'source_path': str(source_path),
                                'source_name': source_path.name,
                                'x0': x0,
                                'y0': y0,
                                'tile_width': x1 - x0,
                                'tile_height': y1 - y0,
                                'source_width': width,
                                'source_height': height,
                            })

            return pd.DataFrame(manifest_rows)


        houston_images = sorted(
            path for path in HOUSTON_INPUT.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not houston_images:
            raise FileNotFoundError(
                f'No Houston images found in {HOUSTON_INPUT}. Add images before continuing.'
            )

        tile_manifest = tile_images(
            houston_images,
            LOCAL_TILE_ROOT,
            tile_size=TILE_SIZE,
            overlap=TILE_OVERLAP,
        )

        tile_manifest_path = HOUSTON_OUTPUT / 'tile_manifest.csv'
        tile_manifest.to_csv(tile_manifest_path, index=False)

        print(f'Source images: {len(houston_images)}')
        print(f'Tiles: {len(tile_manifest)}')
        print('Manifest:', tile_manifest_path)
        display(tile_manifest.head())
        """
    ),
    markdown("## Run tile-level inference"),
    code(
        """
        RUN_HOUSTON_INFERENCE = False  # Change to True after checking the source images and metadata.

        detection_command = [
            sys.executable,
            str(YOLOV5_ROOT / 'detect.py'),
            '--weights', str(FINAL_WEIGHTS),
            '--source', str(LOCAL_TILE_ROOT),
            '--img', str(TILE_SIZE),
            '--conf-thres', str(CONFIDENCE_THRESHOLD),
            '--iou-thres', str(DETECTION_IOU_THRESHOLD),
            '--save-txt',
            '--save-conf',
            '--project', str(LOCAL_DETECTION_ROOT.parent),
            '--name', LOCAL_DETECTION_ROOT.name,
            '--exist-ok',
        ]

        print(' '.join(detection_command))
        if RUN_HOUSTON_INFERENCE:
            if LOCAL_DETECTION_ROOT.exists():
                shutil.rmtree(LOCAL_DETECTION_ROOT)
            subprocess.run(detection_command, cwd=YOLOV5_ROOT, check=True)
        else:
            print('Houston inference was not started.')
        """
    ),
    markdown("## Map tile detections back to full images and remove duplicates"),
    code(
        """
        def yolo_detection_to_global(
            parts: list[str],
            tile_row: pd.Series,
        ) -> dict:
            class_id, x_center, y_center, width, height, confidence = map(float, parts)

            tile_width = float(tile_row['tile_width'])
            tile_height = float(tile_row['tile_height'])

            x1 = (x_center - width / 2) * tile_width + float(tile_row['x0'])
            y1 = (y_center - height / 2) * tile_height + float(tile_row['y0'])
            x2 = (x_center + width / 2) * tile_width + float(tile_row['x0'])
            y2 = (y_center + height / 2) * tile_height + float(tile_row['y0'])

            return {
                'source_path': tile_row['source_path'],
                'source_name': tile_row['source_name'],
                'class_id': int(class_id),
                'confidence': confidence,
                'x1': max(0.0, x1),
                'y1': max(0.0, y1),
                'x2': min(float(tile_row['source_width']), x2),
                'y2': min(float(tile_row['source_height']), y2),
                'tile_name': tile_row['tile_name'],
            }


        def collect_global_detections(
            manifest: pd.DataFrame,
            labels_root: Path,
        ) -> pd.DataFrame:
            columns = [
                'source_path', 'source_name', 'class_id', 'confidence',
                'x1', 'y1', 'x2', 'y2', 'tile_name',
            ]
            detections = []
            for _, row in manifest.iterrows():
                label_path = labels_root / f"{Path(row['tile_name']).stem}.txt"
                if not label_path.exists():
                    continue

                for line in label_path.read_text(encoding='utf-8').splitlines():
                    parts = line.split()
                    if len(parts) != 6:
                        continue
                    detections.append(yolo_detection_to_global(parts, row))

            return pd.DataFrame(detections, columns=columns)


        def apply_global_nms(
            detections: pd.DataFrame,
            iou_threshold: float,
        ) -> pd.DataFrame:
            if detections.empty:
                return detections.copy()

            retained_frames = []
            for source_path, frame in detections.groupby('source_path', sort=True):
                boxes = torch.tensor(
                    frame[['x1', 'y1', 'x2', 'y2']].to_numpy(),
                    dtype=torch.float32,
                )
                scores = torch.tensor(frame['confidence'].to_numpy(), dtype=torch.float32)
                keep_indices = nms(boxes, scores, iou_threshold).cpu().numpy()
                retained_frames.append(frame.iloc[keep_indices])

            return pd.concat(retained_frames, ignore_index=True)


        labels_root = LOCAL_DETECTION_ROOT / 'labels'
        if not labels_root.exists():
            raise FileNotFoundError(
                f'Detection labels not found: {labels_root}. Run the inference cell first.'
            )

        raw_detections = collect_global_detections(tile_manifest, labels_root)
        final_detections = apply_global_nms(
            raw_detections,
            iou_threshold=GLOBAL_NMS_IOU_THRESHOLD,
        )

        raw_detections.to_csv(HOUSTON_OUTPUT / 'raw_tile_detections.csv', index=False)
        final_detections.to_csv(HOUSTON_OUTPUT / 'final_global_detections.csv', index=False)

        print('Raw tile detections:', len(raw_detections))
        print('Detections after global NMS:', len(final_detections))
        display(final_detections.head())
        """
    ),
    markdown("## Save full-image detection visualizations"),
    code(
        """
        visualization_root = HOUSTON_OUTPUT / 'visualizations'
        visualization_root.mkdir(parents=True, exist_ok=True)

        for source_path_string in sorted(final_detections['source_path'].unique()):
            source_path = Path(source_path_string)
            with Image.open(source_path) as image:
                image = image.convert('RGB')
                draw = ImageDraw.Draw(image)

                frame = final_detections[
                    final_detections['source_path'] == source_path_string
                ].sort_values('confidence', ascending=False)

                line_width = max(2, min(image.size) // 1000)
                for _, detection in frame.iterrows():
                    box = (
                        detection['x1'],
                        detection['y1'],
                        detection['x2'],
                        detection['y2'],
                    )
                    draw.rectangle(box, outline='red', width=line_width)
                    label = f"aircraft {detection['confidence']:.2f}"
                    draw.text((detection['x1'], max(0, detection['y1'] - 14)), label, fill='red')

                output_path = visualization_root / f'{source_path.stem}_detections.jpg'
                image.save(output_path, quality=95)
                print('Saved:', output_path)
        """
    ),
    markdown("## Preview results"),
    code(
        """
        visualization_paths = sorted(visualization_root.glob('*.jpg'))

        if not visualization_paths:
            print('No visualizations were produced.')
        else:
            figure, axes = plt.subplots(
                len(visualization_paths),
                1,
                figsize=(16, 8 * len(visualization_paths)),
            )
            if len(visualization_paths) == 1:
                axes = [axes]

            for axis, visualization_path in zip(axes, visualization_paths):
                image = Image.open(visualization_path)
                axis.imshow(image)
                axis.set_title(visualization_path.name)
                axis.axis('off')

            plt.tight_layout()
            plt.show()
        """
    ),
    markdown(
        """
        ## Interpretation rule

        Report these Houston results as qualitative external inference unless complete ground-truth annotations are created. Discuss representative correct detections, apparent false positives, apparent false negatives, dense parking areas, small aircraft, and background complexity. Do not report mAP, Precision, Recall, or F1 from unannotated Houston imagery.
        """
    ),
]


write_notebook('00_Data_QA_and_Splits.ipynb', notebook_00)
write_notebook('01_Sequential_Transfer_Learning_YOLOv5x.ipynb', notebook_01)
write_notebook('02_Houston_External_Inference.ipynb', notebook_02)

