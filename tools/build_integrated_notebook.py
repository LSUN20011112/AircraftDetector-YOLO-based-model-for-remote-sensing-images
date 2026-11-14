from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAME = os.environ.get(
    "INTEGRATED_NOTEBOOK_OUTPUT",
    "03_Full_Sequential_Transfer_Learning_Pipeline.ipynb",
)


def load_notebook(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


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


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def replace_code_cell(cells: list[dict], needle: str, replacement: dict) -> None:
    matches = [index for index, cell in enumerate(cells) if needle in source_text(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell containing {needle!r}; found {len(matches)}")
    cells[matches[0]] = replacement


def remove_cells_containing(cells: list[dict], needles: list[str]) -> list[dict]:
    return [
        cell for cell in cells
        if not any(needle in source_text(cell) for needle in needles)
    ]


notebook_00 = load_notebook("00_Data_QA_and_Splits.ipynb")
notebook_01 = load_notebook("01_Sequential_Transfer_Learning_YOLOv5x.ipynb")
notebook_02 = load_notebook("02_Houston_External_Inference.ipynb")

cells_00 = copy.deepcopy(notebook_00["cells"])
cells_01 = copy.deepcopy(notebook_01["cells"])
cells_02 = copy.deepcopy(notebook_02["cells"])

# The integrated notebook mounts Drive only once and supplies its own section titles.
cells_00 = remove_cells_containing(
    cells_00,
    ["# 00 - Data quality control", "from google.colab import drive"],
)
cells_01 = remove_cells_containing(
    cells_01,
    [
        "# 01 - Sequential transfer learning",
        "from google.colab import drive",
        "## Resume interrupted training",
        "# Example only. Set the path to the relevant last.pt",
    ],
)
cells_02 = remove_cells_containing(
    cells_02,
    ["# 02 - Qualitative external inference", "from google.colab import drive"],
)

# Current Colab runtimes use PyTorch versions whose checkpoint-loading defaults
# are incompatible with the old YOLOv5 v7.0 snapshot. Use the maintained
# official branch, while recording the exact resolved commit in environment.json.
for notebook_cells in (cells_01, cells_02):
    for notebook_cell in notebook_cells:
        if notebook_cell.get("cell_type") == "code":
            notebook_cell["source"] = source_text(notebook_cell).replace(
                "YOLOV5_REF = 'v7.0'",
                "YOLOV5_REF = 'master'",
            )

replace_code_cell(
    cells_00,
    "# Change only this line if RS_PRO is stored elsewhere in Google Drive.",
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

        PROJECT_ROOT = Path('/content/drive/MyDrive/RS_PRO')
        WORK_ROOT = Path('/content/rs_pro_work')
        ARTIFACT_ROOT = PROJECT_ROOT / 'artifacts'

        RANDOM_SEED = 42
        CORS_INTERNAL_VAL_FRACTION = 0.15
        STAGE_MODE = 'copy'
        IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

        assert PROJECT_ROOT.exists(), f'Project folder not found: {PROJECT_ROOT}'
        assert str(WORK_ROOT).startswith('/content/'), 'WORK_ROOT must remain under /content.'

        required_prepared_files = [
            WORK_ROOT / 'configs' / 'hrplanesv2.yaml',
            WORK_ROOT / 'configs' / 'cors_add_hbb.yaml',
        ]
        REBUILD_PREPARED = not all(path.exists() for path in required_prepared_files)

        if REBUILD_PREPARED and WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)

        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_ROOT / 'splits').mkdir(parents=True, exist_ok=True)
        (ARTIFACT_ROOT / 'data_qa').mkdir(parents=True, exist_ok=True)

        print('PROJECT_ROOT:', PROJECT_ROOT)
        print('WORK_ROOT:', WORK_ROOT)
        print('STAGE_MODE:', STAGE_MODE)
        print('Rebuilding temporary prepared data:', REBUILD_PREPARED)
        """
    ),
)

replace_code_cell(
    cells_01,
    "if not YOLOV5_ROOT.exists():",
    code(
        """
        from collections import deque


        def run_streaming(command: list[str], cwd: Path, tail_lines: int = 120) -> None:
            # Run a long command with live logs and retain a useful error tail.
            print(' '.join(map(str, command)))
            recent_output = deque(maxlen=tail_lines)
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end='')
                recent_output.append(line.rstrip())
            return_code = process.wait()
            if return_code != 0:
                error_tail = '\\n'.join(recent_output)
                raise RuntimeError(
                    f'Command failed with exit code {return_code}. '
                    f'The final {len(recent_output)} log lines were:\\n{error_tail}'
                )


        # A previous attempt may have left the old v7.0 tag in /content.
        # Replace it automatically with the maintained official branch.
        if YOLOV5_ROOT.exists():
            current_branch = subprocess.run(
                ['git', '-C', str(YOLOV5_ROOT), 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if current_branch != YOLOV5_REF:
                print(
                    f'Removing incompatible YOLOv5 checkout '
                    f'(current={current_branch or "unknown"}, required={YOLOV5_REF}).'
                )
                shutil.rmtree(YOLOV5_ROOT)

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
        print('YOLOv5 compatibility checkout is ready:', commit)
        """
    ),
)

replace_code_cell(
    cells_01,
    "RUN_STAGE1_TRAINING = False",
    code(
        """
        stage1_name = 'stage1_hrplanesv2_yolov5x'
        stage1_dir = RUNS_ROOT / stage1_name
        HR_BEST = stage1_dir / 'weights' / 'best.pt'
        HR_LAST = stage1_dir / 'weights' / 'last.pt'
        STAGE1_COMPLETE = stage1_dir / 'stage_complete.txt'

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

        if RUN_FULL_PIPELINE:
            if STAGE1_COMPLETE.exists() and HR_BEST.exists():
                print('Stage 1 is already complete; skipping:', HR_BEST)
            else:
                if AUTO_RESUME and HR_LAST.exists():
                    command = [
                        sys.executable,
                        str(YOLOV5_ROOT / 'train.py'),
                        '--resume', str(HR_LAST),
                    ]
                    print('Resuming Stage 1 from:', HR_LAST)
                else:
                    command = stage1_command
                    print('Starting Stage 1 from COCO-pretrained yolov5x.pt')

                run_streaming(command, cwd=YOLOV5_ROOT)
                if not HR_BEST.exists():
                    raise FileNotFoundError(f'Stage-1 best.pt was not produced: {HR_BEST}')
                STAGE1_COMPLETE.write_text('complete\\n', encoding='utf-8')
                print('Stage 1 complete:', HR_BEST)
        else:
            print('Pipeline execution is disabled. Set RUN_FULL_PIPELINE = True near the top.')
        """
    ),
)

replace_code_cell(
    cells_01,
    "RUN_HR_TEST = False",
    code(
        """
        hr_test_name = 'hrplanesv2_stage1_test'
        hr_test_dir = EVALUATION_ROOT / hr_test_name
        HR_TEST_COMPLETE = hr_test_dir / 'evaluation_complete.txt'

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
            '--name', hr_test_name,
            '--exist-ok',
        ]

        if RUN_FULL_PIPELINE:
            if HR_TEST_COMPLETE.exists():
                print('HRPlanesv2 held-out test is already complete; skipping.')
            else:
                if not HR_BEST.exists():
                    raise FileNotFoundError(f'Stage-1 best.pt not found: {HR_BEST}')
                run_streaming(hr_test_command, cwd=YOLOV5_ROOT)
                hr_test_dir.mkdir(parents=True, exist_ok=True)
                HR_TEST_COMPLETE.write_text('complete\\n', encoding='utf-8')
                print('HRPlanesv2 held-out test complete:', hr_test_dir)
        """
    ),
)

replace_code_cell(
    cells_01,
    "RUN_STAGE2_TRAINING = False",
    code(
        """
        stage2_name = 'stage2_cors_add_hbb_yolov5x'
        stage2_dir = RUNS_ROOT / stage2_name
        CORS_BEST = stage2_dir / 'weights' / 'best.pt'
        CORS_LAST = stage2_dir / 'weights' / 'last.pt'
        STAGE2_COMPLETE = stage2_dir / 'stage_complete.txt'

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

        if RUN_FULL_PIPELINE:
            if STAGE2_COMPLETE.exists() and CORS_BEST.exists():
                print('Stage 2 is already complete; skipping:', CORS_BEST)
            else:
                if AUTO_RESUME and CORS_LAST.exists():
                    command = [
                        sys.executable,
                        str(YOLOV5_ROOT / 'train.py'),
                        '--resume', str(CORS_LAST),
                    ]
                    print('Resuming Stage 2 from:', CORS_LAST)
                else:
                    if not HR_BEST.exists():
                        raise FileNotFoundError(f'Stage-1 best.pt not found: {HR_BEST}')
                    command = stage2_command
                    print('Starting Stage 2 from the Stage-1 HRPlanesv2 checkpoint')

                run_streaming(command, cwd=YOLOV5_ROOT)
                if not CORS_BEST.exists():
                    raise FileNotFoundError(f'Stage-2 best.pt was not produced: {CORS_BEST}')
                STAGE2_COMPLETE.write_text('complete\\n', encoding='utf-8')
                print('Stage 2 complete:', CORS_BEST)
        """
    ),
)

replace_code_cell(
    cells_01,
    "RUN_FINAL_CORS_TEST = False",
    code(
        """
        cors_test_name = 'cors_add_hbb_final_test'
        cors_test_dir = EVALUATION_ROOT / cors_test_name
        CORS_TEST_COMPLETE = cors_test_dir / 'evaluation_complete.txt'

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
            '--name', cors_test_name,
            '--exist-ok',
        ]

        if RUN_FULL_PIPELINE:
            if CORS_TEST_COMPLETE.exists():
                print('Final held-out CORS test is already complete; skipping.')
            else:
                if not CORS_BEST.exists():
                    raise FileNotFoundError(f'Final best.pt not found: {CORS_BEST}')
                run_streaming(cors_test_command, cwd=YOLOV5_ROOT)
                cors_test_dir.mkdir(parents=True, exist_ok=True)
                CORS_TEST_COMPLETE.write_text('complete\\n', encoding='utf-8')
                print('Final held-out CORS test complete:', cors_test_dir)
        """
    ),
)

replace_code_cell(
    cells_02,
    "RUN_HOUSTON_INFERENCE = False",
    code(
        """
        RUN_HOUSTON_INFERENCE = RUN_FULL_PIPELINE

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
            run_streaming(detection_command, cwd=YOLOV5_ROOT)
        else:
            raise RuntimeError('Set RUN_FULL_PIPELINE = True near the top before Run all.')
        """
    ),
)

intro_cells = [
    markdown(
        """
        # Full sequential transfer-learning pipeline: YOLOv5x aircraft detection

        This integrated Colab notebook implements the complete course-project workflow in one file:

        ```text
        COCO-pretrained YOLOv5x
        -> HRPlanesv2 training/validation
        -> HRPlanesv2 held-out test
        -> CORS-ADD-HBB training/internal validation
        -> official CORS val2017 held-out final test
        -> qualitative inference on independently sourced Houston imagery
        ```

        All persistent checkpoints, evaluations, manifests, and visualizations are written to `MyDrive/RS_PRO/artifacts`. Temporary prepared data under `/content` is rebuilt automatically after a Colab runtime reset. Interrupted Stage-1 or Stage-2 training resumes from the corresponding `last.pt` saved in Drive.
        """
    ),
    markdown(
        """
        ## Before running

        1. In Colab choose a GPU runtime.
        2. Confirm the project is located at `MyDrive/RS_PRO`.
        3. Review the settings below. Leave `RUN_FULL_PIPELINE = True` to run the entire workflow from top to bottom.
        4. Use **Runtime -> Run all**.

        If GPU memory is insufficient, change `BATCH_SIZE_OVERRIDE` from `4` to `2` or `1` and run again. Completed stages are skipped, while an interrupted training stage resumes from `last.pt`.
        """
    ),
    code(
        """
        from google.colab import drive

        drive.mount('/content/drive')

        # Master controls for the integrated notebook.
        RUN_FULL_PIPELINE = True
        AUTO_RESUME = True
        BATCH_SIZE_OVERRIDE = 4
        """
    ),
]

section_00 = [markdown("# Part 1 - Data quality control and reproducible splits")]
section_01 = [markdown("# Part 2 - Sequential YOLOv5x transfer learning and evaluation")]
section_02 = [markdown("# Part 3 - Houston external qualitative inference")]

# Apply the single top-level batch-size setting after notebook 01 defines its defaults.
batch_override_cell = code(
    """
    BATCH_SIZE = BATCH_SIZE_OVERRIDE
    print('Effective training/evaluation batch size:', BATCH_SIZE)
    """
)
setup_index = next(
    index for index, cell in enumerate(cells_01)
    if "BATCH_SIZE = 4" in source_text(cell)
)
cells_01.insert(setup_index + 1, batch_override_cell)

all_cells = (
    intro_cells
    + section_00
    + cells_00
    + section_01
    + cells_01
    + section_02
    + cells_02
)

output = {
    "cells": all_cells,
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

output_path = ROOT / OUTPUT_NAME
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(output, indent=1, ensure_ascii=False),
    encoding="utf-8",
)
print(f"Wrote {output_path}")
