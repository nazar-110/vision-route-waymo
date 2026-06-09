# GitHub Upload Checklist

Use this checklist before pushing VisionRoute to GitHub.

## 1. Verify The Repo

```bash
python -m pytest tests
python scripts/check_environment.py
```

For the real Waymo parser, run the environment check from WSL/Linux after `bash scripts/setup_wsl_waymo.sh`.

## 2. Confirm Large Files Are Ignored

```bash
git status --ignored
```

The following should be ignored:

- `data/raw/`
- `data/processed/`
- `outputs/waymo_multimodal/`
- `outputs/video_gallery/`
- `outputs/predictions/`
- `*.tfrecord`
- `*.pt`
- `*.mp4`
- `__pycache__/`
- `.pytest_cache/`

## 3. Stage Source Files Only

```bash
git add .gitattributes .gitignore LICENSE CITATION.cff README.md pyproject.toml
git add assets configs data docs notebooks outputs scripts src tests
git add requirements.txt requirements-waymo.txt environment.yml setup.sh setup_windows.ps1
git status
```

Do not use `git add -f` on raw data, checkpoints, or generated videos.

## 4. Suggested First Commit

```bash
git commit -m "Initial VisionRoute Waymo camera LiDAR planner"
```

## 5. After Cloning Fresh

```bash
bash scripts/setup_wsl_waymo.sh
source ~/visionroute-py310/bin/activate
python scripts/check_environment.py
bash scripts/download_waymo_perception_subset.sh --split validation --num-files 4 --output data/raw/waymo_perception
bash scripts/train_waymo.sh
bash scripts/render_waymo_videos.sh
```
