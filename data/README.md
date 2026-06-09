# Data

Place Waymo Perception TFRecords here:

- `data/raw/waymo_perception/`

The project expects real Waymo Perception segments with FRONT camera images, LiDAR range images, camera calibration, and ego poses.

Waymo data is not committed to git. Use `scripts/download_waymo_perception_subset.sh` after accepting the Waymo Open Dataset terms and authenticating with Google Cloud.

The `.gitkeep` files preserve this directory structure without uploading raw dataset files.
