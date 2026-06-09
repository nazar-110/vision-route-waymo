#!/usr/bin/env bash
set -euo pipefail

SPLIT="training"
NUM_FILES=3
OUTPUT="data/raw/waymo_perception"
BUCKET="${WAYMO_PERCEPTION_GCS_URI:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      SPLIT="$2"; shift 2 ;;
    --num-files)
      NUM_FILES="$2"; shift 2 ;;
    --output)
      OUTPUT="$2"; shift 2 ;;
    --bucket)
      BUCKET="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"; exit 2 ;;
  esac
done

find_cloud_tool() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return 0
  fi
  if command -v "${name}.cmd" >/dev/null 2>&1; then
    command -v "${name}.cmd"
    return 0
  fi
  if command -v cygpath >/dev/null 2>&1 && [ -n "${LOCALAPPDATA:-}" ]; then
    local candidate
    candidate="$(cygpath -u "${LOCALAPPDATA}\\Google\\Cloud SDK\\google-cloud-sdk\\bin\\${name}.cmd")"
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  fi
  local user_candidate="/mnt/c/Users/${USERNAME:-${USER:-}}/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/${name}.cmd"
  if [ -f "$user_candidate" ]; then
    echo "$user_candidate"
    return 0
  fi
  return 1
}

GCLOUD_BIN="$(find_cloud_tool gcloud || true)"
GSUTIL_BIN="$(find_cloud_tool gsutil || true)"

if [[ -z "${CLOUDSDK_CONFIG:-}" && -n "${USER:-}" && -d "/mnt/c/Users/${USER}/AppData/Roaming/gcloud" ]]; then
  if command -v wslpath >/dev/null 2>&1; then
    export CLOUDSDK_CONFIG
    CLOUDSDK_CONFIG="$(wslpath -w "/mnt/c/Users/${USER}/AppData/Roaming/gcloud")"
  fi
fi

if [[ -z "$GCLOUD_BIN" || -z "$GSUTIL_BIN" ]]; then
  echo "ERROR: gcloud/gsutil are required. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install"
  exit 1
fi

ACTIVE_ACCOUNT="$("$GCLOUD_BIN" auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
if [[ -z "$ACTIVE_ACCOUNT" ]]; then
  ADC_FILE=""
  if [[ -n "${USER:-}" && -f "/mnt/c/Users/${USER}/AppData/Roaming/gcloud/application_default_credentials.json" ]]; then
    ADC_FILE="/mnt/c/Users/${USER}/AppData/Roaming/gcloud/application_default_credentials.json"
  elif [[ -n "${HOME:-}" && -f "${HOME}/.config/gcloud/application_default_credentials.json" ]]; then
    ADC_FILE="${HOME}/.config/gcloud/application_default_credentials.json"
  fi
  if [[ -z "$ADC_FILE" ]]; then
    echo "ERROR: gcloud is not authenticated."
    echo "Run:"
    echo "  gcloud auth login"
    echo "  gcloud auth application-default login"
    exit 1
  fi
  echo "gcloud active account was not visible from this shell, but ADC exists at: $ADC_FILE"
else
  echo "gcloud authenticated as: $ACTIVE_ACCOUNT"
fi

if [[ -z "$BUCKET" ]]; then
  echo "No Perception bucket URI was provided."
  echo "Accept Waymo terms and copy the current Perception GCS URI from https://waymo.com/open/download"
  echo "Then run with --bucket gs://CURRENT_WAYMO_PERCEPTION_BUCKET"
  exit 1
fi

mkdir -p "$OUTPUT"
LIST_CMD=("$GSUTIL_BIN" ls -r "${BUCKET}/**")
echo "Discovering TFRecords with: ${LIST_CMD[*]}"
mapfile -t FILES < <("${LIST_CMD[@]}" 2>/tmp/visionroute_perception_gsutil_error.log | grep -Ei "(${SPLIT}).*\.tfrecord|(${SPLIT}).*\.tf_record|\.tfrecord$|\.tf_record$" | head -n "$NUM_FILES" || true)
if [[ "${#FILES[@]}" -eq 0 ]]; then
  echo "ERROR: No Perception TFRecords were listed from $BUCKET"
  echo "Failed command: ${LIST_CMD[*]}"
  cat /tmp/visionroute_perception_gsutil_error.log || true
  echo "Accept Waymo terms, verify authentication, and rerun this script."
  exit 1
fi

for uri in "${FILES[@]}"; do
  echo "$GSUTIL_BIN cp $uri $OUTPUT/"
  "$GSUTIL_BIN" cp "$uri" "$OUTPUT/"
done
echo "Waymo Perception subset downloaded to $OUTPUT"
