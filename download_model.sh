#!/usr/bin/env bash
set -e

MODEL_DIR="${MODEL_DIR:-/opt/render/project/data/models}"
MODEL_PATH="$MODEL_DIR/clip_vit_b16_finetuned.pth"
FILE_ID="${GDRIVE_MODEL_FILE_ID}"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_PATH" ]; then
  echo "Model already exists at $MODEL_PATH"
  exit 0
fi

echo "Downloading model from Google Drive..."
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies \
  --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILE_ID}" -O- \
  | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')

wget --load-cookies /tmp/cookies.txt \
  "https://docs.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
  -O "$MODEL_PATH"

rm -f /tmp/cookies.txt
echo "Model downloaded to $MODEL_PATH"