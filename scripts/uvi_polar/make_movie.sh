#!/usr/bin/env bash
# Assemble polar remapped frames into an mp4 movie.
set -euo pipefail
FRAME_DIR=${1:?frame dir}
OUT=${2:?output mp4}
ffmpeg -y -framerate 8 -pattern_type glob -i "${FRAME_DIR}/frame_*.png" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  "$OUT"
echo "Wrote $OUT"
