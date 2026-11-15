#!/usr/bin/env bash
# Download the DFL open data (Bassek et al. 2025, CC-BY 4.0; figshare doi 10.6084/m9.figshare.28196177)
# into $DFL_DIR (default ../external_data/dfl) and verify every file against its published MD5.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${DFL_DIR:-$HERE/../external_data/dfl}"
checksum() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }
mkdir -p "$DEST"
cd "$DEST"
fail=0
while read -r url name md5; do
  if [ ! -f "$name" ] || [ "$(checksum "$name")" != "$md5" ]; then
    curl -sS -L --retry 3 -o "$name" "$url"
  fi
  if [ "$(checksum "$name")" = "$md5" ]; then echo "ok   $name"; else echo "BAD  $name"; fail=1; fi
done < "$HERE/calibration/dfl_manifest.txt"
exit $fail
