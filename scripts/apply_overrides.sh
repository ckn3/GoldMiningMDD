#!/usr/bin/env bash
set -euo pipefail

# Apply local third-party overrides (patches + untracked files) from this repo.
#
# Usage:
#   bash scripts/apply_overrides.sh /path/to/third_party_root
# Example:
#   bash scripts/apply_overrides.sh /deac/csc/yangGrp/cuij/third_party

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/third_party_root"
  exit 2
fi

ROOT="$1"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVR_DIR="${SELF_DIR}/third_party_overrides"

if [[ ! -d "${OVR_DIR}" ]]; then
  echo "Missing overrides dir: ${OVR_DIR}"
  exit 2
fi

echo "Applying overrides from: ${OVR_DIR}"
echo "Target third_party root: ${ROOT}"

for d in "${OVR_DIR}"/*; do
  [[ -d "${d}" ]] || continue
  name="$(basename "${d}")"
  target="${ROOT}/${name}"
  if [[ ! -d "${target}" ]]; then
    echo "[SKIP] ${name}: target repo not found at ${target}"
    continue
  fi

  patch_file="${d}/${name}.patch"
  if [[ -s "${patch_file}" ]]; then
    echo "[PATCH] ${name}"
    git -C "${target}" apply --reject --whitespace=fix "${patch_file}" || {
      echo "[WARN] patch apply had conflicts for ${name}; inspect .rej files."
    }
  fi

  untracked="${d}/untracked"
  if [[ -d "${untracked}" ]]; then
    echo "[COPY] ${name} untracked files"
    rsync -a "${untracked}/" "${target}/"
  fi
done

echo "Done."
