#!/usr/bin/env bash
#
# Cross-codebase forward-pass reproducibility smoke test.
#
# Runs scripts/smoke_checkpoint_reproducibility.py inside two repos against the
# same checkpoint, then diffs the outputs. Passes iff the new codebase produces
# the same logits/hidden as the old one (proves backward-compatibility of the
# qk_norm/Muon additions when defaults are off).
#
# The smoke script is copied from the new repo into the old repo so an identical
# script runs against each repo's local `src/`. The copy is removed on exit.
#
# Usage:
#     scripts/smoke_compare_codebases.sh CHECKPOINT_PATH TRADEFM4_DIR TRADEFM5_DIR
#
# Example:
#     scripts/smoke_compare_codebases.sh \
#         /shared/checkpoints/transformer_75m/best.pt \
#         /workspace/tradefm4 \
#         /workspace/tradefm5

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 CHECKPOINT_PATH OLD_REPO_DIR NEW_REPO_DIR" >&2
    exit 2
fi

CKPT="$1"
OLD_DIR="$(cd "$2" && pwd)"
NEW_DIR="$(cd "$3" && pwd)"
SCRIPT="smoke_checkpoint_reproducibility.py"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi
if [[ ! -f "$NEW_DIR/scripts/$SCRIPT" ]]; then
    echo "ERROR: missing $NEW_DIR/scripts/$SCRIPT" >&2
    exit 1
fi

OUT_OLD=$(mktemp -t tfm_old.XXXXXX).pt
OUT_NEW=$(mktemp -t tfm_new.XXXXXX).pt
COPIED_SCRIPT="$OLD_DIR/scripts/$SCRIPT"
COPIED_FLAG=0
if [[ ! -f "$COPIED_SCRIPT" ]]; then
    mkdir -p "$OLD_DIR/scripts"
    cp "$NEW_DIR/scripts/$SCRIPT" "$COPIED_SCRIPT"
    COPIED_FLAG=1
fi

cleanup() {
    rm -f "$OUT_OLD" "$OUT_NEW"
    if [[ "$COPIED_FLAG" == "1" ]]; then
        rm -f "$COPIED_SCRIPT"
    fi
}
trap cleanup EXIT

echo "================================================================"
echo "OLD codebase: $OLD_DIR"
echo "================================================================"
(cd "$OLD_DIR" && PYTHONPATH=. uv run python -m scripts.smoke_checkpoint_reproducibility \
    --checkpoint "$CKPT" --out "$OUT_OLD")

echo
echo "================================================================"
echo "NEW codebase: $NEW_DIR"
echo "================================================================"
(cd "$NEW_DIR" && PYTHONPATH=. uv run python -m scripts.smoke_checkpoint_reproducibility \
    --checkpoint "$CKPT" --out "$OUT_NEW")

echo
echo "================================================================"
echo "COMPARE"
echo "================================================================"
cd "$NEW_DIR"
PYTHONPATH=. uv run python -m scripts.smoke_checkpoint_reproducibility \
    --compare "$OUT_OLD" "$OUT_NEW"
