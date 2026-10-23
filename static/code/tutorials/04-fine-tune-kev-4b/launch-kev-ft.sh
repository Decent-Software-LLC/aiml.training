#!/usr/bin/env bash
# Launch the Kev-4B banking77 delta fine-tune in tmux on your GPU box.
# Paths are scrubbed: point KEV_DIR / KEV_OUT at your own directories.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

KEV_DIR="/path/to/kev"          # where the kev package + kev-train.jsonl live
KEV_OUT="/path/to/kev-fin"      # clean output dir (never clobber a previous run)

cd "$KEV_DIR"
rm -rf "$KEV_OUT"
tmux kill-session -t kevft 2>/dev/null || true
tmux new-session -d -s kevft "export PATH=\$HOME/.local/bin:\$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $KEV_DIR
uv run python -m kev.train --data $KEV_DIR/kev-train.jsonl \
  --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b \
  --lr 2e-5 --epochs 2 --batch 1 --accum 8 --dtype bf16 \
  --weights_dtype bf16 --checkpointing 1 --device cuda \
  --out $KEV_OUT 2>&1 | tee $KEV_OUT.log; echo \"EXIT=\$?\""
echo launched

# follow the run:
#   tmux attach -t kevft        # Ctrl-b d to detach
