#!/usr/bin/env bash
#
# Launch 8 RL experiments in parallel, one per GPU.
#
# Grid: state_mode = "both" (α/σ/κ + h_t + fast tier) fixed.
# Vary algorithm × action_mode where action_mode contrasts the two extremes:
#   A      = residual-over-A-S (A-S as warm-start, RL learns delta)
#   C      = direct β·spread (no A-S anchor)
# Discrete D3QN gets A_disc / C_disc accordingly.
#
#   GPU 0: SAC  × a       × both
#   GPU 1: SAC  × c       × both
#   GPU 2: DDPG × a       × both
#   GPU 3: DDPG × c       × both
#   GPU 4: TD3  × a       × both
#   GPU 5: TD3  × c       × both
#   GPU 6: D3QN × a_disc  × both
#   GPU 7: D3QN × c_disc  × both
#
# Mode B (Alpha-AS-style A-S parameter tuning) is strictly less flexible than
# Mode A and is deferred to a separate batch.
#
# Override defaults via env vars:
#   SEED=42 N_ROLLOUTS=50 ./scripts/exp/run_all_8.sh
#
# Logs: runs/rl/<run_name>/         — TB events + checkpoints (from train.py)
#       runs/rl/<run_name>.log      — stdout/stderr of the process

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ENV_JSON="${ENV_JSON:-configs/exp/rl_smoke.json}"
TRAIN_JSON="${TRAIN_JSON:-configs/exp/rl_train.json}"
SEED="${SEED:-0}"
N_ROLLOUTS="${N_ROLLOUTS:-30}"

# Sanity-check configs exist
for f in "$ENV_JSON" "$TRAIN_JSON" \
         configs/exp/rl_agent_sac.json \
         configs/exp/rl_agent_ddpg.json \
         configs/exp/rl_agent_td3.json \
         configs/exp/rl_agent_d3qn.json; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

# Each entry: "algo:action_mode:state_mode"
declare -a CONFIGS=(
  "sac:a:both"
  "sac:c:both"
  "ddpg:a:both"
  "ddpg:c:both"
  "td3:a:both"
  "td3:c:both"
  "d3qn:a_disc:both"
  "d3qn:c_disc:both"
)

mkdir -p runs/rl
declare -a PIDS=()

for i in "${!CONFIGS[@]}"; do
  IFS=':' read -r algo mode state <<< "${CONFIGS[$i]}"
  gpu="$i"
  run_name="${algo}_${mode}_${state}_seed${SEED}_gpu${gpu}"
  log_file="runs/rl/${run_name}.log"

  echo "[launch] GPU=${gpu}  algo=${algo}  mode=${mode}  state=${state}  → ${run_name}"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. uv run python -m scripts.exp.train_rl \
      --env "$ENV_JSON" \
      --agent "configs/exp/rl_agent_${algo}.json" \
      --train "$TRAIN_JSON" \
      --action-mode "$mode" \
      --state-mode "$state" \
      --device cuda \
      --seed "$SEED" \
      --n-rollouts "$N_ROLLOUTS" \
      --run-name "$run_name" \
      > "$log_file" 2>&1 &
  PIDS+=($!)
done

echo "[parent] 8 jobs launched. PIDs: ${PIDS[*]}"
echo "[parent] tail logs:  tail -F runs/rl/*.log"
echo "[parent] waiting for all to finish..."

FAILED=()
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    IFS=':' read -r algo mode state <<< "${CONFIGS[$i]}"
    FAILED+=("gpu${i}:${algo}/${mode}/${state}")
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[parent] FAILED: ${FAILED[*]}"
  exit 1
fi

echo "[parent] all 8 done."
