#!/bin/bash
# =================================================================
# finalize_llm_results.sh — Merge AGLLM parallel seeds & rebuild
# =================================================================
# Run this AFTER all 5 AGLLM array jobs complete.
# Usage:  bash scripts/finalize_llm_results.sh
# =================================================================

set -e

PROJECT_DIR=/path/to/AddictionGym
WORK_DIR=$PROJECT_DIR/compulsionbench
ARRAY_DIR=$WORK_DIR/outputs/v2_llm_array
MASTER_DIR=$WORK_DIR/outputs/paper_llm_v2

cd $WORK_DIR
module load cuda/12.1 || true
export HF_HOME=$PROJECT_DIR/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

# -----------------------------------------------------------------
# 1. Verify seeds completed
# -----------------------------------------------------------------
echo "Checking AGLLM seed directories..."
COMPLETED=0
for i in {0..4}; do
    SRC=$ARRAY_DIR/seed_$i
    if [ -f "$SRC/run.log" ] && ls $SRC/models/*.pt &>/dev/null 2>&1; then
        echo "  [OK] seed $i"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "  [WARN] seed $i appears incomplete"
    fi
done
if [ $COMPLETED -lt 3 ]; then
    echo "ERROR: Only $COMPLETED/5 seeds completed. Exiting."
    exit 1
fi
echo "$COMPLETED/5 seeds ready."

# -----------------------------------------------------------------
# 2. Merge models and training histories into master bundle
# -----------------------------------------------------------------
echo "Merging AGLLM seed outputs into master bundle..."
mkdir -p $MASTER_DIR/models
mkdir -p $MASTER_DIR/tables
mkdir -p $MASTER_DIR/figures

# Copy metadata. We use the local config_calibrated.json to enforce evaluation thresholds.
FIRST_SEED=$ARRAY_DIR/seed_0
cp $PROJECT_DIR/config_calibrated.json $MASTER_DIR/config_used.json
cp $FIRST_SEED/cards_used.json $MASTER_DIR/
cp $FIRST_SEED/manifest.json $MASTER_DIR/
cp $FIRST_SEED/test_episode_seeds.json $MASTER_DIR/
cp $FIRST_SEED/val_episode_seeds.json $MASTER_DIR/

# Generate the train_seed_list.json
echo "[" > $MASTER_DIR/train_seed_list.json
for i in {0..4}; do
    if [ -f "$ARRAY_DIR/seed_$i/run.log" ]; then
        if [ $i -gt 0 ]; then echo -n "," >> $MASTER_DIR/train_seed_list.json; fi
        echo -n "$i" >> $MASTER_DIR/train_seed_list.json
    fi
done
echo "]" >> $MASTER_DIR/train_seed_list.json

for i in {0..4}; do
    SRC=$ARRAY_DIR/seed_$i
    if [ ! -d "$SRC" ]; then continue; fi
    ls $SRC/models/*.pt &>/dev/null 2>&1 && cp $SRC/models/*.pt $MASTER_DIR/models/ && echo "  Copied models seed $i"
    ls $SRC/training_history_*.csv &>/dev/null 2>&1 && cp $SRC/training_history_*.csv $MASTER_DIR/
done
cp $ARRAY_DIR/seed_0/config_used.json $MASTER_DIR/

# -----------------------------------------------------------------
# 2b. Fix manifest metadata
# -----------------------------------------------------------------
echo "Patching manifest metadata..."
python3 -c "
import json, pathlib
manifest_path = pathlib.Path('$MASTER_DIR/manifest.json')
seeds_path = pathlib.Path('$MASTER_DIR/train_seed_list.json')
m = json.loads(manifest_path.read_text())
seeds = json.loads(seeds_path.read_text())
m['num_train_seeds'] = len(seeds)
# Remove references to per-seed artifacts that are not aggregated
for key in ['table_frontier_rebuilt', 'fig_frontier_overcap_rebuilt',
            'table_cap_sensitivity_rebuilt', 'table_backend_sensitivity',
            'table_llm_recovery', 'table_llm_ablation',
            'llm_cache_stats', 'agllm_release_manifest']:
    m.pop(key, None)
manifest_path.write_text(json.dumps(m, indent=2))
print(f'  Manifest patched: num_train_seeds={len(seeds)}')
"

# -----------------------------------------------------------------
# 3. Rebuild backend sensitivity tables (LLM vs Param)
# -----------------------------------------------------------------
echo "Rebuilding official scorecards (AGLLM)..."
python compulsionbench.py rebuild_official_scorecards \
  --bundle_dir $MASTER_DIR \
  --device cpu \
  --log_level INFO

# -----------------------------------------------------------------
# 3b. Rebuild mechanism ablations table
# -----------------------------------------------------------------
echo "Rebuilding mechanism ablations..."
python compulsionbench.py rebuild_mechanism_ablations \
  --bundle_dir $MASTER_DIR \
  --device cpu \
  --log_level INFO

# -----------------------------------------------------------------
# 4. Check LLM fusion weights
# -----------------------------------------------------------------
echo ""
echo "LLM Fusion Weights (want > 0.0):"
python3 -c "
import json, pathlib
for i in range(5):
    p = pathlib.Path('$ARRAY_DIR') / f'seed_{i}' / 'llm_fusion_fit.json'
    if p.exists():
        d = json.load(open(p))
        print(f'  seed {i}: omega_r_llm={d.get(\"omega_r_llm\", \"N/A\"):.4f}, omega_c_llm={d.get(\"omega_c_llm\", \"N/A\"):.4f}')
    else:
        print(f'  seed {i}: llm_fusion_fit.json not found')
" 2>/dev/null || echo "(Could not read fusion weights)"

# -----------------------------------------------------------------
# 5. Render figures
# -----------------------------------------------------------------
echo "Rendering publication figures..."
python $PROJECT_DIR/scripts/render_results.py \
  --bundle_dir $MASTER_DIR \
  --out_dir $MASTER_DIR/figures \
  --style paper \
  --format png \
  --dpi 300

echo ""
echo "============================================================"
echo "AGLLM finalization complete!"
echo "Master results: $MASTER_DIR"
echo "============================================================"
