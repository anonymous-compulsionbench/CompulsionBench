#!/bin/bash
# =================================================================
# finalize_param_results.sh — Merge parallel seeds & rebuild tables
# =================================================================
# Run this AFTER all 5 array jobs complete.
# Usage:  bash scripts/finalize_param_results.sh
#
# It will:
#   1. Verify all 5 seeds finished successfully
#   2. Merge models and training logs into a single master bundle
#   3. Rebuild the official scorecards (averaged across seeds)
#   4. Rebuild the mechanism ablations table
#   5. Render publication-quality figures
# =================================================================

set -e  # Exit immediately if any command fails

PROJECT_DIR=/path/to/AddictionGym
WORK_DIR=$PROJECT_DIR/compulsionbench
ARRAY_DIR=$WORK_DIR/outputs/v2_array
MASTER_DIR=$WORK_DIR/outputs/paper_param_v2

cd $WORK_DIR

# -----------------------------------------------------------------
# 0. Set environment paths
# -----------------------------------------------------------------
module load cuda/12.1 || true
export HF_HOME=$PROJECT_DIR/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

# -----------------------------------------------------------------
# 1. Verify seeds completed — warn (don't abort) on missing seeds
# -----------------------------------------------------------------
echo "Checking seed directories..."
COMPLETED=0
for i in {0..4}; do
    SRC=$ARRAY_DIR/seed_$i
    # A seed is complete if it wrote its run.log and at least one model
    if [ -f "$SRC/run.log" ] && ls $SRC/models/*.pt &>/dev/null 2>&1; then
        echo "  [OK] seed $i"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "  [WARN] seed $i appears incomplete — check $SRC/run.log"
    fi
done

if [ $COMPLETED -lt 3 ]; then
    echo "ERROR: Only $COMPLETED/5 seeds completed. Need at least 3 for valid aggregation. Exiting."
    exit 1
fi
echo "$COMPLETED/5 seeds ready. Proceeding with aggregation."

# -----------------------------------------------------------------
# 2. Merge models and training histories into master bundle
# -----------------------------------------------------------------
echo "Merging seed outputs into master bundle..."
mkdir -p $MASTER_DIR/models
mkdir -p $MASTER_DIR/tables
mkdir -p $MASTER_DIR/figures

# Copy metadata. We deliberately use the local reference config_calibrated.json
# to ensure it serves as the ultimate source of truth for evaluation configuration thresholds.
FIRST_SEED=$ARRAY_DIR/seed_0
cp $PROJECT_DIR/config_calibrated.json $MASTER_DIR/config_used.json
cp $FIRST_SEED/cards_used.json $MASTER_DIR/
cp $FIRST_SEED/test_episode_seeds.json $MASTER_DIR/
cp $FIRST_SEED/val_episode_seeds.json $MASTER_DIR/
cp $FIRST_SEED/manifest.json $MASTER_DIR/

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

    # Models (ppo_seed{i}.pt, lagppo_seed{i}.pt, etc.)
    if ls $SRC/models/*.pt &>/dev/null 2>&1; then
        cp $SRC/models/*.pt $MASTER_DIR/models/
        echo "  Copied models for seed $i"
    fi

    # Training history CSVs
    if ls $SRC/training_history_*.csv &>/dev/null 2>&1; then
        cp $SRC/training_history_*.csv $MASTER_DIR/
        echo "  Copied training history for seed $i"
    fi
done

# Copy representative files from seed_0
cp $ARRAY_DIR/seed_0/config_used.json $MASTER_DIR/

# -----------------------------------------------------------------
# 3. Rebuild official scorecards (aggregates across all seeds)
# -----------------------------------------------------------------
echo "Rebuilding official scorecards..."
python compulsionbench.py rebuild_official_scorecards \
  --bundle_dir $MASTER_DIR \
  --device cpu \
  --log_level INFO

# -----------------------------------------------------------------
# 4. Rebuild mechanism ablations table
# -----------------------------------------------------------------
echo "Rebuilding mechanism ablations..."
python compulsionbench.py rebuild_mechanism_ablations \
  --bundle_dir $MASTER_DIR \
  --device cpu \
  --log_level INFO

# -----------------------------------------------------------------
# 5. Render publication-quality figures
# -----------------------------------------------------------------
echo "Rendering publication figures (paper style, PNG)..."
python $PROJECT_DIR/scripts/render_results.py \
  --bundle_dir $MASTER_DIR \
  --out_dir $MASTER_DIR/figures \
  --style paper \
  --format png \
  --dpi 300 || echo "WARNING: Plot rendering failed, but evaluation artifacts are intact."

echo ""
echo "============================================================"
echo "Aggregation complete!"
echo "Master results: $MASTER_DIR"
echo "   Tables:   $MASTER_DIR/tables/"
echo "   Figures:  $MASTER_DIR/rendered/"
echo "   Models:   $MASTER_DIR/models/ ($COMPLETED seeds)"
echo "============================================================"

