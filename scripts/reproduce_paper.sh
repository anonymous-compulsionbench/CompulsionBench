#!/bin/bash
# =================================================================
# reproduce_paper.sh — CompulsionBench Publication Reproducibility
# =================================================================
#
# This script executes the core Main-Text and Appendix results for
# the paper "CompulsionBench: A Benchmark for Auditing..."
#
# It is designed to be run on a Slurm cluster with NVIDIA A100 GPUs.
#
# Usage:
#   sbatch reproduce_paper.sh
#

#SBATCH --job-name=cb-reproduce
#SBATCH --account=your_account
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=reproduce_%j.log

# Configure paths (Update these for your system)
# -----------------------------------------------------------------
PROJECT_DIR=/path/to/AddictionGym
WORK_DIR=$PROJECT_DIR/compulsionbench
ENV_PATH=$PROJECT_DIR/conda-envs/addictiongym
CONFIG_PATH=$PROJECT_DIR/config_calibrated.json

# Environment Setup
# -----------------------------------------------------------------
module load cuda/12.1
source activate $ENV_PATH

export HF_HOME=$PROJECT_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$PROJECT_DIR/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

cd $WORK_DIR
mkdir -p outputs/reproduction

# -----------------------------------------------------------------
# 1. Main Text Results (AGParam - Default)
# -----------------------------------------------------------------
echo "Executing Main Text Comparison (AGParam - Default)..."
python compulsionbench.py run_paper \
  --outdir outputs/reproduction/main_results \
  --run_profile main \
  --config_json $CONFIG_PATH \
  --threshold_source fixed_paper \
  --device cuda \
  --seed 0

# -----------------------------------------------------------------
# 2. Appendix: AGLLM Robustness Extension
# -----------------------------------------------------------------
echo "Executing AGLLM Robustness Review (Appendix)..."
python compulsionbench.py run_paper \
  --outdir outputs/reproduction/llm_results \
  --run_profile main \
  --config_json $CONFIG_PATH \
  --threshold_source fixed_paper \
  --with_llm \
  --llm_mode hf \
  --llm_model_id Qwen/Qwen2.5-7B-Instruct \
  --fit_llm_fusion \
  --paper_mode \
  --device cuda \
  --seed 0

# -----------------------------------------------------------------
# 3. Figure Generation
# -----------------------------------------------------------------
echo "Generating publication figures..."
python scripts/render_results.py \
  --bundle_dir outputs/reproduction/main_results \
  --style paper \
  --format pdf \
  --dpi 300

echo "REPRODUCTION COMPLETE."
echo "Results available in outputs/reproduction/"
