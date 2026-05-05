# CompulsionBench

This directory contains the draft benchmark runner and calibration utilities used for the paper experiments.

- `compulsionbench.py`: main CLI for smoke tests, calibration, paper runs, and card building
- `calibration_audit.py`: render a markdown audit from an existing `calibration_payload.json`
- `requirements.txt`: base Python dependencies

## Recommended workflow

If you want the paper-style path, run the steps in this order:

1. Create a Python environment and install dependencies.
2. Run the smoke test once.
3. Download KuaiRand and build the calibration CSV.
4. Run `calibrate_optuna` once and freeze the resulting `config_calibrated.json`.
5. Optionally audit the calibration payload.
6. Download KuaiRec and build the card metadata CSV.
7. Run `run_paper` from the frozen config for AGParam.
8. Run `run_paper` again from the same frozen config for AGLLM.

All commands below assume you start in this directory:

```bash
cd /path/to/AddictionGymPaper/compulsionbench
```

If a dataset directory already exists under `data/`, you can skip the corresponding download and extract step.

## 1) Create the environment

Use `python3` to create the virtual environment. After activation, use `python`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install optuna
mkdir -p data data/derived outputs
```

Notes:

- `optuna` is required for `calibrate_optuna` but is not currently listed in `requirements.txt`.
- `curl`, `tar`, and `unzip` are used below for dataset download and extraction.

## 2) Run the smoke test

Run this once to confirm the code path works before downloading large datasets.

```bash
python compulsionbench.py smoke_test \
  --outdir outputs/smoke \
  --log_level INFO \
  --log_file outputs/smoke/run.log
```

Expected outputs:

- `outputs/smoke/tables/`
- `outputs/smoke/figures/`
- `outputs/smoke/models/`
- `outputs/smoke/manifest.json`

## 3) Download KuaiRand for calibration

Use KuaiRand for public-log calibration. Start with `KuaiRand-1K` for development. Use `KuaiRand-27K` only when you are ready for the large final run.

### Development-sized download: KuaiRand-1K

```bash
curl -L https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz -o data/KuaiRand-1K.tar.gz
tar -xzf data/KuaiRand-1K.tar.gz -C data/
```

### Final paper-sized download: KuaiRand-27K

`KuaiRand-27K` is very large, so leave this commented out until you need it.

```bash
# curl -L https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz -o data/KuaiRand-27K.tar.gz
# tar -xzf data/KuaiRand-27K.tar.gz -C data/
```

## 4) Build the calibration CSV from KuaiRand

For the paper-style path, concatenate the KuaiRand `log_standard*.csv` files into one calibration CSV.

### Build from KuaiRand-1K

```bash
python - <<'PY'
from pathlib import Path
import csv

root = Path("data/KuaiRand-1K/data")
parts = sorted(root.glob("log_standard*.csv"))
out = Path("data/derived/kuairand_standard_concat.csv")
out.parent.mkdir(parents=True, exist_ok=True)

if not parts:
    raise SystemExit(f"No KuaiRand standard log files found under {root}")

with out.open("w", newline="", encoding="utf-8") as wf:
    writer = None
    for path in parts:
        with path.open("r", newline="", encoding="utf-8") as rf:
            reader = csv.reader(rf)
            header = next(reader)
            if writer is None:
                writer = csv.writer(wf)
                writer.writerow(header)
            for row in reader:
                writer.writerow(row)

print(f"Wrote {out} from {len(parts)} input files")
PY
```

For the large run, change:

```python
root = Path("data/KuaiRand-1K/data")
```

to:

```python
root = Path("data/KuaiRand-27K/data")
```

Expected output:

- `data/derived/kuairand_standard_concat.csv`

## 5) Calibrate the benchmark from KuaiRand

Run calibration once, keep the resulting `config_calibrated.json` fixed, and reuse that same config for all later policy comparisons.

```bash
python compulsionbench.py calibrate_optuna \
  --log_csv data/derived/kuairand_standard_concat.csv \
  --delta_sess 30 \
  --outdir outputs/calibration_kuairand \
  --threshold_source fixed_paper \
  --exploratory_trials 300 \
  --episodes_per_trial 200 \
  --topk_trials 20 \
  --topk_episodes 1000 \
  --finalists 5 \
  --final_episodes 5000 \
  --seed 0 \
  --log_level INFO \
  --log_file outputs/calibration_kuairand/run.log
```

What this step writes:

- `outputs/calibration_kuairand/study.sqlite3`
- `outputs/calibration_kuairand/calibration_history.csv`
- `outputs/calibration_kuairand/top_trials.csv`
- `outputs/calibration_kuairand/fixed_seed_list.json`
- `outputs/calibration_kuairand/config_calibrated.json`
- `outputs/calibration_kuairand/calibration_payload.json`
- `outputs/calibration_kuairand/table_calibration_audit.md`
- `outputs/calibration_kuairand/table_thresholds.md`
- `outputs/calibration_kuairand/manifest.json`

Important notes:

- Do not recalibrate for every training seed. Calibrate once, then reuse the frozen config.
- Keep `--threshold_source fixed_paper` unless you explicitly want `empirical_target` or `simulator_relative`.
- For the paper-style KuaiRand calibration path, do not pass KuaiRec metadata into `calibrate_optuna`.
- If the audit status in `table_calibration_audit.md` is `failed`, treat that as a real modeling issue rather than a successful calibration.

## 6) Optionally audit calibration feasibility

If you want a structural audit after calibration, run:

```bash
python compulsionbench.py audit_calibration_feasibility \
  --payload_json outputs/calibration_kuairand/calibration_payload.json \
  --outdir outputs/calibration_feasibility \
  --log_level INFO \
  --log_file outputs/calibration_feasibility/run.log
```

This writes:

- `outputs/calibration_feasibility/feasibility_report.md`
- `outputs/calibration_feasibility/feasibility_report.json`
- `outputs/calibration_feasibility/fig_session_item_count_cdf.png`
- `outputs/calibration_feasibility/fig_stop_hazard_support.png`
- `outputs/calibration_feasibility/fig_per_item_watch_cdf.png`

## 7) Download KuaiRec for semantic card metadata

Use KuaiRec to build the metadata-backed cards used by AGLLM and `--paper_mode`.

```bash
curl -L https://zenodo.org/records/18164998/files/KuaiRec.zip -o data/KuaiRec.zip
unzip -o data/KuaiRec.zip -d data/
```

Only run this extra download if `kuairec_caption_category.csv` is missing after unzip:

```bash
curl -L https://zenodo.org/records/18164998/files/kuairec_caption_category.csv -o "data/KuaiRec 2.0/data/kuairec_caption_category.csv"
```

## 8) Build the KuaiRec metadata CSV for AGLLM cards

This produces the exact metadata CSV expected by `--metadata_csv`.

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("data/KuaiRec 2.0/data")
caption_csv = root / "kuairec_caption_category.csv"
interactions_csv = root / "big_matrix.csv"

caps = pd.read_csv(caption_csv, engine="python").rename(columns={
    "video_id": "item_id",
    "first_level_category_name": "category_level1",
    "second_level_category_name": "category_level2",
})
caps["item_id"] = pd.to_numeric(caps["item_id"], errors="coerce")
caps = caps.dropna(subset=["item_id"]).copy()
caps["item_id"] = caps["item_id"].astype("int64")
caps = caps[["item_id", "caption", "category_level1", "category_level2"]].drop_duplicates("item_id")

interactions = pd.read_csv(interactions_csv, usecols=["video_id", "video_duration"])
stats = (
    interactions.groupby("video_id")
    .agg(
        duration_ms=("video_duration", "first"),
        interaction_count=("video_id", "size"),
    )
    .reset_index()
    .rename(columns={"video_id": "item_id"})
)
stats["duration_sec"] = stats["duration_ms"].astype(float) / 1000.0
stats = stats[["item_id", "duration_sec", "interaction_count"]]

meta = caps.merge(stats, on="item_id", how="inner")
meta["caption"] = meta["caption"].fillna("")
meta["category_level1"] = meta["category_level1"].fillna("")
meta["category_level2"] = meta["category_level2"].fillna("")

out = Path("data/derived/kuairec_card_metadata.csv")
out.parent.mkdir(parents=True, exist_ok=True)
meta.to_csv(out, index=False)
print(f"Wrote {out} with {len(meta)} rows")
PY
```

Expected output:

- `data/derived/kuairec_card_metadata.csv`

Required columns in that file:

- `item_id`
- `caption`
- `category_level1`
- `category_level2`
- `duration_sec` or `duration_seconds`
- `interaction_count`

## 9) Run AGParam from the frozen calibrated config

Use the calibrated KuaiRand config for the main parametric benchmark run.

```bash
python compulsionbench.py run_paper \
  --outdir outputs/paper_param_main \
  --run_profile main \
  --config_json outputs/calibration_kuairand/config_calibrated.json \
  --calibration_payload_json outputs/calibration_kuairand/calibration_payload.json \
  --threshold_source fixed_paper \
  --seed 0 \
  --log_level INFO \
  --log_file outputs/paper_param_main/run.log
```

If you want a faster development run, switch to:

```bash
python compulsionbench.py run_paper \
  --outdir outputs/paper_param_dev \
  --run_profile dev \
  --config_json outputs/calibration_kuairand/config_calibrated.json \
  --calibration_payload_json outputs/calibration_kuairand/calibration_payload.json \
  --threshold_source fixed_paper \
  --seed 0 \
  --log_level INFO \
  --log_file outputs/paper_param_dev/run.log
```

If you want appendix-style diagnostics and sweeps, switch to:

```bash
python compulsionbench.py run_paper \
  --outdir outputs/paper_param_full \
  --run_profile full \
  --config_json outputs/calibration_kuairand/config_calibrated.json \
  --calibration_payload_json outputs/calibration_kuairand/calibration_payload.json \
  --threshold_source fixed_paper \
  --seed 0 \
  --log_level INFO \
  --log_file outputs/paper_param_full/run.log
```

## 10) Run AGLLM from the same frozen config

Use the exact same `config_calibrated.json` for AGLLM. The only thing that should change here is the backend.

### Fast local AGLLM check with the surrogate backend

```bash
python compulsionbench.py run_paper \
  --outdir outputs/paper_llm_surrogate \
  --run_profile dev \
  --config_json outputs/calibration_kuairand/config_calibrated.json \
  --calibration_payload_json outputs/calibration_kuairand/calibration_payload.json \
  --threshold_source fixed_paper \
  --metadata_csv data/derived/kuairec_card_metadata.csv \
  --with_llm \
  --llm_mode surrogate \
  --seed 0 \
  --log_level INFO \
  --log_file outputs/paper_llm_surrogate/run.log
```

### Main AGLLM run with a Hugging Face model

```bash
python compulsionbench.py run_paper \
  --outdir outputs/paper_llm_hf \
  --run_profile main \
  --config_json outputs/calibration_kuairand/config_calibrated.json \
  --calibration_payload_json outputs/calibration_kuairand/calibration_payload.json \
  --threshold_source fixed_paper \
  --metadata_csv data/derived/kuairec_card_metadata.csv \
  --with_llm \
  --llm_mode hf \
  --llm_model_id Qwen/Qwen2.5-7B-Instruct \
  --fit_llm_fusion \
  --paper_mode \
  --seed 0 \
  --log_level INFO \
  --log_file outputs/paper_llm_hf/run.log
```

Important notes:

- `--fit_llm_fusion` is the second-stage AGLLM calibration. Use it here, not in `calibrate_optuna`.
- `--paper_mode` requires `--metadata_csv`; it forbids synthetic fallback cards.
- The first Hugging Face run may need to download model weights and may take substantially longer than the surrogate backend.
- AGLLM runs write `llm_cache_stats.json` in addition to the standard tables, figures, models, and manifest.

## 11) Optional: build cards as a standalone JSON artifact

`run_paper` builds cards on the fly from `--metadata_csv`, so this step is optional. Use it only if you want to inspect the card construction separately.

```bash
python compulsionbench.py build_cards \
  --metadata_csv data/derived/kuairec_card_metadata.csv \
  --out_json outputs/cards.json \
  --log_level INFO \
  --log_file outputs/build_cards.log
```

## 12) Output layout

`run_paper` writes these artifacts under the requested `--outdir`:

- `tables/`
- `figures/`
- `models/`
- `manifest.json`

Additional outputs:

- `calibration_payload.json` if calibration is run inline inside `run_paper`
- `llm_cache_stats.json` when `--with_llm` is enabled

## 13) Supported calibration log schemas

The calibration loader accepts any of the following input schemas:

1. Canonical CSV
   Required columns: `user_id`, `timestamp_min`, `watch_time_min`
   Optional columns: `session_id`, `cluster_id`, `item_id`
2. KuaiRec native CSV
   Required columns: `user_id`, `video_id`, `timestamp`, `play_duration`
3. KuaiRand native CSV
   Required columns: `user_id`, `time_ms`, `play_time_ms`, plus one of `video_id`, `final_video_id`, or `item_id`

Behavior notes:

- If `session_id` is missing, the code sessionizes with `--delta_sess`.
- If `cluster_id` is missing, the draft code falls back to metadata-derived clusters or hashed buckets.

## 14) Common failure points

- `ModuleNotFoundError: No module named 'optuna'`
  Install it with `python -m pip install optuna`.
- `paper_mode=True forbids synthetic fallback cards`
  Pass `--metadata_csv data/derived/kuairec_card_metadata.csv`.
- `Metadata CSV is missing columns`
  Rebuild `data/derived/kuairec_card_metadata.csv` and make sure it contains the required fields listed above.
- `threshold_source requires calibration target statistics`
  Pass `--calibration_payload_json` or keep `--threshold_source fixed_paper`.

## Misuse note

This code is for auditing and benchmarking recommendation policies under explicit assumptions. It should not be used as deployment guidance for maximizing harmful engagement.
