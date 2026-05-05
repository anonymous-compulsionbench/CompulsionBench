
#!/usr/bin/env python3
"""
CompulsionBench draft implementation.

This file intentionally keeps the entire benchmark, training, evaluation, and
plotting pipeline in one place so the release stays compact.

What is implemented:
- AGParam reference backend with the transition equations from the paper.
- Optional AGLLM-style backend with deterministic prompt/card/persona rendering,
  cache-first scoring, optional Hugging Face inference, affine score maps, and
  linear fusion.
- Random, myopic reward model, PPO, and Lagrangian PPO baselines.
- Session-cap evaluation wrapper and appendix hooks for break prompts,
  personalization throttling, and a simple autoplay-off friction term.
- Paper-style experiment driver that writes tables and figures.

What remains dataset-dependent:
- Calibration to public logs requires the user to supply their own event log CSV.
- The exact KuaiRec/KuaiRand card pipeline depends on local metadata files.
- HF model downloads are not bundled in this draft.

The code defaults to the appendix configuration values in the draft paper so the
pipeline is runnable even without external datasets.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import logging
import math
import os
import platform
import random
import re
import sys
import tempfile
import textwrap
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.distributions import Categorical

try:
    from sklearn_extra.cluster import KMedoids  # type: ignore
    HAVE_KMEDOIDS = True
except Exception:
    HAVE_KMEDOIDS = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False

try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover - dependency availability is environment-specific
    threadpool_limits = None


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

LOGGER = logging.getLogger("compulsionbench")
METRIC_LOG_KEYS = ("CumWatch", "CVaR_0.95(L)", "ReturnRate60", "NightMinutes", "OverCapMinutes")
RETURN_RATE_THRESHOLDS = (5.0, 15.0, 30.0, 60.0)
RETURN_RATE_METRIC_KEYS = tuple(f"ReturnRate{int(threshold)}" for threshold in RETURN_RATE_THRESHOLDS)
NIGHT_PROXY_METRIC_KEYS = ("NightMinutes", "NightFraction", "LateNightSessionStartRate")
NIGHT_PROXY_PROMOTION_CANDIDATES = ("NightFraction", "LateNightSessionStartRate")
MAIN_POLICY_METRIC_KEYS = ("CumWatch", "CVaR_0.95(L)", *RETURN_RATE_METRIC_KEYS, *NIGHT_PROXY_METRIC_KEYS, "OverCapMinutes")
CAP_FRAGMENTATION_GAP_THRESHOLDS = (1.0, 5.0)
CAP_FRAGMENTATION_POLICY_ORDER = ("PPO", "PPO+Cap(90)", "PPO+Cap(120)", "PPO+Cap(150)")
MECHANISM_ABLATION_NAMES = ("NoHabit", "NoVar", "NoPers", "HomogeneousUsers")
MECHANISM_POLICY_ORDER = ("PPO", "Lagrangian PPO")
OFFICIAL_SCORECARD_BOOTSTRAP_METRICS = (
    "CumWatch",
    "CVaR_0.95(L)",
    "NightMinutes",
    "NightFraction",
    "LateNightSessionStartRate",
    "OverCapMinutes",
    *RETURN_RATE_METRIC_KEYS,
)
OFFICIAL_SCORECARD_SCALAR_EPISODE_METRICS = (
    "CumWatch",
    "NightMinutes",
    "NightFraction",
    "LateNightSessionStartRate",
    "OverCapMinutes",
)
SCORECARD_BOOTSTRAP_RESAMPLES = 400
SCORECARD_BOOTSTRAP_BASE_SEED = 1729
EVAL_MODE_ORDER = ("deterministic", "stochastic")
DIVERSITY_DIAGNOSTIC_POLICY_ORDER = (
    "PPO",
    "Lagrangian PPO",
    "Myopic",
    "Random",
    "RoundRobinPolicy",
    "LeastRecentPolicy",
    "NoveltyGreedyPolicy",
)
DIVERSITY_PERSONALIZATION_POLICIES = frozenset({"PPO", "Lagrangian PPO", "Myopic"})
DIVERSITY_HEURISTIC_POLICIES = frozenset({"Random", "RoundRobinPolicy", "LeastRecentPolicy", "NoveltyGreedyPolicy"})
APPENDIX_DASHBOARD_METRIC_KEYS = (
    "CumWatch",
    "CVaR_0.95(L)",
    "p99_L",
    "ReturnRate60",
    "ReturnRate5",
    "NightMinutes",
    "NightFraction",
    "LateNightSessionStartRate",
    "OverCapMinutes",
    "OverrideRate",
    "AvgHabit",
    "CtrlDepletion",
    "BreakAdherence",
)
APPENDIX_STRESS_TEST_NAMES = (
    "susceptibility_shift",
    "slower_recovery",
    "higher_reward_variability",
)
RETURN_BUCKET_LABELS = ("short", "medium", "long")
RETURN_RATE_MAIN_TEXT_MIN_SPREAD = 0.05
REPO_ROOT = Path(__file__).resolve().parent.parent
AGLLM_RELEASE_MANIFEST_PATH = REPO_ROOT / "release" / "agllm_release_manifest.json"
AGLLM_PARSE_SCRIPT_PATH = REPO_ROOT / "scripts" / "parse_agllm_response.py"
RUN_PROFILE_CHOICES = ("dev", "main", "full")
DEFAULT_CONSTRAINT_SCALES = [0.95, 0.90, 0.85]
NIGHT_CUMWATCH_PROXY_CORR_THRESHOLD = 0.95
ACTIVE_COST_EPS = 1e-8
NIGHT_FRACTION_EPS = 1e-8
POLICY_COMPARISON_TOL = 1e-6
RUN_PAPER_BASE_DEFAULTS: Dict[str, Any] = {
    "calibration_trials": 50,
    "calibration_episodes": 100,
    "num_train_seeds": 5,
    "eval_episodes": 5_000,
    "random_log_steps": 20_000,
    "myopic_epochs": 8,
    "myopic_batch_size": 512,
    "ppo_steps": 1_000_000,
    "rollout_steps": 2048,
    "minibatch_size": 256,
    "update_epochs": 10,
    "validate_every": 20_000,
    "val_episodes": 1000,
    "constraint_scales": list(DEFAULT_CONSTRAINT_SCALES),
    "cap_grid": [],
}
RUN_PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "dev": {
        "calibration_trials": 10,
        "calibration_episodes": 40,
        "num_train_seeds": 1,
        "eval_episodes": 128,
        "random_log_steps": 4_000,
        "myopic_epochs": 3,
        "myopic_batch_size": 256,
        "ppo_steps": 20_000,
        "rollout_steps": 512,
        "minibatch_size": 128,
        "update_epochs": 4,
        "validate_every": 5_000,
        "val_episodes": 128,
        "constraint_scales": [DEFAULT_CONSTRAINT_SCALES[0]],
        "cap_grid": [],
    },
    "main": {},
    "full": {},
}


def default_lag_search_mode(run_profile: str) -> str:
    return "full" if str(run_profile).lower() == "full" else "two_stage"


def apply_run_profile_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "command", None) != "run_paper":
        return
    run_profile = str(getattr(args, "run_profile", "main")).lower()
    profile_defaults = RUN_PROFILE_DEFAULTS.get(run_profile, {})
    for field_name, profile_value in profile_defaults.items():
        if getattr(args, field_name) == RUN_PAPER_BASE_DEFAULTS[field_name]:
            setattr(args, field_name, copy.deepcopy(profile_value))
    if getattr(args, "lag_search_mode", None) is None:
        args.lag_search_mode = default_lag_search_mode(run_profile)


def default_torch_num_threads() -> int:
    cpu_count = os.cpu_count() or 1
    if sys.platform == "darwin" and platform.machine().lower() == "arm64":
        return 4
    return max(1, min(4, cpu_count))


def _mps_is_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def _synchronize_torch_device(device_name: str) -> None:
    if device_name == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _benchmark_torch_device(device_name: str) -> float:
    cfg = BenchConfig(device=device_name)
    obs_dim = cfg.obs_dim()
    num_actions = cfg.num_actions()

    torch.manual_seed(0)
    np.random.seed(0)
    _synchronize_torch_device(device_name)
    start = time.perf_counter()

    ppo_model = build_actor_critic_model(cfg, hidden_size=64, policy_arch="flat").to(device_name).eval()
    ppo_obs = torch.randn(1, obs_dim, dtype=torch.float32, device=device_name)
    with torch.no_grad():
        for _ in range(256):
            ppo_model(ppo_obs)

    reward_model = RewardModel(obs_dim, num_actions, hidden=64).to(device_name)
    optimizer = optim.Adam(reward_model.parameters(), lr=1e-3)
    batch_obs = torch.randn(64, obs_dim, dtype=torch.float32, device=device_name)
    batch_actions = torch.randint(0, num_actions, (64,), device=device_name)
    batch_rewards = torch.randn(64, dtype=torch.float32, device=device_name)
    predictions = reward_model(batch_obs)
    chosen_rewards = predictions.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
    loss = nn.functional.mse_loss(chosen_rewards, batch_rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _synchronize_torch_device(device_name)
    elapsed = time.perf_counter() - start

    del batch_rewards
    del batch_actions
    del batch_obs
    del optimizer
    del reward_model
    del ppo_obs
    del ppo_model
    if device_name == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return elapsed


def configure_torch_runtime(
    device_arg: str,
    torch_num_threads: int,
    torch_num_interop_threads: int,
) -> str:
    if int(torch_num_threads) < 1:
        raise ValueError(f"--torch_num_threads must be >= 1, got {torch_num_threads}")
    if int(torch_num_interop_threads) < 1:
        raise ValueError(
            f"--torch_num_interop_threads must be >= 1, got {torch_num_interop_threads}"
        )

    torch.set_num_threads(int(torch_num_threads))
    torch.set_num_interop_threads(int(torch_num_interop_threads))

    requested_device = str(device_arg).lower()
    if requested_device == "auto":
        candidate_devices = ["cpu"]
        if _mps_is_available():
            candidate_devices.append("mps")
        benchmark_timings: Dict[str, float] = {}
        for candidate_device in candidate_devices:
            try:
                benchmark_timings[candidate_device] = _benchmark_torch_device(candidate_device)
            except Exception as exc:
                LOGGER.warning(
                    "Torch auto-device benchmark failed | candidate_device=%s | error=%s",
                    candidate_device,
                    exc,
                )
        if not benchmark_timings:
            raise RuntimeError("Torch auto-device benchmark failed for all candidate devices")
        for candidate_device, elapsed in benchmark_timings.items():
            LOGGER.info(
                "Torch auto-device benchmark | candidate_device=%s | elapsed_seconds=%.6f",
                candidate_device,
                elapsed,
            )
        resolved_device = min(benchmark_timings, key=benchmark_timings.get)
    elif requested_device == "cpu":
        resolved_device = "cpu"
    elif requested_device == "mps":
        if not _mps_is_available():
            raise ValueError("Requested --device=mps, but torch.backends.mps.is_available() is false")
        resolved_device = "mps"
    elif requested_device == "cuda":
        resolved_device = "cuda"
    else:
        raise ValueError(f"Unsupported --device value: {device_arg}")

    LOGGER.info(
        "Torch runtime configured | resolved_device=%s | torch_num_threads=%s | torch_num_interop_threads=%s | torch_get_num_threads=%s | torch_get_num_interop_threads=%s",
        resolved_device,
        torch_num_threads,
        torch_num_interop_threads,
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )
    return resolved_device


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logger = LOGGER
    level_name = str(log_level).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file)
        ensure_dir(log_path.parent)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def format_elapsed(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


def format_clock_hms(seconds: float) -> str:
    total_seconds = max(int(round(float(seconds))), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    def _fmt(value: Any) -> str:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isnan(num):
            return "nan"
        return f"{num:.3f}"

    return " | ".join(f"{key}={_fmt(metrics.get(key))}" for key in METRIC_LOG_KEYS)


def log_artifact_written(path: Path, artifact_type: str) -> None:
    LOGGER.info("Artifact written | type=%s | path=%s", artifact_type, path)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray | float) -> np.ndarray | float:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def clip_scalar(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def logit(p: float, eps: float = 1e-8) -> float:
    p = clip_scalar(p, eps, 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


def clip_radius(vec: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= radius or norm == 0.0:
        return vec
    return vec * (radius / norm)


def json_dumps(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def json_load(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_progress_task(task_id: str, name: str, unit_name: str, total_units: int) -> Dict[str, Any]:
    return {
        "task_id": str(task_id),
        "name": str(name),
        "unit_name": str(unit_name),
        "total_units": max(1, int(total_units)),
    }


class ProgressTracker:
    def __init__(
        self,
        progress_path: Optional[Path] = None,
        plan: Optional[Sequence[Dict[str, Any]]] = None,
        log_interval_sec: float = 15.0,
        milestone_fraction: float = 0.1,
    ):
        self.progress_path = Path(progress_path) if progress_path is not None else None
        self.plan = [dict(task) for task in (plan or [])]
        self.task_index_by_id = {
            str(task["task_id"]): index
            for index, task in enumerate(self.plan, start=1)
            if "task_id" in task
        }
        self.log_interval_sec = float(log_interval_sec)
        self.milestone_fraction = float(milestone_fraction)
        self.run_started_at: Optional[float] = None
        self.total_tasks = 0
        self.completed_tasks = 0
        self.current_task_id: Optional[str] = None
        self.current_task_name: Optional[str] = None
        self.current_task_unit_name: Optional[str] = None
        self.current_task_total = 0
        self.current_task_done = 0
        self.current_task_index = 0
        self.message = "Not started"
        self.last_log_time: Optional[float] = None
        self.last_logged_task_fraction = 0.0

    def start_run(self, total_tasks: int) -> None:
        self.run_started_at = time.perf_counter()
        self.total_tasks = max(1, int(total_tasks))
        self.completed_tasks = 0
        self.current_task_id = None
        self.current_task_name = None
        self.current_task_unit_name = None
        self.current_task_total = 0
        self.current_task_done = 0
        self.current_task_index = 0
        self.message = "Run initialized"
        self.last_log_time = None
        self.last_logged_task_fraction = 0.0
        self._write_default()
        self._log_progress(force=True)

    def start_task(self, task_id: str, name: str, unit_name: str, total_units: int) -> None:
        self.current_task_id = str(task_id)
        self.current_task_name = str(name)
        self.current_task_unit_name = str(unit_name)
        self.current_task_total = max(1, int(total_units))
        self.current_task_done = 0
        self.current_task_index = int(self.task_index_by_id.get(self.current_task_id, self.completed_tasks + 1))
        self.message = self.current_task_name
        self.last_logged_task_fraction = 0.0
        self._write_default()
        self._log_progress(force=True)

    def update_task(self, done_units: int, extra: Optional[str] = None) -> None:
        self.current_task_done = max(0, min(int(done_units), int(self.current_task_total)))
        if self.current_task_name is not None:
            self.message = f"{self.current_task_name} | {extra}" if extra else self.current_task_name
        elif extra:
            self.message = str(extra)
        self._write_default()
        self._log_progress(force=False)

    def finish_task(self, extra: Optional[str] = None) -> None:
        if self.current_task_name is not None:
            self.current_task_done = int(self.current_task_total)
            self.message = f"{self.current_task_name} complete | {extra}" if extra else f"{self.current_task_name} complete"
            self._write_default()
            self._log_progress(force=True)
        self.completed_tasks = min(int(self.total_tasks), int(self.completed_tasks) + 1)
        self.current_task_id = None
        self.current_task_name = None
        self.current_task_unit_name = None
        self.current_task_total = 0
        self.current_task_done = 0
        self.current_task_index = min(int(self.total_tasks), int(self.completed_tasks))
        self.message = extra if extra else self.message
        self._write_default()

    def snapshot(self) -> Dict[str, Any]:
        elapsed_sec = 0.0 if self.run_started_at is None else max(0.0, time.perf_counter() - self.run_started_at)
        task_fraction = (
            float(self.current_task_done) / float(self.current_task_total)
            if self.current_task_total > 0
            else 0.0
        )
        overall_fraction = (
            (float(self.completed_tasks) + task_fraction) / float(self.total_tasks)
            if self.total_tasks > 0
            else 0.0
        )
        overall_fraction = max(0.0, min(1.0, overall_fraction))
        eta_sec: Optional[float]
        if overall_fraction > 1e-8 and overall_fraction < 1.0:
            eta_sec = elapsed_sec * (1.0 - overall_fraction) / overall_fraction
        elif overall_fraction >= 1.0:
            eta_sec = 0.0
        else:
            eta_sec = None
        return {
            "current_task": self.current_task_name,
            "current_task_id": self.current_task_id,
            "task_index": int(self.current_task_index if self.current_task_name is not None else min(self.completed_tasks, self.total_tasks)),
            "total_tasks": int(self.total_tasks),
            "task_done": int(self.current_task_done),
            "task_total": int(self.current_task_total),
            "task_unit": self.current_task_unit_name,
            "tasks_completed": int(self.completed_tasks),
            "overall_fraction": float(overall_fraction),
            "elapsed_sec": float(elapsed_sec),
            "eta_sec": None if eta_sec is None else float(max(0.0, eta_sec)),
            "message": str(self.message),
        }

    def write_json(self, path: Path) -> None:
        json_dumps(self.snapshot(), path)

    def _write_default(self) -> None:
        if self.progress_path is not None:
            self.write_json(self.progress_path)

    def _log_progress(self, force: bool) -> None:
        snapshot = self.snapshot()
        task_total = max(1, int(snapshot["task_total"])) if snapshot["current_task"] is not None else 1
        task_done = int(snapshot["task_done"]) if snapshot["current_task"] is not None else 0
        task_fraction = float(task_done) / float(task_total)
        now = time.perf_counter()
        should_log = bool(force)
        if not should_log:
            if self.last_log_time is None or (now - self.last_log_time) >= self.log_interval_sec:
                should_log = True
            elif task_fraction >= min(1.0, self.last_logged_task_fraction + self.milestone_fraction):
                should_log = True
        if not should_log:
            return
        eta_sec = snapshot["eta_sec"]
        LOGGER.info(
            'Overall progress | task=%s/%s | current="%s" | task_progress=%s/%s %s | overall=%.1f%% | elapsed=%s | eta=%s',
            max(1, int(snapshot["task_index"])) if self.total_tasks > 0 else 0,
            int(snapshot["total_tasks"]),
            snapshot["current_task"] or "idle",
            task_done,
            task_total if snapshot["current_task"] is not None else 0,
            snapshot["task_unit"] or "units",
            100.0 * float(snapshot["overall_fraction"]),
            format_clock_hms(float(snapshot["elapsed_sec"])),
            "unknown" if eta_sec is None else format_clock_hms(float(eta_sec)),
        )
        self.last_log_time = now
        self.last_logged_task_fraction = task_fraction

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def load_agllm_release_manifest() -> Dict[str, Any]:
    return json_load(AGLLM_RELEASE_MANIFEST_PATH)


@lru_cache(maxsize=1)
def load_agllm_parse_module() -> Any:
    spec = importlib.util.spec_from_file_location("agllm_parse_runtime", AGLLM_PARSE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load AGLLM parse script at {AGLLM_PARSE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=2)
def load_agllm_prompt_template(phase: str) -> Tuple[str, str, Path]:
    manifest = load_agllm_release_manifest()
    prompt_entry = manifest.get("prompt_templates", {}).get(str(phase))
    if not isinstance(prompt_entry, dict):
        raise ValueError(f"Release manifest is missing prompt template metadata for phase={phase}")
    rel_path = prompt_entry.get("path")
    expected_hash = prompt_entry.get("sha256")
    if not rel_path or not expected_hash:
        raise ValueError(f"Release manifest prompt template for phase={phase} is incomplete")
    template_path = REPO_ROOT / str(rel_path)
    template_text = template_path.read_text(encoding="utf-8")
    actual_hash = sha256_text(template_text)
    LOGGER.info(
        "AGLLM prompt template loaded | phase=%s | path=%s | sha256=%s",
        phase,
        template_path,
        actual_hash,
    )
    if actual_hash != str(expected_hash):
        raise ValueError(
            f"AGLLM prompt template hash mismatch for phase={phase}: expected {expected_hash}, got {actual_hash}"
        )
    return template_text, actual_hash, template_path


def render_string_template(template: str, values: Dict[str, Any]) -> str:
    rendered = str(template)
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    missing = sorted(set(re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", rendered)))
    if missing:
        raise ValueError(f"Unresolved template placeholders: {missing}")
    return rendered.strip()


def ci95(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, 0.0
    half = 1.96 * float(arr.std(ddof=1)) / math.sqrt(len(arr))
    return mean, half


def df_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [headers] + df.astype(str).values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    def fmt_row(row: Sequence[str]) -> str:
        return "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"
    out = [fmt_row(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(fmt_row(row) for row in df.astype(str).values.tolist())
    return "\n".join(out)


def save_table(df: pd.DataFrame, path_stem: Path) -> None:
    csv_path = path_stem.with_suffix(".csv")
    md_path = path_stem.with_suffix(".md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df_to_markdown(df))
    log_artifact_written(csv_path, "table_csv")
    log_artifact_written(md_path, "table_markdown")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    log_artifact_written(path, "table_csv")


def fit_kmeans_single_thread(X: Any, *, n_clusters: int, random_state: int, n_init: int = 10) -> KMeans:
    km = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init=int(n_init))
    limit_ctx = threadpool_limits(limits=1) if threadpool_limits is not None else nullcontext()
    with limit_ctx:
        km.fit(X)
    return km


def action_id_to_tuple(action_id: int, P: int) -> Tuple[int, int, int]:
    lam_levels = P + 1
    z = action_id // (lam_levels * 2)
    rem = action_id % (lam_levels * 2)
    lam_idx = rem // 2
    nu = rem % 2
    return z, lam_idx, nu


def tuple_to_action_id(z: int, lam_idx: int, nu: int, P: int) -> int:
    lam_levels = P + 1
    return z * (lam_levels * 2) + lam_idx * 2 + nu


def overlap_with_window(start_abs: float, duration: float, win_start: float, win_end: float, period: float = 1440.0) -> float:
    """Exact overlap in minutes between [start_abs, start_abs+duration] and the repeating window on a 24h clock.
    win_start and win_end are in minutes on the clock. The default assumes a non-wrapping window like 00:00-06:00."""
    end_abs = start_abs + duration
    if duration <= 0:
        return 0.0
    total = 0.0
    # Check the current day and neighboring days.
    k0 = int(math.floor(start_abs / period)) - 1
    k1 = int(math.floor(end_abs / period)) + 1
    for k in range(k0, k1 + 1):
        ws = k * period + win_start
        we = k * period + win_end
        total += max(0.0, min(end_abs, we) - max(start_abs, ws))
    return total


def clock_time_in_window(clock_minute: float, win_start: float, win_end: float, period: float = 1440.0) -> bool:
    minute = float(clock_minute) % float(period)
    start = float(win_start) % float(period)
    end = float(win_end) % float(period)
    if abs(start - end) <= 1e-12:
        return False
    if start < end:
        return bool(start <= minute < end)
    return bool(minute >= start or minute < end)


def safe_mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def safe_quantile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    return float(np.quantile(np.asarray(xs, dtype=np.float64), float(q)))


def safe_relative_error(target: float, estimate: float, floor: float = 1e-8) -> float:
    denom = max(abs(float(target)), float(floor))
    return float(abs(float(estimate) - float(target)) / denom)


def summarize_distribution(values: Sequence[float]) -> Dict[str, float]:
    arr = list(map(float, values))
    return {
        "mean": safe_mean(arr),
        "median": safe_quantile(arr, 0.50),
        "p90": safe_quantile(arr, 0.90),
        "p95": safe_quantile(arr, 0.95),
        "p99": safe_quantile(arr, 0.99),
    }


def safe_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fraction_at_or_below(values: Sequence[float], threshold: float, empty_value: float = float("nan")) -> float:
    arr = np.asarray(list(map(float, values)), dtype=np.float64)
    if arr.size == 0:
        return float(empty_value)
    return float(np.mean(arr <= float(threshold)))


def count_late_night_session_starts(
    session_start_abs_minutes: Sequence[float],
    *,
    night_start: float,
    night_end: float,
) -> int:
    return int(
        sum(
            1
            for start_abs in session_start_abs_minutes
            if clock_time_in_window(float(start_abs), float(night_start), float(night_end))
        )
    )


def build_episode_fragmentation_row(
    summary: Dict[str, Any],
    *,
    cfg: BenchConfig,
    policy_name: str,
    eval_mode: str,
    train_seed: int,
    episode_index: int,
    episode_seed: int,
    backend: str,
    wrappers: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sessions = list(map(float, summary.get("sessions", [])))
    gaps = list(map(float, summary.get("gaps", [])))
    session_starts = list(map(float, summary.get("session_start_abs_minutes", [])))
    session_cap_trigger_count = int(summary.get("SessionCapTriggers_num", 0))
    gap_le1_count = int(sum(1 for gap in gaps if float(gap) <= 1.0))
    gap_le5_count = int(sum(1 for gap in gaps if float(gap) <= 5.0))
    late_night_count = count_late_night_session_starts(
        session_starts,
        night_start=float(cfg.night_start),
        night_end=float(cfg.night_end),
    )
    return {
        "policy": str(policy_name),
        "backend": str(backend),
        "eval_mode": str(eval_mode),
        "train_seed": int(train_seed),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "T_cap": (
            float(wrappers.get("T_cap", float("nan")))
            if isinstance(wrappers, dict) and bool(wrappers.get("session_cap", False))
            else float("nan")
        ),
        "total_watch": float(summary.get("CumWatch", 0.0)),
        "num_sessions": int(len(sessions)),
        "mean_within_episode_gap": safe_mean(gaps) if gaps else float("nan"),
        "median_within_episode_gap": safe_quantile(gaps, 0.50),
        "fraction_returns_within_1_min": fraction_at_or_below(gaps, 1.0),
        "fraction_returns_within_5_min": fraction_at_or_below(gaps, 5.0),
        "late_night_session_start_count": int(late_night_count),
        "late_night_session_start_rate": float(summary.get("LateNightSessionStartRate", float("nan"))),
        "session_cap_triggered": int(session_cap_trigger_count > 0),
        "session_cap_trigger_count": int(session_cap_trigger_count),
        "gap_count": int(len(gaps)),
        "gap_le1_count": int(gap_le1_count),
        "gap_le5_count": int(gap_le5_count),
    }


def metric_is_active(value: Any, eps: float = ACTIVE_COST_EPS) -> bool:
    try:
        return abs(float(value)) > float(eps)
    except (TypeError, ValueError):
        return False


def _policy_comparison_sign(delta: float, tol: float = POLICY_COMPARISON_TOL) -> int:
    if not math.isfinite(float(delta)):
        return 0
    if float(delta) > float(tol):
        return 1
    if float(delta) < -float(tol):
        return -1
    return 0


def _night_proxy_episode_value_key(metric_key: str) -> str:
    mapping = {
        "NightMinutes": "episode_night_values",
        "NightFraction": "episode_night_fraction_values",
        "LateNightSessionStartRate": "episode_late_night_session_start_rate_values",
    }
    if metric_key not in mapping:
        raise KeyError(f"Unknown night proxy metric {metric_key!r}")
    return mapping[metric_key]


def night_proxy_clears_orthogonality_bar(
    policy_mean_corr: float,
    threshold: float = NIGHT_CUMWATCH_PROXY_CORR_THRESHOLD,
) -> bool:
    return bool(math.isfinite(float(policy_mean_corr)) and abs(float(policy_mean_corr)) < float(threshold))


def night_proxy_is_scalar_proxy(
    policy_mean_corr: float,
    threshold: float = NIGHT_CUMWATCH_PROXY_CORR_THRESHOLD,
) -> bool:
    return bool(math.isfinite(float(policy_mean_corr)) and abs(float(policy_mean_corr)) >= float(threshold))


def compare_policy_orderings(
    policy_rows: Sequence[Dict[str, float]],
    metric_key: str,
    reference_key: str = "CumWatch",
    tol: float = POLICY_COMPARISON_TOL,
) -> Dict[str, Any]:
    examples: List[Dict[str, Any]] = []
    changed_pair_count = 0
    for i in range(len(policy_rows)):
        left = policy_rows[i]
        for j in range(i + 1, len(policy_rows)):
            right = policy_rows[j]
            left_ref = float(left.get(reference_key, float("nan")))
            right_ref = float(right.get(reference_key, float("nan")))
            left_metric = float(left.get(metric_key, float("nan")))
            right_metric = float(right.get(metric_key, float("nan")))
            if not all(math.isfinite(value) for value in [left_ref, right_ref, left_metric, right_metric]):
                continue
            ref_sign = _policy_comparison_sign(left_ref - right_ref, tol=tol)
            metric_sign = _policy_comparison_sign(left_metric - right_metric, tol=tol)
            if ref_sign == metric_sign:
                continue
            changed_pair_count += 1
            if len(examples) >= 5:
                continue
            examples.append(
                {
                    "pair": [str(left["policy"]), str(right["policy"])],
                    "cumwatch_order": int(ref_sign),
                    "metric_order": int(metric_sign),
                    "cumwatch_values": [float(left_ref), float(right_ref)],
                    "metric_values": [float(left_metric), float(right_metric)],
                }
            )
    return {
        "changed_policy_comparison": bool(changed_pair_count > 0),
        "changed_pair_count": int(changed_pair_count),
        "examples": examples,
    }


def summarize_night_proxy_candidate(
    policy_results: Dict[str, Sequence[Dict[str, Any]]],
    metric_key: str,
) -> Dict[str, Any]:
    policy_rows: List[Dict[str, float]] = []
    episode_cumwatch: List[float] = []
    episode_metric: List[float] = []
    for policy_name, results in policy_results.items():
        if not results:
            continue
        cumwatch_values = [float(result["CumWatch"]) for result in results if "CumWatch" in result and math.isfinite(float(result["CumWatch"]))]
        metric_values = [float(result[metric_key]) for result in results if metric_key in result and math.isfinite(float(result[metric_key]))]
        if cumwatch_values and metric_values:
            policy_rows.append(
                {
                    "policy": str(policy_name),
                    "CumWatch": safe_mean(cumwatch_values),
                    metric_key: safe_mean(metric_values),
                }
            )
        pooled = aggregate_policy_diagnostics(results)
        episode_cumwatch.extend(list(map(float, pooled.get("episode_cumwatch_values", []))))
        episode_metric.extend(list(map(float, pooled.get(_night_proxy_episode_value_key(metric_key), []))))

    policy_mean_corr = safe_corr(
        [float(row["CumWatch"]) for row in policy_rows],
        [float(row[metric_key]) for row in policy_rows],
    )
    episode_corr = safe_corr(episode_cumwatch, episode_metric)
    comparison_delta = compare_policy_orderings(policy_rows, metric_key)
    eligible_main_text = bool(
        night_proxy_clears_orthogonality_bar(policy_mean_corr)
        and bool(comparison_delta["changed_policy_comparison"])
    )
    return {
        "metric": str(metric_key),
        "policy_mean_corr": float(policy_mean_corr),
        "episode_corr": float(episode_corr),
        "num_policies": int(len(policy_rows)),
        "num_episodes": int(min(len(episode_cumwatch), len(episode_metric))),
        "changed_policy_comparison": bool(comparison_delta["changed_policy_comparison"]),
        "changed_pair_count": int(comparison_delta["changed_pair_count"]),
        "comparison_examples": list(comparison_delta["examples"]),
        "eligible_main_text": bool(eligible_main_text),
        "promotion_reason": (
            "eligible"
            if eligible_main_text
            else "abs(policy-mean corr) too high or no policy comparison changes relative to CumWatch"
        ),
        "night_minutes_is_scalar_proxy": bool(night_proxy_is_scalar_proxy(policy_mean_corr)),
    }


def build_constraint_track_spec(
    ppo_val_metrics: Dict[str, Any],
    proxy_orthogonality_audit: Dict[str, Any],
) -> Dict[str, Any]:
    ppo_night = float(ppo_val_metrics.get("NightMinutes", 0.0))
    ppo_over = float(ppo_val_metrics.get("OverCapMinutes", 0.0))
    policy_corr = float(proxy_orthogonality_audit.get("policy_mean_corr", float("nan")))
    episode_corr = float(proxy_orthogonality_audit.get("episode_corr", float("nan")))
    night_is_scalar_proxy = bool(proxy_orthogonality_audit.get("night_minutes_is_scalar_proxy", False))
    night_active = metric_is_active(ppo_night)
    over_active = metric_is_active(ppo_over)
    promoted_night_proxy = proxy_orthogonality_audit.get("promoted_main_text_proxy")
    night_appendix_only = bool(proxy_orthogonality_audit.get("night_proxy_appendix_only", promoted_night_proxy is None))
    use_night_budget = bool((not over_active) and night_active and (not night_is_scalar_proxy))
    use_over_budget = bool(over_active)
    active_channels = []
    if use_over_budget:
        active_channels.append("OverCapMinutes")
    if use_night_budget:
        active_channels.append("NightMinutes")

    if active_channels:
        status = "active"
        if use_over_budget and promoted_night_proxy:
            reason = (
                f"Using OverCapMinutes as the only constrained channel. {promoted_night_proxy} cleared the "
                "descriptive orthogonality audit for main-text reporting, but it is not a per-step budget term."
            )
        elif use_over_budget:
            reason = (
                "Using OverCapMinutes only because it is active on PPO validation; no independent night-minute budget "
                "is needed while the released observable-risk constraint is live."
            )
        elif promoted_night_proxy:
            reason = (
                f"Using NightMinutes as the constrained channel because OverCapMinutes is inactive on PPO validation and "
                f"{promoted_night_proxy} indicates night behavior is not just a scalar watch-time proxy."
            )
        else:
            reason = (
                "Using NightMinutes as the constrained channel because OverCapMinutes is inactive on PPO validation and "
                "NightMinutes is not behaving like a scalar watch-time proxy."
            )
    else:
        status = "descriptive_only" if promoted_night_proxy else "disabled"
        if promoted_night_proxy:
            reason = (
                f"{promoted_night_proxy} is retained as a descriptive main-text proxy, but OverCapMinutes is "
                "inactive on PPO validation so no constrained budget channel remains active."
            )
        elif night_appendix_only:
            reason = (
                "No constrained cost channel remains after screening. No audited night proxy clears the "
                "orthogonality bar, and OverCapMinutes is inactive on PPO validation."
            )
        else:
            reason = "No constrained cost channel is active on PPO validation."

    return {
        "status": str(status),
        "reason": str(reason),
        "active_channels": list(active_channels),
        "channels_label": ", ".join(active_channels) if active_channels else "none",
        "use_night_budget": bool(use_night_budget),
        "use_over_budget": bool(use_over_budget),
        "night_budget_active": bool(use_night_budget),
        "over_budget_active": bool(use_over_budget),
        "ppo_validation_night_minutes": float(ppo_night),
        "ppo_validation_overcap_minutes": float(ppo_over),
        "night_minutes_cumwatch_corr_policy_means": float(policy_corr),
        "night_minutes_cumwatch_corr_episodes": float(episode_corr),
        "night_minutes_is_scalar_proxy": bool(night_is_scalar_proxy),
        "main_text_night_proxy": (
            str(promoted_night_proxy)
            if promoted_night_proxy is not None
            else ("NightMinutes" if use_night_budget else None)
        ),
        "night_proxy_appendix_only": bool(night_appendix_only),
    }


def build_night_proxy_orthogonality_audit(policy_results: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    candidate_stats = {
        metric_key: summarize_night_proxy_candidate(policy_results, metric_key)
        for metric_key in NIGHT_PROXY_METRIC_KEYS
    }
    promoted_candidates = [
        metric_key
        for metric_key in NIGHT_PROXY_PROMOTION_CANDIDATES
        if bool(candidate_stats[metric_key]["eligible_main_text"])
    ]
    promoted_main_text_proxy: Optional[str] = None
    if promoted_candidates:
        promoted_main_text_proxy = sorted(
            promoted_candidates,
            key=lambda metric_key: (
                abs(float(candidate_stats[metric_key]["policy_mean_corr"])),
                -int(candidate_stats[metric_key]["changed_pair_count"]),
                str(metric_key),
            ),
        )[0]
    if promoted_main_text_proxy is None:
        reason = (
            "No audited alternative night proxy clears the main-text promotion bar. "
            "NightMinutes stays appendix-only and OverCapMinutes remains the only eligible main-text constrained channel."
        )
    else:
        reason = (
            f"Promoting {promoted_main_text_proxy} to the main-text scorecard because its |policy-mean corr| with CumWatch "
            f"is {abs(float(candidate_stats[promoted_main_text_proxy]['policy_mean_corr'])):.3f} and it changes at least one policy comparison."
        )
    legacy = candidate_stats["NightMinutes"]
    return {
        "policy_mean_corr": float(legacy["policy_mean_corr"]),
        "episode_corr": float(legacy["episode_corr"]),
        "num_policies": int(legacy["num_policies"]),
        "num_episodes": int(legacy["num_episodes"]),
        "night_minutes_is_scalar_proxy": bool(legacy["night_minutes_is_scalar_proxy"]),
        "corr_threshold": float(NIGHT_CUMWATCH_PROXY_CORR_THRESHOLD),
        "promoted_main_text_proxy": None if promoted_main_text_proxy is None else str(promoted_main_text_proxy),
        "night_proxy_appendix_only": bool(promoted_main_text_proxy is None),
        "main_text_scorecard_risk_metrics": ["OverCapMinutes"] + ([str(promoted_main_text_proxy)] if promoted_main_text_proxy is not None else []),
        "main_text_constraint_channels": ["OverCapMinutes"],
        "proxy_candidates": candidate_stats,
        "reason": str(reason),
    }


def proxy_orthogonality_audit_dataframe(proxy_orthogonality_audit: Dict[str, Any]) -> pd.DataFrame:
    promoted = proxy_orthogonality_audit.get("promoted_main_text_proxy")
    rows: List[Dict[str, Any]] = []
    for metric_key in NIGHT_PROXY_METRIC_KEYS:
        stats = proxy_orthogonality_audit.get("proxy_candidates", {}).get(metric_key, {})
        example_pairs = [
            f"{pair['pair'][0]} vs {pair['pair'][1]}"
            for pair in stats.get("comparison_examples", [])
            if isinstance(pair, dict) and isinstance(pair.get("pair"), list) and len(pair["pair"]) == 2
        ]
        rows.append(
            {
                "metric": str(metric_key),
                "policy_mean_corr_with_cumwatch": float(stats.get("policy_mean_corr", float("nan"))),
                "episode_corr_with_cumwatch": float(stats.get("episode_corr", float("nan"))),
                "changed_policy_comparison": bool(stats.get("changed_policy_comparison", False)),
                "changed_pair_count": int(stats.get("changed_pair_count", 0)),
                "eligible_main_text": bool(stats.get("eligible_main_text", False)),
                "selected_main_text_proxy": bool(metric_key == promoted),
                "promotion_pool": bool(metric_key in NIGHT_PROXY_PROMOTION_CANDIDATES),
                "policy_count": int(stats.get("num_policies", 0)),
                "episode_count": int(stats.get("num_episodes", 0)),
                "comparison_examples": "; ".join(example_pairs),
                "note": str(stats.get("promotion_reason", "")),
            }
        )
    return pd.DataFrame(rows)


def render_proxy_orthogonality_audit_markdown(proxy_orthogonality_audit: Dict[str, Any]) -> str:
    promoted = proxy_orthogonality_audit.get("promoted_main_text_proxy")
    main_text_risk_metrics = proxy_orthogonality_audit.get("main_text_scorecard_risk_metrics", [])
    lines = [
        "# Proxy Orthogonality Audit",
        "",
        f"- promotion abs(policy-mean corr) threshold: {float(proxy_orthogonality_audit.get('corr_threshold', NIGHT_CUMWATCH_PROXY_CORR_THRESHOLD)):.2f}",
        f"- promoted main-text night proxy: {promoted if promoted is not None else 'none'}",
        f"- main-text scorecard risk metrics: {', '.join(main_text_risk_metrics) if main_text_risk_metrics else 'none'}",
        f"- main-text constrained channels: {', '.join(proxy_orthogonality_audit.get('main_text_constraint_channels', ['none']))}",
        "",
        proxy_orthogonality_audit.get("reason", ""),
        "",
        "## Candidate metrics",
    ]
    for metric_key in NIGHT_PROXY_METRIC_KEYS:
        stats = proxy_orthogonality_audit.get("proxy_candidates", {}).get(metric_key, {})
        example_pairs = [
            f"{pair['pair'][0]} vs {pair['pair'][1]}"
            for pair in stats.get("comparison_examples", [])
            if isinstance(pair, dict) and isinstance(pair.get("pair"), list) and len(pair["pair"]) == 2
        ]
        lines.extend(
            [
                f"- {metric_key}: policy-mean corr={float(stats.get('policy_mean_corr', float('nan'))):.3f}, "
                f"episode corr={float(stats.get('episode_corr', float('nan'))):.3f}, "
                f"changed_pair_count={int(stats.get('changed_pair_count', 0))}, "
                f"eligible_main_text={bool(stats.get('eligible_main_text', False))}",
                f"  examples: {', '.join(example_pairs) if example_pairs else 'none'}",
            ]
        )
    return "\n".join(lines) + "\n"


def scaled_constraint_budgets(
    scale: float,
    ppo_val_metrics: Dict[str, Any],
    constraint_track_spec: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    night_budget = (
        float(scale) * float(ppo_val_metrics["NightMinutes"])
        if bool(constraint_track_spec.get("use_night_budget", False))
        else None
    )
    over_budget = (
        float(scale) * float(ppo_val_metrics["OverCapMinutes"])
        if bool(constraint_track_spec.get("use_over_budget", False))
        else None
    )
    return night_budget, over_budget


def normalize_probability_vector(values: Sequence[float], eps: float = 0.0) -> List[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return []
    arr = np.clip(arr, 0.0, None)
    if eps > 0.0:
        arr = arr + eps
    total = float(arr.sum())
    if not math.isfinite(total) or total <= 0.0:
        return (np.ones(arr.size, dtype=np.float64) / float(arr.size)).tolist()
    return (arr / total).tolist()


def quantile_summary(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p80": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p80": float(np.quantile(arr, 0.80)),
        "median": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


FEASIBILITY_QUANTILES: Tuple[Tuple[str, float], ...] = (
    ("p50", 0.50),
    ("p80", 0.80),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)


def named_quantile_summary(
    values: Sequence[float],
    quantiles: Sequence[Tuple[str, float]] = FEASIBILITY_QUANTILES,
) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    out: Dict[str, Any] = {"count": int(arr.size)}
    if arr.size == 0:
        for label, _ in quantiles:
            out[label] = float("nan")
        return out
    for label, q in quantiles:
        out[label] = float(np.quantile(arr, float(q)))
    return out


def empirical_hazard(values: Sequence[float], edges: Sequence[float]) -> Dict[str, List[float]]:
    arr = np.asarray(list(values), dtype=np.float64)
    clean = arr[np.isfinite(arr)]
    if clean.size == 0:
        return {
            "bin_labels": [f"[{edges[i]},{edges[i + 1]})" for i in range(len(edges) - 1)],
            "hazard": [0.0 for _ in range(max(0, len(edges) - 1))],
            "counts": [0 for _ in range(max(0, len(edges) - 1))],
        }
    counts, _ = np.histogram(clean, bins=np.asarray(list(edges), dtype=np.float64))
    remaining = int(clean.size)
    hazard: List[float] = []
    for count in counts.tolist():
        hazard.append(float(count) / max(1, remaining))
        remaining -= int(count)
    return {
        "bin_labels": [f"[{edges[i]},{edges[i + 1]})" for i in range(len(edges) - 1)],
        "hazard": hazard,
        "counts": [int(x) for x in counts.tolist()],
    }


def susceptibility_bin_summary(
    susceptibility_values: Sequence[float],
    metric_values: Sequence[float],
    n_bins: int = 5,
    q: float = 0.2,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    sus = np.asarray(list(susceptibility_values), dtype=np.float64)
    met = np.asarray(list(metric_values), dtype=np.float64)
    mask = np.isfinite(sus) & np.isfinite(met)
    sus = sus[mask]
    met = met[mask]
    if sus.size == 0 or met.size == 0:
        return {
            "bin_labels": [],
            "bin_means": [],
            "top_bottom_ratio": float("nan"),
            "top_mean": float("nan"),
            "bottom_mean": float("nan"),
        }
    quantiles = np.quantile(sus, np.linspace(0.0, 1.0, n_bins + 1))
    quantiles = np.asarray(quantiles, dtype=np.float64)
    for idx in range(1, quantiles.size):
        if quantiles[idx] <= quantiles[idx - 1]:
            quantiles[idx] = quantiles[idx - 1] + 1e-6
    bin_labels: List[str] = []
    bin_means: List[float] = []
    for idx in range(n_bins):
        lo = float(quantiles[idx])
        hi = float(quantiles[idx + 1])
        if idx == n_bins - 1:
            mask_bin = (sus >= lo) & (sus <= hi)
        else:
            mask_bin = (sus >= lo) & (sus < hi)
        bin_labels.append(f"{lo:.2f}-{hi:.2f}")
        bin_means.append(float(np.mean(met[mask_bin])) if np.any(mask_bin) else float("nan"))
    bottom_cut = float(np.quantile(sus, q))
    top_cut = float(np.quantile(sus, 1.0 - q))
    bottom_mask = sus <= bottom_cut
    top_mask = sus >= top_cut
    bottom_mean = float(np.mean(met[bottom_mask])) if np.any(bottom_mask) else float("nan")
    top_mean = float(np.mean(met[top_mask])) if np.any(top_mask) else float("nan")
    ratio = float(top_mean / (bottom_mean + eps)) if math.isfinite(top_mean) and math.isfinite(bottom_mean) else float("nan")
    return {
        "bin_labels": bin_labels,
        "bin_means": bin_means,
        "top_bottom_ratio": ratio,
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
    }


def compute_return_rate_metrics(
    gaps: Sequence[float],
    thresholds: Sequence[float] = RETURN_RATE_THRESHOLDS,
) -> Dict[str, float]:
    arr = np.asarray(list(gaps), dtype=np.float64)
    metrics: Dict[str, float] = {}
    for threshold in thresholds:
        key = f"ReturnRate{int(threshold) if float(threshold).is_integer() else threshold}"
        metrics[key] = float(np.mean(arr <= float(threshold))) if arr.size else float("nan")
    return metrics


def derive_prev_session_bucket_edges(session_lengths: Sequence[float]) -> List[float]:
    arr = np.asarray(list(session_lengths), dtype=np.float64)
    if arr.size == 0:
        return [30.0, 90.0]
    q1, q2 = [float(x) for x in np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])]
    if q2 <= q1 + 1e-6:
        q1 = float(np.quantile(arr, 0.50))
        q2 = float(np.quantile(arr, 0.90))
    if q2 <= q1 + 1e-6:
        q2 = q1 + 1e-3
    return [q1, q2]


def prev_session_bucket_label(session_length: float, bucket_edges: Sequence[float]) -> str:
    if len(bucket_edges) < 2:
        bucket_edges = derive_prev_session_bucket_edges([float(session_length)])
    first_edge = float(bucket_edges[0])
    second_edge = float(bucket_edges[1])
    if float(session_length) <= first_edge:
        return "short"
    if float(session_length) <= second_edge:
        return "medium"
    return "long"


def summarize_conditional_reentry(
    transitions: Sequence[Dict[str, float]],
    bucket_edges: Optional[Sequence[float]] = None,
    thresholds: Sequence[float] = RETURN_RATE_THRESHOLDS,
) -> Tuple[List[float], Dict[str, Dict[str, float]]]:
    prev_lengths = [float(row["prev_session_length"]) for row in transitions if "prev_session_length" in row]
    edges = list(bucket_edges) if bucket_edges is not None else derive_prev_session_bucket_edges(prev_lengths)
    conditional: Dict[str, Dict[str, float]] = {}
    for bucket in RETURN_BUCKET_LABELS:
        bucket_gaps = [
            float(row["gap"])
            for row in transitions
            if prev_session_bucket_label(float(row["prev_session_length"]), edges) == bucket
        ]
        bucket_summary: Dict[str, float] = {
            "count": float(len(bucket_gaps)),
            "mean_gap": safe_mean(bucket_gaps),
        }
        bucket_summary.update(compute_return_rate_metrics(bucket_gaps, thresholds=thresholds))
        conditional[bucket] = bucket_summary
    return [float(edges[0]), float(edges[1])], conditional


def empirical_entropy_from_counts(counts: Sequence[float]) -> float:
    probs = np.asarray(normalize_probability_vector(counts), dtype=np.float64)
    if probs.size == 0:
        return 0.0
    mask = probs > 0.0
    return float(-np.sum(probs[mask] * np.log(probs[mask])))


def to_probability_dict(values: Sequence[float], labels: Optional[Sequence[str]] = None) -> Dict[str, float]:
    probs = normalize_probability_vector(values)
    if labels is None:
        labels = [str(i) for i in range(len(probs))]
    return {str(label): float(prob) for label, prob in zip(labels, probs)}


def filename_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "item"


def json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def action_diagnostics_to_row(
    policy_name: str,
    split: str,
    eval_mode: str,
    train_seed: int,
    metrics: Dict[str, Any],
    category: str,
) -> Dict[str, Any]:
    return {
        "policy": policy_name,
        "category": category,
        "split": split,
        "eval_mode": eval_mode,
        "train_seed": int(train_seed),
        "CumWatch": float(metrics["CumWatch"]),
        "NightMinutes": float(metrics["NightMinutes"]),
        "NightFraction": float(metrics.get("NightFraction", float("nan"))),
        "LateNightSessionStartRate": float(metrics.get("LateNightSessionStartRate", float("nan"))),
        "OverCapMinutes": float(metrics["OverCapMinutes"]),
        "CVaR_0.95(L)": float(metrics["CVaR_0.95(L)"]),
        "ReturnRate5": float(metrics.get("ReturnRate5", float("nan"))),
        "ReturnRate15": float(metrics.get("ReturnRate15", float("nan"))),
        "ReturnRate30": float(metrics.get("ReturnRate30", float("nan"))),
        "ReturnRate60": float(metrics["ReturnRate60"]),
        "ActionEmpiricalEntropy": float(metrics.get("ActionEmpiricalEntropy", float("nan"))),
        "MeanPolicyEntropy": float(metrics.get("MeanPolicyEntropy", float("nan"))),
        "UniqueClusterCount": float(metrics.get("UniqueClusterCount", float("nan"))),
        "RepeatRate": float(metrics.get("RepeatRate", float("nan"))),
        "ImmediateRepeatRate": float(metrics.get("ImmediateRepeatRate", float("nan"))),
        "FractionNu1": float(metrics.get("FractionNu1", float("nan"))),
        "MarginalZ": json_compact(metrics.get("MarginalZ", {})),
        "MarginalLambdaIdx": json_compact(metrics.get("MarginalLambdaIdx", {})),
        "MarginalNu": json_compact(metrics.get("MarginalNu", {})),
        "PersonalizationHistogram": json_compact(metrics.get("PersonalizationHistogram", {})),
    }


def build_official_scorecard_metric_row(
    policy_name: str,
    backend: str,
    eval_mode: str,
    train_seed: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "policy": str(policy_name),
        "backend": str(backend),
        "eval_mode": str(eval_mode),
        "train_seed": int(train_seed),
        **{key: metrics[key] for key in MAIN_POLICY_METRIC_KEYS},
    }


def _episode_metric_value(values: Sequence[Any], index: int, default: float = float("nan")) -> float:
    if index >= len(values):
        return float(default)
    try:
        value = float(values[index])
    except (TypeError, ValueError):
        return float(default)
    return value


def build_official_episode_output_rows(
    policy_name: str,
    backend: str,
    eval_mode: str,
    train_seed: int,
    episode_seeds: Sequence[int],
    metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    fragmentation_rows = [dict(row) for row in metrics.get("EpisodeFragmentationRows", [])]
    session_length_lists = [list(map(float, row)) for row in metrics.get("EpisodeSessionLengths", [])]
    gap_lists = [list(map(float, row)) for row in metrics.get("EpisodeGapLists", [])]
    cum_watch_values = list(map(float, metrics.get("EpisodeCumWatchValues", [])))
    night_values = list(map(float, metrics.get("EpisodeNightValues", [])))
    night_fraction_values = list(map(float, metrics.get("EpisodeNightFractionValues", [])))
    late_night_values = list(map(float, metrics.get("EpisodeLateNightSessionStartRateValues", [])))
    over_values = list(map(float, metrics.get("EpisodeOverCapValues", [])))

    rows: List[Dict[str, Any]] = []
    total_episodes = len(episode_seeds)
    for episode_index, episode_seed in enumerate(episode_seeds, start=1):
        base_row = fragmentation_rows[episode_index - 1].copy() if episode_index - 1 < len(fragmentation_rows) else {}
        session_lengths = session_length_lists[episode_index - 1] if episode_index - 1 < len(session_length_lists) else []
        gaps = gap_lists[episode_index - 1] if episode_index - 1 < len(gap_lists) else []
        base_row.update(
            {
                "policy": str(policy_name),
                "backend": str(backend),
                "split": "test",
                "eval_mode": str(eval_mode),
                "train_seed": int(train_seed),
                "episode_index": int(episode_index),
                "episode_seed": int(episode_seed),
                "NumEpisodes": int(total_episodes),
                "CumWatch": _episode_metric_value(cum_watch_values, episode_index - 1),
                "NightMinutes": _episode_metric_value(night_values, episode_index - 1),
                "NightFraction": _episode_metric_value(
                    night_fraction_values,
                    episode_index - 1,
                    default=float(base_row.get("NightFraction", float("nan"))),
                ),
                "LateNightSessionStartRate": _episode_metric_value(
                    late_night_values,
                    episode_index - 1,
                    default=float(base_row.get("late_night_session_start_rate", float("nan"))),
                ),
                "OverCapMinutes": _episode_metric_value(over_values, episode_index - 1),
                "SessionLengthsJson": json_compact(session_lengths),
                "GapValuesJson": json_compact(gaps),
                "NumSessionLengths": int(len(session_lengths)),
                "NumGaps": int(len(gaps)),
            }
        )
        rows.append(base_row)
    return rows


def bootstrap_statistic_ci95(
    num_units: int,
    statistic_fn: Any,
    *,
    num_resamples: int = SCORECARD_BOOTSTRAP_RESAMPLES,
    seed: int = SCORECARD_BOOTSTRAP_BASE_SEED,
) -> Tuple[float, float]:
    if int(num_units) <= 0:
        return float("nan"), float("nan")
    observed = float(statistic_fn(np.arange(int(num_units), dtype=np.int64)))
    if int(num_units) == 1 or int(num_resamples) <= 1:
        return observed, 0.0
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(num_resamples), dtype=np.float64)
    for draw_index in range(int(num_resamples)):
        sampled = rng.integers(0, int(num_units), size=int(num_units), endpoint=False)
        draws[draw_index] = float(statistic_fn(sampled))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return observed, float(0.5 * (float(upper) - float(lower)))


def pooled_cvar_from_episode_session_lengths(session_length_lists: Sequence[Sequence[float]], tail_alpha: float) -> float:
    pooled = [float(length) for values in session_length_lists for length in values]
    if not pooled:
        return 0.0
    arr = np.asarray(pooled, dtype=np.float64)
    threshold = float(np.quantile(arr, float(tail_alpha)))
    tail = arr[arr >= threshold]
    if tail.size == 0:
        return 0.0
    return float(np.mean(tail))


def pooled_return_rate_from_episode_gaps(gap_lists: Sequence[Sequence[float]], threshold: float) -> float:
    pooled = [float(gap) for values in gap_lists for gap in values]
    if not pooled:
        return float("nan")
    arr = np.asarray(pooled, dtype=np.float64)
    return float(np.mean(arr <= float(threshold)))


def official_scorecard_metric_from_episode_block(
    block: pd.DataFrame,
    metric: str,
    cfg: BenchConfig,
) -> float:
    if block.empty:
        return float("nan")
    if metric in OFFICIAL_SCORECARD_SCALAR_EPISODE_METRICS:
        return float(block[metric].astype(float).mean())
    if metric == "CVaR_0.95(L)":
        session_lists = [json.loads(value) for value in block["SessionLengthsJson"].astype(str).tolist()]
        return pooled_cvar_from_episode_session_lengths(session_lists, float(cfg.tail_alpha))
    match = re.fullmatch(r"ReturnRate(\d+)", str(metric))
    if match is not None:
        threshold = float(match.group(1))
        gap_lists = [json.loads(value) for value in block["GapValuesJson"].astype(str).tolist()]
        return pooled_return_rate_from_episode_gaps(gap_lists, threshold)
    return float("nan")


def bootstrap_scorecard_metric_ci95(
    block: pd.DataFrame,
    metric: str,
    cfg: BenchConfig,
    *,
    num_resamples: int = SCORECARD_BOOTSTRAP_RESAMPLES,
    seed: int = SCORECARD_BOOTSTRAP_BASE_SEED,
) -> Tuple[float, float]:
    if block.empty:
        return float("nan"), float("nan")
    if metric in OFFICIAL_SCORECARD_SCALAR_EPISODE_METRICS:
        values = block[metric].astype(float).to_numpy(dtype=np.float64)
        return bootstrap_statistic_ci95(
            len(values),
            lambda indices: float(np.mean(values[np.asarray(indices, dtype=np.int64)])),
            num_resamples=num_resamples,
            seed=seed,
        )
    if metric == "CVaR_0.95(L)":
        session_lists = [json.loads(value) for value in block["SessionLengthsJson"].astype(str).tolist()]
        return bootstrap_statistic_ci95(
            len(session_lists),
            lambda indices: pooled_cvar_from_episode_session_lengths([session_lists[int(i)] for i in indices], float(cfg.tail_alpha)),
            num_resamples=num_resamples,
            seed=seed,
        )
    match = re.fullmatch(r"ReturnRate(\d+)", str(metric))
    if match is not None:
        threshold = float(match.group(1))
        gap_lists = [json.loads(value) for value in block["GapValuesJson"].astype(str).tolist()]
        return bootstrap_statistic_ci95(
            len(gap_lists),
            lambda indices: pooled_return_rate_from_episode_gaps([gap_lists[int(i)] for i in indices], threshold),
            num_resamples=num_resamples,
            seed=seed,
        )
    return float("nan"), float("nan")


def bootstrap_paired_metric_delta_vs_ppo(
    block: pd.DataFrame,
    ppo_block: pd.DataFrame,
    metric: str,
    cfg: BenchConfig,
    *,
    num_resamples: int = SCORECARD_BOOTSTRAP_RESAMPLES,
    seed: int = SCORECARD_BOOTSTRAP_BASE_SEED,
) -> Tuple[float, float]:
    if block.empty or ppo_block.empty:
        return float("nan"), float("nan")
    merged = block.merge(
        ppo_block,
        on=["train_seed", "episode_seed"],
        how="inner",
        suffixes=("_policy", "_ppo"),
    )
    if merged.empty:
        return float("nan"), float("nan")
    if metric in OFFICIAL_SCORECARD_SCALAR_EPISODE_METRICS:
        deltas = (
            merged[f"{metric}_policy"].astype(float).to_numpy(dtype=np.float64)
            - merged[f"{metric}_ppo"].astype(float).to_numpy(dtype=np.float64)
        )
        return bootstrap_statistic_ci95(
            len(deltas),
            lambda indices: float(np.mean(deltas[np.asarray(indices, dtype=np.int64)])),
            num_resamples=num_resamples,
            seed=seed,
        )
    if metric == "CVaR_0.95(L)":
        session_lists_policy = [json.loads(value) for value in merged["SessionLengthsJson_policy"].astype(str).tolist()]
        session_lists_ppo = [json.loads(value) for value in merged["SessionLengthsJson_ppo"].astype(str).tolist()]
        return bootstrap_statistic_ci95(
            len(session_lists_policy),
            lambda indices: pooled_cvar_from_episode_session_lengths([session_lists_policy[int(i)] for i in indices], float(cfg.tail_alpha))
            - pooled_cvar_from_episode_session_lengths([session_lists_ppo[int(i)] for i in indices], float(cfg.tail_alpha)),
            num_resamples=num_resamples,
            seed=seed,
        )
    match = re.fullmatch(r"ReturnRate(\d+)", str(metric))
    if match is not None:
        threshold = float(match.group(1))
        gap_lists_policy = [json.loads(value) for value in merged["GapValuesJson_policy"].astype(str).tolist()]
        gap_lists_ppo = [json.loads(value) for value in merged["GapValuesJson_ppo"].astype(str).tolist()]
        return bootstrap_statistic_ci95(
            len(gap_lists_policy),
            lambda indices: pooled_return_rate_from_episode_gaps([gap_lists_policy[int(i)] for i in indices], threshold)
            - pooled_return_rate_from_episode_gaps([gap_lists_ppo[int(i)] for i in indices], threshold),
            num_resamples=num_resamples,
            seed=seed,
        )
    return float("nan"), float("nan")


def augment_official_scorecard_with_episode_uncertainty(
    scorecard_df: pd.DataFrame,
    official_episode_df: pd.DataFrame,
    cfg: BenchConfig,
    *,
    num_resamples: int = SCORECARD_BOOTSTRAP_RESAMPLES,
) -> pd.DataFrame:
    if scorecard_df.empty:
        return scorecard_df
    augmented = scorecard_df.copy()
    episode_df = official_episode_df.copy()
    for metric in OFFICIAL_SCORECARD_BOOTSTRAP_METRICS:
        ci_col = f"{metric}_ci95"
        train_seed_ci_col = f"{metric}_train_seed_ci95"
        if ci_col in augmented.columns:
            augmented = augmented.rename(columns={ci_col: train_seed_ci_col})
        elif train_seed_ci_col not in augmented.columns:
            augmented[train_seed_ci_col] = float("nan")
        augmented[ci_col] = float("nan")
        augmented[f"{metric}_delta_vs_PPO"] = float("nan")
        augmented[f"{metric}_delta_vs_PPO_ci95"] = float("nan")

    for row_index, row in augmented.iterrows():
        policy_name = str(row["policy"])
        backend = str(row["backend"])
        eval_mode = str(row["eval_mode"])
        block = episode_df[
            (episode_df["policy"] == policy_name)
            & (episode_df["backend"] == backend)
            & (episode_df["eval_mode"] == eval_mode)
        ].copy()
        ppo_block = episode_df[
            (episode_df["policy"] == "PPO")
            & (episode_df["backend"] == backend)
            & (episode_df["eval_mode"] == eval_mode)
        ].copy()
        for metric_index, metric in enumerate(OFFICIAL_SCORECARD_BOOTSTRAP_METRICS, start=1):
            metric_seed = SCORECARD_BOOTSTRAP_BASE_SEED + metric_index * 1009 + row_index * 7919
            observed_metric, episode_halfwidth = bootstrap_scorecard_metric_ci95(
                block,
                metric,
                cfg,
                num_resamples=num_resamples,
                seed=metric_seed,
            )
            if math.isfinite(observed_metric):
                augmented.at[row_index, metric] = observed_metric
            augmented.at[row_index, f"{metric}_ci95"] = episode_halfwidth
            if policy_name == "PPO":
                augmented.at[row_index, f"{metric}_delta_vs_PPO"] = 0.0
                augmented.at[row_index, f"{metric}_delta_vs_PPO_ci95"] = 0.0
                continue
            observed_delta, delta_halfwidth = bootstrap_paired_metric_delta_vs_ppo(
                block,
                ppo_block,
                metric,
                cfg,
                num_resamples=num_resamples,
                seed=metric_seed + 313,
            )
            augmented.at[row_index, f"{metric}_delta_vs_PPO"] = observed_delta
            augmented.at[row_index, f"{metric}_delta_vs_PPO_ci95"] = delta_halfwidth
    return augmented


def build_official_scorecard_column_order(
    available_columns: Sequence[str],
    *,
    main_text_night_proxy: Optional[str] = None,
    include_returnrate60: bool = False,
) -> List[str]:
    metric_cols = ["CumWatch", "CVaR_0.95(L)", "OverCapMinutes"]
    if isinstance(main_text_night_proxy, str) and main_text_night_proxy:
        metric_cols.insert(2, str(main_text_night_proxy))
    if bool(include_returnrate60):
        metric_cols.append("ReturnRate60")
    columns = ["policy", "backend", "eval_mode"]
    for metric in metric_cols:
        columns.extend(
            [
                metric,
                f"{metric}_ci95",
                f"{metric}_train_seed_ci95",
                f"{metric}_delta_vs_PPO",
                f"{metric}_delta_vs_PPO_ci95",
            ]
        )
    return [column for column in columns if column in set(available_columns)]


def append_duration_to_hour_bins(hour_bins: np.ndarray, start_abs_minutes: float, duration_minutes: float) -> None:
    remaining = max(0.0, float(duration_minutes))
    current = float(start_abs_minutes)
    while remaining > 1e-9:
        minute_of_day = current % 1440.0
        hour_idx = int(minute_of_day // 60.0) % 24
        hour_end = current - minute_of_day + (hour_idx + 1) * 60.0
        chunk = min(remaining, max(hour_end - current, 1e-9))
        hour_bins[hour_idx] += chunk
        current += chunk
        remaining -= chunk


DEFAULT_DELTA_SWEEP = [15.0, 30.0, 45.0, 60.0]
DEFAULT_T_REF_SWEEP = [4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 30.0, 60.0, 90.0, 120.0]
DEFAULT_ACTIVITY_THRESHOLDS = [90.0, 120.0, 150.0]
THRESHOLD_SOURCE_CHOICES = ("fixed_paper", "empirical_target", "simulator_relative")
DEFAULT_THRESHOLD_SOURCE = "fixed_paper"
FIXED_PAPER_THRESHOLDS = {
    "T_ref": 15.0,
    "T_cap": 15.0,
    "cap_grid": [10.0, 15.0, 20.0],
    "break_T": 30.0,
    "break_J": 20,
}
CALIBRATION_COMPONENT_KEYS = ("session", "session_item_counts", "hazard", "gaps", "cluster", "diurnal", "return_conditional")
CALIBRATION_PRIMARY_COMPONENT_KEYS = ("session", "session_item_counts")
CALIBRATION_SECONDARY_COMPONENT_KEYS = ("gaps", "hazard", "cluster", "diurnal", "return_conditional")
CALIBRATION_QUANTILES = {"p50": 0.50, "p95": 0.95}
CALIBRATION_HAZARD_ELAPSED_MINUTE_BINS = 20
CALIBRATION_HAZARD_SUPPORT_RATIO_SEVERE = 0.50
CALIBRATION_HAZARD_SUPPORT_ABS_GAP_SEVERE = 3
CALIBRATION_LOSS_PRIMARY_MULTIPLIER = 1000.0
CALIBRATION_COMPONENT_MIN_MASS = {
    "session": 0.20,
    "session_item_counts": 0.20,
    "hazard": 0.15,
    "gaps": 0.15,
    "return_conditional": 0.10,
}
CALIBRATION_COMPONENT_MAX_MASS = {
    "cluster": 0.10,
    "diurnal": 0.10,
}
CALIBRATION_ACCEPTANCE_LIMITS = {
    "session_mean_relative_error_max": 0.50,
    "session_p95_relative_error_max": 0.50,
    "session_item_count_p95_relative_error_max": 0.50,
    "gap_mean_relative_error_max": 0.50,
    "gap_p95_relative_error_max": 0.50,
    "return_conditional_mse_max": 0.25,
    "session_ks_max": 0.25,
    "raw_hazard_support_ratio_min": CALIBRATION_HAZARD_SUPPORT_RATIO_SEVERE,
    "raw_hazard_support_abs_gap_min": CALIBRATION_HAZARD_SUPPORT_ABS_GAP_SEVERE,
}


def make_linspace_bins(edges: Sequence[float], value: float) -> str:
    edges = list(edges)
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return f"[{edges[i]},{edges[i+1]})"
    return f"[{edges[-1]},inf)"


def flatten_dict(prefix: str, data: Dict[str, Any], out: Dict[str, Any]) -> None:
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flatten_dict(name, value, out)
        else:
            out[name] = value


def duration_bucket_label(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 15.0:
        return "0-15s"
    if seconds < 30.0:
        return "15-30s"
    if seconds < 60.0:
        return "30-60s"
    return "60s+"


def popularity_bucket_label(score: float, cutpoints: Sequence[float]) -> str:
    q = list(map(float, cutpoints))
    if float(score) < q[0]:
        return "very_low"
    if float(score) < q[1]:
        return "low"
    if float(score) < q[2]:
        return "medium"
    if float(score) < q[3]:
        return "high"
    return "very_high"


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class BenchConfig:
    # Environment topology
    H: float = 1440.0
    d: int = 16
    Z: int = 30
    k: int = 10
    P: int = 5
    scale: float = 1.0
    r_max: float = 5.0
    epsilon: float = 1e-8
    catalog_seed: int = 12345

    # Watch-time model
    sigma_eta0: float = 0.2
    sigma_eta1: float = 0.8
    sigma_xi: float = 0.2
    alpha: float = 1.0
    beta: float = 0.3
    gamma_h: float = 0.5
    gamma_f: float = 0.4
    lambda_rep: float = 0.7
    lambda_r: float = 0.1

    # State dynamics
    h_max: float = 10.0
    eta_u: float = 0.1
    belief_radius: float = 5.0
    rho_f: float = 0.08
    chi_c: float = 0.25
    a_f: float = 0.6

    # Continue model
    omega_h: float = 0.6
    omega_r: float = 0.2
    omega_rm: float = 0.4
    omega_c: float = 1.0
    omega_f: float = 0.6
    omega_o: float = 0.8
    kappa_0: float = -2.0
    kappa_h: float = 1.0
    kappa_c: float = 1.0
    continue_logit_bias: float = 0.0
    continue_logit_temp: float = 4.0
    outside_barV: float = 0.3
    outside_A: float = 0.4
    outside_phi: float = 840.0

    # Gap model
    gap_mu: float = 4.2
    gap_sigma: float = 0.8
    gap_w_h: float = 1.1
    gap_w_r: float = 0.8
    gap_w_f: float = 1.0
    gap_w_c: float = 0.9

    # User priors
    sigma_u: float = 0.15
    v_alpha: float = 2.0
    v_beta: float = 5.0
    c0_alpha: float = 8.0
    c0_beta: float = 2.0
    n_loc: float = -0.2
    n_scale: float = 0.25
    rho_h_loc: float = -3.2188758248682006  # log(0.04)
    rho_h_scale: float = 0.35
    rho_c_loc: float = -2.5257286443082556  # log(0.08)
    rho_c_scale: float = 0.35
    gamma_rh_bar: float = 0.005
    gamma_rc_bar: float = 0.010
    gamma_rf: float = 0.008
    recovery_heterogeneity_scale: float = 0.15

    # Metrics and wrappers
    night_start: float = 0.0
    night_end: float = 360.0
    tail_alpha: float = 0.95
    return_threshold: float = 60.0
    T_ref: float = 15.0
    T_cap: float = 15.0
    break_T: float = 30.0
    break_J: int = 20
    break_friction: float = 0.5
    autoplay_friction: float = 0.25
    lambda_max: float = 0.6
    session_start_hour_probs: List[float] = field(default_factory=list)
    disable_habit_state: bool = False
    threshold_source: str = DEFAULT_THRESHOLD_SOURCE
    paper_thresholds: Dict[str, Any] = field(default_factory=dict)

    # LLM fusion defaults
    a_r: float = 0.0
    b_r: float = 1.0
    a_c: float = 0.0
    b_c: float = 1.0
    omega_r_llm: float = 0.0
    omega_c_llm: float = 0.0
    llm_eps: float = 1e-4
    llm_lambda_penalty: float = 0.01

    # Misc
    device: str = "cpu"

    def lambda_values(self) -> List[float]:
        if int(self.P) <= 0:
            return [0.0]
        return [i / self.P for i in range(self.P + 1)]

    def num_actions(self) -> int:
        return self.Z * (self.P + 1) * 2

    def obs_dim(self) -> int:
        # tau/H, sin(time), cos(time), belief(d), recent_embed(d), hist(Z),
        # mean_recent_watch, last_watch, session_minutes, item_count, gap, break_flag
        return 3 + self.d + self.d + self.Z + 6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: str | Path) -> "BenchConfig":
        return cls(**json_load(path))

    def save(self, path: str | Path) -> None:
        json_dumps(self.to_dict(), Path(path))

    def ablated(self, name: str) -> "BenchConfig":
        cfg = copy.deepcopy(self)
        if name.lower() == "nohabit":
            cfg.disable_habit_state = True
            cfg.gamma_h = 0.0
        elif name.lower() == "novar":
            cfg.sigma_eta1 = cfg.sigma_eta0
        elif name.lower() == "nopers":
            cfg.P = 0
        elif name.lower() == "homogeneoususers":
            # Degenerate priors around the means.
            cfg.v_alpha, cfg.v_beta = 200.0, 500.0
            cfg.c0_alpha, cfg.c0_beta = 800.0, 200.0
            cfg.n_scale = 1e-6
            cfg.rho_h_scale = 1e-6
            cfg.rho_c_scale = 1e-6
            cfg.recovery_heterogeneity_scale = 1e-6
        else:
            raise ValueError(f"Unknown ablation: {name}")
        return cfg

    def stress_test(self, name: str) -> "BenchConfig":
        cfg = copy.deepcopy(self)
        if name.lower() == "susceptibility_shift":
            cfg.v_alpha, cfg.v_beta = 2.0, 2.0
        elif name.lower() == "slower_recovery":
            cfg.gamma_rh_bar *= 0.75
            cfg.gamma_rc_bar *= 0.75
            cfg.gamma_rf *= 0.75
        elif name.lower() == "higher_reward_variability":
            cfg.sigma_eta1 = 1.0
        else:
            raise ValueError(f"Unknown stress test: {name}")
        return cfg


@dataclass
class UserParams:
    u: np.ndarray
    anchor_z: int
    v: float
    c0: float
    n: float
    rho_h: float
    rho_c: float
    gamma_rh: float
    gamma_rc: float


@dataclass
class StepState:
    user: UserParams
    h: float
    c: float
    f: float
    tau: float
    hat_r: float
    hat_u: np.ndarray
    hist_z: List[int]
    hist_r: List[float]
    l: float
    j: int
    g: float
    br: int
    tau_start: float


def compute_post_consumption_state(cfg: BenchConfig, s: StepState, r_t: float) -> Dict[str, float]:
    hat_r_plus = (1.0 - cfg.lambda_r) * s.hat_r + cfg.lambda_r * r_t
    delta = r_t - s.hat_r
    h_plus = clip_scalar((1.0 - s.user.rho_h) * s.h + s.user.rho_h * (1.0 + s.user.v) * max(delta, 0.0), 0.0, cfg.h_max)
    if cfg.disable_habit_state:
        h_plus = 0.0
    c_plus = max(0.0, s.c - s.user.rho_c * cfg.chi_c * (r_t / max(1e-6, cfg.scale)))
    f_plus = min(1.0, (1.0 - cfg.rho_f) * s.f + cfg.rho_f * cfg.a_f * (r_t / max(1e-6, cfg.scale)))
    tau_plus = s.tau + r_t
    l_plus = s.l + r_t
    j_plus = s.j + 1
    bar_tau_plus = (tau_plus + s.tau_start) % 1440.0
    q_r = cfg.omega_h * (h_plus / max(1e-6, cfg.h_max)) + cfg.omega_r * (hat_r_plus / max(1e-6, cfg.scale))
    outside = cfg.outside_barV + cfg.outside_A * math.cos(2.0 * math.pi * (bar_tau_plus - cfg.outside_phi) / 1440.0)
    q_m = cfg.omega_rm * (hat_r_plus / max(1e-6, cfg.scale)) + cfg.omega_c * c_plus - cfg.omega_f * f_plus - cfg.omega_o * outside
    w_plus = float(sigmoid(cfg.kappa_0 + cfg.kappa_h * (h_plus / max(1e-6, cfg.h_max)) - cfg.kappa_c * c_plus))
    q_base = w_plus * math.tanh(q_r) + (1.0 - w_plus) * math.tanh(q_m)
    return {
        "hat_r_plus": float(hat_r_plus),
        "h_plus": float(h_plus),
        "c_plus": float(c_plus),
        "f_plus": float(f_plus),
        "tau_plus": float(tau_plus),
        "l_plus": float(l_plus),
        "j_plus": int(j_plus),
        "bar_tau_plus": float(bar_tau_plus),
        "q_r": float(q_r),
        "q_m": float(q_m),
        "w_plus": float(w_plus),
        "q_base": float(q_base),
    }


# ---------------------------------------------------------------------
# Catalog and semantics
# ---------------------------------------------------------------------

def build_catalog(cfg: BenchConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.catalog_seed)
    g = rng.normal(size=(cfg.Z, cfg.d))
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    x = g / (norms + cfg.epsilon)
    return x.astype(np.float32)


def _normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_default_cards(
    cfg: BenchConfig,
    metadata_csv: Optional[str] = None,
    random_seed: int = 0,
    paper_mode: bool = False,
) -> List[Dict[str, Any]]:
    """If metadata_csv is given, build deterministic card anchors from metadata.
    Otherwise emit simple synthetic placeholder cards."""
    LOGGER.info(
        "Card builder start | metadata_csv_provided=%s | paper_mode=%s",
        bool(metadata_csv),
        bool(paper_mode),
    )
    if metadata_csv is None:
        if paper_mode:
            LOGGER.warning("paper_mode=True but metadata_csv is missing; falling back to synthetic cards manually.")
        cards = []
        for z in range(cfg.Z):
            cards.append(
                {
                    "archetype_id": z,
                    "cluster_id": z,
                    "medoid_item_id": None,
                    "schema_version": "agllm_card_v2",
                    "index_base": 0,
                    "topic_tag": f"archetype-{z}",
                    "category_level1": "Generic",
                    "category_level2": f"Theme-{z%5}",
                    "duration_bucket": ["0-15s", "15-30s", "30-60s", "60s+"][z % 4],
                    "popularity_bucket": ["very_low", "low", "medium", "high", "very_high"][z % 5],
                }
            )
        LOGGER.info("Card builder complete | source=synthetic fallback | num_cards=%s", len(cards))
        return cards

    df = pd.read_csv(metadata_csv)
    duration_col = "duration_sec" if "duration_sec" in df.columns else ("duration_seconds" if "duration_seconds" in df.columns else None)
    required = {"caption", "category_level1", "category_level2", "interaction_count"}
    missing = sorted(required.difference(df.columns))
    if missing or duration_col is None:
        if duration_col is None:
            missing.append("duration_sec|duration_seconds")
        raise ValueError(f"Metadata CSV is missing columns: {missing}")

    texts = (
        df["caption"].fillna("").map(_normalize_text)
        + " [SEP] "
        + df["category_level1"].fillna("").map(_normalize_text)
        + " [SEP] "
        + df["category_level2"].fillna("").map(_normalize_text)
    ).tolist()

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    X = vectorizer.fit_transform(texts)
    if HAVE_KMEDOIDS:
        LOGGER.info("Card builder clustering | method=KMedoids")
        clusterer = KMedoids(n_clusters=cfg.Z, metric="cosine", random_state=random_seed)
        labels = clusterer.fit_predict(X)
        medoid_indices = clusterer.medoid_indices_
    else:
        LOGGER.info("Card builder clustering | method=KMeans fallback")
        warnings.warn("sklearn-extra is not installed; falling back to KMeans + nearest-medoid approximation.")
        km = fit_kmeans_single_thread(X, n_clusters=cfg.Z, random_state=random_seed, n_init=10)
        labels = km.labels_
        medoid_indices = []
        for k in range(cfg.Z):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                medoid_indices.append(0)
                continue
            center = km.cluster_centers_[k]
            block = X[idx]
            dists = ((block.toarray() - center) ** 2).sum(axis=1)
            medoid_indices.append(int(idx[np.argmin(dists)]))
        medoid_indices = np.asarray(medoid_indices, dtype=int)

    df = df.copy()
    df["cluster"] = labels
    cards: List[Dict[str, Any]] = []

    # Sort clusters by descending size for deterministic z assignment.
    order = (
        df.groupby("cluster")
        .size()
        .reset_index(name="size")
        .sort_values(["size", "cluster"], ascending=[False, True])["cluster"]
        .tolist()
    )
    cluster_to_z = {cluster: i for i, cluster in enumerate(order[: cfg.Z])}

    for cluster, z in cluster_to_z.items():
        block = df[df["cluster"] == cluster].copy()
        if block.empty:
            continue
        if HAVE_KMEDOIDS:
            medoid_idx = int(medoid_indices[cluster])
        else:
            medoid_idx = int(medoid_indices[cluster])
        row = df.iloc[medoid_idx]
        topic = _normalize_text(row.get("caption", ""))
        if not topic:
            topic = _normalize_text(row.get("category_level2", "")) or _normalize_text(row.get("category_level1", "")) or f"archetype-{z}"

        cat_counts = (
            block.groupby(["category_level1", "category_level2"])
            .size()
            .reset_index(name="n")
            .sort_values(["n", "category_level1", "category_level2"], ascending=[False, True, True])
        )
        cat1 = _normalize_text(cat_counts.iloc[0]["category_level1"])
        cat2 = _normalize_text(cat_counts.iloc[0]["category_level2"])

        duration = float(row.get(duration_col, 0.0))
        duration_bucket = duration_bucket_label(duration)
        q = np.quantile(df["interaction_count"].fillna(0).astype(float).values, [0.2, 0.4, 0.6, 0.8])
        ic = float(row.get("interaction_count", 0.0))
        pop_bucket = popularity_bucket_label(ic, q)
        medoid_item_id = row.get("item_id", None)

        cards.append(
            {
                "archetype_id": z,
                "cluster_id": z,
                "medoid_item_id": None if medoid_item_id is None or pd.isna(medoid_item_id) else int(medoid_item_id),
                "schema_version": "agllm_card_v2",
                "index_base": 0,
                "topic_tag": topic,
                "category_level1": cat1 or "Unknown",
                "category_level2": cat2 or "Unknown",
                "duration_bucket": duration_bucket,
                "popularity_bucket": pop_bucket,
            }
        )

    cards = sorted(cards, key=lambda x: x["archetype_id"])
    if len(cards) < cfg.Z:
        if paper_mode:
            raise ValueError(
                f"paper_mode=True requires exactly {cfg.Z} metadata-backed cards, but only {len(cards)} were constructed."
            )
        existing = {c["archetype_id"] for c in cards}
        for z in range(cfg.Z):
            if z not in existing:
                cards.append(
                    {
                        "archetype_id": z,
                        "cluster_id": z,
                        "medoid_item_id": None,
                        "schema_version": "agllm_card_v2",
                        "index_base": 0,
                        "topic_tag": f"archetype-{z}",
                        "category_level1": "Generic",
                        "category_level2": f"Theme-{z%5}",
                        "duration_bucket": ["0-15s", "15-30s", "30-60s", "60s+"][z % 4],
                        "popularity_bucket": ["very_low", "low", "medium", "high", "very_high"][z % 5],
                    }
                )
    cards = sorted(cards, key=lambda x: x["archetype_id"])
    LOGGER.info(
        "Card builder complete | source=metadata_csv | num_cards=%s | duration_col=%s",
        len(cards),
        duration_col,
    )
    return cards


def novelty_note(count: int, k: int) -> str:
    if count == 0:
        return f"not seen in the last {k} items"
    if count == 1:
        return f"seen once in the last {k} items"
    return f"repeated in the last {k} items"


def render_card(cards: Sequence[Dict[str, Any]], z: int, lam_idx: int, nu: int, hist_z: Sequence[int], wall_clock: float, cfg: BenchConfig, wrappers: Dict[str, Any]) -> Dict[str, str]:
    card = dict(cards[z])
    rep_count = int(sum(1 for zz in hist_z if zz == z))
    card["novelty_note"] = novelty_note(rep_count, cfg.k)
    intervention = "none"
    if wrappers.get("break_prompt", False):
        intervention = "break reminder active"
    elif wrappers.get("session_cap", False):
        intervention = "session cap active"
    elif wrappers.get("autoplay_off", False):
        intervention = "autoplay disabled"
    elif wrappers.get("throttle_personalization", False):
        intervention = "personalization throttle active"
    card["intervention_text"] = intervention
    card["lambda_bucket"] = str(lam_idx)
    return card


def render_persona(user: UserParams, cfg: BenchConfig) -> Tuple[str, str]:
    novelty = "explore_new_topics" if user.n > math.exp(cfg.n_loc) else "prefer_familiar_topics"
    persistence = "keep_browsing_when_engaged" if (user.v > 0.45 or user.rho_h > math.exp(cfg.rho_h_loc)) else "stop_when_sessions_get_long"
    control = "easy_to_disengage" if user.c0 > 0.75 else "hard_to_disengage"
    style = "steady_browsing" if user.rho_c < math.exp(cfg.rho_c_loc) else "momentum_prone_browsing"
    persona = json_compact(
        {
            "novelty_style": novelty,
            "persistence_style": persistence,
            "control_style": control,
            "browsing_style": style,
            "anchor_archetype_id": int(user.anchor_z),
        }
    )
    bucket = "|".join([
        "novelty_high" if user.n > math.exp(cfg.n_loc) else "novelty_low",
        "persistent_high" if (user.v > 0.45 or user.rho_h > math.exp(cfg.rho_h_loc)) else "persistent_low",
        "control_high" if user.c0 > 0.75 else "control_low",
        f"anchor_{user.anchor_z}",
    ])
    return persona, bucket


# ---------------------------------------------------------------------
# AGLLM-style cache-first scorer
# ---------------------------------------------------------------------

class AGLLMScorer:
    def __init__(
        self,
        cfg: BenchConfig,
        cards: Sequence[Dict[str, Any]],
        mode: str = "surrogate",
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        cache_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.cfg = cfg
        self.cards = cards
        self.mode = mode
        self.model_id = model_id
        self.cache_path = Path(cache_path) if cache_path else None
        self.device = device or cfg.device
        self.release_manifest_path = AGLLM_RELEASE_MANIFEST_PATH
        self.release_manifest = load_agllm_release_manifest()
        self.watch_template, self.watch_template_hash, self.watch_template_path = load_agllm_prompt_template("watch")
        self.continue_template, self.continue_template_hash, self.continue_template_path = load_agllm_prompt_template("continue")
        self.watch_cache: Dict[str, Dict[str, Any]] = {}
        self.cont_cache: Dict[str, Dict[str, Any]] = {}
        self.total_queries = 0
        self.cache_hits = 0
        self.invalid_json = 0
        self.repaired = 0
        self.fallbacks = 0
        self.model = None
        self.tokenizer = None
        self._quality_warning_counts = {
            "invalid_json": 0,
            "repair": 0,
            "fallback": 0,
        }
        LOGGER.info(
            "AGLLM release contract ready | manifest=%s | watch_template=%s | continue_template=%s",
            self.release_manifest_path,
            self.watch_template_hash,
            self.continue_template_hash,
        )
        if self.cache_path and self.cache_path.exists():
            blob = json_load(self.cache_path)
            self.watch_cache = blob.get("watch_cache", {})
            self.cont_cache = blob.get("continue_cache", {})
            LOGGER.info(
                "AGLLM cache loaded | path=%s | watch_entries=%s | continue_entries=%s",
                self.cache_path,
                len(self.watch_cache),
                len(self.cont_cache),
            )

    def reset_stats(self) -> None:
        self.total_queries = 0
        self.cache_hits = 0
        self.invalid_json = 0
        self.repaired = 0
        self.fallbacks = 0
        self._quality_warning_counts = {
            "invalid_json": 0,
            "repair": 0,
            "fallback": 0,
        }

    def save(self) -> None:
        if self.cache_path is None:
            return
        payload = {
            "watch_cache": self.watch_cache,
            "continue_cache": self.cont_cache,
            "stats": self.stats(),
            "model_id": self.model_id,
            "mode": self.mode,
        }
        json_dumps(payload, self.cache_path)
        stats = self.stats()
        LOGGER.info(
            "AGLLM cache saved | path=%s | watch_entries=%s | continue_entries=%s | hit_rate=%.3f | invalid_json_rate=%.3f | repair_rate=%.3f | fallback_rate=%.3f",
            self.cache_path,
            len(self.watch_cache),
            len(self.cont_cache),
            stats["cache_hit_rate"],
            stats["invalid_json_rate"],
            stats["repair_rate"],
            stats["fallback_rate"],
        )

    def stats(self) -> Dict[str, Any]:
        miss = max(1, self.total_queries - self.cache_hits)
        hit_rate = self.cache_hits / max(1, self.total_queries)
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": hit_rate,
            "invalid_json_rate": self.invalid_json / max(1, miss),
            "repair_rate": self.repaired / max(1, miss),
            "fallback_rate": self.fallbacks / max(1, miss),
            "watch_cache_entries": len(self.watch_cache),
            "continue_cache_entries": len(self.cont_cache),
            "watch_template_hash": self.watch_template_hash,
            "continue_template_hash": self.continue_template_hash,
            "model_id": self.model_id,
            "mode": self.mode,
        }

    def _lazy_load_model(self) -> None:
        if self.mode != "hf":
            return
        if self.model is not None and self.tokenizer is not None:
            return
        if not HAVE_TRANSFORMERS:
            raise RuntimeError("transformers is not installed but mode='hf' was requested.")
        LOGGER.info(
            "AGLLM HF backend load | model_id=%s | device=%s",
            self.model_id,
            self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    def _warn_quality_issue(
        self,
        event_type: str,
        message: str,
        *args: Any,
    ) -> None:
        count = self._quality_warning_counts.get(event_type, 0)
        if count < 5:
            LOGGER.warning(message, *args)
        self._quality_warning_counts[event_type] = count + 1

    def _watch_key(self, persona_bucket: str, z: int, bar_tau: float, l: float, j: int, rep_count: int, last_watch: float, gap: float, lam_idx: int, intervention: str) -> str:
        time_bucket = make_linspace_bins([0, 240, 480, 720, 960, 1200, 1440], bar_tau % 1440.0)
        session_bucket = make_linspace_bins([0, 5, 15, 30, 60, 120], l)
        item_bucket = make_linspace_bins([0, 1, 3, 5, 10, 20, 30], float(j))
        rep_bucket = "0" if rep_count == 0 else ("1" if rep_count == 1 else "2+")
        last_watch_bucket = make_linspace_bins([0, 0.5, 1.0, 2.0, 3.0, 5.1], last_watch)
        gap_bucket = make_linspace_bins([0, 5, 30, 60, 180, 720], gap)
        key = (
            persona_bucket,
            f"z{z}",
            time_bucket,
            session_bucket,
            item_bucket,
            rep_bucket,
            last_watch_bucket,
            gap_bucket,
            f"lam{lam_idx}",
            intervention,
        )
        return "||".join(key)

    def _continue_key(self, persona_bucket: str, z: int, bar_tau_plus: float, l_plus: float, j_plus: int, h_plus: float, c_plus: float, f_plus: float, hat_r_plus: float, r_t: float, intervention: str) -> str:
        time_bucket = make_linspace_bins([0, 240, 480, 720, 960, 1200, 1440], bar_tau_plus % 1440.0)
        session_bucket = make_linspace_bins([0, 5, 15, 30, 60, 120], l_plus)
        item_bucket = make_linspace_bins([0, 1, 3, 5, 10, 20, 30], float(j_plus))
        habit_bucket = make_linspace_bins([0, 1, 2.5, 5, 7.5, 10.1], h_plus)
        control_bucket = make_linspace_bins([0, 0.2, 0.4, 0.6, 0.8, 1.01], c_plus)
        fatigue_bucket = make_linspace_bins([0, 0.2, 0.4, 0.6, 0.8, 1.01], f_plus)
        reward_bucket = make_linspace_bins([0, 0.5, 1.0, 2.0, 3.0, 5.1], hat_r_plus)
        realized_bucket = make_linspace_bins([0, 0.5, 1.0, 2.0, 3.0, 5.1], r_t)
        key = (
            persona_bucket,
            f"z{z}",
            time_bucket,
            session_bucket,
            item_bucket,
            habit_bucket,
            control_bucket,
            fatigue_bucket,
            reward_bucket,
            realized_bucket,
            intervention,
        )
        return "||".join(key)

    def _pre_summary(self, bar_tau: float, l: float, j: int, h: float, c: float, f: float, rep_count: int, last_watch: float, gap: float) -> str:
        payload = {
            "time_bucket": make_linspace_bins([0, 240, 480, 720, 960, 1200, 1440], bar_tau % 1440.0),
            "session_minutes_bucket": make_linspace_bins([0, 5, 15, 30, 60, 120], l),
            "item_count_bucket": make_linspace_bins([0, 1, 3, 5, 10, 20, 30], float(j)),
            "habit_bucket": make_linspace_bins([0, 1, 2.5, 5, 7.5, 10.1], h),
            "control_bucket": make_linspace_bins([0, 0.2, 0.4, 0.6, 0.8, 1.01], c),
            "fatigue_bucket": make_linspace_bins([0, 0.2, 0.4, 0.6, 0.8, 1.01], f),
            "recent_repetition_bucket": "0" if rep_count == 0 else ("1" if rep_count == 1 else "2+"),
            "last_watch_bucket": make_linspace_bins([0, 0.5, 1.0, 2.0, 3.0, 5.1], last_watch),
            "gap_bucket": make_linspace_bins([0, 5, 30, 60, 180, 720], gap),
        }
        return json_compact(payload)

    def _realized_watch_bucket(self, r_t: float) -> str:
        return make_linspace_bins([0, 0.5, 1.0, 2.0, 3.0, 5.1], float(r_t))

    def _post_summary(self, bar_tau_plus: float, l_plus: float, j_plus: int, h_plus: float, c_plus: float, f_plus: float, hat_r_plus: float, r_t: float) -> str:
        payload = {
            "post_time_bucket": make_linspace_bins([0, 240, 480, 720, 960, 1200, 1440], bar_tau_plus % 1440.0),
            "post_session_minutes_bucket": make_linspace_bins([0, 5, 15, 30, 60, 120], l_plus),
            "post_item_count_bucket": make_linspace_bins([0, 1, 3, 5, 10, 20, 30], float(j_plus)),
            "habit_bucket": make_linspace_bins([0, 1, 2.5, 5, 7.5, 10.1], h_plus),
            "control_bucket": make_linspace_bins([0, 0.2, 0.4, 0.6, 0.8, 1.01], c_plus),
            "fatigue_bucket": make_linspace_bins([0, 0.2, 0.4, 0.6, 0.8, 1.01], f_plus),
            "reward_expectation_bucket": make_linspace_bins([0, 0.5, 1.0, 2.0, 3.0, 5.1], hat_r_plus),
        }
        return json_compact(payload)

    def _card_prompt_payload(self, card: Dict[str, Any]) -> str:
        stable_fields = {
            "archetype_id": int(card.get("archetype_id", 0)),
            "cluster_id": card.get("cluster_id"),
            "topic_tag": card.get("topic_tag", ""),
            "category_level1": card.get("category_level1", ""),
            "category_level2": card.get("category_level2", ""),
            "duration_bucket": card.get("duration_bucket", ""),
            "popularity_bucket": card.get("popularity_bucket", ""),
            "novelty_note": card.get("novelty_note", ""),
            "intervention_text": card.get("intervention_text", "none"),
            "lambda_bucket": card.get("lambda_bucket", "0"),
        }
        if card.get("medoid_item_id") is not None:
            stable_fields["medoid_item_id"] = int(card["medoid_item_id"])
        return json_compact(stable_fields)

    def build_watch_prompt(self, persona: str, summary: str, card: Dict[str, Any]) -> str:
        return render_string_template(
            self.watch_template,
            {
                "persona": persona,
                "state_summary": summary,
                "card": self._card_prompt_payload(card),
            },
        )

    def build_continue_prompt(self, persona: str, summary: str, card: Dict[str, Any], r_t: float) -> str:
        return render_string_template(
            self.continue_template,
            {
                "persona": persona,
                "post_state_summary": summary,
                "card": self._card_prompt_payload(card),
                "realized_watch_bucket": self._realized_watch_bucket(r_t),
            },
        )

    def _surrogate_watch(self, user: UserParams, card: Dict[str, Any], rep_count: int, l: float, j: int, c: float, f: float, z: int) -> float:
        score = 0.50
        if user.anchor_z == z:
            score += 0.12
        if user.n > math.exp(self.cfg.n_loc):
            score += 0.04 if rep_count == 0 else -0.02
        else:
            score += 0.03 if rep_count >= 2 else 0.00
        pop = card["popularity_bucket"]
        score += {"very_low": -0.04, "low": -0.01, "medium": 0.01, "high": 0.03, "very_high": 0.04}.get(pop, 0.0)
        dur = card["duration_bucket"]
        score += {"0-15s": 0.00, "15-30s": 0.02, "30-60s": 0.01, "60s+": -0.03}.get(dur, 0.0)
        score += -0.06 * f + 0.02 * c
        score += -0.02 * (l / max(1.0, self.cfg.T_ref)) - 0.01 * (j / 20.0)
        return clip_scalar(score, 0.01, 0.99)

    def _surrogate_continue(self, user: UserParams, card: Dict[str, Any], rep_count: int, bar_tau_plus: float, l_plus: float, j_plus: int, h_plus: float, c_plus: float, f_plus: float, z: int) -> float:
        score = 0.50
        if user.anchor_z == z:
            score += 0.04
        score += 0.10 if (user.v > 0.45 or user.rho_h > math.exp(self.cfg.rho_h_loc)) else -0.05
        score += 0.04 * (h_plus / max(1.0, self.cfg.h_max))
        score += 0.08 * (1.0 - c_plus)
        score += -0.08 * f_plus
        score += -0.04 if rep_count >= 2 else (0.02 if rep_count == 0 else 0.0)
        # Later clock time and longer session weakly reduce continuation
        clock = bar_tau_plus % 1440.0
        if clock >= 22 * 60 or clock <= 5 * 60:
            score -= 0.05
        score += -0.05 * min(1.0, l_plus / max(1.0, self.cfg.T_ref))
        score += -0.02 * min(1.0, j_plus / 20.0)
        if card["intervention_text"] != "none":
            score -= 0.06
        return clip_scalar(score, 0.01, 0.99)

    def _parse_json_score(
        self,
        raw: str,
        key: str,
        phase: Optional[str] = None,
        cache_key: Optional[str] = None,
        prompt_hash: Optional[str] = None,
    ) -> Tuple[Optional[float], bool, bool]:
        parser_module = load_agllm_parse_module()
        result = parser_module.parse_response(raw, key)
        if bool(result.repaired):
            self.invalid_json += 1
            self.repaired += 1
            self._warn_quality_issue(
                "invalid_json",
                "AGLLM parse invalid_json | phase=%s | score_key=%s | prompt_hash=%s | cache_key=%s | raw_len=%s",
                phase or "unknown",
                key,
                prompt_hash or "unknown",
                cache_key or "unknown",
                len(raw),
            )
            self._warn_quality_issue(
                "repair",
                "AGLLM parse repaired | phase=%s | score_key=%s | prompt_hash=%s | cache_key=%s",
                phase or "unknown",
                key,
                prompt_hash or "unknown",
                cache_key or "unknown",
            )
        elif bool(result.fallback):
            self.invalid_json += 1
            self.fallbacks += 1
            self._warn_quality_issue(
                "invalid_json",
                "AGLLM parse invalid_json | phase=%s | score_key=%s | prompt_hash=%s | cache_key=%s | raw_len=%s",
                phase or "unknown",
                key,
                prompt_hash or "unknown",
                cache_key or "unknown",
                len(raw),
            )
            self._warn_quality_issue(
                "fallback",
                "AGLLM parse fallback_to_0.5 | phase=%s | score_key=%s | prompt_hash=%s | cache_key=%s",
                phase or "unknown",
                key,
                prompt_hash or "unknown",
                cache_key or "unknown",
            )
        return float(result.score), bool(result.repaired), bool(result.fallback)

    def _hf_generate(
        self,
        prompt: str,
        key: str,
        phase: Optional[str] = None,
        cache_key: Optional[str] = None,
        prompt_hash: Optional[str] = None,
    ) -> Tuple[float, str, bool]:
        self._lazy_load_model()
        assert self.model is not None and self.tokenizer is not None
        toks = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **toks,
                do_sample=False,
                max_new_tokens=16,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(output_ids[0][toks["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        value, _, fallback = self._parse_json_score(
            raw,
            key,
            phase=phase,
            cache_key=cache_key,
            prompt_hash=prompt_hash,
        )
        if value is None:
            value = 0.5
        return float(value), raw, fallback

    def watch_score(
        self,
        user: UserParams,
        z: int,
        lam_idx: int,
        bar_tau: float,
        l: float,
        j: int,
        h: float,
        c: float,
        f: float,
        hist_z: Sequence[int],
        last_watch: float,
        gap: float,
        wrappers: Dict[str, Any],
    ) -> float:
        persona, persona_bucket = render_persona(user, self.cfg)
        card = render_card(self.cards, z, lam_idx, 0, hist_z, bar_tau, self.cfg, wrappers)
        rep_count = int(sum(1 for zz in hist_z if zz == z))
        intervention = card["intervention_text"]
        key = self._watch_key(persona_bucket, z, bar_tau, l, j, rep_count, last_watch, gap, lam_idx, intervention)
        self.total_queries += 1
        if key in self.watch_cache:
            self.cache_hits += 1
            return float(self.watch_cache[key]["score"])
        LOGGER.debug("AGLLM cache miss | phase=watch | cache_key=%s", key)

        summary = self._pre_summary(bar_tau, l, j, h, c, f, rep_count, last_watch, gap)
        prompt = self.build_watch_prompt(persona, summary, card)
        prompt_hash = sha256_text(prompt)
        if self.mode == "surrogate":
            score = self._surrogate_watch(user, card, rep_count, l, j, c, f, z)
            raw_output = json.dumps({"watch_score": score})
            fallback = False
        elif self.mode == "hf":
            score, raw_output, fallback = self._hf_generate(
                prompt,
                "watch_score",
                phase="watch",
                cache_key=key,
                prompt_hash=prompt_hash,
            )
        else:
            raise ValueError(f"Unknown scorer mode: {self.mode}")

        self.watch_cache[key] = {
            "score": float(score),
            "prompt_hash": prompt_hash,
            "template_hash": self.watch_template_hash,
            "raw_output": raw_output,
            "fallback": fallback,
            "phase": "watch",
        }
        return float(score)

    def continue_score(
        self,
        user: UserParams,
        z: int,
        lam_idx: int,
        bar_tau_plus: float,
        l_plus: float,
        j_plus: int,
        h_plus: float,
        c_plus: float,
        f_plus: float,
        hat_r_plus: float,
        r_t: float,
        hist_z: Sequence[int],
        wrappers: Dict[str, Any],
    ) -> float:
        persona, persona_bucket = render_persona(user, self.cfg)
        card = render_card(self.cards, z, lam_idx, 0, hist_z, bar_tau_plus, self.cfg, wrappers)
        rep_count = int(sum(1 for zz in hist_z if zz == z))
        intervention = card["intervention_text"]
        key = self._continue_key(persona_bucket, z, bar_tau_plus, l_plus, j_plus, h_plus, c_plus, f_plus, hat_r_plus, r_t, intervention)
        self.total_queries += 1
        if key in self.cont_cache:
            self.cache_hits += 1
            return float(self.cont_cache[key]["score"])
        LOGGER.debug("AGLLM cache miss | phase=continue | cache_key=%s", key)

        summary = self._post_summary(bar_tau_plus, l_plus, j_plus, h_plus, c_plus, f_plus, hat_r_plus, r_t)
        prompt = self.build_continue_prompt(persona, summary, card, r_t)
        prompt_hash = sha256_text(prompt)
        if self.mode == "surrogate":
            score = self._surrogate_continue(user, card, rep_count, bar_tau_plus, l_plus, j_plus, h_plus, c_plus, f_plus, z)
            raw_output = json.dumps({"continue_score": score})
            fallback = False
        elif self.mode == "hf":
            score, raw_output, fallback = self._hf_generate(
                prompt,
                "continue_score",
                phase="continue",
                cache_key=key,
                prompt_hash=prompt_hash,
            )
        else:
            raise ValueError(f"Unknown scorer mode: {self.mode}")

        self.cont_cache[key] = {
            "score": float(score),
            "prompt_hash": prompt_hash,
            "template_hash": self.continue_template_hash,
            "raw_output": raw_output,
            "fallback": fallback,
            "phase": "continue",
        }
        return float(score)


# ---------------------------------------------------------------------
# Benchmark environment
# ---------------------------------------------------------------------

class CompulsionBenchEnv:
    def __init__(
        self,
        cfg: BenchConfig,
        catalog: np.ndarray,
        cards: Sequence[Dict[str, Any]],
        backend: str = "param",
        scorer: Optional[AGLLMScorer] = None,
        wrappers: Optional[Dict[str, Any]] = None,
        ablation: Optional[str] = None,
        seed: int = 0,
    ):
        self.cfg = cfg.ablated(ablation) if ablation else copy.deepcopy(cfg)
        self.catalog = catalog
        self.cards = cards
        self.backend = backend
        self.scorer = scorer
        self.wrappers = wrappers or {}
        self.rng = np.random.default_rng(seed)
        self.state: Optional[StepState] = None

        self.episode_total_watch = 0.0
        self.episode_night_minutes = 0.0
        self.episode_overcap_minutes = 0.0
        self.episode_sessions: List[float] = []
        self.episode_session_start_abs_minutes: List[float] = []
        self.episode_gaps: List[float] = []
        self.episode_cluster_watch: List[Tuple[int, float]] = []
        self.episode_session_cap_triggers = 0
        self.override_count = 0
        self.break_prompt_states = 0
        self.break_prompt_stops = 0
        self.sum_habit = 0.0
        self.sum_ctrl_dep = 0.0
        self.num_steps = 0
        self._finalized = False

    def _sample_user(self) -> UserParams:
        cfg = self.cfg
        g = self.rng.normal(size=cfg.d)
        g = g / (np.linalg.norm(g) + cfg.epsilon)
        anchor = int(self.rng.integers(0, cfg.Z))
        u = g + cfg.sigma_u * self.catalog[anchor]
        u = u / (np.linalg.norm(u) + cfg.epsilon)
        v = float(self.rng.beta(cfg.v_alpha, cfg.v_beta))
        c0 = float(self.rng.beta(cfg.c0_alpha, cfg.c0_beta))
        n = float(self.rng.lognormal(cfg.n_loc, cfg.n_scale))
        rho_h = float(self.rng.lognormal(cfg.rho_h_loc, cfg.rho_h_scale))
        rho_c = float(self.rng.lognormal(cfg.rho_c_loc, cfg.rho_c_scale))
        z1 = float(self.rng.lognormal(0.0, cfg.recovery_heterogeneity_scale))
        z2 = float(self.rng.lognormal(0.0, cfg.recovery_heterogeneity_scale))
        gamma_rh = cfg.gamma_rh_bar * z1
        gamma_rc = cfg.gamma_rc_bar * z2
        return UserParams(
            u=u.astype(np.float32),
            anchor_z=anchor,
            v=v,
            c0=c0,
            n=n,
            rho_h=rho_h,
            rho_c=rho_c,
            gamma_rh=gamma_rh,
            gamma_rc=gamma_rc,
        )

    def _wall_clock(self, tau: float, tau_start: float) -> float:
        return (tau + tau_start) % 1440.0

    def _sample_tau_start(self) -> float:
        probs = list(getattr(self.cfg, "session_start_hour_probs", []))
        if len(probs) == 24 and float(np.sum(probs)) > 0.0:
            hour_idx = int(self.rng.choice(np.arange(24, dtype=np.int64), p=np.asarray(normalize_probability_vector(probs), dtype=np.float64)))
            return float(60.0 * hour_idx + self.rng.uniform(0.0, 60.0))
        return float(self.rng.uniform(0.0, 1440.0))

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        user = self._sample_user()
        tau_start = self._sample_tau_start()
        initial_h = 0.0
        self.state = StepState(
            user=user,
            h=initial_h,
            c=user.c0,
            f=0.0,
            tau=0.0,
            hat_r=0.0,
            hat_u=np.zeros(self.cfg.d, dtype=np.float32),
            hist_z=[],
            hist_r=[],
            l=0.0,
            j=0,
            g=0.0,
            br=0,
            tau_start=tau_start,
        )
        self.episode_total_watch = 0.0
        self.episode_night_minutes = 0.0
        self.episode_overcap_minutes = 0.0
        self.episode_sessions = []
        self.episode_session_start_abs_minutes = []
        self.episode_gaps = []
        self.episode_cluster_watch = []
        self.episode_session_cap_triggers = 0
        self.override_count = 0
        self.break_prompt_states = 0
        self.break_prompt_stops = 0
        self.sum_habit = 0.0
        self.sum_ctrl_dep = 0.0
        self.num_steps = 0
        self._finalized = False
        return self._observe()

    def _observe(self) -> np.ndarray:
        assert self.state is not None
        s = self.state
        cfg = self.cfg
        wall = self._wall_clock(s.tau, s.tau_start)
        recent_embed = np.zeros(cfg.d, dtype=np.float32)
        if s.hist_z:
            recent_embed = self.catalog[s.hist_z].mean(axis=0).astype(np.float32)
        hist = np.zeros(cfg.Z, dtype=np.float32)
        if s.hist_z:
            for z in s.hist_z:
                hist[z] += 1.0
            hist /= max(1, len(s.hist_z))
        mean_recent_watch = float(np.mean(s.hist_r)) if s.hist_r else 0.0
        last_watch = float(s.hist_r[-1]) if s.hist_r else 0.0

        obs = np.concatenate(
            [
                np.asarray([s.tau / cfg.H, math.sin(2 * math.pi * wall / 1440.0), math.cos(2 * math.pi * wall / 1440.0)], dtype=np.float32),
                s.hat_u.astype(np.float32),
                recent_embed.astype(np.float32),
                hist.astype(np.float32),
                np.asarray(
                    [
                        mean_recent_watch / 10.0,
                        last_watch / 10.0,
                        s.l / 120.0,
                        s.j / 30.0,
                        s.g / 180.0,
                        float(s.br),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs

    def _current_tilde_x(self, z: int, lam_idx: int) -> np.ndarray:
        assert self.state is not None
        cfg = self.cfg
        lam = cfg.lambda_values()[lam_idx]
        if self.wrappers.get("throttle_personalization", False):
            lam = min(lam, float(self.wrappers.get("lambda_max", cfg.lambda_max)))
        hat_u_dir = self.state.hat_u / (np.linalg.norm(self.state.hat_u) + cfg.epsilon)
        return ((1.0 - lam) * self.catalog[z] + lam * hat_u_dir).astype(np.float32)

    def _gap_log_mean(self, h: float, hat_r: float, f: float, c: float) -> float:
        cfg = self.cfg
        h_norm = float(h) / max(1e-6, float(cfg.h_max))
        r_norm = float(hat_r) / max(1e-6, float(cfg.scale))
        gap_log_mean = (
            float(cfg.gap_mu)
            - float(cfg.gap_w_h) * h_norm
            - float(cfg.gap_w_r) * r_norm
            + float(cfg.gap_w_f) * float(f)
            + float(cfg.gap_w_c) * float(c)
        )
        return clip_scalar(float(gap_log_mean), -2.0, 8.0)

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self.state is not None
        cfg = self.cfg
        s = self.state
        session_minutes_before = float(s.l)
        if self._finalized:
            raise RuntimeError("Episode already done; call reset() before stepping again.")

        z, lam_idx, nu = action_id_to_tuple(int(action_id), cfg.P)
        tilde_x = self._current_tilde_x(z, lam_idx)
        rep_count = int(sum(1 for zz in s.hist_z if zz == z))
        nov = math.exp(-cfg.lambda_rep * rep_count)
        base_watch_logit = (
            cfg.alpha * float(np.dot(s.user.u, tilde_x))
            + cfg.beta * s.user.n * nov
            + cfg.gamma_h * s.h
            - cfg.gamma_f * s.f
        )
        watch_logit = base_watch_logit

        wall = self._wall_clock(s.tau, s.tau_start)
        last_watch = float(s.hist_r[-1]) if s.hist_r else 0.0
        if self.backend == "llm" and float(cfg.omega_r_llm) > 0.0:
            if self.scorer is None:
                raise RuntimeError("backend='llm' requires a scorer.")
            g_watch = self.scorer.watch_score(s.user, z, lam_idx, wall, s.l, s.j, s.h, s.c, s.f, s.hist_z, last_watch, s.g, self.wrappers)
            m_llm = cfg.a_r + cfg.b_r * logit(g_watch, cfg.llm_eps)
            watch_logit = (1.0 - cfg.omega_r_llm) * base_watch_logit + cfg.omega_r_llm * m_llm

        sigma_eta = cfg.sigma_eta0 if nu == 0 else cfg.sigma_eta1
        eta = float(self.rng.lognormal(-0.5 * sigma_eta**2, sigma_eta))
        xi = float(self.rng.lognormal(-0.5 * cfg.sigma_xi**2, cfg.sigma_xi))
        r_t = float(min(cfg.r_max, cfg.scale * softplus(watch_logit) * eta * xi))

        post = compute_post_consumption_state(cfg, s, r_t)
        hat_r_plus = float(post["hat_r_plus"])
        h_plus = float(post["h_plus"])
        c_plus = float(post["c_plus"])
        f_plus = float(post["f_plus"])
        tau_plus = float(post["tau_plus"])
        l_plus = float(post["l_plus"])
        j_plus = int(post["j_plus"])
        bar_tau_plus = float(post["bar_tau_plus"])
        q_r = float(post["q_r"])
        q_m = float(post["q_m"])
        w_plus = float(post["w_plus"])

        br_plus = 0
        friction = 0.0
        if self.wrappers.get("break_prompt", False):
            br_plus = 1 if (l_plus >= cfg.break_T or j_plus >= cfg.break_J) else 0
            friction += cfg.break_friction * br_plus
        if self.wrappers.get("autoplay_off", False):
            friction += cfg.autoplay_friction

        q_base = float(post["q_base"]) - friction
        q_score = q_base
        if self.backend == "llm" and float(cfg.omega_c_llm) > 0.0:
            assert self.scorer is not None
            g_cont = self.scorer.continue_score(s.user, z, lam_idx, bar_tau_plus, l_plus, j_plus, h_plus, c_plus, f_plus, hat_r_plus, r_t, s.hist_z, self.wrappers)
            q_llm = cfg.a_c + cfg.b_c * logit(g_cont, cfg.llm_eps)
            q_score = (1.0 - cfg.omega_c_llm) * q_base + cfg.omega_c_llm * q_llm

        p_continue = float(sigmoid(cfg.continue_logit_bias + cfg.continue_logit_temp * q_score))
        continue_flag = bool(self.rng.uniform() < p_continue)
        session_cap_triggered = False

        if self.wrappers.get("session_cap", False) and l_plus > float(self.wrappers.get("T_cap", cfg.T_cap)) and continue_flag:
            continue_flag = False
            session_cap_triggered = True

        # Append history and update platform belief before branching.
        hist_z = list((s.hist_z + [z])[-cfg.k :])
        hist_r = list((s.hist_r + [r_t])[-cfg.k :])
        hat_u_next = clip_radius((1.0 - cfg.eta_u) * s.hat_u + cfg.eta_u * (r_t / max(1e-6, cfg.scale)) * tilde_x, cfg.belief_radius).astype(np.float32)

        step_abs_start = s.tau + s.tau_start
        if s.j == 0 and session_minutes_before <= 1e-12:
            self.episode_session_start_abs_minutes.append(float(step_abs_start))

        # Per-step observable costs.
        night_cost = overlap_with_window(step_abs_start, r_t, cfg.night_start, cfg.night_end)
        over_cost = max(s.l + r_t - cfg.T_ref, 0.0) - max(s.l - cfg.T_ref, 0.0)

        gap = None
        session_ended = False
        session_length = None

        if continue_flag:
            self.state = StepState(
                user=s.user,
                h=h_plus,
                c=c_plus,
                f=f_plus,
                tau=tau_plus,
                hat_r=hat_r_plus,
                hat_u=hat_u_next,
                hist_z=hist_z,
                hist_r=hist_r,
                l=l_plus,
                j=j_plus,
                g=s.g,
                br=br_plus,
                tau_start=s.tau_start,
            )
        else:
            gap_log_mean = self._gap_log_mean(h_plus, hat_r_plus, f_plus, c_plus)
            gap = float(self.rng.lognormal(gap_log_mean, cfg.gap_sigma))
            h_next = h_plus * math.exp(-s.user.gamma_rh * gap)
            if cfg.disable_habit_state:
                h_next = 0.0
            c_next = c_plus + (s.user.c0 - c_plus) * (1.0 - math.exp(-s.user.gamma_rc * gap))
            f_next = f_plus * math.exp(-cfg.gamma_rf * gap)
            self.state = StepState(
                user=s.user,
                h=h_next,
                c=c_next,
                f=f_next,
                tau=tau_plus + gap,
                hat_r=hat_r_plus,
                hat_u=hat_u_next,
                hist_z=hist_z,
                hist_r=hist_r,
                l=0.0,
                j=0,
                g=gap,
                br=0,
                tau_start=s.tau_start,
            )
            session_ended = True
            session_length = l_plus
            self.episode_sessions.append(float(l_plus))
            self.episode_gaps.append(gap)

        self.episode_total_watch += r_t
        self.episode_night_minutes += night_cost
        self.episode_overcap_minutes += over_cost
        self.episode_cluster_watch.append((z, r_t))
        self.episode_session_cap_triggers += int(session_cap_triggered)
        self.override_count += int(continue_flag and q_m < 0.0)
        self.sum_habit += s.h
        self.sum_ctrl_dep += (1.0 - s.c)
        self.num_steps += 1
        if br_plus == 1:
            self.break_prompt_states += 1
            if not continue_flag:
                self.break_prompt_stops += 1

        done = self.state.tau >= cfg.H
        if done:
            self._finalize_if_needed()

        info = {
            "night_cost": night_cost,
            "over_cost": over_cost,
            "continue_prob": p_continue,
            "continued": continue_flag,
            "watch_time": r_t,
            "action_id": action_id,
            "z": z,
            "lambda_idx": lam_idx,
            "nu": nu,
            "session_minutes_before": session_minutes_before,
            "session_minutes_after": float(self.state.l),
            "session_ended": session_ended,
            "session_length": session_length,
            "gap": gap,
            "session_cap_triggered": session_cap_triggered,
            "q_m_plus": q_m,
            "q_r_plus": q_r,
            "w_plus": w_plus,
        }
        return self._observe(), r_t, done, info

    def _finalize_if_needed(self) -> None:
        if self._finalized or self.state is None:
            return
        # Horizon-censored terminal session: include realized length but do not add a gap.
        if self.state.l > 0.0:
            if not self.episode_sessions or abs(self.episode_sessions[-1] - self.state.l) > 1e-8:
                self.episode_sessions.append(float(self.state.l))
        self._finalized = True

    def episode_summary(self) -> Dict[str, Any]:
        self._finalize_if_needed()
        cluster_watch: Dict[int, List[float]] = {}
        for z, r in self.episode_cluster_watch:
            cluster_watch.setdefault(z, []).append(r)
        cluster_mean_watch = {int(k): float(np.mean(v)) for k, v in cluster_watch.items()}
        night_fraction = float(self.episode_night_minutes / max(float(self.episode_total_watch), float(NIGHT_FRACTION_EPS)))
        late_night_session_start_rate = (
            float(
                np.mean(
                    [
                        1.0 if clock_time_in_window(start_abs, self.cfg.night_start, self.cfg.night_end) else 0.0
                        for start_abs in self.episode_session_start_abs_minutes
                    ]
                )
            )
            if self.episode_session_start_abs_minutes
            else 0.0
        )
        return {
            "CumWatch": float(self.episode_total_watch),
            "NightMinutes": float(self.episode_night_minutes),
            "NightFraction": float(night_fraction),
            "LateNightSessionStartRate": float(late_night_session_start_rate),
            "OverCapMinutes": float(self.episode_overcap_minutes),
            "sessions": list(map(float, self.episode_sessions)),
            "session_start_abs_minutes": list(map(float, self.episode_session_start_abs_minutes)),
            "gaps": list(map(float, self.episode_gaps)),
            "cluster_mean_watch": cluster_mean_watch,
            "SessionCapTriggers_num": int(self.episode_session_cap_triggers),
            "OverrideRate_num": int(self.override_count),
            "OverrideRate_den": int(self.num_steps),
            "AvgHabit_num": float(self.sum_habit),
            "CtrlDepletion_num": float(self.sum_ctrl_dep),
            "latent_steps": int(self.num_steps),
            "BreakAdherence_num": int(self.break_prompt_stops),
            "BreakAdherence_den": int(self.break_prompt_states),
        }


# ---------------------------------------------------------------------
# Policy models
# ---------------------------------------------------------------------

class RandomPolicy:
    def __init__(self, num_actions: int, seed: int = 0):
        self.num_actions = num_actions
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        _ = obs, deterministic
        return int(self.rng.integers(0, self.num_actions))

    def act_with_info(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        env: Optional[CompulsionBenchEnv] = None,
        need_info: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        _ = env
        action = self.act(obs, deterministic=deterministic)
        return action, {"policy_entropy": math.log(max(1, self.num_actions))} if need_info else {}


class RewardModel(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _obs_to_model_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    obs_t = torch.from_numpy(obs).unsqueeze(0)
    return obs_t.to(device=device, dtype=torch.float32)


class MyopicPolicy:
    def __init__(self, model: RewardModel, seed: int = 0):
        self.model = model.eval()
        self.device = next(self.model.parameters()).device
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.act_with_info(obs, deterministic=deterministic)
        return action

    def act_with_info(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: Optional[CompulsionBenchEnv] = None,
        need_info: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        _ = env
        with torch.no_grad():
            logits = self.model(_obs_to_model_tensor(obs, self.device))[0]
            if deterministic and not need_info:
                action = int(torch.argmax(logits).item())
                return action, {}
            if deterministic:
                action = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits, dim=0).cpu().numpy()
                action = int(self.rng.choice(len(probs), p=probs))
            info: Dict[str, Any] = {}
            if need_info:
                info["policy_entropy"] = float(Categorical(logits=logits.unsqueeze(0)).entropy()[0].item())
            return action, info


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_reward = nn.Linear(hidden, 1)
        self.value_costs = nn.Linear(hidden, 2)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        logits = self.policy_head(x)
        v_r = self.value_reward(x).squeeze(-1)
        v_c = self.value_costs(x)
        return logits, v_r, v_c

    def dist_and_values(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        logits, v_r, v_c = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist, v_r, v_c


class FactorizedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, Z: int, P: int, hidden: int = 128):
        super().__init__()
        self.Z = int(Z)
        self.P = int(P)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.z_head = nn.Linear(hidden, self.Z)
        self.lambda_head = nn.Linear(hidden, self.P + 1)
        self.nu_head = nn.Linear(hidden, 2)
        self.value_reward = nn.Linear(hidden, 1)
        self.value_costs = nn.Linear(hidden, 2)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        z_logits = self.z_head(x)
        lambda_logits = self.lambda_head(x)
        nu_logits = self.nu_head(x)
        joint_logits = (
            z_logits[:, :, None, None]
            + lambda_logits[:, None, :, None]
            + nu_logits[:, None, None, :]
        ).reshape(obs.shape[0], -1)
        v_r = self.value_reward(x).squeeze(-1)
        v_c = self.value_costs(x)
        return joint_logits, v_r, v_c

    def dist_and_values(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        logits, v_r, v_c = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist, v_r, v_c


def build_actor_critic_model(cfg: BenchConfig, hidden_size: int = 128, policy_arch: str = "flat") -> nn.Module:
    arch = str(policy_arch).lower()
    if arch == "flat":
        return ActorCritic(cfg.obs_dim(), cfg.num_actions(), hidden=hidden_size)
    if arch == "factorized":
        return FactorizedActorCritic(cfg.obs_dim(), cfg.Z, cfg.P, hidden=hidden_size)
    raise ValueError(f"Unknown policy_arch: {policy_arch}")


class PPOPolicy:
    def __init__(self, model: nn.Module):
        self.model = model.eval()
        self.device = next(self.model.parameters()).device

    def reset(self) -> None:
        return None

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.act_with_info(obs, deterministic=deterministic)
        return action

    def act_with_info(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: Optional[CompulsionBenchEnv] = None,
        need_info: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        _ = env
        with torch.no_grad():
            logits, _, _ = self.model(_obs_to_model_tensor(obs, self.device))
            if deterministic and not need_info:
                action = int(torch.argmax(logits[0]).item())
                return action, {}
            dist = Categorical(logits=logits) if (need_info or not deterministic) else None
            if deterministic:
                action = int(torch.argmax(logits[0]).item())
            else:
                assert dist is not None
                action = int(dist.sample()[0].item())
            info: Dict[str, Any] = {}
            if need_info:
                assert dist is not None
                info["policy_entropy"] = float(dist.entropy()[0].item())
            return action, info


class ProjectedActionSpacePolicy:
    def __init__(self, base_policy: Any, source_P: int, target_P: int):
        self.base_policy = base_policy
        self.source_P = int(source_P)
        self.target_P = int(target_P)

    def reset(self) -> None:
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset()

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.act_with_info(obs, deterministic=deterministic)
        return action

    def act_with_info(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: Optional[CompulsionBenchEnv] = None,
        need_info: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        action, info = self.base_policy.act_with_info(
            obs,
            deterministic=deterministic,
            env=env,
            need_info=need_info,
        )
        if self.source_P == self.target_P:
            return int(action), info
        z, lam_idx, nu = action_id_to_tuple(int(action), self.source_P)
        projected_lambda = max(0, min(int(lam_idx), int(self.target_P)))
        projected_action = tuple_to_action_id(int(z), projected_lambda, int(nu), self.target_P)
        return projected_action, info


class RandomZLowLambdaHighVarPolicy:
    def __init__(self, cfg: BenchConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.act_with_info(obs, deterministic=deterministic)
        return action

    def act_with_info(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: Optional[CompulsionBenchEnv] = None,
        need_info: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        _ = obs, deterministic, env
        z = int(self.rng.integers(0, self.cfg.Z))
        action = tuple_to_action_id(z, 0, 1, self.cfg.P)
        return action, {"policy_entropy": math.log(max(1, self.cfg.Z))} if need_info else {}


class CyclicZLowLambdaHighVarPolicy:
    def __init__(self, cfg: BenchConfig):
        self.cfg = cfg
        self.next_z = 0

    def reset(self) -> None:
        self.next_z = 0

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.act_with_info(obs, deterministic=deterministic)
        return action

    def act_with_info(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: Optional[CompulsionBenchEnv] = None,
        need_info: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        _ = obs, deterministic, env
        z = self.next_z % self.cfg.Z
        self.next_z += 1
        action = tuple_to_action_id(z, 0, 1, self.cfg.P)
        return action, {"policy_entropy": 0.0} if need_info else {}


class RoundRobinPolicy:
    def __init__(self, cfg: BenchConfig):
        self.cfg = cfg
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.act_with_info(obs, deterministic=deterministic)
        return action

    def act_with_info(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: Optional[CompulsionBenchEnv] = None,
        need_info: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        _ = obs, deterministic, env
        z = self.step % self.cfg.Z
        lam_idx = (self.step // self.cfg.Z) % (self.cfg.P + 1)
        nu = (self.step // (self.cfg.Z * (self.cfg.P + 1))) % 2
        self.step += 1
        action = tuple_to_action_id(int(z), int(lam_idx), int(nu), self.cfg.P)
        return action, {"policy_entropy": 0.0} if need_info else {}


class LeastRecentPolicy:
    def __init__(self, cfg: BenchConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.act_with_info(obs, deterministic=deterministic)
        return action

    def act_with_info(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: Optional[CompulsionBenchEnv] = None,
        need_info: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        _ = obs, deterministic
        hist = list(env.state.hist_z) if env is not None and env.state is not None else []
        if not hist:
            z = 0 if deterministic else int(self.rng.integers(0, self.cfg.Z))
        else:
            scores = []
            for z_idx in range(self.cfg.Z):
                if z_idx not in hist:
                    scores.append((float("inf"), z_idx))
                else:
                    last_seen = max(i for i, zz in enumerate(hist) if zz == z_idx)
                    scores.append((len(hist) - last_seen, z_idx))
            best = max(scores, key=lambda item: (item[0], -item[1]))
            z = int(best[1])
        action = tuple_to_action_id(z, 0, 1, self.cfg.P)
        return action, {"policy_entropy": 0.0} if need_info else {}


class NoveltyGreedyPolicy:
    def __init__(self, cfg: BenchConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.priority_order = self._build_priority_order()
        self.priority_rank = {int(z): int(rank) for rank, z in enumerate(self.priority_order)}

    def _build_priority_order(self) -> List[int]:
        rng = np.random.default_rng(self.seed)
        return list(map(int, rng.permutation(self.cfg.Z)))

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.act_with_info(obs, deterministic=deterministic)
        return action

    def act_with_info(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: Optional[CompulsionBenchEnv] = None,
        need_info: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        _ = obs, deterministic
        hist = list(env.state.hist_z) if env is not None and env.state is not None else []
        counts = {z: hist.count(z) for z in range(self.cfg.Z)}
        min_count = min(counts.values()) if counts else 0
        candidates = [z for z, count in counts.items() if count == min_count] if counts else list(range(self.cfg.Z))
        if deterministic:
            z = int(min(candidates, key=lambda candidate: self.priority_rank.get(int(candidate), int(candidate))))
        else:
            z = int(candidates[self.rng.integers(0, len(candidates))])
        action = tuple_to_action_id(z, 0, 1, self.cfg.P)
        if not need_info:
            return action, {}
        return action, {"policy_entropy": math.log(max(1, len(candidates))) if len(candidates) > 1 else 0.0}


# ---------------------------------------------------------------------
# Data collection and metrics
# ---------------------------------------------------------------------

def make_env(
    cfg: BenchConfig,
    backend: str,
    catalog: np.ndarray,
    cards: Sequence[Dict[str, Any]],
    scorer: Optional[AGLLMScorer],
    wrappers: Optional[Dict[str, Any]],
    seed: int,
    ablation: Optional[str] = None,
) -> CompulsionBenchEnv:
    return CompulsionBenchEnv(cfg, catalog, cards, backend=backend, scorer=scorer, wrappers=wrappers, ablation=ablation, seed=seed)


def policy_act_with_info(
    policy: Any,
    obs: np.ndarray,
    deterministic: bool,
    env: Optional[CompulsionBenchEnv] = None,
    need_info: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    if hasattr(policy, "act_with_info"):
        action, info = policy.act_with_info(obs, deterministic=deterministic, env=env, need_info=need_info)
        return int(action), dict(info or {})
    return int(policy.act(obs, deterministic=deterministic)), {}


def reset_policy_state(policy: Any) -> None:
    reset_fn = getattr(policy, "reset", None)
    if callable(reset_fn):
        reset_fn()


def evaluate_policy(
    policy: Any,
    cfg: BenchConfig,
    catalog: np.ndarray,
    cards: Sequence[Dict[str, Any]],
    backend: str = "param",
    scorer: Optional[AGLLMScorer] = None,
    wrappers: Optional[Dict[str, Any]] = None,
    ablation: Optional[str] = None,
    episode_seeds: Optional[Sequence[int]] = None,
    num_episodes: int = 100,
    deterministic: bool = True,
    label: Optional[str] = None,
    log_progress: bool = False,
    log_every_episodes: Optional[int] = None,
    trace_first_episode: bool = False,
    trace_max_steps: int = 25,
    collect_policy_diagnostics: bool = True,
    train_seed: Optional[int] = None,
    progress_tracker: Optional[ProgressTracker] = None,
    progress_task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if episode_seeds is None:
        episode_seeds = list(range(num_episodes))
    if progress_tracker is not None and progress_task is not None:
        progress_tracker.start_task(
            str(progress_task["task_id"]),
            str(progress_task["name"]),
            str(progress_task["unit_name"]),
            int(progress_task["total_units"]),
        )
    reset_policy_state(policy)
    eval_label = label or policy.__class__.__name__
    eval_mode = "deterministic" if deterministic else "stochastic"
    resolved_train_seed = -1 if train_seed is None else int(train_seed)
    total_episodes = len(episode_seeds)
    LOGGER.info(
        "Evaluation start | label=%s | backend=%s | num_episodes=%s | wrappers_active=%s | ablation_active=%s | deterministic=%s",
        eval_label,
        backend,
        total_episodes,
        bool(wrappers),
        bool(ablation),
        deterministic,
    )
    summaries = []
    cum_watch_values: List[float] = []
    night_values: List[float] = []
    night_fraction_values: List[float] = []
    late_night_session_start_rate_values: List[float] = []
    over_values: List[float] = []
    all_sessions: List[float] = []
    all_gaps: List[float] = []
    cluster_watch_values: Dict[int, List[float]] = {}
    override_num = 0
    override_den = 0
    avg_habit_num = 0.0
    ctrl_dep_num = 0.0
    latent_steps = 0
    break_num = 0
    break_den = 0
    session_cap_trigger_num = 0
    episodes_with_cap_trigger = 0
    episode_session_lists: List[List[float]] = []
    episode_gap_lists: List[List[float]] = []
    episode_fragmentation_rows: List[Dict[str, Any]] = []
    episode_susceptibility_values: List[float] = []
    action_counts = np.zeros(cfg.num_actions(), dtype=np.float64)
    z_counts = np.zeros(cfg.Z, dtype=np.float64)
    lambda_counts = np.zeros(cfg.P + 1, dtype=np.float64)
    nu_counts = np.zeros(2, dtype=np.float64)
    policy_entropy_values: List[float] = []
    unique_clusters_per_episode: List[float] = []
    repeat_events = 0
    immediate_repeat_events = 0
    total_decisions = 0
    hazard_pos_counts: Dict[int, int] = {}
    hazard_stop_counts: Dict[int, int] = {}
    next_progress_pct = 10
    progress_interval = max(1, int(log_every_episodes)) if log_every_episodes is not None else None

    trace_limit = max(0, int(trace_max_steps))

    for episode_idx, seed in enumerate(episode_seeds, start=1):
        env = make_env(cfg, backend, catalog, cards, scorer, wrappers, seed=seed, ablation=ablation)
        obs = env.reset(seed=seed)
        if env.state is not None:
            episode_susceptibility_values.append(float(env.state.user.v))
        done = False
        should_trace_episode = bool(trace_first_episode and episode_idx == 1)
        trace_step_count = 0
        trace_truncated = False
        episode_unique_clusters: set[int] = set()
        current_items_in_session = 0
        if should_trace_episode:
            LOGGER.info(
                "Episode trace start | label=%s | backend=%s | episode=%s/%s | seed=%s | trace_max_steps=%s",
                eval_label,
                backend,
                episode_idx,
                total_episodes,
                seed,
                trace_limit,
            )
        while not done:
            hist_before = list(env.state.hist_z) if env.state is not None else []
            action, action_info = policy_act_with_info(
                policy,
                obs,
                deterministic=deterministic,
                env=env,
                need_info=collect_policy_diagnostics,
            )
            z_action, lam_idx_action, nu_action = action_id_to_tuple(int(action), cfg.P)
            episode_unique_clusters.add(int(z_action))
            if collect_policy_diagnostics:
                action_counts[int(action)] += 1.0
                z_counts[int(z_action)] += 1.0
                lambda_counts[int(lam_idx_action)] += 1.0
                nu_counts[int(nu_action)] += 1.0
                total_decisions += 1
                if hist_before and int(z_action) in hist_before:
                    repeat_events += 1
                if hist_before and int(z_action) == int(hist_before[-1]):
                    immediate_repeat_events += 1
                entropy_value = float(action_info.get("policy_entropy", float("nan")))
                if math.isfinite(entropy_value):
                    policy_entropy_values.append(entropy_value)
            obs, _, done, info = env.step(action)
            current_items_in_session += 1
            if info["session_ended"]:
                for position in range(1, current_items_in_session + 1):
                    hazard_pos_counts[position] = hazard_pos_counts.get(position, 0) + 1
                hazard_stop_counts[current_items_in_session] = hazard_stop_counts.get(current_items_in_session, 0) + 1
                current_items_in_session = 0
            if should_trace_episode:
                trace_step_count += 1
                if trace_step_count <= trace_limit:
                    outcome = "continue" if bool(info["continued"]) else "stop"
                    gap_text = f"{float(info['gap']):.3f}" if info["gap"] is not None else "-"
                    LOGGER.info(
                        "Episode trace step | label=%s | step=%s | action_id=%s | decoded=(z=%s, lambda_idx=%s, nu=%s) | r_t=%.3f | continue_prob=%.3f | outcome=%s | session_minutes=%.3f->%.3f | night_cost=%.3f | over_cost=%.3f | q_r_plus=%.3f | q_m_plus=%.3f | w_plus=%.3f | gap=%s",
                        eval_label,
                        trace_step_count,
                        info["action_id"],
                        info["z"],
                        info["lambda_idx"],
                        info["nu"],
                        float(info["watch_time"]),
                        float(info["continue_prob"]),
                        outcome,
                        float(info["session_minutes_before"]),
                        float(info["session_minutes_after"]),
                        float(info["night_cost"]),
                        float(info["over_cost"]),
                        float(info["q_r_plus"]),
                        float(info["q_m_plus"]),
                        float(info["w_plus"]),
                        gap_text,
                    )
                elif trace_step_count == trace_limit + 1:
                    trace_truncated = True
        unique_clusters_per_episode.append(float(len(episode_unique_clusters)))
        summary = env.episode_summary()
        if should_trace_episode:
            LOGGER.info(
                "Episode trace end | label=%s | backend=%s | episode=%s/%s | total_steps=%s | traced_steps=%s | truncated=%s | CumWatch=%.3f | NightMinutes=%.3f | OverCapMinutes=%.3f",
                eval_label,
                backend,
                episode_idx,
                total_episodes,
                trace_step_count,
                min(trace_step_count, trace_limit),
                trace_truncated,
                float(summary["CumWatch"]),
                float(summary["NightMinutes"]),
                float(summary["OverCapMinutes"]),
            )
        summaries.append(summary)
        episode_fragmentation_rows.append(
            build_episode_fragmentation_row(
                summary,
                cfg=cfg,
                policy_name=eval_label,
                eval_mode=eval_mode,
                train_seed=resolved_train_seed,
                episode_index=episode_idx,
                episode_seed=int(seed),
                backend=backend,
                wrappers=wrappers,
            )
        )
        cum_watch_values.append(float(summary["CumWatch"]))
        night_values.append(float(summary["NightMinutes"]))
        night_fraction_values.append(float(summary.get("NightFraction", float("nan"))))
        late_night_session_start_rate_values.append(float(summary.get("LateNightSessionStartRate", float("nan"))))
        over_values.append(float(summary["OverCapMinutes"]))
        all_sessions.extend(summary["sessions"])
        all_gaps.extend(summary["gaps"])
        episode_session_lists.append(list(map(float, summary["sessions"])))
        episode_gap_lists.append(list(map(float, summary["gaps"])))
        session_cap_trigger_num += int(summary.get("SessionCapTriggers_num", 0))
        episodes_with_cap_trigger += int(int(summary.get("SessionCapTriggers_num", 0)) > 0)
        override_num += summary["OverrideRate_num"]
        override_den += summary["OverrideRate_den"]
        avg_habit_num += summary["AvgHabit_num"]
        ctrl_dep_num += summary["CtrlDepletion_num"]
        latent_steps += summary["latent_steps"]
        break_num += summary["BreakAdherence_num"]
        break_den += summary["BreakAdherence_den"]
        for z, r in summary["cluster_mean_watch"].items():
            cluster_watch_values.setdefault(int(z), []).append(float(r))
        if total_episodes > 0:
            should_log = False
            if progress_interval is not None:
                should_log = (episode_idx % progress_interval == 0) or (episode_idx == total_episodes)
            else:
                completed_pct = int((episode_idx * 100) / total_episodes)
                if completed_pct >= next_progress_pct:
                    should_log = True
                    while next_progress_pct <= completed_pct and next_progress_pct <= 100:
                        next_progress_pct += 10
            if should_log and log_progress:
                LOGGER.info(
                    "Evaluation progress | label=%s | episodes=%s/%s | running_CumWatch=%.3f | running_NightMinutes=%.3f | running_OverCapMinutes=%.3f | observed_sessions=%s | observed_gaps=%s",
                    eval_label,
                    episode_idx,
                    total_episodes,
                    safe_mean(cum_watch_values),
                    safe_mean(night_values),
                    safe_mean(over_values),
                    len(all_sessions),
                    len(all_gaps),
                )
            if should_log and progress_tracker is not None and progress_task is not None:
                progress_tracker.update_task(
                    episode_idx,
                    extra=f"{eval_label} | running_CumWatch={safe_mean(cum_watch_values):.3f}",
                )

    cum_watch = cum_watch_values
    night = night_values
    over = over_values
    if all_sessions:
        q = np.quantile(all_sessions, cfg.tail_alpha)
        cvar = float(np.mean([x for x in all_sessions if x >= q]))
        p99 = float(np.quantile(all_sessions, 0.99))
    else:
        cvar = 0.0
        p99 = 0.0
    return_rate_metrics = compute_return_rate_metrics(all_gaps, thresholds=RETURN_RATE_THRESHOLDS)
    fragmentation_return_rate_metrics = compute_return_rate_metrics(all_gaps, thresholds=CAP_FRAGMENTATION_GAP_THRESHOLDS)
    return_rate = float(np.mean(np.asarray(all_gaps, dtype=np.float64) <= float(cfg.return_threshold))) if all_gaps else float("nan")
    session_count_values = [float(row["num_sessions"]) for row in episode_fragmentation_rows]
    late_night_session_start_count_values = [float(row["late_night_session_start_count"]) for row in episode_fragmentation_rows]
    mean_gap_values = [
        float(row["mean_within_episode_gap"])
        for row in episode_fragmentation_rows
        if math.isfinite(float(row["mean_within_episode_gap"]))
    ]
    median_gap_values = [
        float(row["median_within_episode_gap"])
        for row in episode_fragmentation_rows
        if math.isfinite(float(row["median_within_episode_gap"]))
    ]
    cluster_means = {int(z): safe_mean(v) for z, v in cluster_watch_values.items()}
    lambda_labels = [f"{value:.2f}" for value in cfg.lambda_values()]
    action_empirical_entropy = empirical_entropy_from_counts(action_counts.tolist()) if collect_policy_diagnostics else float("nan")
    marginal_z = to_probability_dict(z_counts.tolist()) if collect_policy_diagnostics else {}
    marginal_lambda = to_probability_dict(lambda_counts.tolist()) if collect_policy_diagnostics else {}
    marginal_nu = to_probability_dict(nu_counts.tolist()) if collect_policy_diagnostics else {}
    personalization_histogram = to_probability_dict(lambda_counts.tolist(), labels=lambda_labels) if collect_policy_diagnostics else {}
    max_hazard_pos = max(hazard_pos_counts) if hazard_pos_counts else 0
    stop_hazard = [
        float(hazard_stop_counts.get(position, 0) / max(1, hazard_pos_counts.get(position, 0)))
        for position in range(1, max_hazard_pos + 1)
    ]

    result = {
        "CumWatch": safe_mean(cum_watch),
        "NightMinutes": safe_mean(night),
        "NightFraction": safe_mean([value for value in night_fraction_values if math.isfinite(value)]),
        "LateNightSessionStartRate": safe_mean([value for value in late_night_session_start_rate_values if math.isfinite(value)]),
        "OverCapMinutes": safe_mean(over),
        "CVaR_0.95(L)": cvar,
        "p99_L": p99,
        "ReturnRate60": return_rate,
        "FractionReturnsWithin1Min": float(fragmentation_return_rate_metrics.get("ReturnRate1", float("nan"))),
        "FractionReturnsWithin5Min": float(fragmentation_return_rate_metrics.get("ReturnRate5", float("nan"))),
        "ReturnRate5": float(return_rate_metrics.get("ReturnRate5", float("nan"))),
        "ReturnRate15": float(return_rate_metrics.get("ReturnRate15", float("nan"))),
        "ReturnRate30": float(return_rate_metrics.get("ReturnRate30", float("nan"))),
        "NumEpisodes": len(summaries),
        "NumSessions": len(all_sessions),
        "SessionsPerEpisode": safe_mean(session_count_values),
        "LateNightSessionStartsPerEpisode": safe_mean(late_night_session_start_count_values),
        "MeanWithinEpisodeGap": safe_mean(mean_gap_values) if mean_gap_values else float("nan"),
        "MedianWithinEpisodeGap": safe_mean(median_gap_values) if median_gap_values else float("nan"),
        "OverrideRate": override_num / max(1, override_den),
        "AvgHabit": avg_habit_num / max(1, latent_steps),
        "CtrlDepletion": ctrl_dep_num / max(1, latent_steps),
        "BreakAdherence": (break_num / break_den) if break_den > 0 else float("nan"),
        "BreakAdherence_num": int(break_num),
        "BreakAdherence_den": int(break_den),
        "EpisodeCumWatchValues": list(map(float, cum_watch)),
        "EpisodeNightValues": list(map(float, night)),
        "EpisodeNightFractionValues": [float(value) for value in night_fraction_values if math.isfinite(value)],
        "EpisodeLateNightSessionStartRateValues": [float(value) for value in late_night_session_start_rate_values if math.isfinite(value)],
        "EpisodeOverCapValues": list(map(float, over)),
        "EpisodeSessionLengths": episode_session_lists,
        "EpisodeGapLists": episode_gap_lists,
        "EpisodeFragmentationRows": copy.deepcopy(episode_fragmentation_rows),
        "EpisodeSusceptibilityValues": list(map(float, episode_susceptibility_values)),
        "EpisodeOverCapPositiveRate": float(np.mean(np.asarray(over, dtype=np.float64) > 0.0)) if over else 0.0,
        "NightCumWatchCorr": safe_corr(cum_watch, night),
        "SessionCapTriggerRate": session_cap_trigger_num / max(1, len(all_sessions)),
        "EpisodeSessionCapTriggerRate": episodes_with_cap_trigger / max(1, len(summaries)),
        "SessionCapTriggerCount": int(session_cap_trigger_num),
        "SessionLengths": all_sessions,
        "Gaps": all_gaps,
        "StopHazard": stop_hazard,
        "StopHazardPositions": list(range(1, len(stop_hazard) + 1)),
        "ClusterMeanWatch": cluster_means,
        "ActionEmpiricalEntropy": float(action_empirical_entropy),
        "MeanPolicyEntropy": safe_mean(policy_entropy_values),
        "UniqueClusterCount": safe_mean(unique_clusters_per_episode),
        "RepeatRate": (repeat_events / max(1, total_decisions)) if collect_policy_diagnostics else float("nan"),
        "ImmediateRepeatRate": (immediate_repeat_events / max(1, total_decisions)) if collect_policy_diagnostics else float("nan"),
        "MarginalZ": marginal_z,
        "MarginalLambdaIdx": marginal_lambda,
        "MarginalNu": marginal_nu,
        "PersonalizationHistogram": personalization_histogram,
        "FractionNu1": float(marginal_nu.get("1", 0.0)) if collect_policy_diagnostics else float("nan"),
    }
    LOGGER.info(
        "Evaluation complete | label=%s | backend=%s | CumWatch=%.3f | CVaR_0.95(L)=%.3f | ReturnRate60=%.3f | NightMinutes=%.3f | OverCapMinutes=%.3f | OverrideRate=%.3f | BreakAdherence=%.3f",
        eval_label,
        backend,
        result["CumWatch"],
        result["CVaR_0.95(L)"],
        result["ReturnRate60"],
        result["NightMinutes"],
        result["OverCapMinutes"],
        result["OverrideRate"],
        result["BreakAdherence"],
    )
    if progress_tracker is not None and progress_task is not None:
        progress_tracker.finish_task(extra=f"{eval_label} | CumWatch={result['CumWatch']:.3f}")
    return result


def aggregate_across_train_seeds(records: List[Dict[str, Any]], group_cols: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    rows = []
    metric_cols = [c for c in df.columns if c not in group_cols and c not in {"train_seed"}]
    for keys, block in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        for metric in metric_cols:
            mean, half = ci95(block[metric].astype(float).tolist())
            row[metric] = mean
            row[f"{metric}_ci95"] = half
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols))


def collect_logged_dataset(
    cfg: BenchConfig,
    catalog: np.ndarray,
    cards: Sequence[Dict[str, Any]],
    num_transitions: int,
    seed: int,
    progress_tracker: Optional[ProgressTracker] = None,
    progress_task: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    num_actions = cfg.num_actions()
    if progress_tracker is not None and progress_task is not None:
        progress_tracker.start_task(
            str(progress_task["task_id"]),
            str(progress_task["name"]),
            str(progress_task["unit_name"]),
            int(progress_task["total_units"]),
        )
    LOGGER.info(
        "Logged dataset collection start | num_transitions=%s | seed=%s | num_actions=%s",
        num_transitions,
        seed,
        num_actions,
    )
    env = make_env(cfg, "param", catalog, cards, scorer=None, wrappers=None, seed=seed)
    obs = env.reset(seed=seed)
    X, A, R = [], [], []
    next_progress_pct = 10
    while len(X) < num_transitions:
        action = int(env.rng.integers(0, num_actions))
        next_obs, reward, done, _ = env.step(action)
        X.append(obs.copy())
        A.append(action)
        R.append(reward)
        obs = env.reset(seed=int(env.rng.integers(0, 2**31 - 1))) if done else next_obs
        completed_pct = int((len(X) * 100) / num_transitions)
        if completed_pct >= next_progress_pct:
            LOGGER.info(
                "Logged dataset collection progress | collected=%s/%s (%s%%)",
                len(X),
                num_transitions,
                completed_pct,
            )
            if progress_tracker is not None and progress_task is not None:
                progress_tracker.update_task(len(X), extra=f"collected={len(X)}/{num_transitions}")
            while next_progress_pct <= completed_pct and next_progress_pct < 100:
                next_progress_pct += 10
    LOGGER.info(
        "Logged dataset collection complete | final_transition_count=%s",
        len(X),
    )
    if progress_tracker is not None and progress_task is not None:
        progress_tracker.finish_task(extra=f"transitions={len(X)}")
    return {
        "obs": np.asarray(X, dtype=np.float32),
        "action": np.asarray(A, dtype=np.int64),
        "reward": np.asarray(R, dtype=np.float32),
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_myopic_reward_model(
    dataset: Dict[str, np.ndarray],
    obs_dim: int,
    num_actions: int,
    seed: int,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu",
    progress_tracker: Optional[ProgressTracker] = None,
    progress_task: Optional[Dict[str, Any]] = None,
) -> RewardModel:
    set_global_seed(seed)
    if progress_tracker is not None and progress_task is not None:
        progress_tracker.start_task(
            str(progress_task["task_id"]),
            str(progress_task["name"]),
            str(progress_task["unit_name"]),
            int(progress_task["total_units"]),
        )
    LOGGER.info(
        "Myopic training start | samples=%s | obs_dim=%s | num_actions=%s | device=%s | epochs=%s | batch_size=%s | lr=%s",
        len(dataset["obs"]),
        obs_dim,
        num_actions,
        device,
        epochs,
        batch_size,
        lr,
    )
    model = RewardModel(obs_dim, num_actions).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    obs = torch.tensor(dataset["obs"], dtype=torch.float32, device=device)
    act = torch.tensor(dataset["action"], dtype=torch.long, device=device)
    rew = torch.tensor(dataset["reward"], dtype=torch.float32, device=device)
    LOGGER.debug(
        "Myopic training tensors | obs_shape=%s | action_shape=%s | reward_shape=%s | model_device=%s",
        tuple(obs.shape),
        tuple(act.shape),
        tuple(rew.shape),
        next(model.parameters()).device,
    )

    n = obs.shape[0]
    idx = np.arange(n)
    final_epoch_loss = float("nan")
    for epoch in range(epochs):
        np.random.shuffle(idx)
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            pred = model(obs[batch])
            chosen = pred.gather(1, act[batch].unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(chosen, rew[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss_sum += float(loss.item())
            epoch_batches += 1
        final_epoch_loss = epoch_loss_sum / max(epoch_batches, 1)
        LOGGER.info(
            "Myopic training epoch %s/%s | avg_mse_loss=%.6f",
            epoch + 1,
            epochs,
            final_epoch_loss,
        )
        if progress_tracker is not None and progress_task is not None:
            progress_tracker.update_task(epoch + 1, extra=f"avg_mse_loss={final_epoch_loss:.6f}")
    LOGGER.info("Myopic training complete | final_training_loss=%.6f", final_epoch_loss)
    if progress_tracker is not None and progress_task is not None:
        progress_tracker.finish_task(extra=f"final_training_loss={final_epoch_loss:.6f}")
    return model


class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = torch.empty((self.capacity, int(obs_dim)), dtype=torch.float32, device=self.device)
        self.actions = torch.empty(self.capacity, dtype=torch.long, device=self.device)
        self.logp = torch.empty(self.capacity, dtype=torch.float32, device=self.device)
        self.rewards = torch.empty(self.capacity, dtype=torch.float32, device=self.device)
        self.cost1 = torch.empty(self.capacity, dtype=torch.float32, device=self.device)
        self.cost2 = torch.empty(self.capacity, dtype=torch.float32, device=self.device)
        self.done = torch.empty(self.capacity, dtype=torch.float32, device=self.device)
        self.vr = torch.empty(self.capacity, dtype=torch.float32, device=self.device)
        self.vc1 = torch.empty(self.capacity, dtype=torch.float32, device=self.device)
        self.vc2 = torch.empty(self.capacity, dtype=torch.float32, device=self.device)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        logp: torch.Tensor,
        reward: float,
        cost1: float,
        cost2: float,
        done: bool,
        vr: torch.Tensor,
        vc1: torch.Tensor,
        vc2: torch.Tensor,
    ) -> None:
        if self.ptr >= self.capacity:
            raise IndexError("RolloutBuffer capacity exceeded")
        self.obs[self.ptr] = torch.from_numpy(obs).to(device=self.device, dtype=torch.float32)
        self.actions[self.ptr] = int(action)
        self.logp[self.ptr] = logp
        self.rewards[self.ptr] = float(reward)
        self.cost1[self.ptr] = float(cost1)
        self.cost2[self.ptr] = float(cost2)
        self.done[self.ptr] = float(done)
        self.vr[self.ptr] = vr
        self.vc1[self.ptr] = vc1
        self.vc2[self.ptr] = vc2
        self.ptr += 1

    def as_tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "obs": self.obs[: self.ptr],
            "actions": self.actions[: self.ptr],
            "logp": self.logp[: self.ptr],
            "rewards": self.rewards[: self.ptr],
            "cost1": self.cost1[: self.ptr],
            "cost2": self.cost2[: self.ptr],
            "done": self.done[: self.ptr],
            "vr": self.vr[: self.ptr],
            "vc1": self.vc1[: self.ptr],
            "vc2": self.vc2[: self.ptr],
        }

    def __len__(self) -> int:
        return int(self.ptr)


def gae_torch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    last_value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.empty_like(rewards)
    lastgaelam = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    next_value = last_value.to(device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        advantages[t] = lastgaelam
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def validation_status(
    metrics: Dict[str, Any],
    night_budget: Optional[float] = None,
    over_budget: Optional[float] = None,
) -> Dict[str, Any]:
    night_violation = max(0.0, float(metrics["NightMinutes"]) - float(night_budget)) if night_budget is not None else 0.0
    over_violation = max(0.0, float(metrics["OverCapMinutes"]) - float(over_budget)) if over_budget is not None else 0.0
    feasible = (night_violation <= 1e-12) and (over_violation <= 1e-12)
    total_violation = float(night_violation + over_violation)
    return {
        "feasible": bool(feasible),
        "night_violation": float(night_violation),
        "over_violation": float(over_violation),
        "total_violation": float(total_violation),
    }


def lagrangian_cost_scale(budget: Optional[float], normalization: str) -> float:
    mode = str(normalization).strip().lower()
    if mode not in {"none", "budget"}:
        raise ValueError(f"Unsupported lagrangian cost normalization mode: {normalization!r}")
    if mode == "budget" and budget is not None and math.isfinite(float(budget)):
        return max(abs(float(budget)), 1.0)
    return 1.0


def train_ppo(
    cfg: BenchConfig,
    catalog: np.ndarray,
    cards: Sequence[Dict[str, Any]],
    seed: int,
    total_steps: int = 1_000_000,
    rollout_steps: int = 2048,
    minibatch_size: int = 256,
    update_epochs: int = 10,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    hidden_size: int = 128,
    policy_arch: str = "flat",
    device: str = "cpu",
    lagrangian: bool = False,
    night_budget: Optional[float] = None,
    over_budget: Optional[float] = None,
    dual_lr: float = 0.05,
    cost_normalization: str = "none",
    validate_every: int = 20_000,
    val_episodes: int = 1000,
    val_episode_seeds: Optional[Sequence[int]] = None,
    progress_every_rollouts: int = 1,
    progress_tracker: Optional[ProgressTracker] = None,
    progress_task: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    set_global_seed(seed)
    if progress_tracker is not None and progress_task is not None:
        progress_tracker.start_task(
            str(progress_task["task_id"]),
            str(progress_task["name"]),
            str(progress_task["unit_name"]),
            int(progress_task["total_units"]),
        )
    device = device or cfg.device
    torch_device = torch.device(device)
    model = build_actor_critic_model(cfg, hidden_size=hidden_size, policy_arch=policy_arch).to(torch_device)
    opt = optim.Adam(model.parameters(), lr=lr)
    trainer_name = "Lagrangian PPO" if lagrangian else "PPO"
    LOGGER.info(
        "%s training start | seed=%s | total_steps=%s | rollout_steps=%s | minibatch_size=%s | update_epochs=%s | lr=%s | ent_coef=%s | hidden_size=%s | policy_arch=%s | lagrangian=%s",
        trainer_name,
        seed,
        total_steps,
        rollout_steps,
        minibatch_size,
        update_epochs,
        lr,
        ent_coef,
        hidden_size,
        policy_arch,
        lagrangian,
    )
    if lagrangian:
        LOGGER.info(
            "%s budgets | night_budget=%s | over_budget=%s | dual_lr=%s | cost_normalization=%s",
            trainer_name,
            night_budget,
            over_budget,
            dual_lr,
            cost_normalization,
        )

    env = make_env(cfg, "param", catalog, cards, scorer=None, wrappers=None, seed=seed)
    obs = env.reset(seed=seed)
    history: List[Dict[str, Any]] = []
    rollout_history: List[Dict[str, Any]] = []
    lambda1 = 0.0
    lambda2 = 0.0
    rollout_index = 0
    validation_seeds = list(val_episode_seeds) if val_episode_seeds is not None else list(range(seed + 10_000, seed + 10_000 + val_episodes))

    best_unconstrained_state = copy.deepcopy(model.state_dict())
    best_unconstrained_reward = -float("inf")
    best_unconstrained_step: Optional[int] = None

    best_feasible_state: Optional[Dict[str, torch.Tensor]] = None
    best_feasible_reward = -float("inf")
    best_feasible_step: Optional[int] = None

    least_violating_state = copy.deepcopy(model.state_dict())
    least_violating_key = (float("inf"), float("-inf"))
    least_violating_step: Optional[int] = None

    global_step = 0
    cost1_scale = lagrangian_cost_scale(night_budget, cost_normalization) if lagrangian else 1.0
    cost2_scale = lagrangian_cost_scale(over_budget, cost_normalization) if lagrangian else 1.0
    while global_step < total_steps:
        buffer = RolloutBuffer(rollout_steps, cfg.obs_dim(), torch_device)
        rollout_reward_sum = 0.0
        rollout_cost1_sum = 0.0
        rollout_cost2_sum = 0.0
        rollout_entropy_sum = 0.0
        completed_episode_costs: List[Tuple[float, float]] = []

        for _ in range(rollout_steps):
            obs_t = torch.from_numpy(obs).to(device=torch_device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist, vr, vc = model.dist_and_values(obs_t)
                action_t = dist.sample()[0]
                action = int(action_t.item())
                logp_t = dist.log_prob(action_t.unsqueeze(0))[0]
                entropy_sample = float(dist.entropy()[0].item())
                vr_t = vr[0]
                vc1_t = vc[0, 0]
                vc2_t = vc[0, 1]

            next_obs, reward, done, info = env.step(action)
            cost1 = float(info["night_cost"])
            cost2 = float(info["over_cost"])
            buffer.add(obs, action, logp_t, reward, cost1, cost2, done, vr_t, vc1_t, vc2_t)
            obs = next_obs
            rollout_reward_sum += reward
            rollout_cost1_sum += cost1
            rollout_cost2_sum += cost2
            rollout_entropy_sum += entropy_sample
            global_step += 1

            if done:
                summary = env.episode_summary()
                completed_episode_costs.append((summary["NightMinutes"], summary["OverCapMinutes"]))
                obs = env.reset(seed=int(env.rng.integers(0, 2**31 - 1)))

            if global_step >= total_steps:
                break

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device=torch_device, dtype=torch.float32).unsqueeze(0)
            _, last_vr_t, last_vc_t = model.dist_and_values(obs_t)
            last_vr = last_vr_t[0]
            last_vc1 = last_vc_t[0, 0]
            last_vc2 = last_vc_t[0, 1]

        data = buffer.as_tensors()
        adv_r, ret_r = gae_torch(data["rewards"], data["vr"], data["done"], gamma, gae_lambda, last_vr)
        adv_c1, ret_c1 = gae_torch(data["cost1"], data["vc1"], data["done"], gamma, gae_lambda, last_vc1)
        adv_c2, ret_c2 = gae_torch(data["cost2"], data["vc2"], data["done"], gamma, gae_lambda, last_vc2)

        adv = adv_r.clone()
        if lagrangian:
            # Normalize reward and cost advantages SEPARATELY before combining.
            # This prevents the reward signal from drowning out the cost signal
            # when over_cost is sparse (mostly zero per step).  Standard practice
            # in constrained RL (Stooke et al., "Responsive Safety in RL").
            adv_r_normed = (adv_r - adv_r.mean()) / (adv_r.std(unbiased=False) + 1e-8)
            adv_c1_normed = (adv_c1 - adv_c1.mean()) / (adv_c1.std(unbiased=False) + 1e-8) if adv_c1.std() > 1e-8 else adv_c1 * 0.0
            adv_c2_normed = (adv_c2 - adv_c2.mean()) / (adv_c2.std(unbiased=False) + 1e-8) if adv_c2.std() > 1e-8 else adv_c2 * 0.0
            adv = adv_r_normed - lambda1 * adv_c1_normed / cost1_scale - lambda2 * adv_c2_normed / cost2_scale
        else:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        n = len(buffer)
        for _ in range(update_epochs):
            idx = torch.randperm(n, device=torch_device)
            for start in range(0, n, minibatch_size):
                mb = idx[start : start + minibatch_size]
                obs_mb = data["obs"][mb]
                act_mb = data["actions"][mb]
                old_logp_mb = data["logp"][mb]
                adv_mb = adv[mb]
                ret_r_mb = ret_r[mb]
                ret_c1_mb = ret_c1[mb]
                ret_c2_mb = ret_c2[mb]

                dist, v_r_pred, v_c_pred = model.dist_and_values(obs_mb)
                new_logp = dist.log_prob(act_mb)
                ratio = torch.exp(new_logp - old_logp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(v_r_pred, ret_r_mb)
                value_loss = value_loss + nn.functional.mse_loss(v_c_pred[:, 0], ret_c1_mb)
                value_loss = value_loss + nn.functional.mse_loss(v_c_pred[:, 1], ret_c2_mb)

                entropy = dist.entropy().mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        if lagrangian and completed_episode_costs:
            avg_c1 = float(np.mean([x[0] for x in completed_episode_costs]))
            avg_c2 = float(np.mean([x[1] for x in completed_episode_costs]))
            if night_budget is not None:
                lambda1 = max(0.0, lambda1 + dual_lr * ((avg_c1 - night_budget) / cost1_scale))
            if over_budget is not None:
                lambda2 = max(0.0, lambda2 + dual_lr * ((avg_c2 - over_budget) / cost2_scale))

        rollout_row = {
            "rollout_index": int(rollout_index + 1),
            "global_step": int(global_step),
            "reward_per_rollout": float(rollout_reward_sum),
            "NightMinutes_per_rollout": float(rollout_cost1_sum),
            "OverCapMinutes_per_rollout": float(rollout_cost2_sum),
            "reward_per_step": float(rollout_reward_sum / max(len(buffer), 1)),
            "NightMinutes_per_step": float(rollout_cost1_sum / max(len(buffer), 1)),
            "OverCapMinutes_per_step": float(rollout_cost2_sum / max(len(buffer), 1)),
            "policy_entropy": float(rollout_entropy_sum / max(len(buffer), 1)),
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "completed_episodes": int(len(completed_episode_costs)),
            "selected_checkpoint_step": None,
        }
        rollout_history.append(rollout_row)

        rollout_index += 1
        if progress_every_rollouts and progress_every_rollouts > 0 and (rollout_index % progress_every_rollouts == 0):
            LOGGER.info(
                "%s rollout %s | global_step=%s | buffer_size=%s | completed_episodes=%s | avg_reward=%.4f | avg_cost1=%.4f | avg_cost2=%.4f | policy_entropy=%.4f | lambda1=%.4f | lambda2=%.4f",
                trainer_name,
                rollout_index,
                global_step,
                len(buffer),
                len(completed_episode_costs),
                rollout_reward_sum / max(len(buffer), 1),
                rollout_cost1_sum / max(len(buffer), 1),
                rollout_cost2_sum / max(len(buffer), 1),
                rollout_entropy_sum / max(len(buffer), 1),
                lambda1,
                lambda2,
            )
            if progress_tracker is not None and progress_task is not None:
                progress_tracker.update_task(global_step, extra=f"rollout={rollout_index} | lambda1={lambda1:.3f} | lambda2={lambda2:.3f}")

        if validate_every and (global_step % validate_every < rollout_steps or global_step >= total_steps):
            if progress_tracker is not None and progress_task is not None:
                progress_tracker.update_task(global_step, extra=f"validation_checkpoint={global_step}")
            policy = PPOPolicy(model)
            metrics = evaluate_policy(
                policy,
                cfg,
                catalog,
                cards,
                backend="param",
                scorer=None,
                wrappers=None,
                num_episodes=len(validation_seeds),
                episode_seeds=validation_seeds,
                deterministic=True,
                label=f"{trainer_name} validation",
                collect_policy_diagnostics=False,
            )
            status = validation_status(metrics, night_budget=night_budget if lagrangian else None, over_budget=over_budget if lagrangian else None)
            LOGGER.info(
                "%s validation | global_step=%s | CumWatch=%.3f | NightMinutes=%.3f | OverCapMinutes=%.3f | CVaR_0.95(L)=%.3f | feasible=%s | total_violation=%.6f",
                trainer_name,
                global_step,
                metrics["CumWatch"],
                metrics["NightMinutes"],
                metrics["OverCapMinutes"],
                metrics["CVaR_0.95(L)"],
                status["feasible"],
                status["total_violation"],
            )
            row = {
                "global_step": global_step,
                "validation_CumWatch": metrics["CumWatch"],
                "validation_NightMinutes": metrics["NightMinutes"],
                "validation_OverCapMinutes": metrics["OverCapMinutes"],
                "validation_CVaR_0.95(L)": metrics["CVaR_0.95(L)"],
                "lambda1": lambda1,
                "lambda2": lambda2,
                "feasible": bool(status["feasible"]),
                "night_violation": float(status["night_violation"]),
                "over_violation": float(status["over_violation"]),
                "total_violation": float(status["total_violation"]),
                "selected": False,
            }
            history.append(row)

            if not lagrangian:
                if (
                    float(metrics["CumWatch"]) > best_unconstrained_reward + 1e-12
                    or (
                        abs(float(metrics["CumWatch"]) - best_unconstrained_reward) <= 1e-12
                        and (best_unconstrained_step is None or int(global_step) > int(best_unconstrained_step))
                    )
                ):
                    prev_best = best_unconstrained_reward
                    best_unconstrained_reward = float(metrics["CumWatch"])
                    best_unconstrained_state = copy.deepcopy(model.state_dict())
                    best_unconstrained_step = int(global_step)
                    LOGGER.info(
                        "%s new best checkpoint | global_step=%s | validation_CumWatch=%.3f | previous_best=%s",
                        trainer_name,
                        global_step,
                        best_unconstrained_reward,
                        "none" if math.isinf(prev_best) and prev_best < 0 else f"{prev_best:.3f}",
                    )
            else:
                if bool(status["feasible"]) and (
                    float(metrics["CumWatch"]) > best_feasible_reward + 1e-12
                    or (
                        abs(float(metrics["CumWatch"]) - best_feasible_reward) <= 1e-12
                        and (best_feasible_step is None or int(global_step) > int(best_feasible_step))
                    )
                ):
                    prev_best = best_feasible_reward
                    best_feasible_reward = float(metrics["CumWatch"])
                    best_feasible_state = copy.deepcopy(model.state_dict())
                    best_feasible_step = int(global_step)
                    LOGGER.info(
                        "%s new best feasible checkpoint | global_step=%s | validation_CumWatch=%.3f | previous_best=%s",
                        trainer_name,
                        global_step,
                        best_feasible_reward,
                        "none" if math.isinf(prev_best) and prev_best < 0 else f"{prev_best:.3f}",
                    )
                violation_key = (float(status["total_violation"]), -float(metrics["CumWatch"]))
                if violation_key < least_violating_key or (
                    violation_key == least_violating_key
                    and (least_violating_step is None or int(global_step) > int(least_violating_step))
                ):
                    least_violating_key = violation_key
                    least_violating_state = copy.deepcopy(model.state_dict())
                    least_violating_step = int(global_step)

    selected_step: Optional[int]
    selected_feasible: bool
    selected_source: str
    if not lagrangian:
        model.load_state_dict(best_unconstrained_state)
        selected_step = best_unconstrained_step
        selected_feasible = True
        selected_source = "reward_best"
    elif best_feasible_state is not None:
        model.load_state_dict(best_feasible_state)
        selected_step = best_feasible_step
        selected_feasible = True
        selected_source = "best_feasible"
    else:
        model.load_state_dict(least_violating_state)
        selected_step = least_violating_step
        selected_feasible = False
        selected_source = "least_violating"
        LOGGER.warning(
            "%s finished with no feasible validation checkpoint | selected_global_step=%s | total_violation=%.6f",
            trainer_name,
            selected_step,
            0.0 if least_violating_key[0] == float("inf") else float(least_violating_key[0]),
        )

    for row in history:
        row["selected"] = bool(selected_step is not None and int(row["global_step"]) == int(selected_step))
    for row in rollout_history:
        row["selected_checkpoint_step"] = int(selected_step) if selected_step is not None else None

    LOGGER.info(
        "%s training complete | selected_source=%s | selected_feasible=%s | validation_checkpoints=%s | final_lambda1=%.4f | final_lambda2=%.4f",
        trainer_name,
        selected_source,
        selected_feasible,
        len(history),
        lambda1,
        lambda2,
    )
    summary = {
        "history": history,
        "rollout_history": rollout_history,
        "best_score": best_unconstrained_reward if not lagrangian else best_feasible_reward,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "dual_lr": float(dual_lr),
        "total_steps": int(total_steps),
        "lagrangian": lagrangian,
        "night_budget": night_budget,
        "over_budget": over_budget,
        "cost_normalization": str(cost_normalization),
        "ent_coef": ent_coef,
        "hidden_size": int(hidden_size),
        "policy_arch": str(policy_arch),
        "selected_global_step": selected_step,
        "selected_feasible": bool(selected_feasible),
        "selected_source": selected_source,
        "selected_validation": next((copy.deepcopy(row) for row in history if bool(row["selected"])), None),
        "best_feasible_validation": None if best_feasible_step is None else next((copy.deepcopy(row) for row in history if int(row["global_step"]) == int(best_feasible_step)), None),
        "least_violating_validation": None if least_violating_step is None else next((copy.deepcopy(row) for row in history if int(row["global_step"]) == int(least_violating_step)), None),
        "has_feasible_checkpoint": bool(best_feasible_state is not None) if lagrangian else True,
        "validation_seeds": list(map(int, validation_seeds)),
    }
    if progress_tracker is not None and progress_task is not None:
        progress_tracker.finish_task(extra=f"selected_source={selected_source} | selected_step={selected_step}")
    return model, summary


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

def _first_present_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for name in candidates:
        if name in colset:
            return name
    return None


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _hash_bucket(value: Any, modulo: int) -> int:
    text = "" if value is None else str(value)
    return int(sha256_text(text)[:12], 16) % max(1, int(modulo))


def load_public_log_dataframe(log_csv: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = pd.read_csv(log_csv)
    out = pd.DataFrame()
    source = "generic_csv"

    if {"user_id", "timestamp_min", "watch_time_min"}.issubset(raw.columns):
        out["user_id"] = raw["user_id"]
        out["timestamp_min"] = _coerce_numeric_series(raw["timestamp_min"])
        out["watch_time_min"] = _coerce_numeric_series(raw["watch_time_min"])
        if "item_id" in raw.columns:
            out["item_id"] = raw["item_id"]
        elif "video_id" in raw.columns:
            out["item_id"] = raw["video_id"]
        elif "final_video_id" in raw.columns:
            out["item_id"] = raw["final_video_id"]
        if "session_id" in raw.columns:
            out["session_id"] = raw["session_id"]
        if "cluster_id" in raw.columns:
            out["cluster_id"] = raw["cluster_id"]
    elif {"user_id", "video_id", "timestamp", "play_duration"}.issubset(raw.columns):
        source = "KuaiRec"
        out["user_id"] = raw["user_id"]
        out["item_id"] = raw["video_id"]
        out["timestamp_min"] = _coerce_numeric_series(raw["timestamp"]) / 60.0
        out["watch_time_min"] = _coerce_numeric_series(raw["play_duration"]) / 60000.0
        if "session_id" in raw.columns:
            out["session_id"] = raw["session_id"]
        if "cluster_id" in raw.columns:
            out["cluster_id"] = raw["cluster_id"]
    elif {"user_id", "time_ms", "play_time_ms"}.issubset(raw.columns):
        item_col = _first_present_column(raw.columns, ["final_video_id", "video_id", "item_id"])
        if item_col is None:
            raise ValueError("KuaiRand-style logs require one of final_video_id, video_id, or item_id.")
        source = "KuaiRand"
        out["user_id"] = raw["user_id"]
        out["item_id"] = raw[item_col]
        out["timestamp_min"] = _coerce_numeric_series(raw["time_ms"]) / 60000.0
        out["watch_time_min"] = _coerce_numeric_series(raw["play_time_ms"]) / 60000.0
        if "session_id" in raw.columns:
            out["session_id"] = raw["session_id"]
        if "cluster_id" in raw.columns:
            out["cluster_id"] = raw["cluster_id"]
    else:
        raise ValueError(
            "Unsupported public log format. Expected canonical columns "
            "{user_id,timestamp_min,watch_time_min}, KuaiRec columns "
            "{user_id,video_id,timestamp,play_duration}, or KuaiRand columns "
            "{user_id,time_ms,play_time_ms,(video_id|final_video_id)}."
        )

    out = out.replace([np.inf, -np.inf], np.nan)
    required = ["user_id", "timestamp_min", "watch_time_min"]
    out = out.dropna(subset=required).copy()
    raw_rows = len(out)
    out = out[out["watch_time_min"] > 0.0].copy()
    out["user_id"] = out["user_id"].astype(str)
    if "item_id" in out.columns:
        out["item_id"] = out["item_id"].astype(str)
    if "session_id" in out.columns:
        out["session_id"] = out["session_id"].astype(str)
    if "cluster_id" in out.columns:
        out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce")
    out = out.sort_values(["user_id", "timestamp_min"]).reset_index(drop=True)
    info = {
        "source": source,
        "rows_loaded": int(len(raw)),
        "rows_valid_timestamp_watch": int(raw_rows),
        "rows_used_positive_watch": int(len(out)),
    }
    LOGGER.info(
        "Public-log preprocessing | source=%s | rows_loaded=%s | rows_after_type_clean=%s | rows_used=%s",
        source,
        info["rows_loaded"],
        info["rows_valid_timestamp_watch"],
        info["rows_used_positive_watch"],
    )
    return out, info


@lru_cache(maxsize=8)
def build_cluster_id_map(metadata_csv: Optional[str], Z: Optional[int], random_seed: int = 0) -> Dict[str, int]:
    if metadata_csv is None or Z is None or int(Z) <= 0:
        return {}
    df = pd.read_csv(metadata_csv)
    item_col = _first_present_column(df.columns, ["item_id", "video_id", "final_video_id"])
    caption_col = _first_present_column(df.columns, ["caption", "manual_cover_text", "topic_tag"])
    cat1_col = _first_present_column(df.columns, ["category_level1", "first_level_category_name", "level1_category_name"])
    cat2_col = _first_present_column(df.columns, ["category_level2", "second_level_category_name", "level2_category_name"])
    if item_col is None or (caption_col is None and cat1_col is None and cat2_col is None):
        return {}

    meta = pd.DataFrame(
        {
            "item_id": df[item_col].astype(str),
            "caption": df[caption_col].fillna("").map(_normalize_text) if caption_col else "",
            "category_level1": df[cat1_col].fillna("").map(_normalize_text) if cat1_col else "",
            "category_level2": df[cat2_col].fillna("").map(_normalize_text) if cat2_col else "",
        }
    ).drop_duplicates("item_id")
    if meta.empty:
        return {}

    texts = (
        meta["caption"].fillna("")
        + " [SEP] "
        + meta["category_level1"].fillna("")
        + " [SEP] "
        + meta["category_level2"].fillna("")
    ).tolist()
    n_clusters = min(int(Z), len(meta))
    if n_clusters <= 0:
        return {}

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    X = vectorizer.fit_transform(texts)
    if HAVE_KMEDOIDS and len(meta) >= n_clusters:
        clusterer = KMedoids(n_clusters=n_clusters, metric="cosine", random_state=random_seed)
        labels = clusterer.fit_predict(X)
    else:
        km = fit_kmeans_single_thread(X, n_clusters=n_clusters, random_state=random_seed, n_init=10)
        labels = km.labels_

    meta = meta.copy()
    meta["cluster"] = labels
    order = (
        meta.groupby("cluster")
        .size()
        .reset_index(name="size")
        .sort_values(["size", "cluster"], ascending=[False, True])["cluster"]
        .tolist()
    )
    cluster_to_z = {cluster: idx for idx, cluster in enumerate(order)}
    meta["cluster_id"] = meta["cluster"].map(cluster_to_z).astype(int)
    return dict(zip(meta["item_id"].tolist(), meta["cluster_id"].tolist()))


def attach_cluster_ids(df: pd.DataFrame, metadata_csv: Optional[str], Z: Optional[int], random_seed: int = 0) -> Tuple[pd.DataFrame, str]:
    if "cluster_id" in df.columns and df["cluster_id"].notna().any():
        out = df.copy()
        out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce")
        return out, "provided"

    out = df.copy()
    cluster_map = build_cluster_id_map(metadata_csv, Z, random_seed=random_seed)
    if cluster_map and "item_id" in out.columns:
        out["cluster_id"] = out["item_id"].map(cluster_map)
        if out["cluster_id"].notna().any():
            out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce")
            return out, "metadata_cluster_map"

    if Z is not None and "item_id" in out.columns:
        out["cluster_id"] = out["item_id"].map(
            lambda item: int(item) % int(Z) if str(item).isdigit() else _hash_bucket(item, int(Z))
        ).astype(float)
        return out, "hashed_item_fallback"

    return out, "missing"


CALIBRATION_POLICY_CHOICES = ("random", "logged_cluster_marginal")
DEFAULT_CALIBRATION_POLICY = "random"
DEFAULT_CALIBRATION_POLICY_LAMBDA_IDX = 0
DEFAULT_CALIBRATION_POLICY_NU = 0


def normalize_calibration_policy_name(calibration_policy: str) -> str:
    policy = str(calibration_policy).strip().lower()
    if policy not in CALIBRATION_POLICY_CHOICES:
        raise ValueError(f"Unsupported calibration_policy={calibration_policy!r}. Expected one of {CALIBRATION_POLICY_CHOICES}.")
    return policy


def _project_calibration_policy_cluster(value: Any, Z: int) -> int:
    try:
        cluster = int(float(value))
    except (TypeError, ValueError):
        return _hash_bucket(value, int(Z))
    if 0 <= cluster < int(Z):
        return int(cluster)
    return _hash_bucket(cluster, int(Z))


def build_logged_cluster_marginal_policy_context(
    df: pd.DataFrame,
    *,
    Z: int,
    cluster_source: str,
    fixed_lambda_idx: int = DEFAULT_CALIBRATION_POLICY_LAMBDA_IDX,
    fixed_nu: int = DEFAULT_CALIBRATION_POLICY_NU,
) -> Dict[str, Any]:
    if int(Z) <= 0:
        raise ValueError("logged_cluster_marginal calibration policy requires Z > 0.")
    if "timestamp_min" not in df.columns:
        raise ValueError("logged_cluster_marginal calibration policy requires timestamp_min in the public-log dataframe.")

    valid = df.dropna(subset=["timestamp_min"]).copy()
    if valid.empty:
        raise ValueError("logged_cluster_marginal calibration policy requires at least one timestamped log event.")

    projection_source = "cluster_id"
    if "cluster_id" in valid.columns and valid["cluster_id"].notna().any():
        raw_clusters = valid["cluster_id"]
    elif "item_id" in valid.columns:
        projection_source = "item_id_hash_fallback"
        raw_clusters = valid["item_id"]
    else:
        raise ValueError("logged_cluster_marginal calibration policy requires cluster_id or item_id for cluster estimation.")

    projected_clusters = raw_clusters.map(lambda value: _project_calibration_policy_cluster(value, int(Z))).astype(int)
    timestamp_minutes = pd.to_numeric(valid["timestamp_min"], errors="coerce").fillna(0.0)
    hours = np.mod(np.floor(timestamp_minutes.to_numpy(dtype=np.float64) / 60.0).astype(np.int64), 24)

    overall_counts = np.zeros(int(Z), dtype=np.float64)
    hourly_counts = np.zeros((24, int(Z)), dtype=np.float64)
    for hour, cluster_id in zip(hours.tolist(), projected_clusters.tolist()):
        overall_counts[int(cluster_id)] += 1.0
        hourly_counts[int(hour), int(cluster_id)] += 1.0

    overall_probs = normalize_probability_vector(overall_counts.tolist())
    hourly_probs: List[List[float]] = []
    for hour in range(24):
        hour_total = float(hourly_counts[hour].sum())
        hourly_probs.append(
            normalize_probability_vector(hourly_counts[hour].tolist())
            if hour_total > 0.0
            else []
        )

    return {
        "policy": "logged_cluster_marginal",
        "conditioning": "hour_of_day",
        "cluster_source": str(cluster_source),
        "cluster_projection_source": str(projection_source),
        "cluster_probabilities": [float(value) for value in overall_probs],
        "hourly_cluster_probabilities": [[float(value) for value in row] for row in hourly_probs],
        "fixed_lambda_idx": int(fixed_lambda_idx),
        "fixed_nu": int(fixed_nu),
        "fallback_when_hour_empty": "overall_cluster_marginal",
        "rows_used": int(len(valid)),
        "hourly_rows_with_support": int(sum(1 for row in hourly_probs if row)),
    }


def resolve_calibration_rollout_policy_context(
    targets: Dict[str, Any],
    calibration_policy: str,
) -> Optional[Dict[str, Any]]:
    policy = normalize_calibration_policy_name(calibration_policy)
    if policy == "random":
        return None
    contexts = targets.get("calibration_rollout_policies", {})
    context = contexts.get(policy) if isinstance(contexts, dict) else None
    if not isinstance(context, dict):
        raise ValueError(
            f"Calibration targets are missing rollout policy context for calibration_policy={policy!r}."
        )
    return context


def describe_calibration_rollout_policy(
    calibration_policy: str,
    calibration_policy_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    policy = normalize_calibration_policy_name(calibration_policy)
    if policy == "random":
        return {
            "name": "random",
            "action_sampling": "uniform_over_full_action_space",
        }
    context = dict(calibration_policy_context or {})
    return {
        "name": policy,
        "action_sampling": "empirical_cluster_marginal",
        "conditioning": str(context.get("conditioning", "none")),
        "cluster_source": str(context.get("cluster_source", "unknown")),
        "cluster_projection_source": str(context.get("cluster_projection_source", "unknown")),
        "fixed_lambda_idx": int(context.get("fixed_lambda_idx", DEFAULT_CALIBRATION_POLICY_LAMBDA_IDX)),
        "fixed_nu": int(context.get("fixed_nu", DEFAULT_CALIBRATION_POLICY_NU)),
        "fallback_when_hour_empty": str(context.get("fallback_when_hour_empty", "overall_cluster_marginal")),
    }


def sample_calibration_rollout_action(
    cfg: BenchConfig,
    env: CompulsionBenchEnv,
    calibration_policy: str,
    calibration_policy_context: Optional[Dict[str, Any]] = None,
) -> int:
    policy = normalize_calibration_policy_name(calibration_policy)
    if policy == "random":
        return int(env.rng.integers(0, cfg.num_actions()))

    context = calibration_policy_context or {}
    overall_probs = np.asarray(context.get("cluster_probabilities", []), dtype=np.float64)
    if overall_probs.size != int(cfg.Z):
        raise ValueError(
            f"logged_cluster_marginal calibration policy expects {int(cfg.Z)} cluster probabilities; got {overall_probs.size}."
        )
    hourly_probs = context.get("hourly_cluster_probabilities", [])
    hour_probs = np.asarray([], dtype=np.float64)
    if env.state is not None and isinstance(hourly_probs, list) and len(hourly_probs) == 24:
        hour = int(env._wall_clock(env.state.tau, env.state.tau_start) // 60.0) % 24
        hour_probs = np.asarray(hourly_probs[hour], dtype=np.float64)
    probs = hour_probs if hour_probs.size == int(cfg.Z) else overall_probs
    probs = np.asarray(normalize_probability_vector(probs.tolist()), dtype=np.float64)
    z = int(env.rng.choice(int(cfg.Z), p=probs))
    lam_idx = int(context.get("fixed_lambda_idx", DEFAULT_CALIBRATION_POLICY_LAMBDA_IDX))
    lam_idx = max(0, min(lam_idx, int(cfg.P)))
    nu = 1 if int(context.get("fixed_nu", DEFAULT_CALIBRATION_POLICY_NU)) > 0 else 0
    return tuple_to_action_id(z, lam_idx, nu, cfg.P)


def sessionize_logs(df: pd.DataFrame, delta_sess_minutes: float) -> pd.DataFrame:
    required = {"user_id", "timestamp_min", "watch_time_min"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Log CSV is missing columns: {sorted(missing)}")
    df = df.sort_values(["user_id", "timestamp_min"]).copy()
    df["event_end_ts"] = df["timestamp_min"].astype(float) + df["watch_time_min"].astype(float)
    df["session_break"] = 0
    prev_end = None
    prev_user = None
    session_ids = []
    sid = 0
    for row in df.itertuples(index=False):
        ts = float(getattr(row, "timestamp_min"))
        event_end_ts = float(getattr(row, "event_end_ts"))
        user = getattr(row, "user_id")
        if prev_user != user:
            sid += 1
            session_ids.append(sid)
            prev_end = event_end_ts
            prev_user = user
            continue
        gap = ts - float(prev_end)
        if gap > delta_sess_minutes:
            sid += 1
        session_ids.append(sid)
        prev_end = event_end_ts
        prev_user = user
    df["session_id"] = session_ids
    return df


def summarize_gap_targets_from_session_aggregate(
    sess: pd.DataFrame,
    *,
    prev_end_column: str,
    gap_bucket_edges: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    gaps: List[float] = []
    session_transitions: List[Dict[str, float]] = []
    for _, block in sess.groupby("user_id"):
        block = block.sort_values("start_ts")
        prev_end = None
        prev_watch = None
        for row in block.itertuples(index=False):
            if prev_end is not None and prev_watch is not None:
                gap = max(0.0, float(row.start_ts) - float(prev_end))
                gaps.append(gap)
                session_transitions.append(
                    {
                        "prev_session_length": float(prev_watch),
                        "gap": gap,
                    }
                )
            prev_end = float(getattr(row, prev_end_column))
            prev_watch = float(row.session_watch)
    bucket_edges, conditional_reentry = summarize_conditional_reentry(
        session_transitions,
        bucket_edges=gap_bucket_edges,
        thresholds=RETURN_RATE_THRESHOLDS,
    )
    gap_summary = summarize_distribution(gaps)
    return {
        "gaps": gaps,
        "session_transitions": session_transitions,
        "gap_bucket_edges": [float(x) for x in bucket_edges],
        "conditional_return_rates": conditional_reentry,
        "gap_summary": {
            "mean": float(gap_summary["mean"]),
            "p95": float(gap_summary["p95"]),
        },
    }


def build_gap_extraction_diagnostic(session_gap_targets: Dict[str, Any], legacy_gap_targets: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "legacy_definition": "next_session_start - (prev_session_start + summed_watch_time)",
        "corrected_definition": "next_session_start - prev_end_ts",
        "legacy": {
            "gap_summary": copy.deepcopy(legacy_gap_targets["gap_summary"]),
            "conditional_return_rates": copy.deepcopy(legacy_gap_targets["conditional_return_rates"]),
        },
        "corrected": {
            "gap_summary": copy.deepcopy(session_gap_targets["gap_summary"]),
            "conditional_return_rates": copy.deepcopy(session_gap_targets["conditional_return_rates"]),
        },
        "delta": {
            "mean_gap": float(
                session_gap_targets["gap_summary"]["mean"] - legacy_gap_targets["gap_summary"]["mean"]
            ),
            "p95_gap": float(
                session_gap_targets["gap_summary"]["p95"] - legacy_gap_targets["gap_summary"]["p95"]
            ),
        },
    }


def render_gap_extraction_diagnostic_markdown(diagnostic: Dict[str, Any], *, delta_sess: Optional[float] = None) -> str:
    legacy_summary = diagnostic["legacy"]["gap_summary"]
    corrected_summary = diagnostic["corrected"]["gap_summary"]
    legacy_conditional = diagnostic["legacy"]["conditional_return_rates"]
    corrected_conditional = diagnostic["corrected"]["conditional_return_rates"]
    lines = [
        "# Gap Extraction Diagnostic",
        "",
    ]
    if delta_sess is not None:
        lines.append(f"- Sessionization `delta_sess`: `{float(delta_sess):g}`")
    lines.extend(
        [
            f"- Legacy gap definition: `{diagnostic['legacy_definition']}`",
            f"- Corrected gap definition: `{diagnostic['corrected_definition']}`",
            "",
            "## Gap summary",
            "",
            "| Statistic | Legacy target | Corrected target | Delta (corrected - legacy) |",
            "| --- | ---: | ---: | ---: |",
            f"| Mean gap | {float(legacy_summary['mean']):.6f} | {float(corrected_summary['mean']):.6f} | {float(diagnostic['delta']['mean_gap']):.6f} |",
            f"| p95 gap | {float(legacy_summary['p95']):.6f} | {float(corrected_summary['p95']):.6f} | {float(diagnostic['delta']['p95_gap']):.6f} |",
            "",
            "## Conditional return rates",
            "",
            "| Prev-session bucket | Metric | Legacy target | Corrected target | Delta (corrected - legacy) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    metric_order = [f"ReturnRate{int(threshold)}" for threshold in RETURN_RATE_THRESHOLDS]
    for bucket in RETURN_BUCKET_LABELS:
        for metric in metric_order:
            legacy_value = float(legacy_conditional.get(bucket, {}).get(metric, float("nan")))
            corrected_value = float(corrected_conditional.get(bucket, {}).get(metric, float("nan")))
            delta_value = corrected_value - legacy_value if math.isfinite(legacy_value) and math.isfinite(corrected_value) else float("nan")
            lines.append(
                f"| {bucket} | `{metric}` | {legacy_value:.6f} | {corrected_value:.6f} | {delta_value:.6f} |"
            )
    return "\n".join(lines) + "\n"


def write_gap_extraction_diagnostic_artifact(
    targets: Dict[str, Any],
    outpath: Path,
    *,
    delta_sess: Optional[float] = None,
) -> None:
    diagnostic = targets.get("gap_extraction_diagnostic")
    if not isinstance(diagnostic, dict):
        raise ValueError("Targets are missing gap_extraction_diagnostic.")
    outpath.write_text(
        render_gap_extraction_diagnostic_markdown(diagnostic, delta_sess=delta_sess),
        encoding="utf-8",
    )
    log_artifact_written(outpath, "gap_extraction_diagnostic_markdown")


def summarize_targets_from_sessionized_df(
    df: pd.DataFrame,
    Z: Optional[int] = None,
    gap_bucket_edges: Optional[Sequence[float]] = None,
    include_gap_extraction_diagnostic: bool = False,
) -> Dict[str, Any]:
    required = {"user_id", "timestamp_min", "watch_time_min", "session_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Log CSV is missing columns after sessionization: {sorted(missing)}")
    df = df.copy()
    df["event_end_ts"] = df["timestamp_min"].astype(float) + df["watch_time_min"].astype(float)

    sess = (
        df.groupby(["user_id", "session_id"])
        .agg(
            start_ts=("timestamp_min", "min"),
            session_watch=("watch_time_min", "sum"),
            end_ts=("event_end_ts", "max"),
            n_items=("watch_time_min", "size"),
        )
        .reset_index()
        .sort_values(["user_id", "start_ts"])
    )
    sess["legacy_end_ts"] = sess["start_ts"].astype(float) + sess["session_watch"].astype(float)
    item_watch_times = df["watch_time_min"].astype(float).tolist()
    session_lengths = sess["session_watch"].astype(float).tolist()
    session_item_counts = sess["n_items"].astype(float).tolist()
    start_hour_hist = np.zeros(24, dtype=np.float64)
    for start_ts in sess["start_ts"].astype(float).tolist():
        hour_idx = int((start_ts % 1440.0) // 60.0) % 24
        start_hour_hist[hour_idx] += 1.0

    watch_minutes_by_hour = np.zeros(24, dtype=np.float64)
    for row in df.itertuples(index=False):
        append_duration_to_hour_bins(
            watch_minutes_by_hour,
            float(getattr(row, "timestamp_min")),
            float(getattr(row, "watch_time_min")),
        )

    session_gap_targets = summarize_gap_targets_from_session_aggregate(
        sess,
        prev_end_column="end_ts",
        gap_bucket_edges=gap_bucket_edges,
    )
    gaps = session_gap_targets["gaps"]

    # Stop hazard over within-session item positions.
    pos_counts: Dict[int, int] = {}
    stop_counts: Dict[int, int] = {}
    for _, block in df.sort_values(["user_id", "session_id", "timestamp_min"]).groupby(["user_id", "session_id"]):
        n = len(block)
        for j in range(1, n + 1):
            pos_counts[j] = pos_counts.get(j, 0) + 1
        stop_counts[n] = stop_counts.get(n, 0) + 1
    max_pos = max(pos_counts) if pos_counts else 0
    hazard = []
    for j in range(1, max_pos + 1):
        hazard.append(stop_counts.get(j, 0) / max(1, pos_counts.get(j, 0)))

    cluster_mean_watch: Dict[int, float] = {}
    if "cluster_id" in df.columns and df["cluster_id"].notna().any():
        cluster_mean_watch = (
            df.dropna(subset=["cluster_id"])
            .assign(cluster_id=lambda x: x["cluster_id"].astype(int))
            .groupby("cluster_id")["watch_time_min"].mean().astype(float).to_dict()
        )
    elif Z is not None and "item_id" in df.columns:
        cluster_mean_watch = (
            (df.assign(cluster_id=(df["item_id"].map(lambda item: int(item) % Z if str(item).isdigit() else _hash_bucket(item, Z))))
               .groupby("cluster_id")["watch_time_min"].mean().astype(float).to_dict())
        )

    result = {
        "item_watch_times": item_watch_times,
        "session_lengths": session_lengths,
        "session_item_counts": session_item_counts,
        "stop_hazard": hazard,
        "gaps": gaps,
        "cluster_mean_watch": {int(k): float(v) for k, v in cluster_mean_watch.items()},
        "session_start_histogram": normalize_probability_vector(start_hour_hist.tolist()),
        "watch_minutes_by_hour": normalize_probability_vector(watch_minutes_by_hour.tolist()),
        "gap_bucket_edges": list(session_gap_targets["gap_bucket_edges"]),
        "conditional_return_rates": copy.deepcopy(session_gap_targets["conditional_return_rates"]),
        "num_events": int(len(df)),
        "num_sessions": int(len(sess)),
        "num_users": int(df["user_id"].nunique()),
    }
    if include_gap_extraction_diagnostic:
        legacy_gap_targets = summarize_gap_targets_from_session_aggregate(
            sess,
            prev_end_column="legacy_end_ts",
            gap_bucket_edges=session_gap_targets["gap_bucket_edges"],
        )
        result["gap_extraction_diagnostic"] = build_gap_extraction_diagnostic(
            session_gap_targets,
            legacy_gap_targets,
        )
    return result


def summarize_raw_stop_hazard_support(
    target_stop_hazard: Sequence[float],
    sim_stop_hazard: Sequence[float],
) -> Dict[str, Any]:
    target = list(target_stop_hazard)
    sim = list(sim_stop_hazard)
    target_len = int(len(target))
    sim_len = int(len(sim))
    overlap_len = int(min(target_len, sim_len))
    truncated = bool(target_len != sim_len)
    support_ratio = (
        float(sim_len) / float(target_len)
        if target_len > 0
        else (1.0 if sim_len == 0 else 0.0)
    )
    severe_mismatch = bool(
        target_len > 0
        and sim_len < target_len
        and support_ratio <= CALIBRATION_HAZARD_SUPPORT_RATIO_SEVERE
        and (target_len - sim_len) >= CALIBRATION_HAZARD_SUPPORT_ABS_GAP_SEVERE
    )
    if target_len == 0 and sim_len == 0:
        note = "Neither target nor simulator provided stop-hazard support."
    elif not truncated:
        note = f"No stop-hazard truncation: target and simulator both provide {overlap_len} within-session positions."
    else:
        shorter_label = "simulator" if sim_len < target_len else "target"
        longer_len = max(target_len, sim_len)
        note = (
            f"Overlap-only stop-hazard comparison uses the first {overlap_len} positions "
            f"(the `min(len(target_stop_hazard), len(sim_stop_hazard))` overlap) because "
            f"`len(target_stop_hazard)={target_len}` and `len(sim_stop_hazard)={sim_len}`. "
            f"The {shorter_label} support is shorter, so positions {overlap_len + 1}..{longer_len} are truncated from any direct pointwise comparison."
        )
    return {
        "target_len": target_len,
        "sim_len": sim_len,
        "overlap_len": overlap_len,
        "truncated": truncated,
        "simulator_support_shorter_than_target": bool(sim_len < target_len),
        "support_ratio_shorter_to_target": float(support_ratio),
        "severe_mismatch": severe_mismatch,
        "note": note,
    }


def derive_elapsed_session_minute_hazard_edges(
    target_session_lengths: Sequence[float],
    sim_session_lengths: Sequence[float],
    num_bins: int = CALIBRATION_HAZARD_ELAPSED_MINUTE_BINS,
) -> List[float]:
    clean_values = [
        float(value)
        for value in list(target_session_lengths) + list(sim_session_lengths)
        if math.isfinite(float(value)) and float(value) >= 0.0
    ]
    upper = max(clean_values) if clean_values else 1.0
    upper = max(float(upper), 1.0)
    edges = np.linspace(0.0, upper + 1e-6, int(num_bins) + 1, dtype=np.float64)
    return [float(edge) for edge in edges.tolist()]


def build_elapsed_session_minute_hazard(
    values: Sequence[float],
    edges: Sequence[float],
) -> Dict[str, Any]:
    hazard = empirical_hazard(values, edges)
    return {
        "basis": "elapsed_session_minutes",
        "edges": [float(edge) for edge in edges],
        "bin_labels": list(hazard["bin_labels"]),
        "hazard": [float(value) for value in hazard["hazard"]],
        "counts": [int(value) for value in hazard["counts"]],
    }


def calibration_component_has_signal(full_targets: Dict[str, Any], key: str) -> bool:
    if key == "cluster":
        return bool(full_targets.get("cluster_mean_watch"))
    if key == "hazard":
        return bool(full_targets.get("session_lengths")) or bool(full_targets.get("stop_hazard"))
    if key == "diurnal":
        return bool(full_targets.get("session_start_histogram"))
    if key == "return_conditional":
        return any(float(bucket_stats.get("count", 0.0)) > 0.0 for bucket_stats in full_targets.get("conditional_return_rates", {}).values())
    if key == "session":
        return bool(full_targets.get("session_lengths"))
    if key == "session_item_counts":
        return bool(full_targets.get("session_item_counts"))
    if key == "gaps":
        return bool(full_targets.get("gaps"))
    raise KeyError(f"Unknown calibration component {key!r}")


def calibration_quantile_relative_errors(
    target_values: Sequence[float],
    sim_values: Sequence[float],
    quantiles: Dict[str, float] = CALIBRATION_QUANTILES,
) -> Dict[str, float]:
    if not target_values or not sim_values:
        return {}
    target_arr = np.asarray(target_values, dtype=np.float64)
    sim_arr = np.asarray(sim_values, dtype=np.float64)
    out: Dict[str, float] = {}
    for label, q in quantiles.items():
        target_q = float(np.quantile(target_arr, float(q)))
        sim_q = float(np.quantile(sim_arr, float(q)))
        out[f"{label}_relative_error"] = safe_relative_error(target_q, sim_q)
    return out


def calibration_loss_component_details(targets: Dict[str, Any], sim: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    details: Dict[str, Dict[str, Any]] = {key: {} for key in CALIBRATION_COMPONENT_KEYS}
    if targets.get("session_lengths") and sim.get("session_lengths"):
        details["session"]["ks"] = float(ks_2samp(targets["session_lengths"], sim["session_lengths"]).statistic)
        details["session"].update(calibration_quantile_relative_errors(targets["session_lengths"], sim["session_lengths"]))
    if targets.get("session_item_counts") and sim.get("session_item_counts"):
        details["session_item_counts"]["ks"] = float(ks_2samp(targets["session_item_counts"], sim["session_item_counts"]).statistic)
        details["session_item_counts"].update(
            calibration_quantile_relative_errors(targets["session_item_counts"], sim["session_item_counts"])
        )
    if targets.get("gaps") and sim.get("gaps"):
        details["gaps"]["ks"] = float(ks_2samp(targets["gaps"], sim["gaps"]).statistic)
        details["gaps"].update(calibration_quantile_relative_errors(targets["gaps"], sim["gaps"]))
    raw_support = summarize_raw_stop_hazard_support(targets.get("stop_hazard", []), sim.get("stop_hazard", []))
    details["hazard"].update(
        {
            "raw_item_position_target_len": float(raw_support["target_len"]),
            "raw_item_position_sim_len": float(raw_support["sim_len"]),
            "raw_item_position_overlap_len": float(raw_support["overlap_len"]),
            "raw_item_position_truncated": float(1.0 if raw_support["truncated"] else 0.0),
            "raw_item_position_support_ratio_shorter_to_target": float(raw_support["support_ratio_shorter_to_target"]),
            "raw_item_position_severe_support_mismatch": float(1.0 if raw_support["severe_mismatch"] else 0.0),
        }
    )
    hz_t = np.asarray(targets.get("stop_hazard", []), dtype=np.float32)
    hz_s = np.asarray(sim.get("stop_hazard", []), dtype=np.float32)
    if hz_t.size and hz_s.size:
        m = min(hz_t.size, hz_s.size)
        details["hazard"]["raw_item_position_l2_overlap_only"] = float(np.mean((hz_t[:m] - hz_s[:m]) ** 2))
    if targets.get("session_lengths") and sim.get("session_lengths"):
        elapsed_edges = derive_elapsed_session_minute_hazard_edges(targets["session_lengths"], sim["session_lengths"])
        target_elapsed = build_elapsed_session_minute_hazard(targets["session_lengths"], elapsed_edges)
        sim_elapsed = build_elapsed_session_minute_hazard(sim["session_lengths"], elapsed_edges)
        elapsed_l2 = float(
            np.mean(
                (
                    np.asarray(target_elapsed["hazard"], dtype=np.float32)
                    - np.asarray(sim_elapsed["hazard"], dtype=np.float32)
                )
                ** 2
            )
        )
        details["hazard"].update(
            {
                "basis": "elapsed_session_minutes",
                "bin_labels": target_elapsed["bin_labels"],
                "elapsed_minute_edges": target_elapsed["edges"],
                "target_hazard": target_elapsed["hazard"],
                "simulator_hazard": sim_elapsed["hazard"],
                "l2": elapsed_l2,
                "support_mismatch_addressed": True,
                "used_raw_item_position_fallback": False,
            }
        )
    elif hz_t.size and hz_s.size:
        details["hazard"].update(
            {
                "basis": "raw_item_position_overlap_only",
                "l2": float(details["hazard"].get("raw_item_position_l2_overlap_only", 0.0)),
                "support_mismatch_addressed": bool(not raw_support["severe_mismatch"]),
                "used_raw_item_position_fallback": True,
            }
        )
    else:
        details["hazard"].update(
            {
                "basis": "missing",
                "l2": 0.0,
                "support_mismatch_addressed": True,
                "used_raw_item_position_fallback": False,
            }
        )
    details["hazard"]["severe_support_mismatch_unaddressed"] = float(
        1.0
        if raw_support["severe_mismatch"] and not bool(details["hazard"].get("support_mismatch_addressed", True))
        else 0.0
    )
    cluster_t = targets.get("cluster_mean_watch", {})
    cluster_s = sim.get("cluster_mean_watch", {})
    common = sorted(set(cluster_t).intersection(cluster_s))
    if common:
        diff = [(cluster_t[k] - cluster_s[k]) ** 2 for k in common]
        details["cluster"]["mse"] = float(np.mean(diff))
        details["cluster"]["matched_clusters"] = float(len(common))
    diurnal_t = np.asarray(targets.get("session_start_histogram", []), dtype=np.float32)
    diurnal_s = np.asarray(sim.get("session_start_histogram", []), dtype=np.float32)
    if diurnal_t.size and diurnal_s.size:
        m = min(diurnal_t.size, diurnal_s.size)
        details["diurnal"]["mse"] = float(np.mean((diurnal_t[:m] - diurnal_s[:m]) ** 2))
    cond_t = targets.get("conditional_return_rates", {})
    cond_s = sim.get("conditional_return_rates", {})
    cond_diffs: List[float] = []
    for bucket in RETURN_BUCKET_LABELS:
        target_bucket = cond_t.get(bucket, {})
        sim_bucket = cond_s.get(bucket, {})
        if float(target_bucket.get("count", 0.0)) <= 0.0 or float(sim_bucket.get("count", 0.0)) <= 0.0:
            continue
        for metric_key in RETURN_RATE_METRIC_KEYS:
            target_value = target_bucket.get(metric_key)
            sim_value = sim_bucket.get(metric_key)
            if target_value is None or sim_value is None:
                continue
            if math.isfinite(float(target_value)) and math.isfinite(float(sim_value)):
                cond_diffs.append((float(target_value) - float(sim_value)) ** 2)
    if cond_diffs:
        details["return_conditional"]["mse"] = float(np.mean(cond_diffs))
        details["return_conditional"]["matched_bucket_metrics"] = float(len(cond_diffs))
    return details


def calibration_loss_components(targets: Dict[str, Any], sim: Dict[str, Any]) -> Dict[str, float]:
    details = calibration_loss_component_details(targets, sim)
    components = {key: 0.0 for key in CALIBRATION_COMPONENT_KEYS}
    session_terms = [details["session"].get("ks", 0.0)]
    session_terms.extend(float(value) for name, value in details["session"].items() if name.endswith("_relative_error"))
    if any(float(value) > 0.0 for value in session_terms):
        components["session"] = float(safe_mean(session_terms))
    session_item_terms = [details["session_item_counts"].get("ks", 0.0)]
    session_item_terms.extend(
        float(value)
        for name, value in details["session_item_counts"].items()
        if name.endswith("_relative_error")
    )
    if any(float(value) > 0.0 for value in session_item_terms):
        components["session_item_counts"] = float(safe_mean(session_item_terms))
    gap_terms = [details["gaps"].get("ks", 0.0)]
    gap_terms.extend(float(value) for name, value in details["gaps"].items() if name.endswith("_relative_error"))
    if any(float(value) > 0.0 for value in gap_terms):
        components["gaps"] = float(safe_mean(gap_terms))
    components["hazard"] = float(details["hazard"].get("l2", 0.0))
    components["cluster"] = float(details["cluster"].get("mse", 0.0))
    components["diurnal"] = float(details["diurnal"].get("mse", 0.0))
    components["return_conditional"] = float(details["return_conditional"].get("mse", 0.0))
    return components


def renormalized_component_weights(
    weights: Dict[str, float],
    keys: Sequence[str],
) -> Dict[str, float]:
    active_keys = [key for key in keys if float(weights.get(key, 0.0)) > 0.0]
    if not active_keys:
        return {key: 1.0 / max(1, len(keys)) for key in keys}
    total = float(sum(float(weights.get(key, 0.0)) for key in active_keys))
    if total <= 0.0:
        return {key: 1.0 / len(active_keys) for key in active_keys}
    return {key: float(weights.get(key, 0.0)) / total for key in active_keys}


def weighted_component_average(
    components: Dict[str, float],
    weights: Dict[str, float],
    keys: Sequence[str],
) -> float:
    active_keys = [key for key in keys if key in components and math.isfinite(float(components[key]))]
    if not active_keys:
        return 0.0
    normalized = renormalized_component_weights(weights, active_keys)
    return float(sum(float(normalized.get(key, 0.0)) * float(components[key]) for key in active_keys))


def calibration_loss_breakdown(
    targets: Dict[str, Any],
    sim: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    *,
    components: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    effective_weights = weights or targets.get(
        "calibration_weights",
        {
            "session": 1.0,
            "session_item_counts": 1.0,
            "hazard": 1.0,
            "gaps": 1.0,
            "cluster": 1.0,
            "diurnal": 1.0,
            "return_conditional": 1.0,
        },
    )
    components = components or calibration_loss_components(targets, sim)
    primary = weighted_component_average(components, effective_weights, CALIBRATION_PRIMARY_COMPONENT_KEYS)
    secondary = weighted_component_average(components, effective_weights, CALIBRATION_SECONDARY_COMPONENT_KEYS)
    lexicographic_scalar = float(primary * CALIBRATION_LOSS_PRIMARY_MULTIPLIER + secondary)
    return {
        "primary": float(primary),
        "secondary": float(secondary),
        "lexicographic_scalar": lexicographic_scalar,
    }


def normalize_calibration_weights(
    raw_weights: Dict[str, float],
    active_components: Dict[str, bool],
) -> Dict[str, float]:
    weights = {key: 0.0 for key in CALIBRATION_COMPONENT_KEYS}
    active_keys = [key for key in CALIBRATION_COMPONENT_KEYS if bool(active_components.get(key, False))]
    if not active_keys:
        uniform = 1.0 / max(1, len(CALIBRATION_COMPONENT_KEYS))
        return {key: uniform for key in CALIBRATION_COMPONENT_KEYS}

    lower_bounds = {
        key: float(CALIBRATION_COMPONENT_MIN_MASS.get(key, 0.0)) if key in active_keys else 0.0
        for key in CALIBRATION_COMPONENT_KEYS
    }
    upper_bounds = {
        key: float(CALIBRATION_COMPONENT_MAX_MASS.get(key, 1.0)) if key in active_keys else 0.0
        for key in CALIBRATION_COMPONENT_KEYS
    }
    lower_sum = float(sum(lower_bounds[key] for key in active_keys))
    upper_sum = float(sum(upper_bounds[key] for key in active_keys))
    if lower_sum > 1.0 + 1e-9:
        raise ValueError(f"Calibration weight floors are infeasible: sum={lower_sum:.4f}")
    if upper_sum < 1.0 - 1e-9:
        raise ValueError(f"Calibration weight caps are infeasible: sum={upper_sum:.4f}")

    positive_sum = float(sum(max(0.0, float(raw_weights.get(key, 0.0))) for key in active_keys))
    if positive_sum > 0.0:
        base_weights = {
            key: max(0.0, float(raw_weights.get(key, 0.0))) / positive_sum
            for key in active_keys
        }
    else:
        base_weights = {key: 1.0 / len(active_keys) for key in active_keys}

    for key in active_keys:
        weights[key] = lower_bounds[key]

    remaining = 1.0 - float(sum(weights[key] for key in active_keys))
    free_keys = {key for key in active_keys if upper_bounds[key] - weights[key] > 1e-12}
    while remaining > 1e-12 and free_keys:
        total_score = float(sum(max(base_weights.get(key, 0.0), 1e-12) for key in free_keys))
        saturated: List[str] = []
        residual_taken = 0.0
        provisional: Dict[str, float] = {}
        for key in free_keys:
            desired = remaining * max(base_weights.get(key, 0.0), 1e-12) / max(total_score, 1e-12)
            room = max(0.0, upper_bounds[key] - weights[key])
            if desired >= room - 1e-12:
                provisional[key] = room
                residual_taken += room
                saturated.append(key)
            else:
                provisional[key] = desired
        if saturated:
            for key in saturated:
                weights[key] += provisional[key]
            remaining -= residual_taken
            for key in saturated:
                free_keys.discard(key)
            continue
        for key, value in provisional.items():
            weights[key] += value
        remaining = 0.0

    if remaining > 1e-9:
        fallback_keys = [key for key in active_keys if upper_bounds[key] - weights[key] > 1e-12]
        if not fallback_keys:
            raise ValueError(f"Unable to satisfy calibration weight constraints; remaining mass={remaining:.6f}")
        bonus = remaining / len(fallback_keys)
        for key in fallback_keys:
            room = max(0.0, upper_bounds[key] - weights[key])
            take = min(room, bonus)
            weights[key] += take
            remaining -= take
        if remaining > 1e-6:
            raise ValueError(f"Unable to distribute calibration weight mass; remaining={remaining:.6f}")

    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("Calibration weights collapsed to zero.")
    return {key: float(value / total) for key, value in weights.items()}


def subsample_bootstrap_source_df(
    df: pd.DataFrame,
    seed: int,
    max_users: int = 250,
    max_rows: int = 1_000_000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    users = list(df["user_id"].astype(str).unique())
    if len(users) <= max_users and len(df) <= max_rows:
        return df, {
            "subsampled": False,
            "rows": int(len(df)),
            "users": int(len(users)),
            "row_cap": int(max_rows),
            "user_cap": int(max_users),
        }

    rng = np.random.default_rng(seed)
    user_blocks = {user: block.copy() for user, block in df.groupby("user_id", sort=False)}
    selected_users: List[str] = []
    selected_blocks: List[pd.DataFrame] = []
    selected_rows = 0
    for user in rng.permutation(users):
        if len(selected_users) >= max_users:
            break
        block = user_blocks[str(user)]
        block_rows = int(len(block))
        if selected_blocks and selected_rows + block_rows > max_rows:
            continue
        selected_users.append(str(user))
        selected_blocks.append(block)
        selected_rows += block_rows
        if selected_rows >= max_rows:
            break

    if not selected_blocks:
        smallest_users = sorted(users, key=lambda user: len(user_blocks[str(user)]))
        fallback_user = str(smallest_users[0])
        selected_blocks = [user_blocks[fallback_user]]
        selected_users = [fallback_user]
        selected_rows = int(len(selected_blocks[0]))

    subset = pd.concat(selected_blocks, ignore_index=True)
    return subset, {
        "subsampled": True,
        "rows": int(selected_rows),
        "users": int(len(selected_users)),
        "row_cap": int(max_rows),
        "user_cap": int(max_users),
    }


def estimate_bootstrap_weights(
    df: pd.DataFrame,
    full_targets: Dict[str, Any],
    Z: Optional[int],
    bootstrap_samples: int = 32,
    seed: int = 0,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
    boot_df, subsample_info = subsample_bootstrap_source_df(df, seed=seed + 1729)
    users = list(boot_df["user_id"].astype(str).unique())
    active_components = {key: calibration_component_has_signal(full_targets, key) for key in CALIBRATION_COMPONENT_KEYS}
    if bootstrap_samples <= 1 or len(users) <= 1:
        raw_weights = {key: (1.0 if active_components.get(key, False) else 0.0) for key in CALIBRATION_COMPONENT_KEYS}
        weights = normalize_calibration_weights(raw_weights, active_components)
        variances = {key: 0.0 for key in CALIBRATION_COMPONENT_KEYS}
        return weights, variances, raw_weights, active_components, subsample_info

    rng = np.random.default_rng(seed)
    user_blocks = {user: block.copy() for user, block in boot_df.groupby("user_id", sort=False)}
    component_samples: Dict[str, List[float]] = {k: [] for k in CALIBRATION_COMPONENT_KEYS}
    for boot_idx in range(int(bootstrap_samples)):
        draws = rng.choice(users, size=len(users), replace=True)
        blocks = []
        for draw_idx, user in enumerate(draws):
            block = user_blocks[str(user)].copy()
            block["user_id"] = f"boot{boot_idx}_{draw_idx}_{user}"
            blocks.append(block)
        boot_df = pd.concat(blocks, ignore_index=True) if blocks else df.iloc[0:0].copy()
        boot_targets = summarize_targets_from_sessionized_df(
            boot_df,
            Z=Z,
            gap_bucket_edges=full_targets.get("gap_bucket_edges"),
        )
        comps = calibration_loss_components(full_targets, boot_targets)
        for key, value in comps.items():
            component_samples[key].append(float(value))

    variances: Dict[str, float] = {}
    raw_weights: Dict[str, float] = {}
    for key, values in component_samples.items():
        arr = np.asarray(values, dtype=np.float64)
        var = float(np.var(arr, ddof=1)) if arr.size >= 2 else 0.0
        variances[key] = var
        raw_weights[key] = (1.0 / max(var, 1e-8)) if active_components.get(key, False) else 0.0

    weights = normalize_calibration_weights(raw_weights, active_components)
    return weights, variances, raw_weights, active_components, subsample_info


def extract_targets_from_logs(
    log_csv: str,
    delta_sess: float,
    Z: Optional[int] = None,
    metadata_csv: Optional[str] = None,
    random_seed: int = 0,
    bootstrap_samples: int = 32,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    df, source_info = load_public_log_dataframe(log_csv)
    df, cluster_source = attach_cluster_ids(df, metadata_csv=metadata_csv, Z=Z, random_seed=random_seed)
    if "session_id" not in df.columns:
        df = sessionize_logs(df, delta_sess)
        session_source = f"inactivity_threshold_{int(delta_sess) if float(delta_sess).is_integer() else delta_sess}"
    else:
        session_source = "provided"

    targets = summarize_targets_from_sessionized_df(df, Z=Z, include_gap_extraction_diagnostic=True)
    weights, variances, raw_weights, active_components, bootstrap_source = estimate_bootstrap_weights(
        df,
        targets,
        Z=Z,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    targets["calibration_weights"] = {key: float(value) for key, value in weights.items()}
    targets["calibration_weight_raw_inverse_variance"] = {key: float(value) for key, value in raw_weights.items()}
    targets["bootstrap_component_variance"] = {key: float(value) for key, value in variances.items()}
    targets["calibration_weight_signal"] = {key: bool(value) for key, value in active_components.items()}
    targets["calibration_weight_constraints"] = {
        "minimum_mass": {key: float(value) for key, value in CALIBRATION_COMPONENT_MIN_MASS.items()},
        "maximum_mass": {key: float(value) for key, value in CALIBRATION_COMPONENT_MAX_MASS.items()},
        "normalization": "inverse_bootstrap_variance_with_bounded_component_mass",
    }
    rollout_policies: Dict[str, Any] = {}
    if Z is not None and int(Z) > 0:
        rollout_policies["logged_cluster_marginal"] = build_logged_cluster_marginal_policy_context(
            df,
            Z=int(Z),
            cluster_source=cluster_source,
        )
    targets["calibration_rollout_policies"] = rollout_policies
    targets["delta_sess"] = float(delta_sess)
    targets["preprocessing"] = {
        **source_info,
        "cluster_source": cluster_source,
        "session_source": session_source,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_weight_source": bootstrap_source,
    }
    return targets


def simulate_targets(
    cfg: BenchConfig,
    catalog: np.ndarray,
    cards: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    backend: str = "param",
    scorer: Optional[AGLLMScorer] = None,
    gap_bucket_edges: Optional[Sequence[float]] = None,
    calibration_policy: str = DEFAULT_CALIBRATION_POLICY,
    calibration_policy_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    calibration_policy = normalize_calibration_policy_name(calibration_policy)
    item_watch_times: List[float] = []
    session_lengths: List[float] = []
    session_item_counts: List[float] = []
    gaps: List[float] = []
    session_transitions: List[Dict[str, float]] = []
    cluster_watch_values: Dict[int, List[float]] = {}
    hazard_pos_counts: Dict[int, int] = {}
    hazard_stop_counts: Dict[int, int] = {}
    session_start_hist = np.zeros(24, dtype=np.float64)
    watch_minutes_by_hour = np.zeros(24, dtype=np.float64)
    for seed in seeds:
        env = make_env(cfg, backend, catalog, cards, scorer, wrappers=None, seed=seed)
        obs = env.reset(seed=seed)
        assert env.state is not None
        start_hour = int(env._wall_clock(env.state.tau, env.state.tau_start) // 60.0) % 24
        session_start_hist[start_hour] += 1.0
        done = False
        current_items = 0
        current_watch = 0.0
        while not done:
            assert env.state is not None
            step_abs_start = env.state.tau + env.state.tau_start
            action = sample_calibration_rollout_action(
                cfg,
                env,
                calibration_policy,
                calibration_policy_context=calibration_policy_context,
            )
            obs, reward, done, info = env.step(action)
            item_watch_times.append(float(reward))
            current_items += 1
            current_watch += reward
            z = info["z"]
            cluster_watch_values.setdefault(z, []).append(reward)
            append_duration_to_hour_bins(watch_minutes_by_hour, step_abs_start, reward)
            if info["session_ended"]:
                session_lengths.append(float(info["session_length"]))
                session_item_counts.append(float(current_items))
                gaps.append(float(info["gap"]))
                session_transitions.append(
                    {
                        "prev_session_length": float(info["session_length"]),
                        "gap": float(info["gap"]),
                    }
                )
                for j in range(1, current_items + 1):
                    hazard_pos_counts[j] = hazard_pos_counts.get(j, 0) + 1
                hazard_stop_counts[current_items] = hazard_stop_counts.get(current_items, 0) + 1
                current_items = 0
                current_watch = 0.0
                if not done and env.state is not None:
                    next_hour = int(env._wall_clock(env.state.tau, env.state.tau_start) // 60.0) % 24
                    session_start_hist[next_hour] += 1.0
        summary = env.episode_summary()
        if summary["sessions"]:
            # The terminal session may have been censored; include realized length in session_lengths.
            # Hazard is not updated because no stop was observed.
            if current_items > 0:
                session_lengths.append(summary["sessions"][-1])
                session_item_counts.append(float(current_items))

    max_pos = max(hazard_pos_counts) if hazard_pos_counts else 0
    hazard = []
    for j in range(1, max_pos + 1):
        hazard.append(hazard_stop_counts.get(j, 0) / max(1, hazard_pos_counts.get(j, 0)))
    cluster_mean_watch = {int(k): safe_mean(v) for k, v in cluster_watch_values.items()}
    bucket_edges, conditional_reentry = summarize_conditional_reentry(
        session_transitions,
        bucket_edges=gap_bucket_edges,
        thresholds=RETURN_RATE_THRESHOLDS,
    )
    return {
        "item_watch_times": item_watch_times,
        "session_lengths": session_lengths,
        "session_item_counts": session_item_counts,
        "stop_hazard": hazard,
        "gaps": gaps,
        "cluster_mean_watch": cluster_mean_watch,
        "session_start_histogram": normalize_probability_vector(session_start_hist.tolist()),
        "watch_minutes_by_hour": normalize_probability_vector(watch_minutes_by_hour.tolist()),
        "gap_bucket_edges": [float(x) for x in bucket_edges],
        "conditional_return_rates": conditional_reentry,
        "calibration_policy": str(calibration_policy),
        "calibration_policy_details": describe_calibration_rollout_policy(
            calibration_policy,
            calibration_policy_context=calibration_policy_context,
        ),
    }


def calibration_loss(targets: Dict[str, Any], sim: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> float:
    comps = calibration_loss_components(targets, sim)
    loss_breakdown = calibration_loss_breakdown(targets, sim, weights=weights, components=comps)
    return float(loss_breakdown["lexicographic_scalar"])


def coerce_targets_by_delta(targets_by_delta: Dict[Any, Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    return {float(key): value for key, value in targets_by_delta.items()}


def resolve_selected_delta_sess(payload: Dict[str, Any], explicit_delta_sess: Optional[float] = None) -> float:
    if explicit_delta_sess is not None:
        return float(explicit_delta_sess)
    if payload.get("selected_delta_sess") is not None:
        return float(payload["selected_delta_sess"])
    if payload.get("reference_delta") is not None:
        return float(payload["reference_delta"])
    targets_by_delta = coerce_targets_by_delta(payload.get("targets", payload.get("targets_by_delta", {})))
    if len(targets_by_delta) == 1:
        return float(next(iter(targets_by_delta)))
    raise ValueError("Calibration payload has multiple delta_sess targets and no explicit selected_delta_sess.")


def build_calibration_audit(
    targets_by_delta: Dict[Any, Dict[str, Any]],
    sim: Dict[str, Any],
    selected_delta_sess: float,
) -> Dict[str, Any]:
    normalized_targets = coerce_targets_by_delta(targets_by_delta)
    selected_delta_sess = float(selected_delta_sess)
    if selected_delta_sess not in normalized_targets:
        raise KeyError(f"selected_delta_sess={selected_delta_sess} not found in targets_by_delta")
    target = normalized_targets[selected_delta_sess]
    session_target = summarize_distribution(target.get("session_lengths", []))
    session_sim = summarize_distribution(sim.get("session_lengths", []))
    item_count_target = summarize_distribution(target.get("session_item_counts", []))
    item_count_sim = summarize_distribution(sim.get("session_item_counts", []))
    gap_target_full = summarize_distribution(target.get("gaps", []))
    gap_sim_full = summarize_distribution(sim.get("gaps", []))
    gap_target = {"mean": gap_target_full["mean"], "p95": gap_target_full["p95"]}
    gap_sim = {"mean": gap_sim_full["mean"], "p95": gap_sim_full["p95"]}
    component_details = calibration_loss_component_details(target, sim)
    components = calibration_loss_components(target, sim)
    weights = {key: float(value) for key, value in target.get("calibration_weights", {}).items()}
    loss_breakdown = calibration_loss_breakdown(target, sim, weights=weights or None, components=components)
    session_ks = float(component_details.get("session", {}).get("ks", 0.0))
    session_item_count_ks = float(component_details.get("session_item_counts", {}).get("ks", 0.0))
    hazard_l2 = float(component_details.get("hazard", {}).get("l2", 0.0))
    mean_relative_error = safe_relative_error(session_target["mean"], session_sim["mean"])
    p95_relative_error = safe_relative_error(session_target["p95"], session_sim["p95"])
    gap_mean_relative_error = (
        safe_relative_error(gap_target["mean"], gap_sim["mean"])
        if math.isfinite(float(gap_target["mean"])) and math.isfinite(float(gap_sim["mean"]))
        else float("nan")
    )
    gap_p95_relative_error = (
        safe_relative_error(gap_target["p95"], gap_sim["p95"])
        if math.isfinite(float(gap_target["p95"])) and math.isfinite(float(gap_sim["p95"]))
        else float("nan")
    )
    session_item_count_p50_relative_error = float(
        component_details.get("session_item_counts", {}).get("p50_relative_error", float("nan"))
    )
    session_item_count_p95_relative_error = float(
        component_details.get("session_item_counts", {}).get("p95_relative_error", float("nan"))
    )
    return_conditional_mse = float(component_details.get("return_conditional", {}).get("mse", float("nan")))
    return_conditional_realism_available = bool(
        float(component_details.get("return_conditional", {}).get("matched_bucket_metrics", 0.0)) > 0.0
        and math.isfinite(return_conditional_mse)
    )
    raw_hazard_support = summarize_raw_stop_hazard_support(target.get("stop_hazard", []), sim.get("stop_hazard", []))
    severe_hazard_support_unaddressed = bool(
        component_details.get("hazard", {}).get("severe_support_mismatch_unaddressed", 0.0)
    )
    acceptance = {
        "session_mean_within_50pct": bool(mean_relative_error <= CALIBRATION_ACCEPTANCE_LIMITS["session_mean_relative_error_max"]),
        "session_p95_within_50pct": bool(p95_relative_error <= CALIBRATION_ACCEPTANCE_LIMITS["session_p95_relative_error_max"]),
        "session_ks_le_0_25": bool(session_ks <= CALIBRATION_ACCEPTANCE_LIMITS["session_ks_max"]),
        "session_item_count_p95_within_50pct": bool(
            session_item_count_p95_relative_error <= CALIBRATION_ACCEPTANCE_LIMITS["session_item_count_p95_relative_error_max"]
        ),
        "gap_mean_within_50pct": bool(
            (not math.isfinite(gap_mean_relative_error))
            or gap_mean_relative_error <= CALIBRATION_ACCEPTANCE_LIMITS["gap_mean_relative_error_max"]
        ),
        "gap_p95_within_50pct": bool(
            (not math.isfinite(gap_p95_relative_error))
            or gap_p95_relative_error <= CALIBRATION_ACCEPTANCE_LIMITS["gap_p95_relative_error_max"]
        ),
        "return_conditional_realistic_if_available": bool(
            (not return_conditional_realism_available)
            or return_conditional_mse <= CALIBRATION_ACCEPTANCE_LIMITS["return_conditional_mse_max"]
        ),
        "hazard_support_mismatch_addressed": bool(not severe_hazard_support_unaddressed),
    }
    acceptance["passed"] = bool(all(acceptance.values()))
    status = "passed" if acceptance["passed"] else "failed"
    return {
        "selected_delta_sess": selected_delta_sess,
        "status": status,
        "session_length_summary": {
            "target": session_target,
            "simulator": session_sim,
        },
        "session_item_count_summary": {
            "target": item_count_target,
            "simulator": item_count_sim,
        },
        "gap_summary": {
            "target": gap_target,
            "simulator": gap_sim,
        },
        "main_hazard_basis": str(component_details.get("hazard", {}).get("basis", "missing")),
        "raw_stop_hazard_support": raw_hazard_support,
        "stop_hazard_l2": hazard_l2,
        "ks_session_lengths": session_ks,
        "ks_session_item_counts": session_item_count_ks,
        "session_mean_relative_error": mean_relative_error,
        "session_p95_relative_error": p95_relative_error,
        "gap_mean_relative_error": gap_mean_relative_error,
        "gap_p95_relative_error": gap_p95_relative_error,
        "session_item_count_p50_relative_error": session_item_count_p50_relative_error,
        "session_item_count_p95_relative_error": session_item_count_p95_relative_error,
        "return_conditional_mse": return_conditional_mse,
        "return_conditional_realism_available": bool(return_conditional_realism_available),
        "calibration_loss_primary": float(loss_breakdown["primary"]),
        "calibration_loss_secondary": float(loss_breakdown["secondary"]),
        "calibration_loss": float(loss_breakdown["lexicographic_scalar"]),
        "calibration_loss_components": components,
        "calibration_loss_component_details": component_details,
        "calibration_weights": weights,
        "acceptance": acceptance,
    }


def build_calibration_acceptance_checks(audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    limits = CALIBRATION_ACCEPTANCE_LIMITS
    checks: List[Dict[str, Any]] = []

    def add_numeric_check(
        name: str,
        label: str,
        audit_key: str,
        limit_key: str,
    ) -> None:
        raw_value = audit.get(audit_key, float("nan"))
        value = float(raw_value) if raw_value is not None else float("nan")
        limit = float(limits[limit_key])
        passed = bool(audit.get("acceptance", {}).get(name, False))
        if math.isfinite(value):
            margin = float(limit - value)
            violation = float(max(0.0, value - limit))
        else:
            margin = float("inf")
            violation = 0.0 if passed else float("inf")
        checks.append(
            {
                "name": str(name),
                "label": str(label),
                "kind": "numeric",
                "value": value,
                "threshold": limit,
                "passed": passed,
                "margin": margin,
                "violation": violation,
            }
        )

    add_numeric_check(
        "session_mean_within_50pct",
        "Mean simulated session length within 50%",
        "session_mean_relative_error",
        "session_mean_relative_error_max",
    )
    add_numeric_check(
        "session_p95_within_50pct",
        "Simulated p95 session length within 50%",
        "session_p95_relative_error",
        "session_p95_relative_error_max",
    )
    add_numeric_check(
        "session_item_count_p95_within_50pct",
        "Simulated p95 session item count within 50%",
        "session_item_count_p95_relative_error",
        "session_item_count_p95_relative_error_max",
    )
    add_numeric_check(
        "gap_mean_within_50pct",
        "Simulated gap mean within 50%",
        "gap_mean_relative_error",
        "gap_mean_relative_error_max",
    )
    add_numeric_check(
        "gap_p95_within_50pct",
        "Simulated gap p95 within 50%",
        "gap_p95_relative_error",
        "gap_p95_relative_error_max",
    )
    add_numeric_check(
        "return_conditional_realistic_if_available",
        "Conditional return realism when matched buckets exist",
        "return_conditional_mse",
        "return_conditional_mse_max",
    )
    add_numeric_check(
        "session_ks_le_0_25",
        "KS(session_lengths)",
        "ks_session_lengths",
        "session_ks_max",
    )
    hazard_passed = bool(audit.get("acceptance", {}).get("hazard_support_mismatch_addressed", False))
    checks.append(
        {
            "name": "hazard_support_mismatch_addressed",
            "label": "Severe raw stop-hazard support mismatch addressed",
            "kind": "binary",
            "value": 1.0 if hazard_passed else 0.0,
            "threshold": 1.0,
            "passed": hazard_passed,
            "margin": 1.0 if hazard_passed else -1.0,
            "violation": 0.0 if hazard_passed else 1.0,
        }
    )
    return checks


def summarize_calibration_acceptance(audit: Dict[str, Any]) -> Dict[str, Any]:
    checks = build_calibration_acceptance_checks(audit)
    failed_checks = [check for check in checks if not bool(check["passed"])]
    finite_margins = [float(check["margin"]) for check in checks if math.isfinite(float(check["margin"]))]
    worst_margin = min(finite_margins) if finite_margins else float("inf")
    worst_margin_violation = max((float(check["violation"]) for check in checks), default=0.0)
    return {
        "checks": checks,
        "failed_checks": failed_checks,
        "failed_check_count": int(len(failed_checks)),
        "failed_check_names": [str(check["name"]) for check in failed_checks],
        "failed_check_labels": [str(check["label"]) for check in failed_checks],
        "worst_margin": float(worst_margin),
        "worst_margin_violation": float(worst_margin_violation),
    }


def apply_calibration_audit_to_row(row: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
    acceptance_summary = summarize_calibration_acceptance(audit)
    row["status"] = str(audit["status"])
    row["acceptance_passed"] = bool(audit["acceptance"]["passed"])
    row["acceptance_failed_checks"] = int(acceptance_summary["failed_check_count"])
    row["acceptance_failed_check_names"] = ",".join(acceptance_summary["failed_check_names"])
    row["acceptance_failed_check_labels"] = " | ".join(acceptance_summary["failed_check_labels"])
    row["acceptance_worst_margin"] = float(acceptance_summary["worst_margin"])
    row["acceptance_worst_margin_violation"] = float(acceptance_summary["worst_margin_violation"])
    row["session_mean_relative_error"] = float(audit["session_mean_relative_error"])
    row["session_p95_relative_error"] = float(audit["session_p95_relative_error"])
    row["session_item_count_p95_relative_error"] = float(audit["session_item_count_p95_relative_error"])
    row["gap_mean_relative_error"] = float(audit["gap_mean_relative_error"])
    row["gap_p95_relative_error"] = float(audit["gap_p95_relative_error"])
    row["return_conditional_mse"] = float(audit["return_conditional_mse"])
    row["ks_session_lengths"] = float(audit["ks_session_lengths"])
    row["acceptance_margin_session_mean_within_50pct"] = float(
        next(check["margin"] for check in acceptance_summary["checks"] if check["name"] == "session_mean_within_50pct")
    )
    row["acceptance_margin_session_p95_within_50pct"] = float(
        next(check["margin"] for check in acceptance_summary["checks"] if check["name"] == "session_p95_within_50pct")
    )
    row["acceptance_margin_session_item_count_p95_within_50pct"] = float(
        next(check["margin"] for check in acceptance_summary["checks"] if check["name"] == "session_item_count_p95_within_50pct")
    )
    row["acceptance_margin_gap_mean_within_50pct"] = float(
        next(check["margin"] for check in acceptance_summary["checks"] if check["name"] == "gap_mean_within_50pct")
    )
    row["acceptance_margin_gap_p95_within_50pct"] = float(
        next(check["margin"] for check in acceptance_summary["checks"] if check["name"] == "gap_p95_within_50pct")
    )
    row["acceptance_margin_return_conditional_realistic_if_available"] = float(
        next(
            check["margin"]
            for check in acceptance_summary["checks"]
            if check["name"] == "return_conditional_realistic_if_available"
        )
    )
    row["acceptance_margin_session_ks_le_0_25"] = float(
        next(check["margin"] for check in acceptance_summary["checks"] if check["name"] == "session_ks_le_0_25")
    )
    row["acceptance_margin_hazard_support_mismatch_addressed"] = float(
        next(check["margin"] for check in acceptance_summary["checks"] if check["name"] == "hazard_support_mismatch_addressed")
    )
    return row


def build_calibration_payload(
    targets_by_delta: Dict[Any, Dict[str, Any]],
    sim: Dict[str, Any],
    selected_delta_sess: float,
    log_csv: str,
    n_trials: int,
    episodes_per_trial: int,
    seed: int,
    calibration_policy: str = DEFAULT_CALIBRATION_POLICY,
    thresholds: Optional[Dict[str, Any]] = None,
    calibrated_cfg: Optional[BenchConfig] = None,
) -> Dict[str, Any]:
    audit = build_calibration_audit(targets_by_delta, sim, selected_delta_sess)
    policy = normalize_calibration_policy_name(sim.get("calibration_policy", calibration_policy))
    policy_details = describe_calibration_rollout_policy(
        policy,
        calibration_policy_context=sim.get("calibration_policy_details"),
    )
    fitted_continue_params = (
        None
        if calibrated_cfg is None
        else {
            "continue_logit_bias": float(calibrated_cfg.continue_logit_bias),
            "continue_logit_temp": float(calibrated_cfg.continue_logit_temp),
            "r_max": float(calibrated_cfg.r_max),
        }
    )
    return {
        "selected_delta_sess": float(selected_delta_sess),
        "reference_delta": float(selected_delta_sess),
        "targets": coerce_targets_by_delta(targets_by_delta),
        "sim": sim,
        "calibration_policy": str(policy),
        "calibration_policy_details": copy.deepcopy(policy_details),
        "fitted_config": None if calibrated_cfg is None else calibrated_cfg.to_dict(),
        "fitted_continue_params": copy.deepcopy(fitted_continue_params),
        "calibration_loss": float(audit["calibration_loss"]),
        "calibration_loss_primary": float(audit["calibration_loss_primary"]),
        "calibration_loss_secondary": float(audit["calibration_loss_secondary"]),
        "calibration_loss_components": {key: float(value) for key, value in audit["calibration_loss_components"].items()},
        "manifest": {
            "kind": "calibration_manifest",
            "status": audit["status"],
            "model_family_assessment": "adequate" if audit["acceptance"]["passed"] else "inadequate",
            "selected_delta_sess": float(selected_delta_sess),
            "threshold_source": None if thresholds is None else thresholds.get("threshold_source"),
            "log_csv": str(log_csv),
            "n_trials": int(n_trials),
            "episodes_per_trial": int(episodes_per_trial),
            "seed": int(seed),
            "calibration_policy": str(policy),
            "calibration_policy_details": copy.deepcopy(policy_details),
            "thresholds": copy.deepcopy(thresholds or {}),
            "acceptance": audit["acceptance"],
            "fitted_continue_params": copy.deepcopy(fitted_continue_params),
        },
        "thresholds": copy.deepcopy(thresholds or {}),
        "audit": audit,
    }


def render_calibration_audit_markdown(audit: Dict[str, Any]) -> str:
    session_target = audit["session_length_summary"]["target"]
    session_sim = audit["session_length_summary"]["simulator"]
    item_count_target = audit["session_item_count_summary"]["target"]
    item_count_sim = audit["session_item_count_summary"]["simulator"]
    gap_target = audit["gap_summary"]["target"]
    gap_sim = audit["gap_summary"]["simulator"]
    raw_hazard_support = audit.get("raw_stop_hazard_support", {})
    limits = CALIBRATION_ACCEPTANCE_LIMITS
    def render_acceptance_value(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return f"{numeric:.3f}" if math.isfinite(numeric) else "n/a"
    lines = [
        "# Calibration audit",
        "",
        f"- Selected `delta_sess`: `{audit['selected_delta_sess']:g}`",
        f"- Status: **{str(audit['status']).upper()}**",
        f"- `calibration_loss`: `{audit['calibration_loss']:.6f}`",
        f"- `calibration_loss_stage_a`: `{audit['calibration_loss_primary']:.6f}`",
        f"- `calibration_loss_stage_b`: `{audit['calibration_loss_secondary']:.6f}`",
        f"- Main hazard basis: `{audit.get('main_hazard_basis', 'missing')}`",
        "",
        "## Session lengths",
        "",
        "| Metric | Target | Simulator | Relative error |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in ("mean", "median", "p90", "p95", "p99"):
        rel_error = safe_relative_error(float(session_target[metric]), float(session_sim[metric]))
        lines.append(
            f"| `{metric}` | {float(session_target[metric]):.3f} | {float(session_sim[metric]):.3f} | {rel_error:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Session item counts",
            "",
            "| Metric | Target | Simulator | Relative error |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric in ("mean", "median", "p90", "p95", "p99"):
        rel_error = safe_relative_error(float(item_count_target[metric]), float(item_count_sim[metric]))
        lines.append(
            f"| `{metric}` | {float(item_count_target[metric]):.3f} | {float(item_count_sim[metric]):.3f} | {rel_error:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Gaps",
            "",
            "| Metric | Target | Simulator | Relative error |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric in ("mean", "p95"):
        rel_error = safe_relative_error(float(gap_target[metric]), float(gap_sim[metric]))
        lines.append(
            f"| `{metric}` | {float(gap_target[metric]):.3f} | {float(gap_sim[metric]):.3f} | {rel_error:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Loss components",
            "",
            "| Component | Weight | Value |",
            "| --- | ---: | ---: |",
        ]
    )
    for key in CALIBRATION_COMPONENT_KEYS:
        lines.append(
            f"| `{key}` | {float(audit.get('calibration_weights', {}).get(key, 0.0)):.3f} | {float(audit['calibration_loss_components'].get(key, 0.0)):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- `KS(session_lengths)`: `{float(audit['ks_session_lengths']):.6f}`",
            f"- `KS(session_item_counts)`: `{float(audit['ks_session_item_counts']):.6f}`",
            f"- Main hazard `l2`: `{float(audit['stop_hazard_l2']):.6f}`",
            "- Raw item-position stop hazard is retained as an appendix-only diagnostic and is not used for the main loss when session-length support is available.",
            f"- Raw item-position stop-hazard support: `len(target)={int(raw_hazard_support.get('target_len', 0))}`, `len(sim)={int(raw_hazard_support.get('sim_len', 0))}`, overlap=`{int(raw_hazard_support.get('overlap_len', 0))}`",
            f"- Raw item-position support note: {raw_hazard_support.get('note', 'n/a')}",
            "",
            "## Acceptance",
            "",
            "| Check | Threshold | Value | Pass |",
            "| --- | --- | ---: | --- |",
            f"| Mean simulated session length within 50% | `<= {limits['session_mean_relative_error_max']:.2f}` | {float(audit['session_mean_relative_error']):.3f} | {'yes' if audit['acceptance']['session_mean_within_50pct'] else 'no'} |",
            f"| Simulated p95 session length within 50% | `<= {limits['session_p95_relative_error_max']:.2f}` | {float(audit['session_p95_relative_error']):.3f} | {'yes' if audit['acceptance']['session_p95_within_50pct'] else 'no'} |",
            f"| Simulated p95 session item count within 50% | `<= {limits['session_item_count_p95_relative_error_max']:.2f}` | {float(audit['session_item_count_p95_relative_error']):.3f} | {'yes' if audit['acceptance']['session_item_count_p95_within_50pct'] else 'no'} |",
            f"| Simulated gap mean within 50% | `<= {limits['gap_mean_relative_error_max']:.2f}` | {render_acceptance_value(audit.get('gap_mean_relative_error'))} | {'yes' if audit['acceptance']['gap_mean_within_50pct'] else 'no'} |",
            f"| Simulated gap p95 within 50% | `<= {limits['gap_p95_relative_error_max']:.2f}` | {render_acceptance_value(audit.get('gap_p95_relative_error'))} | {'yes' if audit['acceptance']['gap_p95_within_50pct'] else 'no'} |",
            f"| Conditional return realism when matched buckets exist | `<= {limits['return_conditional_mse_max']:.2f}` or `n/a` | {render_acceptance_value(audit.get('return_conditional_mse'))} | {'yes' if audit['acceptance']['return_conditional_realistic_if_available'] else 'no'} |",
            f"| `KS(session_lengths)` | `<= {limits['session_ks_max']:.2f}` | {float(audit['ks_session_lengths']):.3f} | {'yes' if audit['acceptance']['session_ks_le_0_25'] else 'no'} |",
            f"| Severe raw stop-hazard support mismatch addressed | `yes` | {1.0 if audit['acceptance']['hazard_support_mismatch_addressed'] else 0.0:.1f} | {'yes' if audit['acceptance']['hazard_support_mismatch_addressed'] else 'no'} |",
        ]
    )
    return "\n".join(lines) + "\n"


def build_calibration_confirmation_manifest(
    selected_final_row: Dict[str, Any],
    confirmation_audit: Dict[str, Any],
    confirmation_seed_list: Sequence[int],
    *,
    selected_trial_number: int,
    selected_reason: str,
    selected_delta_sess: float,
    calibration_policy: str,
) -> Dict[str, Any]:
    final_acceptance_passed = bool(selected_final_row.get("acceptance_passed", calibration_row_acceptance_passed(selected_final_row)))
    confirmation_acceptance_summary = summarize_calibration_acceptance(confirmation_audit)
    confirmation_acceptance_passed = bool(confirmation_audit.get("acceptance", {}).get("passed", False))
    selected_config_status = "confirmed" if (final_acceptance_passed and confirmation_acceptance_passed) else "unconfirmed"
    return {
        "kind": "calibration_confirmation_manifest",
        "selected_trial_number": int(selected_trial_number),
        "selected_reason": str(selected_reason),
        "selected_delta_sess": float(selected_delta_sess),
        "calibration_policy": str(normalize_calibration_policy_name(calibration_policy)),
        "final_status": str(selected_final_row.get("status", "passed" if final_acceptance_passed else "failed")),
        "final_acceptance_passed": bool(final_acceptance_passed),
        "final_failed_checks": int(selected_final_row.get("acceptance_failed_checks", 0 if final_acceptance_passed else 1)),
        "final_worst_margin_violation": float(
            selected_final_row.get("acceptance_worst_margin_violation", 0.0 if final_acceptance_passed else 1.0)
        ),
        "confirmation_audit_status": str(confirmation_audit.get("status", "failed")),
        "confirmation_acceptance_passed": bool(confirmation_acceptance_passed),
        "confirmation_failed_checks": int(confirmation_acceptance_summary["failed_check_count"]),
        "confirmation_worst_margin_violation": float(confirmation_acceptance_summary["worst_margin_violation"]),
        "confirmation_seed_count": int(len(confirmation_seed_list)),
        "confirmation_seed_list": [int(value) for value in confirmation_seed_list],
        "confirmation_status": str(selected_config_status),
        "confirmed": bool(selected_config_status == "confirmed"),
    }


def render_calibration_confirmation_markdown(
    confirmation_manifest: Dict[str, Any],
    confirmation_audit: Dict[str, Any],
) -> str:
    audit_md = render_calibration_audit_markdown(confirmation_audit)
    if audit_md.startswith("# Calibration audit"):
        audit_md = audit_md.replace("# Calibration audit", "## Confirmation audit", 1)
    lines = [
        "# Calibration confirmation",
        "",
        f"- Selected trial: `{int(confirmation_manifest['selected_trial_number'])}`",
        f"- Selected reason: `{confirmation_manifest['selected_reason']}`",
        f"- Calibration policy: `{confirmation_manifest['calibration_policy']}`",
        f"- Final reevaluation status: **{str(confirmation_manifest['final_status']).upper()}**",
        f"- Confirmation audit status: **{str(confirmation_manifest['confirmation_audit_status']).upper()}**",
        f"- Selected config status: **{str(confirmation_manifest['confirmation_status']).upper()}**",
        f"- Confirmation seeds: `{int(confirmation_manifest['confirmation_seed_count'])}` fresh episodes",
        "",
    ]
    return "\n".join(lines) + "\n" + audit_md


def build_calibration_audit_from_payload(payload: Dict[str, Any], selected_delta_sess: Optional[float] = None) -> Dict[str, Any]:
    delta = resolve_selected_delta_sess(payload, explicit_delta_sess=selected_delta_sess)
    targets_by_delta = payload.get("targets", payload.get("targets_by_delta", {}))
    return build_calibration_audit(targets_by_delta, payload["sim"], delta)


def resolve_item_watch_times(metrics: Dict[str, Any]) -> List[float]:
    for key in ("item_watch_times", "per_item_watch_times", "watch_time_min_values"):
        values = metrics.get(key)
        if values:
            return [float(value) for value in values]
    return []


def build_quantile_comparison(target_values: Sequence[float], sim_values: Sequence[float]) -> Dict[str, Any]:
    target_summary = named_quantile_summary(target_values)
    sim_summary = named_quantile_summary(sim_values)
    delta_summary: Dict[str, float] = {}
    for label, _ in FEASIBILITY_QUANTILES:
        target_value = float(target_summary.get(label, float("nan")))
        sim_value = float(sim_summary.get(label, float("nan")))
        delta_summary[label] = float(sim_value - target_value) if math.isfinite(target_value) and math.isfinite(sim_value) else float("nan")
    return {
        "target": target_summary,
        "simulator": sim_summary,
        "sim_minus_target": delta_summary,
    }


def build_stop_hazard_support_diagnostics(
    target_stop_hazard: Sequence[float],
    sim_stop_hazard: Sequence[float],
) -> Dict[str, Any]:
    return summarize_raw_stop_hazard_support(target_stop_hazard, sim_stop_hazard)


def bounded_continue_quantile_item_count(p_continue: float, q: float = 0.95) -> int:
    if p_continue <= 0.0:
        return 1
    if p_continue >= 1.0:
        return int(10**9)
    target_tail = max(1e-12, 1.0 - float(q))
    return int(math.ceil(math.log(target_tail) / math.log(float(p_continue))))


def bounded_continue_survival_mass(p_continue: float, item_count: int) -> float:
    if item_count <= 1:
        return 1.0
    return float(float(p_continue) ** max(0, int(item_count) - 1))


def resolve_fitted_bench_config_from_payload(payload: Dict[str, Any]) -> Optional[BenchConfig]:
    config_field_names = set(BenchConfig.__dataclass_fields__.keys())
    for key in ("fitted_config", "config_calibrated", "config", "cfg"):
        candidate = payload.get(key)
        if not isinstance(candidate, dict):
            continue
        config_kwargs = {name: value for name, value in candidate.items() if name in config_field_names}
        if config_kwargs:
            return BenchConfig(**config_kwargs)
    continue_params = payload.get("fitted_continue_params")
    if not isinstance(continue_params, dict):
        continue_params = payload.get("manifest", {}).get("fitted_continue_params")
    if isinstance(continue_params, dict):
        cfg = BenchConfig()
        for key in ("continue_logit_bias", "continue_logit_temp", "r_max"):
            if key in continue_params:
                setattr(cfg, key, float(continue_params[key]))
        return cfg
    return None


def build_structural_feasibility_note(
    target_session_lengths: Sequence[float],
    target_session_item_counts: Sequence[float],
    fitted_cfg: Optional[BenchConfig] = None,
) -> Dict[str, Any]:
    effective_cfg = BenchConfig() if fitted_cfg is None else fitted_cfg
    q_base_lower = -1.0
    q_base_upper = 1.0
    continue_logit_bias = float(effective_cfg.continue_logit_bias)
    continue_logit_temp = float(effective_cfg.continue_logit_temp)
    continue_logit_at_q_base_lower = float(continue_logit_bias + continue_logit_temp * q_base_lower)
    continue_logit_at_q_base_upper = float(continue_logit_bias + continue_logit_temp * q_base_upper)
    continue_logit_min = float(min(continue_logit_at_q_base_lower, continue_logit_at_q_base_upper))
    continue_logit_max = float(max(continue_logit_at_q_base_lower, continue_logit_at_q_base_upper))
    q_base_at_max_p_continue = float(q_base_lower if continue_logit_at_q_base_lower >= continue_logit_at_q_base_upper else q_base_upper)
    p_continue_max = float(sigmoid(continue_logit_max))
    r_max = float(effective_cfg.r_max)
    target_p95_session_length = safe_quantile(target_session_lengths, 0.95)
    target_p95_session_item_count = safe_quantile(target_session_item_counts, 0.95)
    minimum_items_needed = (
        int(max(1, math.ceil(float(target_p95_session_length) / max(r_max, 1e-8))))
        if math.isfinite(float(target_p95_session_length))
        else 0
    )
    rounded_target_item_count = (
        int(max(1, math.ceil(float(target_p95_session_item_count))))
        if math.isfinite(float(target_p95_session_item_count))
        else 0
    )
    max_p95_item_count = int(bounded_continue_quantile_item_count(p_continue_max, q=0.95))
    max_p95_session_length = float(r_max * max_p95_item_count)
    length_tail_survival_mass = (
        bounded_continue_survival_mass(p_continue_max, minimum_items_needed)
        if minimum_items_needed > 0
        else float("nan")
    )
    item_tail_survival_mass = (
        bounded_continue_survival_mass(p_continue_max, rounded_target_item_count)
        if rounded_target_item_count > 0
        else float("nan")
    )
    minutes_tail_plausible = bool(minimum_items_needed > 0 and minimum_items_needed <= max_p95_item_count)
    item_count_tail_plausible = bool(rounded_target_item_count > 0 and rounded_target_item_count <= max_p95_item_count)
    structurally_capable = bool(minutes_tail_plausible and item_count_tail_plausible)
    verdict = (
        "The current simulator family is structurally capable in principle of matching the public-log p95 session tail under the current step semantics."
        if structurally_capable
        else "The current simulator family is structurally incapable of matching the public-log p95 session tail under the current step semantics."
    )
    explanation = (
        "The bounded continue score leaves enough headroom in the item-count tail to make the target p95 session length and p95 session-item count plausible."
        if structurally_capable
        else "The bounded continue score implies a p95 item-count ceiling that is below the public-log tail requirement, even before accounting for per-item watch times below r_max."
    )
    return {
        "continue_parameter_source": "default_bench_config_fallback" if fitted_cfg is None else "payload_fitted_config",
        "q_base_range": {
            "lower": float(q_base_lower),
            "upper": float(q_base_upper),
            "interpretation": "q_base is a convex mixture of tanh(q_r) and tanh(q_m), so the parametric continue score is bounded to [-1, 1] before any wrapper friction or LLM fusion.",
        },
        "continue_logit_bias": float(continue_logit_bias),
        "continue_logit_temp": float(continue_logit_temp),
        "continue_logit_range": {
            "at_q_base_lower": float(continue_logit_at_q_base_lower),
            "at_q_base_upper": float(continue_logit_at_q_base_upper),
            "min": float(continue_logit_min),
            "max": float(continue_logit_max),
        },
        "q_base_at_max_parametric_p_continue": float(q_base_at_max_p_continue),
        "max_parametric_p_continue": float(p_continue_max),
        "r_max": float(r_max),
        "target_p95_session_length": float(target_p95_session_length),
        "target_p95_session_item_count": float(target_p95_session_item_count),
        "minimum_items_needed_for_target_p95_session_length": int(minimum_items_needed),
        "ceiled_target_p95_session_item_count": int(rounded_target_item_count),
        "max_p95_session_item_count_under_bounded_continue": int(max_p95_item_count),
        "max_p95_session_length_under_bounded_continue": float(max_p95_session_length),
        "tail_survival_mass_at_minimum_items_needed": float(length_tail_survival_mass),
        "tail_survival_mass_at_target_p95_item_count": float(item_tail_survival_mass),
        "minutes_tail_plausible": bool(minutes_tail_plausible),
        "item_count_tail_plausible": bool(item_count_tail_plausible),
        "public_log_p95_session_tail_plausible": bool(structurally_capable),
        "verdict": verdict,
        "explanation": explanation,
    }


def build_calibration_feasibility_report(
    targets_by_delta: Dict[Any, Dict[str, Any]],
    sim: Dict[str, Any],
    selected_delta_sess: float,
    fitted_cfg: Optional[BenchConfig] = None,
) -> Dict[str, Any]:
    normalized_targets = coerce_targets_by_delta(targets_by_delta)
    selected_delta_sess = float(selected_delta_sess)
    if selected_delta_sess not in normalized_targets:
        raise KeyError(f"selected_delta_sess={selected_delta_sess} not found in targets_by_delta")
    target = normalized_targets[selected_delta_sess]
    audit = build_calibration_audit(normalized_targets, sim, selected_delta_sess)
    target_item_watch_times = resolve_item_watch_times(target)
    sim_item_watch_times = resolve_item_watch_times(sim)
    stop_hazard_support = build_stop_hazard_support_diagnostics(
        target.get("stop_hazard", []),
        sim.get("stop_hazard", []),
    )
    structural_feasibility = build_structural_feasibility_note(
        target.get("session_lengths", []),
        target.get("session_item_counts", []),
        fitted_cfg=fitted_cfg,
    )
    per_item_watch_available = bool(target_item_watch_times and sim_item_watch_times)
    per_item_watch_note = (
        "Per-item watch-time quantiles were derived from the payload samples."
        if per_item_watch_available
        else "Per-item watch-time samples are missing on at least one side of the payload; regenerate calibration_payload.json with the current code to populate this section."
    )
    return {
        "selected_delta_sess": float(selected_delta_sess),
        "current_fit_status": str(audit["status"]),
        "current_fit_acceptance": copy.deepcopy(audit["acceptance"]),
        "session_length_quantiles": build_quantile_comparison(
            target.get("session_lengths", []),
            sim.get("session_lengths", []),
        ),
        "session_item_count_quantiles": build_quantile_comparison(
            target.get("session_item_counts", []),
            sim.get("session_item_counts", []),
        ),
        "per_item_watch_quantiles": build_quantile_comparison(
            target_item_watch_times,
            sim_item_watch_times,
        ),
        "stop_hazard_support": stop_hazard_support,
        "per_item_watch_available": bool(per_item_watch_available),
        "per_item_watch_note": per_item_watch_note,
        "structural_feasibility": structural_feasibility,
        "current_fit_tail_gap": {
            "session_length_p95_sim_minus_target": float(
                safe_quantile(sim.get("session_lengths", []), 0.95) - safe_quantile(target.get("session_lengths", []), 0.95)
            ),
            "session_item_count_p95_sim_minus_target": float(
                safe_quantile(sim.get("session_item_counts", []), 0.95) - safe_quantile(target.get("session_item_counts", []), 0.95)
            ),
            "gap_mean_relative_error": float(audit.get("gap_mean_relative_error", float("nan"))),
            "gap_p95_relative_error": float(audit.get("gap_p95_relative_error", float("nan"))),
        },
    }


def build_calibration_feasibility_report_from_payload(
    payload: Dict[str, Any],
    selected_delta_sess: Optional[float] = None,
) -> Dict[str, Any]:
    delta = resolve_selected_delta_sess(payload, explicit_delta_sess=selected_delta_sess)
    targets_by_delta = payload.get("targets", payload.get("targets_by_delta", {}))
    fitted_cfg = resolve_fitted_bench_config_from_payload(payload)
    return build_calibration_feasibility_report(targets_by_delta, payload["sim"], delta, fitted_cfg=fitted_cfg)


def format_report_number(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def render_quantile_comparison_markdown(title: str, comparison: Dict[str, Any]) -> List[str]:
    lines = [
        f"## {title}",
        "",
        "| Quantile | Target | Simulator | Sim - Target |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, _ in FEASIBILITY_QUANTILES:
        lines.append(
            f"| `{label}` | {format_report_number(comparison['target'].get(label))} | "
            f"{format_report_number(comparison['simulator'].get(label))} | "
            f"{format_report_number(comparison['sim_minus_target'].get(label))} |"
        )
    lines.append("")
    return lines


def render_calibration_feasibility_markdown(report: Dict[str, Any]) -> str:
    stop_support = report["stop_hazard_support"]
    structural = report["structural_feasibility"]
    lines = [
        "# Calibration feasibility audit",
        "",
        f"- Selected `delta_sess`: `{report['selected_delta_sess']:g}`",
        f"- Current calibration audit status: **{str(report['current_fit_status']).upper()}**",
        f"- Structural verdict: **{'CAPABLE IN PRINCIPLE' if structural['public_log_p95_session_tail_plausible'] else 'STRUCTURALLY INCAPABLE'}**",
        f"- Tail verdict: {structural['verdict']}",
        f"- Current fitted simulator `p95` session length gap (sim - target): `{format_report_number(report['current_fit_tail_gap']['session_length_p95_sim_minus_target'])}` min",
        f"- Current fitted simulator `p95` session-item-count gap (sim - target): `{format_report_number(report['current_fit_tail_gap']['session_item_count_p95_sim_minus_target'])}` items",
        f"- Current fitted simulator gap mean relative error: `{format_report_number(report['current_fit_tail_gap']['gap_mean_relative_error'])}`",
        f"- Current fitted simulator gap `p95` relative error: `{format_report_number(report['current_fit_tail_gap']['gap_p95_relative_error'])}`",
        "",
    ]
    lines.extend(render_quantile_comparison_markdown("Session Length Quantiles (min)", report["session_length_quantiles"]))
    lines.extend(render_quantile_comparison_markdown("Session Item Count Quantiles", report["session_item_count_quantiles"]))
    lines.extend(render_quantile_comparison_markdown("Per-Item Watch-Time Quantiles (min)", report["per_item_watch_quantiles"]))
    lines.extend(
        [
            "## Stop-Hazard Support",
            "",
            f"- `len(target_stop_hazard) = {int(stop_support['target_len'])}`",
            f"- `len(sim_stop_hazard) = {int(stop_support['sim_len'])}`",
            f"- Overlap used for direct pointwise comparison: `{int(stop_support['overlap_len'])}` positions",
            f"- Truncated overlap-only comparison: `{'yes' if stop_support['truncated'] else 'no'}`",
            f"- Note: {stop_support['note']}",
            "",
            "## Analytic Feasibility Note",
            "",
            f"- Continue-parameter source: `{structural['continue_parameter_source']}`.",
            f"- Current parametric `q_base` range under the code path: `[{format_report_number(structural['q_base_range']['lower'])}, {format_report_number(structural['q_base_range']['upper'])}]`.",
            f"- Fitted `continue_logit_bias`: `{format_report_number(structural['continue_logit_bias'])}`.",
            f"- Fitted `continue_logit_temp`: `{format_report_number(structural['continue_logit_temp'])}`.",
            f"- Parametric continue-logit range over `q_base \\in [-1,1]`: `[{format_report_number(structural['continue_logit_range']['min'])}, {format_report_number(structural['continue_logit_range']['max'])}]`.",
            f"- Current max parametric `p_continue`: `{format_report_number(structural['max_parametric_p_continue'], digits=4)}`.",
            f"- Current `r_max`: `{format_report_number(structural['r_max'])}` min/item.",
            f"- Target `p95` session length: `{format_report_number(structural['target_p95_session_length'])}` min.",
            f"- Target `p95` session-item count: `{format_report_number(structural['target_p95_session_item_count'])}` items.",
            f"- Minimum item count needed to reach the target `p95` session length at the current `r_max`: `{int(structural['minimum_items_needed_for_target_p95_session_length'])}`.",
            f"- Best-case bounded-model `p95` session-item-count ceiling: `{int(structural['max_p95_session_item_count_under_bounded_continue'])}` items.",
            f"- Best-case bounded-model `p95` session-length ceiling at the current `r_max`: `{format_report_number(structural['max_p95_session_length_under_bounded_continue'])}` min.",
            f"- Tail survival mass at the minimum required item count: `{format_report_number(structural['tail_survival_mass_at_minimum_items_needed'], digits=4)}`.",
            f"- Tail survival mass at the target `p95` session-item count: `{format_report_number(structural['tail_survival_mass_at_target_p95_item_count'], digits=4)}`.",
            f"- Minute-tail plausibility under the bounded continue model: `{'yes' if structural['minutes_tail_plausible'] else 'no'}`.",
            f"- Item-count-tail plausibility under the bounded continue model: `{'yes' if structural['item_count_tail_plausible'] else 'no'}`.",
            f"- Interpretation: {structural['q_base_range']['interpretation']}",
            f"- Conclusion: {structural['explanation']}",
            "",
            f"{report['per_item_watch_note']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_calibration_feasibility_artifacts(
    payload: Dict[str, Any],
    outdir: Path | str,
    selected_delta_sess: Optional[float] = None,
) -> Dict[str, Any]:
    outdir = ensure_dir(outdir)
    report = build_calibration_feasibility_report_from_payload(payload, selected_delta_sess=selected_delta_sess)
    markdown = render_calibration_feasibility_markdown(report)
    markdown_path = outdir / "feasibility_report.md"
    json_path = outdir / "feasibility_report.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    log_artifact_written(markdown_path, "calibration_feasibility_markdown")
    json_dumps(report, json_path)
    log_artifact_written(json_path, "calibration_feasibility_json")

    resolved_delta_sess = resolve_selected_delta_sess(payload, explicit_delta_sess=selected_delta_sess)
    targets_by_delta = coerce_targets_by_delta(payload.get("targets", payload.get("targets_by_delta", {})))
    target = targets_by_delta[resolved_delta_sess]
    sim = payload["sim"]
    save_cdf_plot(
        target.get("session_item_counts", []),
        sim.get("session_item_counts", []),
        "Session item count",
        "Calibration feasibility: session item count CDF",
        outdir / "fig_session_item_count_cdf.png",
    )
    save_stop_hazard_support_plot(
        target.get("stop_hazard", []),
        sim.get("stop_hazard", []),
        outdir / "fig_stop_hazard_support.png",
    )
    save_cdf_plot(
        resolve_item_watch_times(target),
        resolve_item_watch_times(sim),
        "Per-item watch time (min)",
        "Calibration feasibility: per-item watch-time CDF",
        outdir / "fig_per_item_watch_cdf.png",
    )
    return report


CALIBRATABLE_KEYS = [
    "scale",
    "alpha",
    "beta",
    "gamma_h",
    "gamma_f",
    "lambda_rep",
    "gap_mu",
    "gap_sigma",
    "gap_w_h",
    "gap_w_r",
    "gap_w_f",
    "gap_w_c",
    "omega_h",
    "omega_r",
    "omega_c",
    "omega_f",
    "omega_o",
    "kappa_0",
    "kappa_h",
    "kappa_c",
]
CALIBRATION_HISTORY_PARAM_KEYS = tuple(CALIBRATABLE_KEYS + ["continue_logit_bias", "continue_logit_temp"])
OPTUNA_WIDE_MULTIPLICATIVE_KEYS = (
    "alpha",
    "beta",
    "gamma_h",
    "gamma_f",
    "lambda_rep",
    "gap_w_h",
    "gap_w_r",
    "gap_w_f",
    "gap_w_c",
    "omega_h",
    "omega_r",
    "omega_c",
    "omega_f",
    "omega_o",
)
OPTUNA_ADDITIVE_KEYS = ("gap_mu", "kappa_0", "kappa_h", "kappa_c")


def require_optuna() -> Any:
    try:
        import optuna  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Optuna is required for calibrate_optuna. Install it with `pip install optuna`."
        ) from exc
    return optuna


def normalize_calibration_targets(
    targets: Dict[str, Any],
    selection_delta: Optional[float] = None,
) -> Tuple[Dict[float, Dict[str, Any]], float, Dict[str, Any]]:
    if "session_lengths" in targets:
        delta_targets = {
            float(selection_delta if selection_delta is not None else targets.get("delta_sess", 0.0)): targets
        }
    else:
        delta_targets = {float(k): v for k, v in targets.items()}
    if not delta_targets:
        raise ValueError("Calibration requires at least one target set.")
    if selection_delta is None:
        selection_delta = sorted(delta_targets)[0]
    selection_delta = float(selection_delta)
    if selection_delta not in delta_targets:
        raise ValueError(f"selection_delta={selection_delta} is not present in targets.")
    return delta_targets, selection_delta, delta_targets[selection_delta]


def generate_calibration_seed_list(seed: int, count: int, *, offset: int = 0) -> List[int]:
    rng = np.random.default_rng(int(seed) + int(offset))
    if int(count) <= 0:
        return []
    seeds = rng.integers(0, 2**31 - 1, size=int(count), dtype=np.int64)
    return [int(value) for value in seeds.tolist()]


def generate_disjoint_calibration_seed_list(
    seed: int,
    count: int,
    *,
    offset: int = 0,
    exclude: Optional[Sequence[int]] = None,
) -> List[int]:
    if int(count) <= 0:
        return []
    rng = np.random.default_rng(int(seed) + int(offset))
    excluded = {int(value) for value in list(exclude or [])}
    output: List[int] = []
    while len(output) < int(count):
        candidate = int(rng.integers(0, 2**31 - 1, dtype=np.int64))
        if candidate in excluded:
            continue
        excluded.add(candidate)
        output.append(candidate)
    return output


def calibration_param_snapshot(
    cfg: BenchConfig,
    keys: Sequence[str] = CALIBRATION_HISTORY_PARAM_KEYS,
) -> Dict[str, float]:
    return {str(key): float(getattr(cfg, key)) for key in keys}


def prepare_calibration_candidate_cfg(
    base_cfg: BenchConfig,
    reference_targets: Dict[str, Any],
) -> BenchConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.session_start_hour_probs = list(reference_targets.get("session_start_histogram", []))
    return cfg


def evaluate_calibration_candidate(
    cfg: BenchConfig,
    catalog: np.ndarray,
    cards: Sequence[Dict[str, Any]],
    targets: Dict[str, Any],
    *,
    seeds: Sequence[int],
    selection_delta: Optional[float] = None,
    trial_id: int = -1,
    stage: str = "score",
    source_trial: Optional[int] = None,
    build_audit_row: bool = False,
    calibration_policy: str = DEFAULT_CALIBRATION_POLICY,
) -> Tuple[BenchConfig, Dict[str, Any], Dict[str, Any]]:
    delta_targets, resolved_delta, reference_targets = normalize_calibration_targets(
        targets,
        selection_delta=selection_delta,
    )
    trial_cfg = prepare_calibration_candidate_cfg(cfg, reference_targets)
    calibration_policy_context = resolve_calibration_rollout_policy_context(reference_targets, calibration_policy)
    sim = simulate_targets(
        trial_cfg,
        catalog,
        cards,
        seeds=list(map(int, seeds)),
        gap_bucket_edges=reference_targets.get("gap_bucket_edges"),
        calibration_policy=calibration_policy,
        calibration_policy_context=calibration_policy_context,
    )
    row: Dict[str, Any] = {
        "trial": int(trial_id),
        "stage": str(stage),
        "source_trial": None if source_trial is None else int(source_trial),
        "num_episodes": int(len(seeds)),
        "calibration_policy": str(normalize_calibration_policy_name(calibration_policy)),
        **calibration_param_snapshot(trial_cfg),
    }
    losses = {}
    for delta_value, delta_target in sorted(delta_targets.items()):
        loss_value = calibration_loss(delta_target, sim, weights=delta_target.get("calibration_weights"))
        losses[delta_value] = float(loss_value)
        row[f"loss_delta_{int(delta_value) if float(delta_value).is_integer() else delta_value}"] = float(loss_value)
    ref_components = calibration_loss_components(reference_targets, sim)
    ref_loss_breakdown = calibration_loss_breakdown(
        reference_targets,
        sim,
        weights=reference_targets.get("calibration_weights"),
        components=ref_components,
    )
    for key, value in ref_components.items():
        row[f"component_{key}"] = float(value)
    row["loss_primary"] = float(ref_loss_breakdown["primary"])
    row["loss_secondary"] = float(ref_loss_breakdown["secondary"])
    row["loss"] = float(ref_loss_breakdown["lexicographic_scalar"])
    row["loss_mean"] = float(np.mean(list(losses.values())))
    row["loss_max"] = float(max(losses.values()))
    row["selection_delta"] = float(resolved_delta)
    row["session_start_weight"] = float(reference_targets.get("calibration_weights", {}).get("diurnal", 1.0))
    if build_audit_row:
        audit = build_calibration_audit(delta_targets, sim, resolved_delta)
        apply_calibration_audit_to_row(row, audit)
    return trial_cfg, sim, row


def random_search_calibration(
    base_cfg: BenchConfig,
    catalog: np.ndarray,
    cards: Sequence[Dict[str, Any]],
    targets: Dict[str, Any],
    n_trials: int,
    episodes_per_trial: int,
    seed: int,
    selection_delta: Optional[float] = None,
    scoring_seeds: Optional[Sequence[int]] = None,
    calibration_policy: str = DEFAULT_CALIBRATION_POLICY,
) -> Tuple[BenchConfig, List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    delta_targets, resolved_delta, reference_targets = normalize_calibration_targets(
        targets,
        selection_delta=selection_delta,
    )
    best_cfg = prepare_calibration_candidate_cfg(base_cfg, reference_targets)
    best_loss_pair = (float("inf"), float("inf"))
    history: List[Dict[str, Any]] = []
    base_seeds = (
        list(map(int, scoring_seeds))
        if scoring_seeds is not None
        else list(range(seed, seed + episodes_per_trial))
    )
    _, _, base_row = evaluate_calibration_candidate(
        best_cfg,
        catalog,
        cards,
        delta_targets,
        seeds=base_seeds,
        selection_delta=resolved_delta,
        trial_id=-1,
        stage="baseline",
        calibration_policy=calibration_policy,
    )
    history.append(base_row)
    best_loss_pair = (float(base_row["loss_primary"]), float(base_row["loss_secondary"]))
    for trial in range(n_trials):
        cfg = prepare_calibration_candidate_cfg(base_cfg, reference_targets)
        for key in CALIBRATABLE_KEYS:
            val = getattr(cfg, key)
            if key in {"gap_mu"}:
                new_val = float(val + rng.normal(0.0, 0.2))
            elif key in {"gap_sigma"}:
                new_val = float(max(0.05, val + rng.normal(0.0, 0.1)))
            elif key.startswith("kappa_"):
                new_val = float(val + rng.normal(0.0, 0.3))
            else:
                new_val = float(max(1e-4, val * math.exp(rng.normal(0.0, 0.25))))
            setattr(cfg, key, new_val)
        trial_seeds = (
            list(map(int, scoring_seeds))
            if scoring_seeds is not None
            else list(range(seed + (trial + 1) * episodes_per_trial, seed + (trial + 2) * episodes_per_trial))
        )
        trial_cfg, _, row = evaluate_calibration_candidate(
            cfg,
            catalog,
            cards,
            delta_targets,
            seeds=trial_seeds,
            selection_delta=resolved_delta,
            trial_id=trial,
            stage="random_search",
            calibration_policy=calibration_policy,
        )
        history.append(row)
        candidate_loss_pair = (float(row["loss_primary"]), float(row["loss_secondary"]))
        if candidate_loss_pair < best_loss_pair:
            best_loss_pair = candidate_loss_pair
            best_cfg = trial_cfg
    return best_cfg, history


def sample_optuna_calibration_cfg(trial: Any, base_cfg: BenchConfig) -> BenchConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.scale = float(trial.suggest_float("scale", 0.1, 20.0, log=True))
    cfg.continue_logit_temp = float(trial.suggest_float("continue_logit_temp", 0.5, 20.0, log=True))
    cfg.continue_logit_bias = float(trial.suggest_float("continue_logit_bias", -4.0, 4.0))
    cfg.gap_sigma = float(trial.suggest_float("gap_sigma", 0.05, 5.0, log=True))
    for key in OPTUNA_WIDE_MULTIPLICATIVE_KEYS:
        base_value = max(1e-4, float(getattr(base_cfg, key)))
        low = max(1e-4, base_value * math.exp(-2.0))
        high = max(low * 1.0001, base_value * math.exp(2.0))
        setattr(cfg, key, float(trial.suggest_float(key, low, high, log=True)))
    cfg.gap_mu = float(trial.suggest_float("gap_mu", 0.0, 8.0))
    cfg.kappa_0 = float(trial.suggest_float("kappa_0", -8.0, 8.0))
    cfg.kappa_h = float(trial.suggest_float("kappa_h", -8.0, 8.0))
    cfg.kappa_c = float(trial.suggest_float("kappa_c", -8.0, 8.0))
    return cfg


def trial_params_to_cfg(base_cfg: BenchConfig, params: Dict[str, Any]) -> BenchConfig:
    cfg = copy.deepcopy(base_cfg)
    for key, value in params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, float(value))
    return cfg


def sort_calibration_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        list(rows),
        key=lambda row: (
            float(row.get("loss_primary", float("inf"))),
            float(row.get("loss_secondary", float("inf"))),
            float(row.get("loss", float("inf"))),
            int(row.get("trial", 10**9)),
        ),
    )


def calibration_row_acceptance_passed(row: Dict[str, Any]) -> bool:
    value = row.get("acceptance_passed", False)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def calibration_acceptance_rank_key(row: Dict[str, Any]) -> Tuple[int, float, float, float, float, int]:
    default_failed_checks = 0 if calibration_row_acceptance_passed(row) else 1
    return (
        int(row.get("acceptance_failed_checks", default_failed_checks)),
        float(row.get("acceptance_worst_margin_violation", 0.0 if default_failed_checks == 0 else 1.0)),
        float(row.get("loss_primary", float("inf"))),
        float(row.get("loss_secondary", float("inf"))),
        float(row.get("loss", float("inf"))),
        int(row.get("trial", 10**9)),
    )


def sort_calibration_rows_by_acceptance(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(list(rows), key=calibration_acceptance_rank_key)


def _selection_reason_differs(lhs: float, rhs: float, *, tol: float = 1e-12) -> bool:
    if not math.isfinite(lhs) or not math.isfinite(rhs):
        return lhs != rhs
    return not math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=tol)


def build_final_calibration_selection_reason(
    row: Dict[str, Any],
    selected_row: Dict[str, Any],
    selected_reason: str,
) -> str:
    if int(row.get("trial", -1)) == int(selected_row.get("trial", -2)):
        return str(selected_reason)
    row_failed = int(row.get("acceptance_failed_checks", 10**9))
    selected_failed = int(selected_row.get("acceptance_failed_checks", 10**9))
    if row_failed != selected_failed:
        return "more_failed_checks_than_selected"
    row_margin_violation = float(row.get("acceptance_worst_margin_violation", float("inf")))
    selected_margin_violation = float(selected_row.get("acceptance_worst_margin_violation", float("inf")))
    if _selection_reason_differs(row_margin_violation, selected_margin_violation):
        return "worse_acceptance_margin_than_selected"
    row_loss_primary = float(row.get("loss_primary", float("inf")))
    selected_loss_primary = float(selected_row.get("loss_primary", float("inf")))
    if _selection_reason_differs(row_loss_primary, selected_loss_primary):
        return "higher_primary_loss_than_selected"
    row_loss_secondary = float(row.get("loss_secondary", float("inf")))
    selected_loss_secondary = float(selected_row.get("loss_secondary", float("inf")))
    if _selection_reason_differs(row_loss_secondary, selected_loss_secondary):
        return "higher_secondary_loss_than_selected"
    row_loss = float(row.get("loss", float("inf")))
    selected_loss = float(selected_row.get("loss", float("inf")))
    if _selection_reason_differs(row_loss, selected_loss):
        return "higher_scalar_loss_than_selected"
    return "higher_trial_id_than_selected"


def build_calibration_selection_table_dataframe(final_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    ordered_rows = sort_calibration_rows_by_acceptance(final_rows)
    if not ordered_rows:
        return pd.DataFrame(
            columns=[
                "selection_rank",
                "trial",
                "status",
                "acceptance_passed",
                "failed_checks",
                "failed_check_names",
                "worst_margin",
                "worst_margin_violation",
                "loss_primary",
                "loss_secondary",
                "loss",
                "selection_reason",
                "selected_final_winner",
            ]
        )
    selected_row = ordered_rows[0]
    selected_reason = str(selected_row.get("selected_reason", ""))
    selection_rows: List[Dict[str, Any]] = []
    for rank, row in enumerate(ordered_rows, start=1):
        selection_rows.append(
            {
                "selection_rank": int(rank),
                "trial": int(row.get("trial", -1)),
                "status": str(row.get("status", "")),
                "acceptance_passed": bool(row.get("acceptance_passed", False)),
                "failed_checks": int(row.get("acceptance_failed_checks", 0)),
                "failed_check_names": str(row.get("acceptance_failed_check_names", "")),
                "worst_margin": float(row.get("acceptance_worst_margin", float("inf"))),
                "worst_margin_violation": float(row.get("acceptance_worst_margin_violation", float("inf"))),
                "loss_primary": float(row.get("loss_primary", float("inf"))),
                "loss_secondary": float(row.get("loss_secondary", float("inf"))),
                "loss": float(row.get("loss", float("inf"))),
                "selection_reason": build_final_calibration_selection_reason(row, selected_row, selected_reason),
                "selected_final_winner": bool(row.get("selected_final_winner", False)),
            }
        )
    return pd.DataFrame(selection_rows)


def build_calibration_policy_comparison_table_dataframe(payloads: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        manifest = payload.get("manifest", {})
        audit = payload.get("audit", {})
        components = audit.get("calibration_loss_components", {})
        rows.append(
            {
                "calibration_policy": str(manifest.get("calibration_policy", payload.get("calibration_policy", ""))),
                "status": str(manifest.get("status", audit.get("status", ""))),
                "selected_trial_number": manifest.get("selected_trial_number"),
                "selected_reason": manifest.get("selected_reason"),
                "session_fit": float(components.get("session", 0.0)),
                "item_count_fit": float(components.get("session_item_counts", 0.0)),
                "gap_fit": float(components.get("gaps", 0.0)),
                "conditional_return_fit": float(components.get("return_conditional", 0.0)),
                "loss_primary": float(audit.get("calibration_loss_primary", payload.get("calibration_loss_primary", 0.0))),
                "loss_secondary": float(audit.get("calibration_loss_secondary", payload.get("calibration_loss_secondary", 0.0))),
                "calibration_loss": float(audit.get("calibration_loss", payload.get("calibration_loss", 0.0))),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "calibration_policy",
                "status",
                "selected_trial_number",
                "selected_reason",
                "session_fit",
                "item_count_fit",
                "gap_fit",
                "conditional_return_fit",
                "loss_primary",
                "loss_secondary",
                "calibration_loss",
            ]
        )
    return pd.DataFrame(rows).sort_values(["calibration_policy"]).reset_index(drop=True)


def select_final_calibration_row(final_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not final_rows:
        raise ValueError("No final calibration rows are available for selection.")
    ranked_rows = sort_calibration_rows_by_acceptance(final_rows)
    selected_row = ranked_rows[0]
    if calibration_row_acceptance_passed(selected_row):
        return {
            "selected_row": selected_row,
            "selected_reason": "accepted_lowest_loss",
            "selection_warning": None,
            "ranked_rows": ranked_rows,
        }
    selection_warning = (
        "No final_reeval candidate passed acceptance; selecting the finalist with the fewest failed checks, "
        "best acceptance margin, and lowest primary/secondary/scalar loss."
    )
    LOGGER.warning("%s | finalists=%s", selection_warning, len(final_rows))
    return {
        "selected_row": selected_row,
        "selected_reason": "no_accepted_candidate_fallback",
        "selection_warning": selection_warning,
        "ranked_rows": ranked_rows,
    }


def calibrate_with_optuna(
    base_cfg: BenchConfig,
    catalog: np.ndarray,
    cards: Sequence[Dict[str, Any]],
    targets: Dict[str, Any],
    *,
    selection_delta: Optional[float] = None,
    seed: int = 0,
    fixed_seed_list: Optional[Sequence[int]] = None,
    exploratory_trials: int = 300,
    exploratory_episodes: int = 200,
    topk_trials: int = 20,
    topk_episodes: int = 1000,
    finalists: int = 10,
    final_episodes: int = 5000,
    storage_path: Optional[Path] = None,
    study_name: Optional[str] = None,
    initial_trial_params: Optional[Sequence[Dict[str, float]]] = None,
    calibration_policy: str = DEFAULT_CALIBRATION_POLICY,
) -> Dict[str, Any]:
    optuna = require_optuna()
    calibration_policy = normalize_calibration_policy_name(calibration_policy)
    delta_targets, resolved_delta, reference_targets = normalize_calibration_targets(
        targets,
        selection_delta=selection_delta,
    )
    exploratory_seed_list = (
        [int(value) for value in fixed_seed_list]
        if fixed_seed_list is not None
        else generate_disjoint_calibration_seed_list(seed, exploratory_episodes, offset=11)
    )
    topk_seed_list = generate_disjoint_calibration_seed_list(
        seed,
        topk_episodes,
        offset=29,
        exclude=exploratory_seed_list,
    )
    final_seed_list = generate_disjoint_calibration_seed_list(
        seed,
        final_episodes,
        offset=53,
        exclude=list(exploratory_seed_list) + list(topk_seed_list),
    )
    confirmation_seed_list = generate_disjoint_calibration_seed_list(
        seed,
        final_episodes,
        offset=79,
        exclude=list(exploratory_seed_list) + list(topk_seed_list) + list(final_seed_list),
    )
    baseline_cfg = prepare_calibration_candidate_cfg(base_cfg, reference_targets)
    _, _, baseline_row = evaluate_calibration_candidate(
        baseline_cfg,
        catalog,
        cards,
        delta_targets,
        seeds=exploratory_seed_list,
        selection_delta=resolved_delta,
        trial_id=-1,
        stage="baseline",
        calibration_policy=calibration_policy,
    )
    history_rows: List[Dict[str, Any]] = [baseline_row]
    storage_uri = None
    if storage_path is not None:
        storage_uri = f"sqlite:///{Path(storage_path).resolve()}"
    sampler = optuna.samplers.TPESampler(seed=int(seed), multivariate=True)
    actual_study_name = study_name or f"calibration_optuna_seed_{int(seed)}_{int(time.time() * 1000)}"
    study = optuna.create_study(
        study_name=actual_study_name,
        direction="minimize",
        sampler=sampler,
        storage=storage_uri,
        load_if_exists=False,
    )
    for params in initial_trial_params or []:
        study.enqueue_trial({str(key): float(value) for key, value in params.items()})

    def objective(trial: Any) -> float:
        cfg = sample_optuna_calibration_cfg(trial, base_cfg)
        _, _, row = evaluate_calibration_candidate(
            cfg,
            catalog,
            cards,
            delta_targets,
            seeds=exploratory_seed_list,
            selection_delta=resolved_delta,
            trial_id=int(trial.number),
            stage="explore",
            calibration_policy=calibration_policy,
        )
        history_rows.append(row)
        for key, value in row.items():
            if key.startswith("component_") or key in {
                "loss_primary",
                "loss_secondary",
                "loss",
                "loss_mean",
                "loss_max",
            }:
                trial.set_user_attr(key, value)
        return float(row["loss"])

    study.optimize(objective, n_trials=int(exploratory_trials), gc_after_trial=False, show_progress_bar=False)

    exploratory_rows = [row for row in history_rows if row.get("stage") == "explore"]
    top_exploratory_rows = sort_calibration_rows(exploratory_rows)[: max(1, int(topk_trials))]
    top_trial_rows: List[Dict[str, Any]] = []
    reevaluated_top_rows: List[Dict[str, Any]] = []
    for rank, row in enumerate(top_exploratory_rows, start=1):
        cfg = trial_params_to_cfg(base_cfg, {key: row[key] for key in CALIBRATION_HISTORY_PARAM_KEYS if key in row})
        _, _, reevaluated = evaluate_calibration_candidate(
            cfg,
            catalog,
            cards,
            delta_targets,
            seeds=topk_seed_list,
            selection_delta=resolved_delta,
            trial_id=int(row["trial"]),
            stage="top20_reeval",
            source_trial=int(row["trial"]),
            build_audit_row=True,
            calibration_policy=calibration_policy,
        )
        reevaluated["rank_within_stage"] = int(rank)
        reevaluated_top_rows.append(reevaluated)
        top_trial_rows.append(reevaluated)

    requested_finalists = max(10, int(finalists))
    finalist_rows = sort_calibration_rows_by_acceptance(reevaluated_top_rows)[: max(1, requested_finalists)]
    for promotion_rank, row in enumerate(sort_calibration_rows_by_acceptance(reevaluated_top_rows), start=1):
        row["promotion_rank_within_stage"] = int(promotion_rank)
        row["promoted_to_final_reeval"] = bool(promotion_rank <= len(finalist_rows))
    final_rows: List[Dict[str, Any]] = []
    final_cfgs: Dict[int, BenchConfig] = {}
    final_sims: Dict[int, Dict[str, Any]] = {}
    for rank, row in enumerate(finalist_rows, start=1):
        cfg = trial_params_to_cfg(base_cfg, {key: row[key] for key in CALIBRATION_HISTORY_PARAM_KEYS if key in row})
        final_cfg, final_sim, reevaluated = evaluate_calibration_candidate(
            cfg,
            catalog,
            cards,
            delta_targets,
            seeds=final_seed_list,
            selection_delta=resolved_delta,
            trial_id=int(row["trial"]),
            stage="final_reeval",
            source_trial=int(row["trial"]),
            build_audit_row=True,
            calibration_policy=calibration_policy,
        )
        reevaluated["rank_within_stage"] = int(rank)
        final_rows.append(reevaluated)
        top_trial_rows.append(reevaluated)
        final_cfgs[int(row["trial"])] = final_cfg
        final_sims[int(row["trial"])] = final_sim

    final_selection = select_final_calibration_row(final_rows)
    selected_final_row = final_selection["selected_row"]
    selected_trial_number = int(selected_final_row["trial"])
    selected_cfg = final_cfgs[selected_trial_number]
    selected_sim = final_sims[selected_trial_number]
    selected_audit = build_calibration_audit(delta_targets, selected_sim, resolved_delta)
    _, confirmation_sim, confirmation_row = evaluate_calibration_candidate(
        selected_cfg,
        catalog,
        cards,
        delta_targets,
        seeds=confirmation_seed_list,
        selection_delta=resolved_delta,
        trial_id=selected_trial_number,
        stage="confirmation",
        source_trial=selected_trial_number,
        build_audit_row=True,
        calibration_policy=calibration_policy,
    )
    confirmation_audit = build_calibration_audit(delta_targets, confirmation_sim, resolved_delta)
    confirmation_manifest = build_calibration_confirmation_manifest(
        selected_final_row,
        confirmation_audit,
        confirmation_seed_list,
        selected_trial_number=selected_trial_number,
        selected_reason=str(final_selection["selected_reason"]),
        selected_delta_sess=float(resolved_delta),
        calibration_policy=calibration_policy,
    )
    ranked_final_rows = final_selection["ranked_rows"]
    for row in top_trial_rows:
        row["selected_final_winner"] = bool(
            row.get("stage") == "final_reeval" and int(row.get("trial", -1)) == selected_trial_number
        )
    for selection_rank, row in enumerate(ranked_final_rows, start=1):
        row["selection_rank_within_stage"] = int(selection_rank)
        row["selected_reason"] = build_final_calibration_selection_reason(
            row,
            selected_final_row,
            str(final_selection["selected_reason"]),
        )

    return {
        "study": study,
        "study_name": actual_study_name,
        "storage_path": None if storage_path is None else Path(storage_path),
        "fixed_seed_list": list(exploratory_seed_list),
        "topk_seed_list": list(topk_seed_list),
        "final_seed_list": list(final_seed_list),
        "confirmation_seed_list": list(confirmation_seed_list),
        "history_rows": history_rows,
        "top_trial_rows": top_trial_rows,
        "best_cfg": selected_cfg,
        "best_sim": selected_sim,
        "best_audit": selected_audit,
        "confirmation_sim": confirmation_sim,
        "confirmation_row": confirmation_row,
        "confirmation_audit": confirmation_audit,
        "confirmation_manifest": confirmation_manifest,
        "final_rows": [copy.deepcopy(row) for row in ranked_final_rows],
        "finalist_count": int(len(ranked_final_rows)),
        "requested_finalists": int(requested_finalists),
        "selected_trial_number": selected_trial_number,
        "selected_final_row": copy.deepcopy(selected_final_row),
        "selected_reason": str(final_selection["selected_reason"]),
        "selection_warning": final_selection["selection_warning"],
        "confirmation_status": str(confirmation_manifest["confirmation_status"]),
        "baseline_row": baseline_row,
        "reference_targets": reference_targets,
        "calibration_policy": str(calibration_policy),
    }


def fit_llm_fusion(
    cfg: BenchConfig,
    catalog: np.ndarray,
    cards: Sequence[Dict[str, Any]],
    scorer: AGLLMScorer,
    context_seeds: Sequence[int],
    target_loss_seeds: Sequence[int],
    ppo_policy: Optional[Any] = None,
    ppo_deterministic: bool = True,
) -> Tuple[BenchConfig, Dict[str, Any]]:
    """Moment-match score maps and optionally refine with a small optimizer on the aggregate target loss."""
    # Build moment-matching pool from both random and PPO rollouts in AGParam.
    watch_base_logits, watch_raw_logits = [], []
    cont_base_scores, cont_raw_logits = [], []
    context_policy_specs: List[Tuple[str, Any, bool]] = [
        ("random", RandomPolicy(cfg.num_actions(), seed=int(context_seeds[0]) if context_seeds else 0), False),
    ]
    if ppo_policy is not None:
        context_policy_specs.append(("ppo", ppo_policy, bool(ppo_deterministic)))
    else:
        LOGGER.warning("AGLLM fusion fit did not receive a PPO context policy; context pool will use random rollouts only.")

    context_rollout_counts = {"watch_queries": 0, "continue_queries": 0}
    for policy_name, policy_obj, deterministic in context_policy_specs:
        for seed in context_seeds:
            policy_instance = policy_obj if policy_name != "random" else RandomPolicy(cfg.num_actions(), seed=int(seed))
            env = make_env(cfg, "param", catalog, cards, scorer=None, wrappers=None, seed=seed)
            obs = env.reset(seed=seed)
            done = False
            while not done:
                s = copy.deepcopy(env.state)
                assert s is not None
                action, _ = policy_act_with_info(
                    policy_instance,
                    obs,
                    deterministic=deterministic,
                    env=env,
                    need_info=False,
                )
                z, lam_idx, _ = action_id_to_tuple(int(action), cfg.P)
                tilde_x = env._current_tilde_x(z, lam_idx)
                rep_count = int(sum(1 for zz in s.hist_z if zz == z))
                nov = math.exp(-cfg.lambda_rep * rep_count)
                m_base = cfg.alpha * float(np.dot(s.user.u, tilde_x)) + cfg.beta * s.user.n * nov + cfg.gamma_h * s.h - cfg.gamma_f * s.f
                wall = env._wall_clock(s.tau, s.tau_start)
                last_watch = float(s.hist_r[-1]) if s.hist_r else 0.0
                g_watch = scorer.watch_score(s.user, z, lam_idx, wall, s.l, s.j, s.h, s.c, s.f, s.hist_z, last_watch, s.g, {})
                watch_base_logits.append(m_base)
                watch_raw_logits.append(logit(g_watch, cfg.llm_eps))
                context_rollout_counts["watch_queries"] += 1

                obs, reward, done, _ = env.step(action)
                post = compute_post_consumption_state(cfg, s, reward)
                g_cont = scorer.continue_score(
                    s.user,
                    z,
                    lam_idx,
                    float(post["bar_tau_plus"]),
                    float(post["l_plus"]),
                    int(post["j_plus"]),
                    float(post["h_plus"]),
                    float(post["c_plus"]),
                    float(post["f_plus"]),
                    float(post["hat_r_plus"]),
                    float(reward),
                    s.hist_z,
                    {},
                )
                cont_base_scores.append(float(post["q_base"]))
                cont_raw_logits.append(logit(g_cont, cfg.llm_eps))
                context_rollout_counts["continue_queries"] += 1

    cfg2 = copy.deepcopy(cfg)
    wr = np.asarray(watch_raw_logits, dtype=np.float64)
    wb = np.asarray(watch_base_logits, dtype=np.float64)
    cr = np.asarray(cont_raw_logits, dtype=np.float64)
    cb = np.asarray(cont_base_scores, dtype=np.float64)

    cfg2.b_r = float(np.std(wb) / max(1e-8, np.std(wr)))
    cfg2.a_r = float(np.mean(wb) - cfg2.b_r * np.mean(wr))
    cfg2.b_c = float(np.std(cb) / max(1e-8, np.std(cr)))
    cfg2.a_c = float(np.mean(cb) - cfg2.b_c * np.mean(cr))

    targets = simulate_targets(cfg, catalog, cards, target_loss_seeds, backend="param", scorer=None)

    def objective(x: np.ndarray) -> float:
        a_r, log_b_r, a_c, log_b_c, w_r, w_c = map(float, x)
        trial = copy.deepcopy(cfg2)
        trial.a_r = a_r
        trial.b_r = float(softplus(log_b_r) + 1e-4)
        trial.a_c = a_c
        trial.b_c = float(softplus(log_b_c) + 1e-4)
        trial.omega_r_llm = clip_scalar(w_r, 0.0, 1.0)
        trial.omega_c_llm = clip_scalar(w_c, 0.0, 1.0)
        scorer.cfg = trial
        sim = simulate_targets(
            trial,
            catalog,
            cards,
            target_loss_seeds,
            backend="llm",
            scorer=scorer,
            gap_bucket_edges=targets.get("gap_bucket_edges"),
        )
        return calibration_loss(targets, sim) + trial.llm_lambda_penalty * (trial.omega_r_llm**2 + trial.omega_c_llm**2)

    x0 = np.asarray([cfg2.a_r, math.log(math.expm1(max(cfg2.b_r, 1e-4))), cfg2.a_c, math.log(math.expm1(max(cfg2.b_c, 1e-4))), 0.1, 0.1], dtype=np.float64)
    try:
        res = minimize(objective, x0, method="L-BFGS-B")
        x = res.x
        cfg2.a_r = float(x[0])
        cfg2.b_r = float(softplus(x[1]) + 1e-4)
        cfg2.a_c = float(x[2])
        cfg2.b_c = float(softplus(x[3]) + 1e-4)
        cfg2.omega_r_llm = clip_scalar(float(x[4]), 0.0, 1.0)
        cfg2.omega_c_llm = clip_scalar(float(x[5]), 0.0, 1.0)
        history = {
            "optimizer_success": bool(res.success),
            "objective": float(res.fun),
            "message": str(res.message),
        }
    except Exception as exc:
        history = {"optimizer_success": False, "objective": float("nan"), "message": str(exc)}

    scorer.cfg = cfg2
    history.update(
        {
            "context_policies": [policy_name for policy_name, _, _ in context_policy_specs],
            "context_rollout_counts": context_rollout_counts,
            "heldout_target_seed_count": int(len(target_loss_seeds)),
            "heldout_target_seeds": list(map(int, target_loss_seeds)),
            "context_seeds": list(map(int, context_seeds)),
        }
    )
    return cfg2, history


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def save_histogram(data_a: Sequence[float], data_b: Optional[Sequence[float]], xlabel: str, title: str, outpath: Path, label_a: str = "Target", label_b: str = "Simulated") -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(list(data_a), bins=30, alpha=0.6, label=label_a)
    if data_b is not None:
        plt.hist(list(data_b), bins=30, alpha=0.6, label=label_b)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    log_artifact_written(outpath, "figure")


def empirical_cdf_points(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    arr = np.sort(arr)
    probs = np.arange(1, arr.size + 1, dtype=np.float64) / float(arr.size)
    return arr, probs


def save_cdf_plot(
    data_a: Sequence[float],
    data_b: Sequence[float],
    xlabel: str,
    title: str,
    outpath: Path,
    label_a: str = "Target",
    label_b: str = "Simulator",
) -> None:
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    any_series = False
    for values, label in ((data_a, label_a), (data_b, label_b)):
        xs, ys = empirical_cdf_points(values)
        if xs.size == 0:
            continue
        ax.step(xs, ys, where="post", label=label)
        any_series = True
    if not any_series:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    if any_series:
        ax.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    log_artifact_written(outpath, "figure")


def save_lineplot(xs: Sequence[float], ys_a: Sequence[float], ys_b: Optional[Sequence[float]], xlabel: str, ylabel: str, title: str, outpath: Path, label_a: str = "Target", label_b: str = "Simulated") -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys_a, label=label_a)
    if ys_b is not None:
        plt.plot(xs, ys_b, label=label_b)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    log_artifact_written(outpath, "figure")


def save_stop_hazard_support_plot(
    target_stop_hazard: Sequence[float],
    sim_stop_hazard: Sequence[float],
    outpath: Path,
) -> None:
    target = list(map(float, target_stop_hazard))
    sim = list(map(float, sim_stop_hazard))
    support = build_stop_hazard_support_diagnostics(target, sim)
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    if target:
        ax.plot(list(range(1, len(target) + 1)), target, label=f"Target (len={len(target)})")
    if sim:
        ax.plot(list(range(1, len(sim) + 1)), sim, label=f"Simulator (len={len(sim)})")
    if not target and not sim:
        ax.text(0.5, 0.5, "No stop-hazard data", ha="center", va="center", transform=ax.transAxes)
    if support["truncated"] and support["overlap_len"] > 0:
        ax.axvline(float(support["overlap_len"]), color="gray", linestyle="--", linewidth=1.0)
        ax.text(
            0.02,
            0.98,
            f"Overlap-only comparison truncates at position {support['overlap_len']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    ax.set_xlabel("Within-session item position")
    ax.set_ylabel("Stop hazard")
    ax.set_title(
        f"Stop-hazard support (target={support['target_len']}, simulator={support['sim_len']})"
    )
    if target or sim:
        ax.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    log_artifact_written(outpath, "figure")


def save_scatter(x: Sequence[float], y: Sequence[float], labels: Sequence[str], xlabel: str, ylabel: str, title: str, outpath: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y)
    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label, (xi, yi))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    log_artifact_written(outpath, "figure")


def save_barplot(labels: Sequence[str], values: Sequence[float], xlabel: str, ylabel: str, title: str, outpath: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(values)), list(values))
    plt.xticks(range(len(values)), list(labels), rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    log_artifact_written(outpath, "figure")


def save_action_marginals_figure(metrics: Dict[str, Any], title: str, outpath: Path) -> None:
    z_dict = metrics.get("MarginalZ", {})
    lambda_dict = metrics.get("MarginalLambdaIdx", {})
    nu_dict = metrics.get("MarginalNu", {})
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    panels = [
        (axes[0], list(z_dict.keys()), list(z_dict.values()), "z", "Mass"),
        (axes[1], list(lambda_dict.keys()), list(lambda_dict.values()), "lambda_idx", "Mass"),
        (axes[2], list(nu_dict.keys()), list(nu_dict.values()), "nu", "Mass"),
    ]
    for ax, labels, values, xlabel, ylabel in panels:
        ax.bar(range(len(values)), list(values))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(list(labels), rotation=90 if len(labels) > 8 else 0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    log_artifact_written(outpath, "figure")


def parse_cap_value_from_policy_name(policy_name: str) -> float:
    match = re.search(r"PPO\+Cap\(([-+0-9.]+)\)", str(policy_name))
    if match is None:
        return float("nan")
    try:
        return float(match.group(1))
    except ValueError:
        return float("nan")


def cap_fragmentation_policy_sort_key(policy_name: str) -> Tuple[int, float, str]:
    label = str(policy_name)
    if label in CAP_FRAGMENTATION_POLICY_ORDER:
        return (0, float(CAP_FRAGMENTATION_POLICY_ORDER.index(label)), label)
    cap_value = parse_cap_value_from_policy_name(label)
    if math.isfinite(cap_value):
        return (1, float(cap_value), label)
    return (2, float("inf"), label)


def build_cap_fragmentation_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "policy",
                "T_cap",
                "CumWatch",
                "CumWatch_ci95",
                "OverCapMinutes",
                "OverCapMinutes_ci95",
                "CVaR_0.95(L)",
                "CVaR_0.95(L)_ci95",
                "SessionsPerEpisode",
                "SessionsPerEpisode_ci95",
                "FractionReturnsWithin1Min",
                "FractionReturnsWithin1Min_ci95",
                "FractionReturnsWithin5Min",
                "FractionReturnsWithin5Min_ci95",
                "LateNightSessionStartRate",
                "LateNightSessionStartRate_ci95",
            ]
        )
    df = aggregate_across_train_seeds(records, ["policy"])
    df["T_cap"] = df["policy"].map(parse_cap_value_from_policy_name)
    ordered_rows = sorted(df.to_dict(orient="records"), key=lambda row: cap_fragmentation_policy_sort_key(str(row["policy"])))
    ordered_df = pd.DataFrame(ordered_rows)
    ordered_df = ordered_df[
        [
            "policy",
            "T_cap",
            "CumWatch",
            "CumWatch_ci95",
            "OverCapMinutes",
            "OverCapMinutes_ci95",
            "CVaR_0.95(L)",
            "CVaR_0.95(L)_ci95",
            "SessionsPerEpisode",
            "SessionsPerEpisode_ci95",
            "FractionReturnsWithin1Min",
            "FractionReturnsWithin1Min_ci95",
            "FractionReturnsWithin5Min",
            "FractionReturnsWithin5Min_ci95",
            "LateNightSessionStartRate",
            "LateNightSessionStartRate_ci95",
        ]
    ]
    return ordered_df.reset_index(drop=True)


def save_cap_fragmentation_figure(fragmentation_df: pd.DataFrame, outpath: Path) -> None:
    if fragmentation_df.empty:
        raise ValueError("Cannot plot cap fragmentation figure from an empty table.")
    block = fragmentation_df.copy().reset_index(drop=True)
    labels = [
        "PPO" if str(policy) == "PPO" else f"Cap {int(round(float(t_cap)))}"
        for policy, t_cap in zip(block["policy"], block["T_cap"])
    ]
    xs = np.arange(len(block), dtype=np.float64)

    def _series(column: str) -> np.ndarray:
        return block[column].astype(float).to_numpy(dtype=np.float64)

    def _err(column: str) -> np.ndarray:
        ci_col = f"{column}_ci95"
        if ci_col not in block.columns:
            return np.zeros(len(block), dtype=np.float64)
        return block[ci_col].astype(float).fillna(0.0).to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_watch = axes[0, 0]
    ax_watch.bar(xs, _series("CumWatch"), color="#4C78A8", alpha=0.85)
    ax_watch.errorbar(xs, _series("CumWatch"), yerr=_err("CumWatch"), fmt="none", ecolor="black", capsize=3)
    ax_watch.set_ylabel("CumWatch")
    ax_watch.set_title("Watch stays flat while the cap target improves")
    ax_watch_twin = ax_watch.twinx()
    ax_watch_twin.plot(xs, _series("OverCapMinutes"), color="#F58518", marker="o", linewidth=2)
    ax_watch_twin.errorbar(xs, _series("OverCapMinutes"), yerr=_err("OverCapMinutes"), fmt="none", ecolor="#F58518", capsize=3)
    ax_watch_twin.set_ylabel("OverCapMinutes")

    ax_cvar = axes[0, 1]
    ax_cvar.plot(xs, _series("CVaR_0.95(L)"), color="#54A24B", marker="o", linewidth=2)
    ax_cvar.errorbar(xs, _series("CVaR_0.95(L)"), yerr=_err("CVaR_0.95(L)"), fmt="none", ecolor="#54A24B", capsize=3)
    ax_cvar.set_ylabel("CVaR_0.95(L)")
    ax_cvar.set_title("Tail session length")

    ax_sessions = axes[1, 0]
    ax_sessions.plot(xs, _series("SessionsPerEpisode"), color="#E45756", marker="o", linewidth=2)
    ax_sessions.errorbar(xs, _series("SessionsPerEpisode"), yerr=_err("SessionsPerEpisode"), fmt="none", ecolor="#E45756", capsize=3)
    ax_sessions.set_ylabel("Sessions / episode")
    ax_sessions.set_title("Session count per episode")

    ax_gaps = axes[1, 1]
    ax_gaps.plot(xs, _series("FractionReturnsWithin1Min"), color="#B279A2", marker="o", linewidth=2, label="<= 1 min")
    ax_gaps.plot(xs, _series("FractionReturnsWithin5Min"), color="#72B7B2", marker="o", linewidth=2, label="<= 5 min")
    ax_gaps.errorbar(xs, _series("FractionReturnsWithin1Min"), yerr=_err("FractionReturnsWithin1Min"), fmt="none", ecolor="#B279A2", capsize=3)
    ax_gaps.errorbar(xs, _series("FractionReturnsWithin5Min"), yerr=_err("FractionReturnsWithin5Min"), fmt="none", ecolor="#72B7B2", capsize=3)
    ax_gaps.set_ylabel("Fraction of returns")
    ax_gaps.set_title("Short-gap re-entry")
    ax_gaps.legend()

    for ax in axes.reshape(-1):
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.75)

    fig.suptitle("Session-cap fragmentation diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    log_artifact_written(outpath, "figure")


def parse_calibration_gap_audit_markdown(markdown_text: str) -> Dict[str, float | str]:
    parsed: Dict[str, float | str] = {}
    status_match = re.search(r"- Status:\s+\*\*([A-Z]+)\*\*", markdown_text)
    if status_match is not None:
        parsed["status"] = str(status_match.group(1))
    if "## Gaps" not in markdown_text:
        return parsed
    gap_section = markdown_text.split("## Gaps", 1)[1]
    if "## " in gap_section:
        gap_section = gap_section.split("## ", 1)[0]
    for line in gap_section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("| `"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) < 4:
            continue
        metric = parts[0].strip("`")
        try:
            parsed[f"{metric}_target"] = float(parts[1])
            parsed[f"{metric}_simulator"] = float(parts[2])
            parsed[f"{metric}_relative_error"] = float(parts[3])
        except ValueError:
            continue
    return parsed


def render_cap_fragmentation_memo(
    fragmentation_df: pd.DataFrame,
    *,
    calibration_audit_markdown: Optional[str] = None,
) -> str:
    if fragmentation_df.empty:
        return "No cap-fragmentation rows were available."
    block = fragmentation_df.copy()
    baseline_block = block[block["policy"] == "PPO"]
    if baseline_block.empty:
        return "PPO baseline row is missing from the cap-fragmentation table."
    baseline = baseline_block.iloc[0]
    cap_rows = block[block["policy"] != "PPO"].copy()
    if cap_rows.empty:
        return "No session-cap rows were available for the fragmentation comparison."
    preferred_cap = cap_rows[cap_rows["policy"] == "PPO+Cap(120)"]
    cap_focus = preferred_cap.iloc[0] if not preferred_cap.empty else cap_rows.iloc[0]
    strongest_cap = cap_rows.sort_values("OverCapMinutes", ascending=True).iloc[0]
    max_watch_shift_pct = float(
        max(
            abs(float(row["CumWatch"]) - float(baseline["CumWatch"])) / max(1e-8, abs(float(baseline["CumWatch"]))) * 100.0
            for _, row in cap_rows.iterrows()
        )
    )
    watch_direction = "down" if float(cap_focus["CumWatch"]) < float(baseline["CumWatch"]) else "up"
    late_night_direction = "up" if float(cap_focus["LateNightSessionStartRate"]) > float(baseline["LateNightSessionStartRate"]) else "down"
    short_gap_saturated = float(baseline["FractionReturnsWithin1Min"]) >= 0.95
    sentences = [
        (
            f"Across the saved deterministic PPO rerun, the session cap does not materially reduce daily watch: "
            f"`CumWatch` stays within {max_watch_shift_pct:.2f}% of PPO across `T_cap=90/120/150`, and at the default "
            f"`T_cap={int(round(float(cap_focus['T_cap'])))}"
            f"` it moves {watch_direction} only from "
            f"{float(baseline['CumWatch']):.1f} to {float(cap_focus['CumWatch']):.1f}."
        ),
        (
            f"The main effect is episode restructuring: `OverCapMinutes` falls from {float(baseline['OverCapMinutes']):.2f} "
            f"to {float(cap_focus['OverCapMinutes']):.2f} while `CVaR_0.95(L)` drops from {float(baseline['CVaR_0.95(L)']):.2f} "
            f"to {float(cap_focus['CVaR_0.95(L)']):.2f}, but sessions per episode rise from {float(baseline['SessionsPerEpisode']):.2f} "
            f"to {float(cap_focus['SessionsPerEpisode']):.2f} and the fraction of returns within 1 / 5 minutes shifts from "
            f"{float(baseline['FractionReturnsWithin1Min']):.3f} / {float(baseline['FractionReturnsWithin5Min']):.3f} to "
            f"{float(cap_focus['FractionReturnsWithin1Min']):.3f} / {float(cap_focus['FractionReturnsWithin5Min']):.3f}."
        ),
        (
            f"The cap therefore looks more like long-session truncation than a clean mitigation of total use: the strongest cap point "
            f"(`T_cap={int(round(float(strongest_cap['T_cap'])))}"
            f"`) drives `OverCapMinutes` to {float(strongest_cap['OverCapMinutes']):.2f} "
            f"while still leaving `CumWatch` at {float(strongest_cap['CumWatch']):.1f}, and `LateNightSessionStartRate` at the default "
            f"cap moves {late_night_direction} from {float(baseline['LateNightSessionStartRate']):.3f} to {float(cap_focus['LateNightSessionStartRate']):.3f}."
        ),
    ]
    if short_gap_saturated:
        sentences.append(
            (
                f"The short-gap fractions are already near saturation under PPO (`<=1` minute = {float(baseline['FractionReturnsWithin1Min']):.3f}, "
                f"`<=5` minutes = {float(baseline['FractionReturnsWithin5Min']):.3f}), so the cleaner anti-gaming signal is the combination of "
                f"flat watch time with higher session counts and a sharply reduced long-session tail."
            )
        )
    if calibration_audit_markdown:
        gap_audit = parse_calibration_gap_audit_markdown(calibration_audit_markdown)
        if all(key in gap_audit for key in ("mean_target", "mean_simulator", "p95_target", "p95_simulator")):
            sentences.append(
                (
                    f"The short-gap diagnostic should still be read conservatively because the referenced calibration audit reports a severe "
                    f"gap mismatch (`mean` {float(gap_audit['mean_target']):.3f} vs {float(gap_audit['mean_simulator']):.3f}, "
                    f"`p95` {float(gap_audit['p95_target']):.3f} vs {float(gap_audit['p95_simulator']):.3f}), so the fragmentation finding "
                    f"is best interpreted as an in-simulator anti-gaming check rather than an externally validated timing claim."
                )
            )
    return "\n\n".join(sentences)


def resolve_bundle_mechanism_night_metric(bundle_dir: Path | str) -> Tuple[str, bool]:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return "LateNightSessionStartRate", True
    manifest = json_load(manifest_path)
    proxy_diagnostics = manifest.get("constraint_track_proxy_diagnostics", {})
    promoted = proxy_diagnostics.get("promoted_main_text_proxy")
    if isinstance(promoted, str) and promoted:
        return str(promoted), False

    candidates = proxy_diagnostics.get("proxy_candidates", {})
    ranked: List[Tuple[float, str]] = []
    for metric_name in NIGHT_PROXY_PROMOTION_CANDIDATES:
        payload = candidates.get(metric_name)
        if not isinstance(payload, dict):
            continue
        try:
            corr = float(payload.get("policy_mean_corr", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(corr):
            ranked.append((abs(corr), str(metric_name)))
    if ranked:
        ranked.sort(key=lambda item: (item[0], item[1]))
        return ranked[0][1], True
    return "LateNightSessionStartRate", True


def mechanism_policy_sort_key(policy_name: str) -> Tuple[int, str]:
    order = {name: index for index, name in enumerate(MECHANISM_POLICY_ORDER)}
    return order.get(str(policy_name), len(order)), str(policy_name)


def mechanism_ablation_sort_key(ablation_name: str) -> Tuple[int, str]:
    ordered = ("Default", *MECHANISM_ABLATION_NAMES)
    order = {name: index for index, name in enumerate(ordered)}
    return order.get(str(ablation_name), len(order)), str(ablation_name)


def build_mechanism_ablation_table(records: Sequence[Dict[str, Any]], night_metric: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    raw_df = aggregate_across_train_seeds(list(records), ["policy", "ablation"])

    default_rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in records:
        if str(row.get("ablation")) == "Default":
            default_rows[(str(row["policy"]), int(row["train_seed"]))] = dict(row)

    metric_cols = ["CumWatch", "CVaR_0.95(L)", "OverCapMinutes", str(night_metric)]
    delta_records: List[Dict[str, Any]] = []
    for row in records:
        policy = str(row["policy"])
        train_seed = int(row["train_seed"])
        baseline = default_rows.get((policy, train_seed))
        if baseline is None:
            continue
        payload: Dict[str, Any] = {
            "policy": policy,
            "ablation": str(row["ablation"]),
            "train_seed": train_seed,
        }
        for metric in metric_cols:
            baseline_value = float(baseline.get(metric, float("nan")))
            denom = max(abs(baseline_value), 1e-8)
            payload[f"{metric}_pct_change_vs_default"] = 100.0 * (
                float(row.get(metric, float("nan"))) - baseline_value
            ) / denom
        delta_records.append(payload)

    delta_df = aggregate_across_train_seeds(delta_records, ["policy", "ablation"]) if delta_records else pd.DataFrame()
    if not delta_df.empty:
        raw_df = raw_df.merge(delta_df, on=["policy", "ablation"], how="left")

    ordered_rows = sorted(
        raw_df.to_dict(orient="records"),
        key=lambda row: (
            mechanism_policy_sort_key(str(row["policy"])),
            mechanism_ablation_sort_key(str(row["ablation"])),
        ),
    )
    ordered_df = pd.DataFrame(ordered_rows)
    ordered_cols = [
        "policy",
        "ablation",
        "CumWatch",
        "CumWatch_ci95",
        "CVaR_0.95(L)",
        "CVaR_0.95(L)_ci95",
        "OverCapMinutes",
        "OverCapMinutes_ci95",
        night_metric,
        f"{night_metric}_ci95",
        "CumWatch_pct_change_vs_default",
        "CumWatch_pct_change_vs_default_ci95",
        "CVaR_0.95(L)_pct_change_vs_default",
        "CVaR_0.95(L)_pct_change_vs_default_ci95",
        "OverCapMinutes_pct_change_vs_default",
        "OverCapMinutes_pct_change_vs_default_ci95",
        f"{night_metric}_pct_change_vs_default",
        f"{night_metric}_pct_change_vs_default_ci95",
    ]
    return ordered_df[[column for column in ordered_cols if column in ordered_df.columns]].reset_index(drop=True)


def save_mechanism_ablation_figure(mechanism_df: pd.DataFrame, night_metric: str, outpath: Path) -> None:
    if mechanism_df.empty:
        raise ValueError("Cannot plot mechanism ablations from an empty table.")
    plotted = mechanism_df[mechanism_df["ablation"] != "Default"].copy()
    if plotted.empty:
        raise ValueError("Mechanism ablation plot requires at least one non-default row.")

    metric_specs = [
        ("CumWatch_pct_change_vs_default", "CumWatch", "#4C78A8"),
        ("CVaR_0.95(L)_pct_change_vs_default", "CVaR_0.95(L)", "#54A24B"),
        ("OverCapMinutes_pct_change_vs_default", "OverCapMinutes", "#F58518"),
        (f"{night_metric}_pct_change_vs_default", night_metric, "#E45756"),
    ]
    label_map = {
        "CumWatch": "CumWatch",
        "CVaR_0.95(L)": "CVaR_0.95(L)",
        "OverCapMinutes": "OverCapMinutes",
        "LateNightSessionStartRate": "Late-night starts",
        "NightFraction": "Night fraction",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    height = 0.18
    offsets = np.linspace(-1.5 * height, 1.5 * height, len(metric_specs))

    for ax, policy_name in zip(axes, MECHANISM_POLICY_ORDER):
        block = plotted[plotted["policy"] == policy_name].copy()
        block["sort_key"] = block["ablation"].map(lambda value: mechanism_ablation_sort_key(str(value)))
        block = block.sort_values("sort_key").reset_index(drop=True)
        ys = np.arange(len(block), dtype=np.float64)
        for offset, (metric_col, label, color) in zip(offsets, metric_specs):
            if metric_col not in block.columns:
                continue
            err_col = f"{metric_col}_ci95"
            values = block[metric_col].astype(float).to_numpy(dtype=np.float64)
            errors = (
                block[err_col].astype(float).fillna(0.0).to_numpy(dtype=np.float64)
                if err_col in block.columns
                else np.zeros(len(block), dtype=np.float64)
            )
            ax.barh(ys + offset, values, height=height, color=color, alpha=0.88, label=label_map.get(label, label))
            ax.errorbar(values, ys + offset, xerr=errors, fmt="none", ecolor="black", capsize=3, linewidth=0.9)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(policy_name)
        ax.set_xlabel("% change vs default")
        ax.set_yticks(ys)
        ax.set_yticklabels(block["ablation"].astype(str).tolist())
        ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.75)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(metric_specs), frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Relative change from the default mechanism by ablation", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(outpath)
    plt.close(fig)
    log_artifact_written(outpath, "figure")


def render_mechanism_ablation_memo(
    mechanism_df: pd.DataFrame,
    *,
    night_metric: str,
    night_metric_appendix_only: bool,
) -> str:
    if mechanism_df.empty:
        return "# Mechanism ablations\n\nNo mechanism-ablation rows were available for interpretation.\n"

    def _row(policy: str, ablation: str) -> Optional[pd.Series]:
        block = mechanism_df[(mechanism_df["policy"] == policy) & (mechanism_df["ablation"] == ablation)]
        if block.empty:
            return None
        return block.iloc[0]

    def _pct(row: Optional[pd.Series], metric: str) -> float:
        if row is None:
            return float("nan")
        key = f"{metric}_pct_change_vs_default"
        if key not in row:
            return float("nan")
        return float(row[key])

    ppo_nohabit = _row("PPO", "NoHabit")
    lag_nohabit = _row("Lagrangian PPO", "NoHabit")
    ppo_novar = _row("PPO", "NoVar")
    lag_novar = _row("Lagrangian PPO", "NoVar")
    ppo_nopers = _row("PPO", "NoPers")
    lag_nopers = _row("Lagrangian PPO", "NoPers")
    ppo_homo = _row("PPO", "HomogeneousUsers")
    lag_homo = _row("Lagrangian PPO", "HomogeneousUsers")

    paragraph_1 = (
        f"Habit removal remains the clearest mechanism lever. Under `NoHabit`, PPO changes by "
        f"{_pct(ppo_nohabit, 'CumWatch'):+.1f}% in `CumWatch`, {_pct(ppo_nohabit, 'CVaR_0.95(L)'):+.1f}% in `CVaR_0.95(L)`, "
        f"and {_pct(ppo_nohabit, 'OverCapMinutes'):+.1f}% in `OverCapMinutes`; the corresponding Lagrangian PPO shifts are "
        f"{_pct(lag_nohabit, 'CumWatch'):+.1f}%, {_pct(lag_nohabit, 'CVaR_0.95(L)'):+.1f}%, and {_pct(lag_nohabit, 'OverCapMinutes'):+.1f}%. "
        f"Risk therefore does not disappear when habit is removed, but the frontier compresses materially, which supports the claim "
        f"that long-horizon accumulation is doing real work rather than merely decorating a one-step reward model."
    )

    def _magnitude_word(pct_val: float) -> str:
        abs_val = abs(pct_val)
        if abs_val < 0.1:
            return "negligible"
        elif abs_val < 1.0:
            return "modest"
        elif abs_val < 5.0:
            return "notable"
        else:
            return "sizeable"

    nopers_ppo_pct = _pct(ppo_nopers, 'OverCapMinutes')
    nopers_lag_pct = _pct(lag_nopers, 'OverCapMinutes')
    homo_ppo_pct = _pct(ppo_homo, 'OverCapMinutes')
    homo_lag_pct = _pct(lag_homo, 'OverCapMinutes')
    novar_ppo_pct = _pct(ppo_novar, 'OverCapMinutes')
    novar_lag_pct = _pct(lag_novar, 'OverCapMinutes')

    paragraph_2 = (
        f"The broader ablations make the story more specific. `NoPers` moves the frontier by "
        f"{nopers_ppo_pct:+.1f}% for PPO and {nopers_lag_pct:+.1f}% for Lagrangian PPO"
        f"{', so personalization clearly matters' if max(abs(nopers_ppo_pct), abs(nopers_lag_pct)) >= 1.0 else ''}. "
        f"`HomogeneousUsers` produces {_magnitude_word(max(abs(homo_ppo_pct), abs(homo_lag_pct)))} changes "
        f"({homo_ppo_pct:+.1f}% for PPO; {homo_lag_pct:+.1f}% for Lagrangian PPO)"
        f"{', which indicates that user heterogeneity contributes beyond the mean dynamics' if max(abs(homo_ppo_pct), abs(homo_lag_pct)) >= 1.0 else ''}. "
        f"By contrast, `NoVar` shows {_magnitude_word(max(abs(novar_ppo_pct), abs(novar_lag_pct)))} effects on the core frontier metrics "
        f"({novar_ppo_pct:+.1f}% for PPO; {novar_lag_pct:+.1f}% for Lagrangian PPO). "
        f"The reported night-family column is `{night_metric}`"
        f"{'  (kept appendix-only because no night proxy clears the orthogonality bar)' if night_metric_appendix_only else ''}."
    )
    return "# Mechanism ablations\n\n" + paragraph_1 + "\n\n" + paragraph_2 + "\n"


def load_saved_actor_critic_policy(checkpoint_path: Path, cfg: BenchConfig, device: str) -> PPOPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_summary = checkpoint.get("train_summary") or {}
    hidden_size = int(train_summary.get("hidden_size", 128))
    policy_arch = str(train_summary.get("policy_arch", "flat"))
    model = build_actor_critic_model(cfg, hidden_size=hidden_size, policy_arch=policy_arch).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return PPOPolicy(model)


def load_saved_myopic_policy(checkpoint_path: Path, cfg: BenchConfig, device: str) -> MyopicPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RewardModel(cfg.obs_dim(), cfg.num_actions()).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return MyopicPolicy(model)


def resolve_saved_lagrangian_checkpoint(model_dir: Path, train_seed: int) -> Path:
    matches = sorted(model_dir.glob(f"lagppo_scale*_seed{int(train_seed)}.pt"))
    if not matches:
        raise FileNotFoundError(f"No saved Lagrangian PPO checkpoint found for train_seed={train_seed} in {model_dir}.")
    if len(matches) > 1:
        LOGGER.warning(
            "Multiple saved Lagrangian PPO checkpoints found for train_seed=%s; using %s",
            train_seed,
            matches[0],
        )
    return matches[0]


def _evaluate_saved_cap_fragmentation_task(
    bundle_dir: str,
    device: str,
    train_seed: int,
    t_cap: Optional[float],
) -> Dict[str, Any]:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    bundle_path = Path(bundle_dir)
    cfg = BenchConfig.from_path(bundle_path / "config_used.json")
    cfg.device = str(device)
    cards = json_load(bundle_path / "cards_used.json")
    episode_seeds = [int(seed) for seed in list(json_load(bundle_path / "test_episode_seeds.json"))]
    catalog = build_catalog(cfg)
    policy = load_saved_actor_critic_policy(bundle_path / "models" / f"ppo_seed{int(train_seed)}.pt", cfg, str(device))
    policy_label = "PPO" if t_cap is None else (f"PPO+Cap({int(t_cap)})" if float(t_cap).is_integer() else f"PPO+Cap({t_cap})")
    wrappers = None if t_cap is None else {"session_cap": True, "T_cap": float(t_cap)}
    metrics = evaluate_policy(
        policy,
        cfg,
        catalog,
        cards,
        wrappers=wrappers,
        episode_seeds=episode_seeds,
        num_episodes=len(episode_seeds),
        deterministic=True,
        label=policy_label,
        train_seed=int(train_seed),
        log_progress=False,
        collect_policy_diagnostics=False,
    )
    return {
        "policy": policy_label,
        "train_seed": int(train_seed),
        "metrics": {
            "policy": policy_label,
            "train_seed": int(train_seed),
            "CumWatch": float(metrics["CumWatch"]),
            "OverCapMinutes": float(metrics["OverCapMinutes"]),
            "CVaR_0.95(L)": float(metrics["CVaR_0.95(L)"]),
            "SessionsPerEpisode": float(metrics["SessionsPerEpisode"]),
            "FractionReturnsWithin1Min": float(metrics["FractionReturnsWithin1Min"]),
            "FractionReturnsWithin5Min": float(metrics["FractionReturnsWithin5Min"]),
            "LateNightSessionStartRate": float(metrics["LateNightSessionStartRate"]),
        },
        "episode_rows": copy.deepcopy(metrics["EpisodeFragmentationRows"]),
    }


def _evaluate_saved_official_scorecard_seed_task(
    bundle_dir: str,
    device: str,
    train_seed: int,
    episode_limit: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    bundle_path = Path(bundle_dir)
    cfg = BenchConfig.from_path(bundle_path / "config_used.json")
    cfg.device = str(device)
    cards = json_load(bundle_path / "cards_used.json")
    episode_seeds = [int(seed) for seed in list(json_load(bundle_path / "test_episode_seeds.json"))]
    if episode_limit is not None:
        episode_seeds = episode_seeds[: max(1, int(episode_limit))]
    catalog = build_catalog(cfg)
    model_dir = bundle_path / "models"

    scorecard_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []

    def evaluate_and_append(
        policy_name: str,
        eval_mode: str,
        policy: Any,
        *,
        deterministic: bool,
        wrappers: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
    ) -> None:
        metrics = evaluate_policy(
            policy,
            cfg,
            catalog,
            cards,
            wrappers=wrappers,
            episode_seeds=episode_seeds,
            num_episodes=len(episode_seeds),
            deterministic=bool(deterministic),
            label=label or (f"{policy_name} [{eval_mode}]"),
            train_seed=int(train_seed),
            log_progress=False,
            collect_policy_diagnostics=False,
        )
        scorecard_rows.append(build_official_scorecard_metric_row(policy_name, "Param", eval_mode, int(train_seed), metrics))
        episode_rows.extend(
            build_official_episode_output_rows(policy_name, "Param", eval_mode, int(train_seed), episode_seeds, metrics)
        )

    evaluate_and_append(
        "Random",
        "stochastic",
        RandomPolicy(cfg.num_actions(), seed=int(train_seed)),
        deterministic=False,
        label="Random [stochastic]",
    )

    myopic_policy = load_saved_myopic_policy(model_dir / f"myopic_seed{int(train_seed)}.pt", cfg, str(device))
    evaluate_and_append("Myopic", "deterministic", myopic_policy, deterministic=True, label="Myopic [deterministic]")
    evaluate_and_append("Myopic", "stochastic", myopic_policy, deterministic=False, label="Myopic [stochastic]")

    evaluate_and_append("RoundRobinPolicy", "deterministic", RoundRobinPolicy(cfg), deterministic=True)
    evaluate_and_append("LeastRecentPolicy", "deterministic", LeastRecentPolicy(cfg, seed=int(train_seed)), deterministic=True)
    evaluate_and_append("NoveltyGreedyPolicy", "deterministic", NoveltyGreedyPolicy(cfg, seed=int(train_seed)), deterministic=True)

    ppo_policy = load_saved_actor_critic_policy(model_dir / f"ppo_seed{int(train_seed)}.pt", cfg, str(device))
    evaluate_and_append("PPO", "deterministic", ppo_policy, deterministic=True, label="PPO")
    evaluate_and_append("PPO", "stochastic", ppo_policy, deterministic=False, label="PPO [stochastic]")

    default_cap_policy_name = (
        f"PPO + SessionCap({int(cfg.T_cap)})" if float(cfg.T_cap).is_integer() else f"PPO + SessionCap({cfg.T_cap})"
    )
    default_cap_wrappers = {"session_cap": True, "T_cap": float(cfg.T_cap)}
    evaluate_and_append(
        default_cap_policy_name,
        "deterministic",
        ppo_policy,
        deterministic=True,
        wrappers=default_cap_wrappers,
        label=default_cap_policy_name,
    )
    evaluate_and_append(
        default_cap_policy_name,
        "stochastic",
        ppo_policy,
        deterministic=False,
        wrappers=default_cap_wrappers,
        label=f"{default_cap_policy_name} [stochastic]",
    )

    lagppo_policy = load_saved_actor_critic_policy(resolve_saved_lagrangian_checkpoint(model_dir, int(train_seed)), cfg, str(device))
    evaluate_and_append("Lagrangian PPO", "deterministic", lagppo_policy, deterministic=True, label="Lagrangian PPO")
    evaluate_and_append("Lagrangian PPO", "stochastic", lagppo_policy, deterministic=False, label="Lagrangian PPO [stochastic]")

    return {
        "train_seed": int(train_seed),
        "scorecard_rows": scorecard_rows,
        "episode_rows": episode_rows,
    }


def _bundle_main_text_night_proxy(bundle_dir: Path) -> Optional[str]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json_load(manifest_path)
    proxy_diagnostics = manifest.get("constraint_track_proxy_diagnostics", {})
    proxy_name = proxy_diagnostics.get("promoted_main_text_proxy")
    if isinstance(proxy_name, str) and proxy_name:
        return proxy_name
    return None


def _bundle_scorecard_includes_returnrate60(bundle_dir: Path) -> bool:
    for filename in ["table_scorecard_deterministic.csv", "table_scorecard_stochastic.csv"]:
        path = bundle_dir / "tables" / filename
        if not path.exists():
            continue
        try:
            columns = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue
        if "ReturnRate60" in columns:
            return True
    return False


def write_official_scorecard_artifacts(
    bundle_dir: Path | str,
    *,
    device: str = "cpu",
    max_workers: Optional[int] = None,
    episode_limit: Optional[int] = None,
) -> Dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    table_dir = ensure_dir(bundle_dir / "tables")
    train_seed_list = [int(seed) for seed in list(json_load(bundle_dir / "train_seed_list.json"))]
    resolved_max_workers = max(1, int(max_workers if max_workers is not None else min(4, max(1, os.cpu_count() or 1))))

    scorecard_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []
    if resolved_max_workers == 1:
        for train_seed in train_seed_list:
            result = _evaluate_saved_official_scorecard_seed_task(
                str(bundle_dir),
                str(device),
                int(train_seed),
                episode_limit=episode_limit,
            )
            scorecard_rows.extend(copy.deepcopy(result["scorecard_rows"]))
            episode_rows.extend(copy.deepcopy(result["episode_rows"]))
    else:
        try:
            with ProcessPoolExecutor(max_workers=resolved_max_workers) as executor:
                future_map = {
                    executor.submit(
                        _evaluate_saved_official_scorecard_seed_task,
                        str(bundle_dir),
                        str(device),
                        int(train_seed),
                        episode_limit,
                    ): int(train_seed)
                    for train_seed in train_seed_list
                }
                for future in as_completed(future_map):
                    train_seed = future_map[future]
                    result = future.result()
                    scorecard_rows.extend(copy.deepcopy(result["scorecard_rows"]))
                    episode_rows.extend(copy.deepcopy(result["episode_rows"]))
                    LOGGER.info(
                        "Official scorecard rebuild task complete | train_seed=%s | rows=%s | episodes=%s",
                        train_seed,
                        len(result["scorecard_rows"]),
                        len(result["episode_rows"]),
                    )
        except (PermissionError, OSError) as exc:
            LOGGER.warning(
                "Official scorecard parallel workers unavailable; falling back to sequential execution | requested_max_workers=%s | error=%s",
                resolved_max_workers,
                exc,
            )
            for train_seed in train_seed_list:
                result = _evaluate_saved_official_scorecard_seed_task(
                    str(bundle_dir),
                    str(device),
                    int(train_seed),
                    episode_limit=episode_limit,
                )
                scorecard_rows.extend(copy.deepcopy(result["scorecard_rows"]))
                episode_rows.extend(copy.deepcopy(result["episode_rows"]))

    if not scorecard_rows or not episode_rows:
        raise ValueError(f"No official scorecard rows were rebuilt for bundle_dir={bundle_dir}.")

    cfg = BenchConfig.from_path(bundle_dir / "config_used.json")
    official_scorecard_df = aggregate_across_train_seeds(scorecard_rows, ["policy", "backend", "eval_mode"])
    official_episode_output_df = (
        pd.DataFrame(episode_rows)
        .sort_values(["policy", "eval_mode", "train_seed", "episode_index"])
        .reset_index(drop=True)
    )
    save_csv(official_episode_output_df, table_dir / "table_official_policy_episode_outputs.csv")

    official_scorecard_df = augment_official_scorecard_with_episode_uncertainty(
        official_scorecard_df,
        official_episode_output_df,
        cfg,
    )
    scorecard_cols = build_official_scorecard_column_order(
        official_scorecard_df.columns,
        main_text_night_proxy=_bundle_main_text_night_proxy(bundle_dir),
        include_returnrate60=_bundle_scorecard_includes_returnrate60(bundle_dir),
    )
    deterministic_scorecard_df = official_scorecard_df[official_scorecard_df["eval_mode"] == "deterministic"].copy()
    stochastic_scorecard_df = official_scorecard_df[official_scorecard_df["eval_mode"] == "stochastic"].copy()
    save_table(deterministic_scorecard_df[scorecard_cols].copy(), table_dir / "table_scorecard_deterministic")
    save_table(stochastic_scorecard_df[scorecard_cols].copy(), table_dir / "table_scorecard_stochastic")

    manifest_path = bundle_dir / "manifest.json"
    manifest = json_load(manifest_path) if manifest_path.exists() else {}
    manifest["table_scorecard_deterministic"] = "tables/table_scorecard_deterministic.csv"
    manifest["table_scorecard_stochastic"] = "tables/table_scorecard_stochastic.csv"
    manifest["official_policy_episode_outputs"] = "tables/table_official_policy_episode_outputs.csv"
    manifest["official_main_eval_mode"] = "deterministic"
    manifest["official_policy_episode_count"] = int(official_episode_output_df["episode_seed"].nunique())
    manifest["official_policy_episode_limit"] = None if episode_limit is None else int(episode_limit)
    manifest["tables"] = sorted(p.name for p in table_dir.iterdir())
    json_dumps(manifest, manifest_path)
    log_artifact_written(manifest_path, "manifest_json")

    return {
        "official_scorecard_df": official_scorecard_df,
        "official_episode_output_df": official_episode_output_df,
        "deterministic_table_path": table_dir / "table_scorecard_deterministic.csv",
        "stochastic_table_path": table_dir / "table_scorecard_stochastic.csv",
    }


def _evaluate_saved_mechanism_ablation_seed_task(
    bundle_dir: str,
    device: str,
    train_seed: int,
    episode_limit: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    bundle_path = Path(bundle_dir)
    base_cfg = BenchConfig.from_path(bundle_path / "config_used.json")
    base_cfg.device = str(device)
    cards = json_load(bundle_path / "cards_used.json")
    episode_seeds = [int(seed) for seed in list(json_load(bundle_path / "test_episode_seeds.json"))]
    if episode_limit is not None:
        episode_seeds = episode_seeds[: max(1, int(episode_limit))]
    catalog = build_catalog(base_cfg)
    model_dir = bundle_path / "models"

    base_ppo_policy = load_saved_actor_critic_policy(model_dir / f"ppo_seed{int(train_seed)}.pt", base_cfg, str(device))
    base_lag_policy = load_saved_actor_critic_policy(resolve_saved_lagrangian_checkpoint(model_dir, int(train_seed)), base_cfg, str(device))

    rows: List[Dict[str, Any]] = []

    def _project_if_needed(policy_obj: Any, eval_cfg: BenchConfig) -> Any:
        if int(eval_cfg.P) == int(base_cfg.P):
            return policy_obj
        return ProjectedActionSpacePolicy(policy_obj, source_P=int(base_cfg.P), target_P=int(eval_cfg.P))

    def _evaluate(policy_name: str, policy_obj: Any, ablation_name: str) -> None:
        eval_cfg = copy.deepcopy(base_cfg) if ablation_name == "Default" else base_cfg.ablated(ablation_name)
        eval_cfg.device = str(device)
        effective_policy = _project_if_needed(policy_obj, eval_cfg)
        metrics = evaluate_policy(
            effective_policy,
            eval_cfg,
            catalog,
            cards,
            episode_seeds=episode_seeds,
            num_episodes=len(episode_seeds),
            deterministic=True,
            ablation=None if ablation_name == "Default" else ablation_name,
            label=f"{policy_name} [{ablation_name}]",
            log_progress=False,
            collect_policy_diagnostics=False,
        )
        rows.append(
            {
                "policy": str(policy_name),
                "ablation": str(ablation_name),
                "train_seed": int(train_seed),
                "CumWatch": float(metrics["CumWatch"]),
                "CVaR_0.95(L)": float(metrics["CVaR_0.95(L)"]),
                "OverCapMinutes": float(metrics["OverCapMinutes"]),
                "NightFraction": float(metrics.get("NightFraction", float("nan"))),
                "LateNightSessionStartRate": float(metrics.get("LateNightSessionStartRate", float("nan"))),
            }
        )

    for ablation_name in ("Default", *MECHANISM_ABLATION_NAMES):
        _evaluate("PPO", base_ppo_policy, ablation_name)
        _evaluate("Lagrangian PPO", base_lag_policy, ablation_name)

    return {
        "train_seed": int(train_seed),
        "rows": rows,
    }


def write_mechanism_ablation_artifacts(
    bundle_dir: Path | str,
    *,
    device: str = "cpu",
    max_workers: Optional[int] = None,
    episode_limit: Optional[int] = None,
) -> Dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    table_dir = ensure_dir(bundle_dir / "tables")
    fig_dir = ensure_dir(bundle_dir / "figures")
    train_seed_list = [int(seed) for seed in list(json_load(bundle_dir / "train_seed_list.json"))]
    resolved_max_workers = max(1, int(max_workers if max_workers is not None else min(4, max(1, os.cpu_count() or 1))))
    night_metric, night_metric_appendix_only = resolve_bundle_mechanism_night_metric(bundle_dir)

    rows: List[Dict[str, Any]] = []
    if resolved_max_workers == 1:
        for train_seed in train_seed_list:
            result = _evaluate_saved_mechanism_ablation_seed_task(
                str(bundle_dir),
                str(device),
                int(train_seed),
                episode_limit=episode_limit,
            )
            rows.extend(copy.deepcopy(result["rows"]))
    else:
        try:
            with ProcessPoolExecutor(max_workers=resolved_max_workers) as executor:
                future_map = {
                    executor.submit(
                        _evaluate_saved_mechanism_ablation_seed_task,
                        str(bundle_dir),
                        str(device),
                        int(train_seed),
                        episode_limit,
                    ): int(train_seed)
                    for train_seed in train_seed_list
                }
                for future in as_completed(future_map):
                    train_seed = future_map[future]
                    result = future.result()
                    rows.extend(copy.deepcopy(result["rows"]))
                    LOGGER.info(
                        "Mechanism ablation task complete | train_seed=%s | rows=%s",
                        train_seed,
                        len(result["rows"]),
                    )
        except (PermissionError, OSError) as exc:
            LOGGER.warning(
                "Mechanism ablation parallel workers unavailable; falling back to sequential execution | requested_max_workers=%s | error=%s",
                resolved_max_workers,
                exc,
            )
            for train_seed in train_seed_list:
                result = _evaluate_saved_mechanism_ablation_seed_task(
                    str(bundle_dir),
                    str(device),
                    int(train_seed),
                    episode_limit=episode_limit,
                )
                rows.extend(copy.deepcopy(result["rows"]))

    mechanism_df = build_mechanism_ablation_table(rows, night_metric)
    save_table(mechanism_df, table_dir / "table_mechanism_ablations")
    figure_path = fig_dir / "fig_mechanism_ablations.png"
    save_mechanism_ablation_figure(mechanism_df, night_metric, figure_path)

    memo_text = render_mechanism_ablation_memo(
        mechanism_df,
        night_metric=night_metric,
        night_metric_appendix_only=night_metric_appendix_only,
    )
    memo_path = bundle_dir / "mechanism_ablations_memo.md"
    memo_path.write_text(memo_text + "\n", encoding="utf-8")
    log_artifact_written(memo_path, "diagnostic_memo_markdown")

    manifest_path = bundle_dir / "manifest.json"
    manifest = json_load(manifest_path) if manifest_path.exists() else {}
    manifest["table_mechanism_ablations"] = "tables/table_mechanism_ablations.csv"
    manifest["fig_mechanism_ablations"] = "figures/fig_mechanism_ablations.png"
    manifest["mechanism_ablations_memo"] = memo_path.name
    manifest["mechanism_ablation_night_metric"] = str(night_metric)
    manifest["mechanism_ablation_night_metric_appendix_only"] = bool(night_metric_appendix_only)
    manifest["mechanism_ablation_episode_limit"] = None if episode_limit is None else int(episode_limit)
    manifest["tables"] = sorted(p.name for p in table_dir.iterdir())
    manifest["figures"] = sorted(p.name for p in fig_dir.iterdir())
    json_dumps(manifest, manifest_path)
    log_artifact_written(manifest_path, "manifest_json")

    return {
        "table": mechanism_df,
        "figure_path": figure_path,
        "memo_path": memo_path,
        "night_metric": night_metric,
        "night_metric_appendix_only": night_metric_appendix_only,
    }


def write_cap_fragmentation_artifacts(
    bundle_dir: Path | str,
    *,
    device: str = "cpu",
    cap_grid: Optional[Sequence[float]] = None,
    calibration_audit_md: Optional[Path | str] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    train_seed_list = [int(seed) for seed in list(json_load(bundle_dir / "train_seed_list.json"))]
    table_dir = ensure_dir(bundle_dir / "tables")
    fig_dir = ensure_dir(bundle_dir / "figures")
    episode_dir = ensure_dir(bundle_dir / "episode_diagnostics")
    resolved_cap_grid = sorted({float(value) for value in (cap_grid or [90.0, 120.0, 150.0])})
    resolved_max_workers = max(1, int(max_workers if max_workers is not None else min(4, max(1, os.cpu_count() or 1))))
    task_specs: List[Tuple[int, Optional[float]]] = []
    for train_seed in train_seed_list:
        task_specs.append((int(train_seed), None))
        task_specs.extend((int(train_seed), float(t_cap)) for t_cap in resolved_cap_grid)

    metric_rows: List[Dict[str, Any]] = []
    episode_rows_by_policy: Dict[str, List[Dict[str, Any]]] = {}
    if resolved_max_workers == 1:
        for train_seed, t_cap in task_specs:
            result = _evaluate_saved_cap_fragmentation_task(str(bundle_dir), str(device), int(train_seed), t_cap)
            metric_rows.append(copy.deepcopy(result["metrics"]))
            episode_rows_by_policy.setdefault(str(result["policy"]), []).extend(copy.deepcopy(result["episode_rows"]))
    else:
        try:
            with ProcessPoolExecutor(max_workers=resolved_max_workers) as executor:
                future_map = {
                    executor.submit(
                        _evaluate_saved_cap_fragmentation_task,
                        str(bundle_dir),
                        str(device),
                        int(train_seed),
                        t_cap,
                    ): (train_seed, t_cap)
                    for train_seed, t_cap in task_specs
                }
                for future in as_completed(future_map):
                    train_seed, t_cap = future_map[future]
                    result = future.result()
                    metric_rows.append(copy.deepcopy(result["metrics"]))
                    episode_rows_by_policy.setdefault(str(result["policy"]), []).extend(copy.deepcopy(result["episode_rows"]))
                    LOGGER.info(
                        "Cap-fragmentation task complete | train_seed=%s | policy=%s | CumWatch=%.3f | OverCapMinutes=%.3f | SessionsPerEpisode=%.3f | FractionReturnsWithin1Min=%.3f",
                        train_seed,
                        result["policy"],
                        float(result["metrics"]["CumWatch"]),
                        float(result["metrics"]["OverCapMinutes"]),
                        float(result["metrics"]["SessionsPerEpisode"]),
                        float(result["metrics"]["FractionReturnsWithin1Min"]),
                    )
        except (PermissionError, OSError) as exc:
            LOGGER.warning(
                "Cap-fragmentation parallel workers unavailable; falling back to sequential execution | requested_max_workers=%s | error=%s",
                resolved_max_workers,
                exc,
            )
            for train_seed, t_cap in task_specs:
                result = _evaluate_saved_cap_fragmentation_task(str(bundle_dir), str(device), int(train_seed), t_cap)
                metric_rows.append(copy.deepcopy(result["metrics"]))
                episode_rows_by_policy.setdefault(str(result["policy"]), []).extend(copy.deepcopy(result["episode_rows"]))
                LOGGER.info(
                    "Cap-fragmentation task complete | train_seed=%s | policy=%s | CumWatch=%.3f | OverCapMinutes=%.3f | SessionsPerEpisode=%.3f | FractionReturnsWithin1Min=%.3f",
                    train_seed,
                    result["policy"],
                    float(result["metrics"]["CumWatch"]),
                    float(result["metrics"]["OverCapMinutes"]),
                    float(result["metrics"]["SessionsPerEpisode"]),
                    float(result["metrics"]["FractionReturnsWithin1Min"]),
                )

    episode_files: List[str] = []
    for policy_name, rows in sorted(episode_rows_by_policy.items(), key=lambda item: cap_fragmentation_policy_sort_key(item[0])):
        episode_df = pd.DataFrame(rows).sort_values(["train_seed", "episode_index"]).reset_index(drop=True)
        episode_path = episode_dir / f"cap_fragmentation_{filename_slug(policy_name)}.csv"
        save_csv(episode_df, episode_path)
        episode_files.append(str(episode_path.relative_to(bundle_dir)))

    fragmentation_df = build_cap_fragmentation_table(metric_rows)
    save_table(fragmentation_df, table_dir / "table_cap_fragmentation")
    figure_path = fig_dir / "fig_cap_fragmentation.png"
    save_cap_fragmentation_figure(fragmentation_df, figure_path)

    calibration_text = None
    if calibration_audit_md is not None and Path(calibration_audit_md).exists():
        calibration_text = Path(calibration_audit_md).read_text(encoding="utf-8")
    memo_text = render_cap_fragmentation_memo(fragmentation_df, calibration_audit_markdown=calibration_text)
    memo_path = bundle_dir / "cap_fragmentation_memo.md"
    memo_path.write_text(memo_text + "\n", encoding="utf-8")
    log_artifact_written(memo_path, "diagnostic_memo_markdown")

    manifest_path = bundle_dir / "manifest.json"
    manifest = json_load(manifest_path) if manifest_path.exists() else {}
    manifest["table_cap_fragmentation"] = "tables/table_cap_fragmentation.csv"
    manifest["fig_cap_fragmentation"] = "figures/fig_cap_fragmentation.png"
    manifest["cap_fragmentation_memo"] = memo_path.name
    manifest["cap_fragmentation_episode_diagnostics"] = list(episode_files)
    manifest["tables"] = sorted(p.name for p in table_dir.iterdir())
    manifest["figures"] = sorted(p.name for p in fig_dir.iterdir())
    json_dumps(manifest, manifest_path)
    log_artifact_written(manifest_path, "manifest_json")

    return {
        "table": fragmentation_df,
        "memo_path": memo_path,
        "figure_path": figure_path,
        "episode_files": episode_files,
    }


def parse_saved_lagppo_checkpoint_scale(checkpoint_path: Path) -> float:
    match = re.search(r"lagppo_scale(?P<scale>[-+0-9.]+)_seed\d+\.pt$", checkpoint_path.name)
    if match is None:
        raise ValueError(f"Could not parse LagPPO scale from checkpoint path: {checkpoint_path}")
    return float(match.group("scale"))


def select_lagrangian_candidate(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    feasible = [candidate for candidate in candidates if bool(candidate.get("train_summary", {}).get("selected_feasible", False))]
    if feasible:
        return max(
            feasible,
            key=lambda candidate: (
                float((candidate.get("selected_validation") or {}).get("validation_CumWatch", float("-inf"))),
                float((candidate.get("selected_validation") or {}).get("global_step", 0.0)),
            ),
        )
    return min(
        candidates,
        key=lambda candidate: (
            float((candidate.get("selected_validation") or {}).get("total_violation", float("inf"))),
            -float((candidate.get("selected_validation") or {}).get("validation_CumWatch", float("-inf"))),
            -float((candidate.get("selected_validation") or {}).get("global_step", 0.0)),
        ),
    )


def parse_lagppo_feasibility_log(run_log_path: Path | str) -> Dict[str, Any]:
    run_log_path = Path(run_log_path)
    if not run_log_path.exists():
        raise FileNotFoundError(f"Missing run.log at {run_log_path}")

    candidate_start_re = re.compile(
        r"train_seed=(?P<train_seed>\d+) \| Lagrangian PPO (?P<stage>\w+) candidate start \| "
        r"scale=(?P<scale>[-+0-9.eE]+) \| dual_lr=(?P<dual_lr>[-+0-9.eE]+) \| total_steps=(?P<total_steps>\d+) \| "
        r"val_episodes=(?P<val_episodes>\d+)"
    )
    no_feasible_re = re.compile(
        r"Lagrangian PPO finished with no feasible validation checkpoint \| selected_global_step=(?P<selected_global_step>\d+) \| "
        r"total_violation=(?P<total_violation>[-+0-9.eE]+)"
    )
    training_complete_re = re.compile(
        r"Lagrangian PPO training complete \| selected_source=(?P<selected_source>[^|]+) \| "
        r"selected_feasible=(?P<selected_feasible>True|False) \| validation_checkpoints=(?P<validation_checkpoints>\d+) \| "
        r"final_lambda1=(?P<final_lambda1>[-+0-9.eE]+) \| final_lambda2=(?P<final_lambda2>[-+0-9.eE]+)"
    )
    all_scales_infeasible_re = re.compile(r"train_seed=(?P<train_seed>\d+) \| all constrained scales are infeasible on validation")
    selected_scale_infeasible_re = re.compile(
        r"train_seed=(?P<train_seed>\d+) \| selected operating point scale=(?P<scale>[-+0-9.eE]+) is infeasible on validation"
    )

    candidate_rows: List[Dict[str, Any]] = []
    seed_notes: Dict[int, Dict[str, Any]] = {}
    current_candidate: Optional[Dict[str, Any]] = None
    for raw_line in run_log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        start_match = candidate_start_re.search(line)
        if start_match is not None:
            current_candidate = {
                "train_seed": int(start_match.group("train_seed")),
                "stage": str(start_match.group("stage")),
                "scale": float(start_match.group("scale")),
                "dual_lr": float(start_match.group("dual_lr")),
                "total_steps": int(start_match.group("total_steps")),
                "val_episodes": int(start_match.group("val_episodes")),
                "no_feasible_validation_checkpoint": False,
                "least_violating_total_violation": float("nan"),
                "least_violating_selected_global_step": None,
            }
            continue

        no_feasible_match = no_feasible_re.search(line)
        if no_feasible_match is not None and current_candidate is not None:
            current_candidate["no_feasible_validation_checkpoint"] = True
            current_candidate["least_violating_total_violation"] = float(no_feasible_match.group("total_violation"))
            current_candidate["least_violating_selected_global_step"] = int(no_feasible_match.group("selected_global_step"))
            continue

        complete_match = training_complete_re.search(line)
        if complete_match is not None and current_candidate is not None:
            row = dict(current_candidate)
            row.update(
                {
                    "selected_source": str(complete_match.group("selected_source")).strip(),
                    "selected_feasible": str(complete_match.group("selected_feasible")) == "True",
                    "validation_checkpoints": int(complete_match.group("validation_checkpoints")),
                    "final_lambda1": float(complete_match.group("final_lambda1")),
                    "final_lambda2": float(complete_match.group("final_lambda2")),
                }
            )
            candidate_rows.append(row)
            current_candidate = None
            continue

        seed_match = all_scales_infeasible_re.search(line)
        if seed_match is not None:
            payload = seed_notes.setdefault(int(seed_match.group("train_seed")), {})
            payload["all_scales_infeasible"] = True
            continue

        selected_match = selected_scale_infeasible_re.search(line)
        if selected_match is not None:
            payload = seed_notes.setdefault(int(selected_match.group("train_seed")), {})
            payload["selected_operating_point_infeasible"] = True
            payload["warned_scale"] = float(selected_match.group("scale"))

    seed_summary_rows: List[Dict[str, Any]] = []
    grouped_rows = pd.DataFrame(candidate_rows)
    if not grouped_rows.empty:
        for train_seed, block in grouped_rows.groupby("train_seed", sort=True):
            payload = seed_notes.get(int(train_seed), {})
            seed_summary_rows.append(
                {
                    "train_seed": int(train_seed),
                    "candidate_runs": int(len(block)),
                    "feasible_candidate_runs": int(block["selected_feasible"].astype(bool).sum()),
                    "no_feasible_candidate_runs": int(block["no_feasible_validation_checkpoint"].astype(bool).sum()),
                    "candidate_scales": ",".join(
                        sorted({f"{float(value):.2f}" for value in block["scale"].astype(float).tolist()})
                    ),
                    "candidate_dual_lrs": ",".join(
                        sorted({f"{float(value):.4f}" for value in block["dual_lr"].astype(float).tolist()})
                    ),
                    "all_scales_infeasible": bool(payload.get("all_scales_infeasible", False)),
                    "selected_operating_point_infeasible_warning": bool(payload.get("selected_operating_point_infeasible", False)),
                }
            )

    return {
        "candidate_rows": candidate_rows,
        "seed_summary_rows": seed_summary_rows,
        "total_candidate_runs": int(len(candidate_rows)),
        "total_no_feasible_candidate_runs": int(
            sum(bool(row.get("no_feasible_validation_checkpoint", False)) for row in candidate_rows)
        ),
        "total_feasible_candidate_runs": int(sum(bool(row.get("selected_feasible", False)) for row in candidate_rows)),
    }


def _evaluate_policy_on_seed_list(
    policy: Any,
    cfg: BenchConfig,
    catalog: np.ndarray,
    cards: Sequence[Dict[str, Any]],
    *,
    episode_seeds: Sequence[int],
    label: str,
) -> Dict[str, Any]:
    return evaluate_policy(
        policy,
        cfg,
        catalog,
        cards,
        episode_seeds=list(map(int, episode_seeds)),
        num_episodes=len(list(episode_seeds)),
        deterministic=True,
        label=label,
        log_progress=False,
        collect_policy_diagnostics=False,
    )


def write_lagppo_rescue_artifacts(
    bundle_dir: Path | str,
    *,
    device: str = "cpu",
    run_log_path: Optional[Path | str] = None,
    train_seeds: Optional[Sequence[int]] = None,
    rescue_scales: Sequence[float] = (0.90, 0.95),
    rescue_dual_lrs: Sequence[float] = (0.005, 0.01),
    rescue_steps: int = 25_000,
    rescue_validate_every: int = 12_500,
    rescue_val_episodes: int = 500,
    confirm_val_episodes: Optional[int] = None,
    cost_normalization: str = "budget",
) -> Dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    model_dir = bundle_dir / "models"
    manifest_path = bundle_dir / "manifest.json"
    manifest = json_load(manifest_path) if manifest_path.exists() else {}
    run_log_path = Path(run_log_path) if run_log_path is not None else bundle_dir / "run.log"

    cfg = BenchConfig.from_path(bundle_dir / "config_used.json")
    cfg.device = str(device)
    cards = json_load(bundle_dir / "cards_used.json")
    catalog = build_catalog(cfg)
    bundle_train_seeds = [int(seed) for seed in list(json_load(bundle_dir / "train_seed_list.json"))]
    target_train_seeds = bundle_train_seeds if train_seeds is None else [int(seed) for seed in train_seeds]
    confirm_episode_count = int(confirm_val_episodes) if confirm_val_episodes is not None else int(
        manifest.get("lag_full_val_episodes", len(list(json_load(bundle_dir / "val_episode_seeds.json"))))
    )
    all_val_episode_seeds = [int(seed) for seed in list(json_load(bundle_dir / "val_episode_seeds.json"))]
    test_episode_seeds = [int(seed) for seed in list(json_load(bundle_dir / "test_episode_seeds.json"))]
    rescue_val_episode_seeds = all_val_episode_seeds[: max(1, int(rescue_val_episodes))]
    confirm_val_episode_seeds = all_val_episode_seeds[: max(1, int(confirm_episode_count))]

    run_log_summary = parse_lagppo_feasibility_log(run_log_path)
    run_log_seed_rows = {
        int(row["train_seed"]): dict(row)
        for row in run_log_summary.get("seed_summary_rows", [])
    }

    proxy_diagnostics = manifest.get("constraint_track_proxy_diagnostics", {})
    if not isinstance(proxy_diagnostics, dict):
        proxy_diagnostics = {}

    summary_rows: List[Dict[str, Any]] = []
    rescue_rows: List[Dict[str, Any]] = []

    frontier_path = bundle_dir / "tables" / "table_frontier_rebuilt.csv"
    frontier_df = pd.read_csv(frontier_path) if frontier_path.exists() else pd.DataFrame()
    frontier_lag_row = frontier_df[
        (frontier_df.get("method") == "LagrangianPPO") & (frontier_df.get("eval_mode") == "deterministic")
    ]
    frontier_ppo_row = frontier_df[
        (frontier_df.get("method") == "PPO") & (frontier_df.get("eval_mode") == "deterministic")
    ]

    for train_seed in bundle_train_seeds:
        current_checkpoint_path = resolve_saved_lagrangian_checkpoint(model_dir, int(train_seed))
        current_checkpoint = torch.load(current_checkpoint_path, map_location=str(device))
        current_train_summary = copy.deepcopy(current_checkpoint.get("train_summary") or {})
        current_selected_validation = copy.deepcopy(current_train_summary.get("selected_validation") or {})
        current_log_row = run_log_seed_rows.get(int(train_seed), {})
        current_row = {
            "phase": "current_selected",
            "train_seed": int(train_seed),
            "candidate_runs": int(current_log_row.get("candidate_runs", 0)),
            "feasible_candidate_runs": int(current_log_row.get("feasible_candidate_runs", 0)),
            "no_feasible_candidate_runs": int(current_log_row.get("no_feasible_candidate_runs", 0)),
            "all_scales_infeasible": bool(current_log_row.get("all_scales_infeasible", False)),
            "selected_operating_point_infeasible_warning": bool(
                current_log_row.get("selected_operating_point_infeasible_warning", False)
            ),
            "chosen_scale": float(parse_saved_lagppo_checkpoint_scale(current_checkpoint_path)),
            "dual_lr": float(current_checkpoint.get("dual_lr", current_train_summary.get("dual_lr", float("nan")))),
            "total_steps": int(current_checkpoint.get("total_steps", current_train_summary.get("total_steps", 0))),
            "cost_normalization": str(current_train_summary.get("cost_normalization", "none")),
            "selected_source": str(current_train_summary.get("selected_source", "")),
            "selected_feasible": bool(current_train_summary.get("selected_feasible", False)),
            "selected_global_step": int(current_train_summary.get("selected_global_step") or 0),
            "validation_episode_count": len(confirm_val_episode_seeds),
            "validation_CumWatch": float(current_selected_validation.get("validation_CumWatch", float("nan"))),
            "validation_OverCapMinutes": float(current_selected_validation.get("validation_OverCapMinutes", float("nan"))),
            "validation_CVaR_0.95(L)": float(current_selected_validation.get("validation_CVaR_0.95(L)", float("nan"))),
            "validation_total_violation": float(current_selected_validation.get("total_violation", float("nan"))),
            "confirmation_only_feasible": False,
            "test_episode_count": len(test_episode_seeds) if int(train_seed) in target_train_seeds else 0,
            "test_CumWatch": float("nan"),
            "test_OverCapMinutes": float("nan"),
            "test_CVaR_0.95(L)": float("nan"),
            "constraint_channels": "OverCapMinutes",
        }
        if int(train_seed) not in target_train_seeds:
            summary_rows.append(current_row)
            continue

        ppo_policy = load_saved_actor_critic_policy(model_dir / f"ppo_seed{int(train_seed)}.pt", cfg, str(device))
        ppo_val_metrics = _evaluate_policy_on_seed_list(
            ppo_policy,
            cfg,
            catalog,
            cards,
            episode_seeds=confirm_val_episode_seeds,
            label=f"PPO validation reference (seed {int(train_seed)})",
        )
        constraint_track_spec = build_constraint_track_spec(ppo_val_metrics, proxy_diagnostics)

        current_policy = load_saved_actor_critic_policy(current_checkpoint_path, cfg, str(device))
        current_test_metrics = _evaluate_policy_on_seed_list(
            current_policy,
            cfg,
            catalog,
            cards,
            episode_seeds=test_episode_seeds,
            label=f"Current LagPPO test (seed {int(train_seed)})",
        )
        current_row.update(
            {
                "test_CumWatch": float(current_test_metrics["CumWatch"]),
                "test_OverCapMinutes": float(current_test_metrics["OverCapMinutes"]),
                "test_CVaR_0.95(L)": float(current_test_metrics["CVaR_0.95(L)"]),
                "constraint_channels": str(constraint_track_spec.get("channels_label", "none")),
            }
        )
        summary_rows.append(current_row)

        candidate_runs: List[Dict[str, Any]] = []
        for scale_index, scale in enumerate(sorted({float(value) for value in rescue_scales}), start=1):
            night_budget, over_budget = scaled_constraint_budgets(scale, ppo_val_metrics, constraint_track_spec)
            for candidate_index, dual_lr_value in enumerate(sorted({float(value) for value in rescue_dual_lrs}), start=1):
                candidate_seed = int(train_seed) + 400_000 + scale_index * 10_000 + candidate_index * 137
                candidate_model, candidate_summary = train_ppo(
                    cfg,
                    catalog,
                    cards,
                    seed=candidate_seed,
                    total_steps=int(rescue_steps),
                    rollout_steps=2048,
                    minibatch_size=256,
                    update_epochs=10,
                    lr=3e-4,
                    ent_coef=0.01,
                    hidden_size=128,
                    policy_arch="flat",
                    device=str(device),
                    lagrangian=True,
                    night_budget=night_budget,
                    over_budget=over_budget,
                    dual_lr=float(dual_lr_value),
                    cost_normalization=str(cost_normalization),
                    validate_every=int(rescue_validate_every),
                    val_episodes=len(rescue_val_episode_seeds),
                    val_episode_seeds=rescue_val_episode_seeds,
                )
                selected_validation = copy.deepcopy(candidate_summary.get("selected_validation") or {})
                candidate_runs.append(
                    {
                        "model": candidate_model,
                        "policy": PPOPolicy(candidate_model),
                        "train_summary": candidate_summary,
                        "selected_validation": selected_validation,
                        "scale": float(scale),
                        "dual_lr": float(dual_lr_value),
                        "total_steps": int(rescue_steps),
                        "night_budget": None if night_budget is None else float(night_budget),
                        "over_budget": None if over_budget is None else float(over_budget),
                    }
                )

        if not candidate_runs:
            continue

        selected_rescue = select_lagrangian_candidate(candidate_runs)
        rescue_policy = selected_rescue["policy"]
        rescue_validation_metrics = _evaluate_policy_on_seed_list(
            rescue_policy,
            cfg,
            catalog,
            cards,
            episode_seeds=confirm_val_episode_seeds,
            label=f"Rescue LagPPO validation (seed {int(train_seed)})",
        )
        rescue_status = validation_status(
            rescue_validation_metrics,
            night_budget=selected_rescue["night_budget"],
            over_budget=selected_rescue["over_budget"],
        )
        rescue_test_metrics = _evaluate_policy_on_seed_list(
            rescue_policy,
            cfg,
            catalog,
            cards,
            episode_seeds=test_episode_seeds,
            label=f"Rescue LagPPO test (seed {int(train_seed)})",
        )
        rescue_row = {
            "phase": "rescue_selected",
            "train_seed": int(train_seed),
            "candidate_runs": int(len(candidate_runs)),
            "feasible_candidate_runs": int(
                sum(bool(run.get("train_summary", {}).get("selected_feasible", False)) for run in candidate_runs)
            ),
            "no_feasible_candidate_runs": int(
                sum(not bool(run.get("train_summary", {}).get("selected_feasible", False)) for run in candidate_runs)
            ),
            "all_scales_infeasible": bool(
                not any(bool(run.get("train_summary", {}).get("selected_feasible", False)) for run in candidate_runs)
            ),
            "selected_operating_point_infeasible_warning": bool(not rescue_status["feasible"]),
            "chosen_scale": float(selected_rescue["scale"]),
            "dual_lr": float(selected_rescue["dual_lr"]),
            "total_steps": int(selected_rescue["total_steps"]),
            "cost_normalization": str(cost_normalization),
            "selected_source": str(selected_rescue.get("train_summary", {}).get("selected_source", "")),
            "selected_feasible": bool(rescue_status["feasible"]),
            "selected_global_step": int(selected_rescue.get("train_summary", {}).get("selected_global_step") or 0),
            "validation_episode_count": len(confirm_val_episode_seeds),
            "validation_CumWatch": float(rescue_validation_metrics["CumWatch"]),
            "validation_OverCapMinutes": float(rescue_validation_metrics["OverCapMinutes"]),
            "validation_CVaR_0.95(L)": float(rescue_validation_metrics["CVaR_0.95(L)"]),
            "validation_total_violation": float(rescue_status["total_violation"]),
            "confirmation_only_feasible": bool(
                rescue_status["feasible"]
                and not any(bool(run.get("train_summary", {}).get("selected_feasible", False)) for run in candidate_runs)
            ),
            "test_episode_count": len(test_episode_seeds),
            "test_CumWatch": float(rescue_test_metrics["CumWatch"]),
            "test_OverCapMinutes": float(rescue_test_metrics["OverCapMinutes"]),
            "test_CVaR_0.95(L)": float(rescue_test_metrics["CVaR_0.95(L)"]),
            "constraint_channels": str(constraint_track_spec.get("channels_label", "none")),
        }
        summary_rows.append(rescue_row)
        rescue_rows.append(
            {
                **rescue_row,
                "current_test_CumWatch": float(current_test_metrics["CumWatch"]),
                "current_test_OverCapMinutes": float(current_test_metrics["OverCapMinutes"]),
                "current_test_CVaR_0.95(L)": float(current_test_metrics["CVaR_0.95(L)"]),
                "current_validation_total_violation": float(current_selected_validation.get("total_violation", float("nan"))),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["train_seed", "phase"]).reset_index(drop=True)
    summary_path = bundle_dir / "lagppo_feasibility_summary.csv"
    save_csv(summary_df, summary_path)
    log_artifact_written(summary_path, "diagnostic_table_csv")

    total_runs = int(run_log_summary.get("total_candidate_runs", 0))
    total_no_feasible = int(run_log_summary.get("total_no_feasible_candidate_runs", 0))
    total_feasible = int(run_log_summary.get("total_feasible_candidate_runs", 0))
    chosen_scale_rows = [
        row
        for row in summary_rows
        if str(row.get("phase")) == "current_selected"
    ]
    chosen_scale_note = ", ".join(
        f"seed {int(row['train_seed'])}: scale {float(row['chosen_scale']):.2f} ({'feasible' if bool(row['selected_feasible']) else 'infeasible'})"
        for row in chosen_scale_rows
    )
    rescue_success = bool(rescue_rows) and all(bool(row.get("selected_feasible", False)) for row in rescue_rows)
    rescue_clean = rescue_success and all(int(row.get("feasible_candidate_runs", 0)) > 0 for row in rescue_rows)
    if rescue_rows:
        rescue_bits = []
        for row in rescue_rows:
            rescue_bits.append(
                "seed {seed}: scale {scale:.2f}, dual_lr={dual_lr:.4f}, validation total_violation={violation:.3f}, "
                "confirmation_only_feasible={confirm_only}, test CumWatch {watch:.1f} vs current {current_watch:.1f}, "
                "test OverCapMinutes {over:.2f} vs current {current_over:.2f}".format(
                    seed=int(row["train_seed"]),
                    scale=float(row["chosen_scale"]),
                    dual_lr=float(row["dual_lr"]),
                    violation=float(row["validation_total_violation"]),
                    confirm_only=bool(row.get("confirmation_only_feasible", False)),
                    watch=float(row["test_CumWatch"]),
                    current_watch=float(row["current_test_CumWatch"]),
                    over=float(row["test_OverCapMinutes"]),
                    current_over=float(row["current_test_OverCapMinutes"]),
                )
            )
        rescue_summary = "; ".join(rescue_bits)
    else:
        rescue_summary = "No focused rescue rows were executed."

    if rescue_clean:
        decision = (
            "Focused rescue produced feasible checkpoints for every targeted failing seed on the full confirmation validation set. "
            "That is enough to keep constrained RL in the paper, but only as a qualified result tied to the rescued recipe rather than as a clean win of the original released sweep."
        )
    elif rescue_success:
        decision = (
            "Focused rescue moved the failing seed across the feasibility boundary only on the larger confirmation pass, not on the actual selection pass. "
            "That is still too fragile to market constrained RL as a strong positive finding. The paper should therefore demote Lagrangian PPO from the main empirical claim and treat the rescue as a sensitivity note showing that the seed was near-feasible under more conservative tuning."
        )
    else:
        decision = (
            "Focused rescue did not repair the validation story cleanly enough to support a positive method claim. "
            "Lagrangian PPO should therefore be framed as a constrained baseline / negative result, while the main empirical story stays with the session-cap fragmentation audit and the released feasibility diagnostics."
        )

    memo_lines = [
        "# LagPPO rescue decision",
        "",
        "Current saved run summary:",
        "",
        (
            f"The saved `run.log` records {total_runs} Lagrangian PPO candidate trainings across the three seeds. "
            f"{total_no_feasible} of those {total_runs} candidates ended with `no feasible validation checkpoint`, leaving only "
            f"{total_feasible} candidate trainings that ever selected a feasible checkpoint. The final chosen operating points were: {chosen_scale_note}."
        ),
        "",
        "Focused rescue sweep:",
        "",
        (
            f"The rescue rerun targeted train seed(s) {', '.join(str(int(seed)) for seed in target_train_seeds)} with a low-cost sweep over "
            f"`scale in {{{', '.join(f'{float(value):.2f}' for value in sorted({float(v) for v in rescue_scales}))}}}` and "
            f"`dual_lr in {{{', '.join(f'{float(value):.4f}' for value in sorted({float(v) for v in rescue_dual_lrs}))}}}`, "
            f"using budget-normalized dual updates, {int(rescue_steps)} training steps per candidate, a {len(rescue_val_episode_seeds)}-episode selection validation set, "
            f"and a {len(confirm_val_episode_seeds)}-episode confirmation validation pass."
        ),
        "",
        rescue_summary + ".",
        "",
        "Decision:",
        "",
        decision,
    ]
    if not frontier_lag_row.empty and not frontier_ppo_row.empty:
        lag_row = frontier_lag_row.iloc[0]
        ppo_row = frontier_ppo_row.iloc[0]
        memo_lines.extend(
            [
                "",
                (
                    f"For reference, the released aggregate frontier still shows only a modest average tradeoff "
                    f"(`LagrangianPPO`: CumWatch {float(lag_row['CumWatch']):.1f}, OverCapMinutes {float(lag_row['OverCapMinutes']):.2f}, "
                    f"CVaR_0.95(L) {float(lag_row['CVaR_0.95(L)']):.2f}; `PPO`: CumWatch {float(ppo_row['CumWatch']):.1f}, "
                    f"OverCapMinutes {float(ppo_row['OverCapMinutes']):.2f}, CVaR_0.95(L) {float(ppo_row['CVaR_0.95(L)']):.2f}). "
                    f"Given the validation instability above, those aggregate gains are not clean enough to carry the main empirical claim by themselves."
                ),
            ]
        )
    memo_path = bundle_dir / "lagppo_rescue_decision.md"
    memo_path.write_text("\n".join(memo_lines) + "\n", encoding="utf-8")
    log_artifact_written(memo_path, "diagnostic_memo_markdown")

    manifest["lagppo_feasibility_summary"] = summary_path.name
    manifest["lagppo_rescue_decision"] = memo_path.name
    json_dumps(manifest, manifest_path)
    log_artifact_written(manifest_path, "manifest_json")

    return {
        "summary_df": summary_df,
        "summary_path": summary_path,
        "memo_path": memo_path,
        "rescue_success": rescue_success,
        "run_log_summary": run_log_summary,
    }


def round_to_nearest(value: float, step: float) -> float:
    if step <= 0.0:
        return float(value)
    return float(step * round(float(value) / step))


def rounded_positive_quantile(values: Sequence[float], q: float, step: float = 5.0, minimum: float = 5.0) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot derive a threshold from an empty distribution.")
    return float(max(float(minimum), round_to_nearest(float(np.quantile(arr, float(q))), float(step))))


def infer_threshold_source_from_payload(thresholds: Optional[Dict[str, Any]]) -> str:
    thresholds = thresholds or {}
    explicit = str(thresholds.get("threshold_source", "")).strip().lower()
    if explicit in THRESHOLD_SOURCE_CHOICES:
        return explicit
    source = str(thresholds.get("source", "")).strip().lower()
    if source in THRESHOLD_SOURCE_CHOICES:
        return source
    if "calibrated_reference_simulation" in source:
        return "simulator_relative"
    if "public_log_target" in source or "empirical_target" in source:
        return "empirical_target"
    return DEFAULT_THRESHOLD_SOURCE


def fixed_paper_thresholds(
    *,
    selected_delta_sess: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "threshold_source": "fixed_paper",
        "source": "fixed_paper",
        "selected_delta_sess": None if selected_delta_sess is None else float(selected_delta_sess),
        "rounding_step_minutes": None,
        "raw_session_length_quantiles": {},
        "raw_session_item_count_quantiles": {},
        "T_ref": float(FIXED_PAPER_THRESHOLDS["T_ref"]),
        "T_cap": float(FIXED_PAPER_THRESHOLDS["T_cap"]),
        "cap_grid": [float(x) for x in FIXED_PAPER_THRESHOLDS["cap_grid"]],
        "break_T": float(FIXED_PAPER_THRESHOLDS["break_T"]),
        "break_J": int(FIXED_PAPER_THRESHOLDS["break_J"]),
    }


def derive_paper_thresholds_from_reference_distribution(
    reference_metrics: Dict[str, Any],
    *,
    threshold_source: str,
    source_label: str,
    selected_delta_sess: Optional[float] = None,
    step: float = 5.0,
) -> Dict[str, Any]:
    session_lengths = list(map(float, reference_metrics.get("session_lengths", [])))
    session_item_counts = list(map(float, reference_metrics.get("session_item_counts", [])))
    if not session_lengths:
        raise ValueError("Reference metrics are missing session_lengths for threshold derivation.")
    if not session_item_counts:
        raise ValueError("Reference metrics are missing session_item_counts for threshold derivation.")

    raw_session_quantiles = {
        "p80": float(np.quantile(np.asarray(session_lengths, dtype=np.float64), 0.80)),
        "p90": float(np.quantile(np.asarray(session_lengths, dtype=np.float64), 0.90)),
        "p95": float(np.quantile(np.asarray(session_lengths, dtype=np.float64), 0.95)),
        "p99": float(np.quantile(np.asarray(session_lengths, dtype=np.float64), 0.99)),
    }
    raw_item_quantiles = {
        "p80": float(np.quantile(np.asarray(session_item_counts, dtype=np.float64), 0.80)),
    }
    rounded_session_quantiles = {
        label: rounded_positive_quantile(session_lengths, float(percent) / 100.0, step=step, minimum=step)
        for label, percent in [("p80", 80), ("p90", 90), ("p95", 95), ("p99", 99)]
    }
    rounded_item_quantiles = {
        "p80": int(max(1.0, round_to_nearest(raw_item_quantiles["p80"], step))),
    }
    # Keep the low cap near the long-session tail. If the rounded p80 is already within
    # two grid steps of T_ref, use it; otherwise anchor the cap grid around {p90, p95, p99}.
    use_p80_low_cap = bool((rounded_session_quantiles["p95"] - rounded_session_quantiles["p80"]) <= 2.0 * float(step))
    cap_grid = sorted(
        set(
            [
                float(rounded_session_quantiles["p80"] if use_p80_low_cap else rounded_session_quantiles["p90"]),
                float(rounded_session_quantiles["p90"] if use_p80_low_cap else rounded_session_quantiles["p95"]),
                float(rounded_session_quantiles["p99"]),
            ]
        )
    )
    return {
        "threshold_source": str(threshold_source),
        "source": str(source_label),
        "selected_delta_sess": None if selected_delta_sess is None else float(selected_delta_sess),
        "rounding_step_minutes": float(step),
        "raw_session_length_quantiles": raw_session_quantiles,
        "raw_session_item_count_quantiles": raw_item_quantiles,
        "T_ref": float(rounded_session_quantiles["p95"]),
        "T_cap": float(rounded_session_quantiles["p95"]),
        "cap_grid": [float(x) for x in cap_grid],
        "break_T": float(rounded_session_quantiles["p80"]),
        "break_J": int(rounded_item_quantiles["p80"]),
    }


def resolve_threshold_target_from_payload(
    payload: Dict[str, Any],
    *,
    selected_delta_sess: Optional[float] = None,
) -> Dict[str, Any]:
    delta = resolve_selected_delta_sess(payload, explicit_delta_sess=selected_delta_sess)
    targets_by_delta = coerce_targets_by_delta(payload.get("targets", payload.get("targets_by_delta", {})))
    if float(delta) not in targets_by_delta:
        raise ValueError(f"Calibration payload is missing targets for delta_sess={delta}.")
    return targets_by_delta[float(delta)]


def resolve_thresholds_for_source(
    threshold_source: str,
    *,
    selected_delta_sess: Optional[float] = None,
    reference_targets: Optional[Dict[str, Any]] = None,
    sim_targets: Optional[Dict[str, Any]] = None,
    calibration_status: Optional[str] = None,
) -> Dict[str, Any]:
    threshold_source = str(threshold_source).strip().lower()
    if threshold_source not in THRESHOLD_SOURCE_CHOICES:
        raise ValueError(f"Unknown threshold_source={threshold_source!r}. Expected one of {THRESHOLD_SOURCE_CHOICES}.")
    if threshold_source == "fixed_paper":
        return fixed_paper_thresholds(selected_delta_sess=selected_delta_sess)
    if threshold_source == "empirical_target":
        if reference_targets is None:
            raise ValueError("threshold_source='empirical_target' requires public-log target statistics.")
        return derive_paper_thresholds_from_reference_distribution(
            reference_targets,
            threshold_source="empirical_target",
            source_label="public_log_target_statistics",
            selected_delta_sess=selected_delta_sess,
        )
    if calibration_status != "passed":
        raise ValueError(
            "threshold_source='simulator_relative' requires calibration status 'passed'; "
            f"got {calibration_status!r}."
        )
    if sim_targets is None:
        raise ValueError("threshold_source='simulator_relative' requires calibrated simulator statistics.")
    return derive_paper_thresholds_from_reference_distribution(
        sim_targets,
        threshold_source="simulator_relative",
        source_label="calibrated_reference_simulation",
        selected_delta_sess=selected_delta_sess,
    )


def apply_paper_thresholds(cfg: BenchConfig, thresholds: Dict[str, Any]) -> BenchConfig:
    cfg.T_ref = float(thresholds["T_ref"])
    cfg.T_cap = float(thresholds.get("T_cap", thresholds["T_ref"]))
    cfg.break_T = float(thresholds["break_T"])
    cfg.break_J = int(thresholds["break_J"])
    cfg.threshold_source = infer_threshold_source_from_payload(thresholds)
    cfg.paper_thresholds = copy.deepcopy(thresholds)
    return cfg


def resolved_paper_thresholds(
    cfg: BenchConfig,
    *,
    fallback_cap_grid: Optional[Sequence[float]] = None,
    selected_delta_sess: Optional[float] = None,
) -> Dict[str, Any]:
    thresholds = copy.deepcopy(cfg.paper_thresholds or {})
    candidate_cap_grid = thresholds.get("cap_grid", fallback_cap_grid if fallback_cap_grid is not None else [cfg.T_cap])
    cap_grid = sorted(set(float(value) for value in candidate_cap_grid))
    if not cap_grid:
        cap_grid = [float(cfg.T_cap)]
    thresholds.setdefault("threshold_source", str(getattr(cfg, "threshold_source", DEFAULT_THRESHOLD_SOURCE)))
    thresholds.setdefault("source", "config_fields")
    thresholds.setdefault("selected_delta_sess", selected_delta_sess)
    thresholds.setdefault("rounding_step_minutes", 5.0)
    thresholds["T_ref"] = float(thresholds.get("T_ref", cfg.T_ref))
    thresholds["T_cap"] = float(thresholds.get("T_cap", thresholds["T_ref"]))
    thresholds["cap_grid"] = [float(value) for value in cap_grid]
    thresholds["break_T"] = float(thresholds.get("break_T", cfg.break_T))
    thresholds["break_J"] = int(round(float(thresholds.get("break_J", cfg.break_J))))
    return thresholds


def threshold_table_dataframe(thresholds: Dict[str, Any]) -> pd.DataFrame:
    threshold_source = infer_threshold_source_from_payload(thresholds)
    rule_prefix = {
        "fixed_paper": "prespecified paper default",
        "empirical_target": "round_to_nearest(target",
        "simulator_relative": "round_to_nearest(simulator",
    }[threshold_source]
    if threshold_source == "fixed_paper":
        t_ref_rule = "prespecified paper default (120 minutes)"
        t_cap_rule = "prespecified paper default (120 minutes)"
        cap_grid_rule = "prespecified paper grid [90, 120, 150]"
        break_t_rule = "prespecified paper default (30 minutes)"
        break_j_rule = "prespecified paper default (20 items)"
    else:
        t_ref_rule = f"{rule_prefix} p95_session_length, 5)"
        t_cap_rule = "same as T_ref"
        cap_grid_rule = (
            f"adaptive rounded session-length cap grid: use {{p80, p90, p99}} when rounded p80 is within 10 minutes "
            f"of rounded p95; otherwise use {{p90, p95, p99}}"
        )
        break_t_rule = f"{rule_prefix} p80_session_length, 5)"
        break_j_rule = f"{rule_prefix} p80_session_item_count, 5)"
    rows = [
        {
            "parameter": "threshold_source",
            "value": threshold_source,
            "rule": "threshold derivation policy used for this run",
            "threshold_source": threshold_source,
            "source": str(thresholds.get("source", "")),
            "selected_delta_sess": thresholds.get("selected_delta_sess"),
        },
        {
            "parameter": "T_ref",
            "value": float(thresholds["T_ref"]),
            "rule": t_ref_rule,
            "threshold_source": threshold_source,
            "source": str(thresholds.get("source", "")),
            "selected_delta_sess": thresholds.get("selected_delta_sess"),
        },
        {
            "parameter": "T_cap_default",
            "value": float(thresholds.get("T_cap", thresholds["T_ref"])),
            "rule": t_cap_rule,
            "threshold_source": threshold_source,
            "source": str(thresholds.get("source", "")),
            "selected_delta_sess": thresholds.get("selected_delta_sess"),
        },
        {
            "parameter": "cap_grid",
            "value": ", ".join(f"{float(value):g}" for value in thresholds.get("cap_grid", [])),
            "rule": cap_grid_rule,
            "threshold_source": threshold_source,
            "source": str(thresholds.get("source", "")),
            "selected_delta_sess": thresholds.get("selected_delta_sess"),
        },
        {
            "parameter": "break_T",
            "value": float(thresholds["break_T"]),
            "rule": break_t_rule,
            "threshold_source": threshold_source,
            "source": str(thresholds.get("source", "")),
            "selected_delta_sess": thresholds.get("selected_delta_sess"),
        },
        {
            "parameter": "break_J",
            "value": int(thresholds["break_J"]),
            "rule": break_j_rule,
            "threshold_source": threshold_source,
            "source": str(thresholds.get("source", "")),
            "selected_delta_sess": thresholds.get("selected_delta_sess"),
        },
    ]
    return pd.DataFrame(rows)


def discover_calibration_payload_path(
    *,
    config_json: Optional[str] = None,
    calibration_payload_json: Optional[Path] = None,
) -> Optional[Path]:
    if calibration_payload_json is not None:
        path = Path(calibration_payload_json)
        return path if path.exists() else None
    if config_json:
        sibling = Path(config_json).resolve().with_name("calibration_payload.json")
        if sibling.exists():
            return sibling
    return None


def assert_paper_threshold_channels_active(
    main_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    break_prompt_df: pd.DataFrame,
) -> None:
    if main_df.empty:
        raise AssertionError("Paper threshold invariants require non-empty main policy metrics.")
    main_block = main_df[main_df["policy"].isin(["Random", "Myopic", "PPO", "Lagrangian PPO"])].copy()
    if main_block.empty:
        raise AssertionError("Paper threshold invariants could not find the main policy rows.")
    if all(abs(float(value)) <= 1e-12 for value in main_block["OverCapMinutes"].astype(float).tolist()):
        raise AssertionError("All main policies have OverCapMinutes == 0 under the paper config.")

    if cap_df.empty:
        raise AssertionError("Paper threshold invariants require non-empty cap sensitivity metrics.")
    if all(abs(float(value)) <= 1e-12 for value in cap_df["SessionCapTriggerRate"].astype(float).tolist()):
        raise AssertionError("All cap settings have SessionCapTriggerRate == 0 under the paper config.")
    if len(cap_df) > 1:
        metric_cols = [col for col in ["CumWatch", "CVaR_0.95(L)", "OverCapMinutes", "SessionCapTriggerRate"] if col in cap_df.columns]
        if metric_cols:
            distinct_rows = {
                tuple(round(float(row[col]), 8) for col in metric_cols)
                for _, row in cap_df.sort_values("T_cap").iterrows()
            }
            if len(distinct_rows) <= 1:
                raise AssertionError("All cap sensitivity rows are identical across cap values.")
        tightest_cap_row = cap_df.sort_values("T_cap").iloc[0]
        ppo_rows = main_df[main_df["policy"] == "PPO"]
        if ppo_rows.empty:
            raise AssertionError("Paper threshold invariants require a PPO row for cap comparison.")
        ppo_row = ppo_rows.iloc[0]
        over_improves = float(tightest_cap_row["OverCapMinutes"]) < float(ppo_row["OverCapMinutes"]) - 1e-12
        cvar_improves = float(tightest_cap_row["CVaR_0.95(L)"]) < float(ppo_row["CVaR_0.95(L)"]) - 1e-12
        if not (over_improves or cvar_improves):
            raise AssertionError("The tightest cap must improve OverCapMinutes or CVaR_0.95(L) relative to PPO.")

    if break_prompt_df.empty:
        raise AssertionError("Paper threshold invariants require a break-prompt evaluation.")
    break_row = break_prompt_df.iloc[0]
    if int(round(float(break_row.get("BreakAdherence_den", 0.0)))) <= 0:
        raise AssertionError("Break-prompt evaluation has BreakAdherence_den == 0 under the paper config.")
    if not math.isfinite(float(break_row.get("BreakAdherence", float("nan")))):
        raise AssertionError("Break-prompt evaluation must produce finite BreakAdherence.")


def overcap_minutes_for_episode_sessions(session_lengths: Sequence[float], t_ref: float) -> float:
    return float(sum(max(float(length) - float(t_ref), 0.0) for length in session_lengths))


def aggregate_policy_diagnostics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    all_sessions: List[float] = []
    episode_session_lengths: List[List[float]] = []
    episode_over: List[float] = []
    episode_night: List[float] = []
    episode_night_fraction: List[float] = []
    episode_late_night_session_start_rate: List[float] = []
    episode_cumwatch: List[float] = []
    num_sessions = 0
    num_episodes = 0
    session_cap_triggers = 0.0
    episode_cap_triggers = 0.0
    for result in results:
        all_sessions.extend(list(map(float, result.get("SessionLengths", []))))
        episode_session_lengths.extend([list(map(float, xs)) for xs in result.get("EpisodeSessionLengths", [])])
        episode_over.extend(list(map(float, result.get("EpisodeOverCapValues", []))))
        episode_night.extend(list(map(float, result.get("EpisodeNightValues", []))))
        episode_night_fraction.extend(list(map(float, result.get("EpisodeNightFractionValues", []))))
        episode_late_night_session_start_rate.extend(list(map(float, result.get("EpisodeLateNightSessionStartRateValues", []))))
        episode_cumwatch.extend(list(map(float, result.get("EpisodeCumWatchValues", []))))
        num_sessions += int(result.get("NumSessions", len(result.get("SessionLengths", []))))
        num_episodes += int(result.get("NumEpisodes", len(result.get("EpisodeOverCapValues", []))))
        session_cap_triggers += float(result.get("SessionCapTriggerCount", 0))
        episode_cap_triggers += float(result.get("EpisodeSessionCapTriggerRate", 0.0)) * float(result.get("NumEpisodes", 0))
    return {
        "session_lengths": all_sessions,
        "episode_session_lengths": episode_session_lengths,
        "episode_overcap_values": episode_over,
        "episode_night_values": episode_night,
        "episode_night_fraction_values": episode_night_fraction,
        "episode_late_night_session_start_rate_values": episode_late_night_session_start_rate,
        "episode_cumwatch_values": episode_cumwatch,
        "num_sessions": int(num_sessions),
        "num_episodes": int(num_episodes),
        "session_cap_trigger_count": float(session_cap_triggers),
        "episode_cap_trigger_count": float(episode_cap_triggers),
    }


def summarize_threshold_activity(
    policy_results: Dict[str, Sequence[Dict[str, Any]]],
    cap_results: Dict[float, Sequence[Dict[str, Any]]],
    default_t_ref: float,
    cap_grid: Sequence[float],
    exceed_thresholds: Sequence[float],
    t_ref_grid: Sequence[float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    quantiles_payload: Dict[str, Any] = {"policies": {}}
    report: Dict[str, Any] = {
        "default_T_ref": float(default_t_ref),
        "thresholds_checked": [float(x) for x in exceed_thresholds],
        "t_ref_diagnostic_grid": [float(x) for x in t_ref_grid],
        "policies": {},
        "session_cap_wrapper": {},
    }

    pooled_main_sessions: List[float] = []
    max_episode_overcap_rate_120 = 0.0
    max_session_over_120 = 0.0

    for policy_name, results in policy_results.items():
        pooled = aggregate_policy_diagnostics(results)
        sessions = pooled["session_lengths"]
        episode_sessions = pooled["episode_session_lengths"]
        episode_over = pooled["episode_overcap_values"]
        episode_cumwatch = pooled["episode_cumwatch_values"]
        episode_night = pooled["episode_night_values"]
        quantiles = quantile_summary(sessions)
        quantiles_payload["policies"][policy_name] = quantiles
        pooled_main_sessions.extend(sessions)

        fraction_sessions_exceeding = {
            str(int(thr) if float(thr).is_integer() else thr): (
                float(np.mean(np.asarray(sessions, dtype=np.float64) > float(thr))) if sessions else 0.0
            )
            for thr in exceed_thresholds
        }
        t_ref_sweep = {}
        for t_ref in t_ref_grid:
            episode_over_t = [overcap_minutes_for_episode_sessions(xs, t_ref) for xs in episode_sessions]
            t_ref_key = str(int(t_ref) if float(t_ref).is_integer() else t_ref)
            t_ref_sweep[t_ref_key] = {
                "mean_overcap_minutes": safe_mean(episode_over_t),
                "fraction_episodes_overcap_positive": float(np.mean(np.asarray(episode_over_t, dtype=np.float64) > 0.0)) if episode_over_t else 0.0,
                "fraction_sessions_exceeding": float(np.mean(np.asarray(sessions, dtype=np.float64) > float(t_ref))) if sessions else 0.0,
            }

        policy_report = {
            **quantiles,
            "num_sessions": int(pooled["num_sessions"]),
            "num_episodes": int(pooled["num_episodes"]),
            "fraction_sessions_exceeding": fraction_sessions_exceeding,
            "fraction_episodes_overcap_positive": float(np.mean(np.asarray(episode_over, dtype=np.float64) > 0.0)) if episode_over else 0.0,
            "mean_overcap_minutes": safe_mean(episode_over),
            "night_minutes_cumwatch_corr": safe_corr(episode_cumwatch, episode_night),
            "t_ref_sweep": t_ref_sweep,
        }
        report["policies"][policy_name] = policy_report
        max_episode_overcap_rate_120 = max(max_episode_overcap_rate_120, float(policy_report["t_ref_sweep"].get("120", {}).get("fraction_episodes_overcap_positive", 0.0)))
        max_session_over_120 = max(max_session_over_120, float(policy_report["fraction_sessions_exceeding"].get("120", 0.0)))

    quantiles_payload["pooled_main"] = quantile_summary(pooled_main_sessions)
    report["pooled_main_quantiles"] = dict(quantiles_payload["pooled_main"])

    session_cap_wrapper: Dict[str, Any] = {}
    for t_cap in cap_grid:
        pooled = aggregate_policy_diagnostics(cap_results.get(float(t_cap), []))
        key = str(int(t_cap) if float(t_cap).is_integer() else t_cap)
        session_cap_wrapper[key] = {
            "fraction_sessions_triggered": float(pooled["session_cap_trigger_count"]) / max(1, int(pooled["num_sessions"])),
            "fraction_episodes_triggered": float(pooled["episode_cap_trigger_count"]) / max(1, int(pooled["num_episodes"])),
            "num_sessions": int(pooled["num_sessions"]),
            "num_episodes": int(pooled["num_episodes"]),
            "trigger_count": int(round(float(pooled["session_cap_trigger_count"]))),
        }
    report["session_cap_wrapper"] = session_cap_wrapper

    pooled_quantiles = quantiles_payload["pooled_main"]
    recommended_relative_threshold = max(5.0, round_to_nearest(float(pooled_quantiles["p95"]), 5.0))
    cap120_trigger_rate = float(session_cap_wrapper.get("120", {}).get("fraction_sessions_triggered", 0.0))
    inactive = (
        max_episode_overcap_rate_120 < 0.01
        and max_session_over_120 < 0.005
        and cap120_trigger_rate < 0.005
    )
    report["binding_summary"] = {
        "max_main_policy_episode_overcap_rate_at_120": float(max_episode_overcap_rate_120),
        "max_main_policy_session_fraction_over_120": float(max_session_over_120),
        "session_cap_trigger_fraction_at_120": float(cap120_trigger_rate),
        "fixed_120_effectively_inactive": bool(inactive),
        "recommended_relative_threshold_minutes": float(recommended_relative_threshold),
        "recommended_relative_threshold_rule": "pooled calibrated p95 session length" if inactive else None,
    }
    return quantiles_payload, report


def render_120_minutes_memo(report: Dict[str, Any]) -> str:
    pooled = report.get("binding_summary", {})
    policy_lines = []
    for policy_name, stats in report.get("policies", {}).items():
        policy_lines.append(
            f"- {policy_name}: p95={stats.get('p95', 0.0):.1f} min, "
            f"frac(session>120)={stats.get('fraction_sessions_exceeding', {}).get('120', 0.0):.4f}, "
            f"frac(episode overcap>0 @120)={stats.get('t_ref_sweep', {}).get('120', {}).get('fraction_episodes_overcap_positive', 0.0):.4f}, "
            f"corr(NightMinutes,CumWatch)={stats.get('night_minutes_cumwatch_corr', float('nan')):.3f}"
        )

    if pooled.get("fixed_120_effectively_inactive", False):
        conclusion = (
            f"Recommendation: fixed 120-minute thresholds are effectively inactive in the calibrated default. "
            f"Replace the fixed threshold with a benchmark-relative rule set to the pooled calibrated p95 session length "
            f"({pooled.get('recommended_relative_threshold_minutes', 0.0):.1f} minutes, rounded to the nearest 5 minutes)."
        )
    else:
        conclusion = (
            "Conclusion: the calibrated default activates the 120-minute regime often enough that fixed thresholds remain behaviorally meaningful."
        )

    lines = [
        "# Does 120 minutes bind?",
        "",
        "This memo summarizes the calibrated-default threshold diagnostics computed from the main policy evaluations.",
        "",
        "## Pooled calibrated session-length quantiles",
    ]
    quant_pooled = report.get("pooled_main_quantiles", {})
    if quant_pooled:
        lines.extend(
            [
                f"- mean={quant_pooled.get('mean', 0.0):.1f} min",
                f"- median={quant_pooled.get('median', 0.0):.1f} min",
                f"- p90={quant_pooled.get('p90', 0.0):.1f} min",
                f"- p95={quant_pooled.get('p95', 0.0):.1f} min",
                f"- p99={quant_pooled.get('p99', 0.0):.1f} min",
                f"- max={quant_pooled.get('max', 0.0):.1f} min",
            ]
        )
    lines.extend(
        [
            "",
            "## Policy-level binding checks",
            *policy_lines,
            "",
            "## 120-minute activity summary",
            f"- max episode over-cap rate at 120 across main policies: {pooled.get('max_main_policy_episode_overcap_rate_at_120', 0.0):.4f}",
            f"- max session fraction above 120 across main policies: {pooled.get('max_main_policy_session_fraction_over_120', 0.0):.4f}",
            f"- session-cap trigger fraction at T_cap=120: {pooled.get('session_cap_trigger_fraction_at_120', 0.0):.4f}",
            "",
            conclusion,
            "",
            "The NightMinutes/CumWatch correlations above are included to check that NightMinutes is not behaving like a trivial rescaling of total watch time after empirical session-start calibration.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_training_history(rows: Sequence[Dict[str, Any]], outpath: Path) -> None:
    pd.DataFrame(list(rows)).to_csv(outpath, index=False)
    log_artifact_written(outpath, "training_history_csv")


def render_random_vs_ppo_memo(baseline_df: pd.DataFrame) -> str:
    if baseline_df.empty:
        return "# Why Random Beats PPO\n\nNo baseline-expansion rows were available.\n"

    test_block = baseline_df[baseline_df["eval_split"] == "test"].copy()
    if test_block.empty:
        test_block = baseline_df.copy()

    random_row = test_block[test_block["policy"] == "Random"]
    random_cumwatch = float(random_row.iloc[0]["CumWatch"]) if not random_row.empty else float("nan")
    better_than_random = test_block[test_block["CumWatch"].astype(float) > random_cumwatch].copy() if math.isfinite(random_cumwatch) else pd.DataFrame()

    best_row = test_block.sort_values("CumWatch", ascending=False).iloc[0]
    best_heuristic = test_block[test_block["category"] == "heuristic"].sort_values("CumWatch", ascending=False)
    ppo_family_categories = ["ppo_main", "ppo_sweep", "factorized_ppo", "lagrangian", "lagrangian_sweep"]
    best_ppo_variant = test_block[test_block["category"].isin(ppo_family_categories)].sort_values("CumWatch", ascending=False)

    lines = [
        "# Why Random Beats PPO",
        "",
        "This memo summarizes the baseline-expansion diagnostics from the current run.",
        "",
        "## Test-set ranking",
        f"- Top policy: {best_row['policy']} ({best_row['eval_mode']}) with CumWatch={float(best_row['CumWatch']):.3f} and RepeatRate={float(best_row['RepeatRate']):.3f}.",
    ]
    if math.isfinite(random_cumwatch):
        lines.append(f"- Random reference CumWatch={random_cumwatch:.3f}.")
    if not best_heuristic.empty:
        row = best_heuristic.iloc[0]
        lines.append(f"- Best simple heuristic: {row['policy']} with CumWatch={float(row['CumWatch']):.3f}.")
    if not best_ppo_variant.empty:
        row = best_ppo_variant.iloc[0]
        lines.append(f"- Best PPO-family variant: {row['policy']} with CumWatch={float(row['CumWatch']):.3f}.")

    lines.extend(["", "## Interpretation"])
    if not better_than_random.empty:
        top_better = better_than_random.sort_values("CumWatch", ascending=False).iloc[0]
        lines.append(
            f"- At least one non-random policy beats Random on CumWatch: {top_better['policy']} ({top_better['eval_mode']}) reaches {float(top_better['CumWatch']):.3f}."
        )
    else:
        lines.append("- No PPO-family or heuristic policy beat Random on CumWatch in this run.")
        lines.append("- The paper should not claim monotone improvement by policy sophistication unless a stronger variant beats Random in the calibrated regime.")

    deterministic = test_block[test_block["eval_mode"] == "deterministic"].sort_values("CumWatch", ascending=False)
    stochastic = test_block[test_block["eval_mode"] == "stochastic"].sort_values("CumWatch", ascending=False)
    if not deterministic.empty and not stochastic.empty:
        lines.append(
            f"- Best deterministic policy: {deterministic.iloc[0]['policy']} at {float(deterministic.iloc[0]['CumWatch']):.3f}; best stochastic policy: {stochastic.iloc[0]['policy']} at {float(stochastic.iloc[0]['CumWatch']):.3f}."
        )

    return "\n".join(lines) + "\n"


def build_diversity_diagnostic_table(
    baseline_expansion_df: pd.DataFrame,
    frontier_validation_df: pd.DataFrame,
) -> pd.DataFrame:
    if baseline_expansion_df.empty:
        return pd.DataFrame()

    test_block = baseline_expansion_df[baseline_expansion_df["eval_split"] == "test"].copy()
    if test_block.empty:
        return pd.DataFrame()

    direct_block = test_block[test_block["policy"].isin(set(DIVERSITY_DIAGNOSTIC_POLICY_ORDER) - {"Lagrangian PPO"})].copy()
    lag_block = pd.DataFrame()
    if not frontier_validation_df.empty:
        selected_lag = frontier_validation_df[
            (frontier_validation_df["method"] == "LagrangianPPO")
            & (frontier_validation_df["selected_operating_point"].astype(bool))
        ][["train_seed", "scale"]].drop_duplicates()
        if not selected_lag.empty:
            selected_lag = selected_lag.copy()
            selected_lag["policy"] = selected_lag["scale"].map(lambda value: f"LagPPO(scale={float(value)})")
            lag_block = test_block.merge(
                selected_lag[["train_seed", "policy"]],
                on=["train_seed", "policy"],
                how="inner",
            )
            if not lag_block.empty:
                lag_block = lag_block.copy()
                lag_block["policy"] = "Lagrangian PPO"

    combined = pd.concat([direct_block, lag_block], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    combined = combined[
        [
            "policy",
            "eval_mode",
            "train_seed",
            "CumWatch",
            "CVaR_0.95(L)",
            "RepeatRate",
            "ActionEmpiricalEntropy",
            "UniqueClusterCount",
            "FractionNu1",
        ]
    ].copy()
    combined = combined.drop_duplicates(subset=["policy", "eval_mode", "train_seed"])
    aggregated = aggregate_across_train_seeds(combined.to_dict("records"), ["policy", "eval_mode"])
    if aggregated.empty:
        return pd.DataFrame()
    aggregated["policy_order"] = aggregated["policy"].map({name: idx for idx, name in enumerate(DIVERSITY_DIAGNOSTIC_POLICY_ORDER)})
    aggregated["eval_mode_order"] = aggregated["eval_mode"].map({name: idx for idx, name in enumerate(EVAL_MODE_ORDER)})
    table = aggregated[
        [
            "policy",
            "eval_mode",
            "CumWatch",
            "CVaR_0.95(L)",
            "RepeatRate",
            "ActionEmpiricalEntropy",
            "UniqueClusterCount",
            "FractionNu1",
            "policy_order",
            "eval_mode_order",
        ]
    ].sort_values(["eval_mode_order", "policy_order", "policy"]).drop(columns=["policy_order", "eval_mode_order"])
    return table.reset_index(drop=True)


def render_diversity_dominate_personalization_memo(
    diversity_df: pd.DataFrame,
    threshold_payload: Dict[str, Any],
) -> str:
    if diversity_df.empty:
        return "# Does Diversity Dominate Personalization?\n\nNo diversity-diagnostic rows were available.\n"

    lines = [
        "# Does Diversity Dominate Personalization?",
        "",
        "This memo summarizes the matched-mode diversity diagnostic from the repaired benchmark configuration.",
        "",
        "## Repaired benchmark context",
        f"- `T_ref={float(threshold_payload.get('T_ref', float('nan'))):.1f}`",
        f"- `cap_grid={threshold_payload.get('cap_grid', [])}`",
        f"- `break_T={float(threshold_payload.get('break_T', float('nan'))):.1f}`",
        f"- `break_J={int(threshold_payload.get('break_J', 0) or 0)}`",
        "- Deterministic and stochastic results are reported separately; Random is not mixed into the deterministic comparison.",
        "",
    ]

    deterministic_survives = False
    stochastic_survives = False
    for eval_mode in EVAL_MODE_ORDER:
        mode_block = diversity_df[diversity_df["eval_mode"] == eval_mode].copy()
        if mode_block.empty:
            continue
        diversity_block = mode_block[mode_block["policy"].isin(DIVERSITY_HEURISTIC_POLICIES)].copy()
        personalization_block = mode_block[mode_block["policy"].isin(DIVERSITY_PERSONALIZATION_POLICIES)].copy()
        repeat_corr = safe_corr(mode_block["RepeatRate"].astype(float).tolist(), mode_block["CumWatch"].astype(float).tolist())
        cluster_corr = safe_corr(mode_block["UniqueClusterCount"].astype(float).tolist(), mode_block["CumWatch"].astype(float).tolist())
        lines.extend(
            [
                f"## {eval_mode.title()} mode",
                f"- Policy count: {len(mode_block)}",
                f"- corr(CumWatch, RepeatRate) = {repeat_corr:.3f}",
                f"- corr(CumWatch, UniqueClusterCount) = {cluster_corr:.3f}",
            ]
        )
        if diversity_block.empty or personalization_block.empty:
            lines.append("- Matched-mode family comparison was unavailable because one policy family was missing.")
            lines.append("")
            continue

        best_diversity = diversity_block.sort_values("CumWatch", ascending=False).iloc[0]
        best_personalization = personalization_block.sort_values("CumWatch", ascending=False).iloc[0]
        diversity_wins = float(best_diversity["CumWatch"]) > float(best_personalization["CumWatch"]) + 1e-6
        if eval_mode == "deterministic":
            deterministic_survives = diversity_wins
        elif eval_mode == "stochastic":
            stochastic_survives = diversity_wins
        lines.extend(
            [
                f"- Best diversity-first policy: `{best_diversity['policy']}` with CumWatch={float(best_diversity['CumWatch']):.3f}, RepeatRate={float(best_diversity['RepeatRate']):.3f}, UniqueClusterCount={float(best_diversity['UniqueClusterCount']):.3f}.",
                f"- Best personalization policy: `{best_personalization['policy']}` with CumWatch={float(best_personalization['CumWatch']):.3f}, RepeatRate={float(best_personalization['RepeatRate']):.3f}, UniqueClusterCount={float(best_personalization['UniqueClusterCount']):.3f}.",
                "- Verdict: diversity-first policies outrank the personalization family on CumWatch in this matched mode."
                if diversity_wins
                else "- Verdict: the personalization family matches or exceeds the diversity-first baselines in this matched mode.",
                "",
            ]
        )

    repaired_verdict = (
        "Yes. The diversity-over-personalization effect survives the repaired benchmark in deterministic matched-mode evaluation."
        if deterministic_survives
        else "No. Once calibration and evaluation-mode matching are repaired, deterministic personalization is no longer dominated by the simple diversity heuristics."
    )
    stochastic_verdict = (
        "Yes. The same effect also appears in the stochastic comparison."
        if stochastic_survives
        else "No. The stochastic comparison does not show the same dominance once evaluated separately."
    )
    appendix_note = (
        "Recommendation: keep this as an appendix diagnostic or backup narrative."
        if deterministic_survives or stochastic_survives
        else "Recommendation: do not elevate this beyond a diagnostic note."
    )
    lines.extend(
        [
            "## Bottom line",
            f"- {repaired_verdict}",
            f"- {stochastic_verdict}",
            f"- {appendix_note}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_returnrate_sensitivity_table(main_df: pd.DataFrame) -> pd.DataFrame:
    if main_df.empty:
        return pd.DataFrame()

    ordered_cols = ["policy", "backend"]
    for metric in RETURN_RATE_METRIC_KEYS:
        ordered_cols.extend([metric, f"{metric}_ci95"])
    ordered_cols = [col for col in ordered_cols if col in main_df.columns]
    out = main_df[ordered_cols].copy()

    for metric in RETURN_RATE_METRIC_KEYS:
        delta_col = f"{metric}_delta_vs_random"
        out[delta_col] = float("nan")
        if metric not in out.columns:
            continue
        for backend_name, block in out.groupby("backend", dropna=False):
            random_block = block[block["policy"] == "Random"]
            if random_block.empty:
                continue
            random_value = float(random_block.iloc[0][metric])
            mask = out["backend"] == backend_name
            out.loc[mask, delta_col] = out.loc[mask, metric].astype(float) - random_value
    return out.sort_values(["backend", "policy"]).reset_index(drop=True)


def assess_returnrate_main_text_worthiness(main_df: pd.DataFrame) -> Dict[str, Any]:
    if main_df.empty:
        return {
            "any_returnrate_material": False,
            "returnrate60_main_text_worthy": False,
            "best_metric": None,
            "best_metric_spread": 0.0,
            "per_metric": [],
        }

    block = main_df[main_df["backend"] == "Param"].copy() if "backend" in main_df.columns else main_df.copy()
    if block.empty:
        block = main_df.copy()

    per_metric = []
    best_metric = None
    best_metric_spread = -float("inf")
    returnrate60_spread = 0.0
    for metric in RETURN_RATE_METRIC_KEYS:
        if metric not in block.columns:
            continue
        metric_block = block[["policy", metric]].dropna()
        if metric_block.empty:
            continue
        metric_block = metric_block.sort_values(metric)
        min_row = metric_block.iloc[0]
        max_row = metric_block.iloc[-1]
        spread = float(max_row[metric] - min_row[metric])
        row = {
            "metric": metric,
            "min_policy": str(min_row["policy"]),
            "min_value": float(min_row[metric]),
            "max_policy": str(max_row["policy"]),
            "max_value": float(max_row[metric]),
            "spread": spread,
        }
        per_metric.append(row)
        if spread > best_metric_spread:
            best_metric_spread = spread
            best_metric = metric
        if metric == "ReturnRate60":
            returnrate60_spread = spread

    any_material = bool(per_metric) and best_metric_spread >= RETURN_RATE_MAIN_TEXT_MIN_SPREAD
    returnrate60_main_text_worthy = returnrate60_spread >= RETURN_RATE_MAIN_TEXT_MIN_SPREAD
    return {
        "any_returnrate_material": bool(any_material),
        "returnrate60_main_text_worthy": bool(returnrate60_main_text_worthy),
        "best_metric": best_metric,
        "best_metric_spread": float(max(best_metric_spread, 0.0)),
        "returnrate60_spread": float(returnrate60_spread),
        "per_metric": per_metric,
    }


def render_returnrate_main_text_note(assessment: Dict[str, Any]) -> str:
    lines = [
        "# Is rapid re-entry main-text worthy?",
        "",
        f"Decision rule: keep `ReturnRate60` in the main scorecard only if its cross-policy spread is at least {RETURN_RATE_MAIN_TEXT_MIN_SPREAD:.2f}.",
        "",
        "## Cross-policy spread by threshold",
    ]
    for row in assessment.get("per_metric", []):
        lines.append(
            f"- {row['metric']}: spread={row['spread']:.3f} "
            f"({row['min_policy']}={row['min_value']:.3f}, {row['max_policy']}={row['max_value']:.3f})"
        )

    lines.extend(["", "## Recommendation"])
    if assessment.get("returnrate60_main_text_worthy", False):
        lines.append(
            f"- `ReturnRate60` is policy-sensitive enough for main-text reporting (spread={assessment.get('returnrate60_spread', 0.0):.3f})."
        )
    elif assessment.get("any_returnrate_material", False):
        lines.append(
            f"- Some rapid re-entry thresholds moved materially, but `ReturnRate60` did not. "
            f"The strongest signal was {assessment.get('best_metric')} with spread={assessment.get('best_metric_spread', 0.0):.3f}."
        )
        lines.append("- Keep the return-rate family in `table_returnrate_sensitivity`, but remove `ReturnRate60` from the main scorecard and main-text claims.")
    else:
        lines.append("- All return-rate thresholds remained nearly fixed across policies after the gap-model repair.")
        lines.append("- Demote rapid re-entry to appendix-only reporting and avoid using `ReturnRate60` in the main scorecard.")

    return "\n".join(lines) + "\n"


def render_old_vs_corrected_nohabit_note(nohabit_df: pd.DataFrame) -> str:
    lines = [
        "# Old vs corrected NoHabit",
        "",
        "The old `NoHabit` ablation was not faithful because it only set `gamma_h = 0` while allowing the latent habit state to persist in the dynamics.",
        "The corrected implementation now enforces `disable_habit_state = True`, which makes `h_t ≡ 0` at reset, on the continue branch, and on the stop/recovery branch.",
        "",
        "Any main-text interpretation based on the old `NoHabit` implementation should be discarded.",
    ]
    if nohabit_df.empty:
        lines.extend(
            [
                "",
                "No corrected `NoHabit` evaluation rows were available in this run.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(["", "## Corrected results from this run"])
    for condition in ["PPO (Default)", "PPO (NoHabit)", "Lagrangian PPO (Default)", "Lagrangian PPO (NoHabit)"]:
        block = nohabit_df[nohabit_df["condition"] == condition]
        if block.empty:
            continue
        row = block.iloc[0]
        fragments = [
            f"CumWatch={float(row['CumWatch']):.3f}" if "CumWatch" in row else None,
            f"CVaR_0.95(L)={float(row['CVaR_0.95(L)']):.3f}" if "CVaR_0.95(L)" in row else None,
        ]
        for metric in RETURN_RATE_METRIC_KEYS:
            if metric in row:
                fragments.append(f"{metric}={float(row[metric]):.3f}")
        lines.append(f"- {condition}: " + ", ".join(fragment for fragment in fragments if fragment is not None))

    lines.extend(
        [
            "",
            "The broader appendix ablation family (`NoVar`, `NoPers`, `HomogeneousUsers`) should be interpreted only after this corrected `NoHabit` result is in place.",
        ]
    )
    return "\n".join(lines) + "\n"


def no_test_leakage_invariant() -> None:
    source = inspect.getsource(run_paper_pipeline)
    if re.search(r"night_budget\s*=.*ppo_metrics", source):
        raise AssertionError("night_budget must not be derived from test-set PPO metrics.")
    if re.search(r"over_budget\s*=.*ppo_metrics", source):
        raise AssertionError("over_budget must not be derived from test-set PPO metrics.")
    for snippet in [
        "build_constraint_track_spec(",
        "scaled_constraint_budgets(",
    ]:
        if snippet not in source:
            raise AssertionError(f"Expected leakage guard snippet missing from run_paper_pipeline: {snippet}")


def run_invariant_smoke_tests() -> Dict[str, bool]:
    cfg = BenchConfig(T_ref=1.0, T_cap=1.0, device="cpu")
    catalog = build_catalog(cfg)
    cards = build_default_cards(cfg, metadata_csv=None, random_seed=0, paper_mode=False)
    seeds = list(range(32))

    random_policy = RandomPolicy(cfg.num_actions(), seed=0)
    low_threshold_metrics = evaluate_policy(
        random_policy,
        cfg,
        catalog,
        cards,
        episode_seeds=seeds,
        num_episodes=len(seeds),
        deterministic=False,
        label="Invariant low-threshold base",
        collect_policy_diagnostics=False,
    )
    if float(low_threshold_metrics["OverCapMinutes"]) <= 0.0:
        raise AssertionError("Low-threshold OverCapMinutes should be nonzero.")

    capped_metrics = evaluate_policy(
        random_policy,
        cfg,
        catalog,
        cards,
        wrappers={"session_cap": True, "T_cap": float(cfg.T_cap)},
        episode_seeds=seeds,
        num_episodes=len(seeds),
        deterministic=False,
        label="Invariant low-threshold capped",
        collect_policy_diagnostics=False,
    )
    if float(capped_metrics["SessionCapTriggerRate"]) <= 0.0:
        raise AssertionError("Session cap should trigger when thresholds are active.")
    if float(capped_metrics["CVaR_0.95(L)"]) > float(low_threshold_metrics["CVaR_0.95(L)"]) + 1e-8:
        raise AssertionError("Session cap should not increase CVaR_0.95(L) when thresholds are active.")

    nohabit_env = make_env(cfg, "param", catalog, cards, scorer=None, wrappers=None, seed=0, ablation="NoHabit")
    nohabit_env.reset(seed=0)
    if nohabit_env.state is None or abs(float(nohabit_env.state.h)) > 1e-12:
        raise AssertionError("NoHabit reset must initialize h_t to zero.")
    for _ in range(8):
        action = int(nohabit_env.rng.integers(0, cfg.num_actions()))
        _, _, done, _ = nohabit_env.step(action)
        if nohabit_env.state is not None and abs(float(nohabit_env.state.h)) > 1e-12:
            raise AssertionError("NoHabit must enforce h_t ≡ 0.")
        if done:
            break

    parity_env = make_env(cfg, "param", catalog, cards, scorer=None, wrappers=None, seed=7)
    parity_obs = parity_env.reset(seed=7)

    torch.manual_seed(7)
    myopic_policy = MyopicPolicy(RewardModel(cfg.obs_dim(), cfg.num_actions()), seed=7)
    myopic_slow_action, myopic_slow_info = myopic_policy.act_with_info(
        parity_obs,
        deterministic=True,
        need_info=True,
    )
    myopic_fast_action, myopic_fast_info = myopic_policy.act_with_info(
        parity_obs,
        deterministic=True,
        need_info=False,
    )
    if myopic_slow_action != myopic_fast_action:
        raise AssertionError("Myopic deterministic fast path must match the rich deterministic action.")
    if "policy_entropy" not in myopic_slow_info or myopic_fast_info:
        raise AssertionError("Myopic deterministic fast path must only skip auxiliary info.")

    torch.manual_seed(11)
    ppo_policy = PPOPolicy(build_actor_critic_model(cfg, hidden_size=64, policy_arch="flat"))
    ppo_slow_action, ppo_slow_info = ppo_policy.act_with_info(
        parity_obs,
        deterministic=True,
        need_info=True,
    )
    ppo_fast_action, ppo_fast_info = ppo_policy.act_with_info(
        parity_obs,
        deterministic=True,
        need_info=False,
    )
    if ppo_slow_action != ppo_fast_action:
        raise AssertionError("PPO deterministic fast path must match the rich deterministic action.")
    if "policy_entropy" not in ppo_slow_info or ppo_fast_info:
        raise AssertionError("PPO deterministic fast path must only skip auxiliary info.")

    llm_zero = copy.deepcopy(cfg)
    llm_zero.omega_r_llm = 0.0
    llm_zero.omega_c_llm = 0.0
    scorer = AGLLMScorer(llm_zero, cards, mode="surrogate", cache_path=None, device="cpu")
    parity_policy = RoundRobinPolicy(cfg)
    param_metrics = evaluate_policy(
        parity_policy,
        cfg,
        catalog,
        cards,
        episode_seeds=seeds,
        num_episodes=len(seeds),
        deterministic=True,
        label="Invariant AGParam parity",
        collect_policy_diagnostics=False,
    )
    llm_metrics = evaluate_policy(
        parity_policy,
        llm_zero,
        catalog,
        cards,
        backend="llm",
        scorer=scorer,
        episode_seeds=seeds,
        num_episodes=len(seeds),
        deterministic=True,
        label="Invariant AGLLM zero-fusion parity",
        collect_policy_diagnostics=False,
    )
    for metric_key in ["CumWatch", "CVaR_0.95(L)", "ReturnRate60", "NightMinutes", "OverCapMinutes"]:
        if abs(float(param_metrics[metric_key]) - float(llm_metrics[metric_key])) > 1e-8:
            raise AssertionError(f"Zero-fusion AGLLM must match AGParam for {metric_key}.")

    watch_template, watch_hash, _ = load_agllm_prompt_template("watch")
    continue_template, continue_hash, _ = load_agllm_prompt_template("continue")
    manifest = load_agllm_release_manifest()
    if watch_hash != str(manifest["prompt_templates"]["watch"]["sha256"]):
        raise AssertionError("Watch prompt hash mismatch.")
    if continue_hash != str(manifest["prompt_templates"]["continue"]["sha256"]):
        raise AssertionError("Continue prompt hash mismatch.")
    if not watch_template or not continue_template:
        raise AssertionError("Prompt templates must load successfully.")

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "items.csv"
        metadata_path.write_text(
            "\n".join(
                [
                    "item_id,caption,category_level1,category_level2,duration_sec,interaction_count",
                    "1,alpha topic,catA,subA,10,5",
                    "2,beta topic,catA,subA,20,15",
                    "3,gamma topic,catB,subB,40,25",
                    "4,delta topic,catB,subB,70,35",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        small_cfg = replace(cfg, Z=2)
        cards_a = build_default_cards(small_cfg, str(metadata_path), random_seed=0, paper_mode=True)
        cards_b = build_default_cards(small_cfg, str(metadata_path), random_seed=0, paper_mode=True)
        if cards_a != cards_b:
            raise AssertionError("Card builder must be deterministic.")

    no_test_leakage_invariant()
    return {
        "low_threshold_overcap_nonzero": True,
        "session_cap_reduces_tail_when_active": True,
        "nohabit_exact_zero": True,
        "deterministic_fast_path_matches_rich_actions": True,
        "zero_fusion_matches_agparam": True,
        "prompt_hashes_match_manifest": True,
        "card_builder_deterministic": True,
        "no_test_leakage": True,
    }


# ---------------------------------------------------------------------
# Paper experiment driver
# ---------------------------------------------------------------------

def run_paper_pipeline(args: argparse.Namespace) -> None:
    run_start = time.perf_counter()
    trace_next_eval = bool(getattr(args, "trace_first_episode", False))
    trace_max_steps = int(getattr(args, "trace_max_steps", 25))
    evaluation_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    evaluation_cache_hits = 0
    evaluation_cache_misses = 0
    model_hash_cache: Dict[int, str] = {}
    policy_hash_cache: Dict[int, str] = {}

    def prepare_logged_dir(path: Path, label: str) -> Path:
        existed = path.exists()
        prepared = ensure_dir(path)
        LOGGER.info("Directory ready | %s | %s | path=%s", label, "reused" if existed else "created", prepared)
        return prepared

    def log_phase_complete(name: str, started_at: float, extra: Optional[str] = None) -> None:
        message = f"{name} complete in {format_elapsed(time.perf_counter() - started_at)}"
        if extra:
            message = f"{message} | {extra}"
        LOGGER.info(message)

    def log_block_metrics(name: str, started_at: float, metrics: Dict[str, Any], train_seed: Optional[int] = None) -> None:
        prefix = f"train_seed={train_seed} | " if train_seed is not None else ""
        LOGGER.info(
            "%s%s complete in %s | %s",
            prefix,
            name,
            format_elapsed(time.perf_counter() - started_at),
            format_metrics_summary(metrics),
        )

    def stable_json_hash(payload: Any) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def config_hash(eval_cfg: BenchConfig) -> str:
        return stable_json_hash(eval_cfg.to_dict())

    def model_state_hash(model: nn.Module) -> str:
        model_id = id(model)
        cached_hash = model_hash_cache.get(model_id)
        if cached_hash is not None:
            return cached_hash
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            tensor_cpu = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor_cpu.shape)).encode("utf-8"))
            digest.update(str(tensor_cpu.dtype).encode("utf-8"))
            digest.update(tensor_cpu.numpy().tobytes())
        cached_hash = digest.hexdigest()
        model_hash_cache[model_id] = cached_hash
        return cached_hash

    def policy_checkpoint_hash(policy_obj: Any) -> str:
        policy_id = id(policy_obj)
        cached_hash = policy_hash_cache.get(policy_id)
        if cached_hash is not None:
            return cached_hash
        payload: Dict[str, Any] = {"class": policy_obj.__class__.__name__}
        policy_model = getattr(policy_obj, "model", None)
        if isinstance(policy_model, nn.Module):
            payload["model_state_hash"] = model_state_hash(policy_model)
        elif isinstance(policy_obj, nn.Module):
            payload["model_state_hash"] = model_state_hash(policy_obj)
        if hasattr(policy_obj, "seed"):
            payload["seed"] = int(getattr(policy_obj, "seed"))
        if hasattr(policy_obj, "num_actions"):
            payload["num_actions"] = int(getattr(policy_obj, "num_actions"))
        policy_cfg = getattr(policy_obj, "cfg", None)
        if hasattr(policy_cfg, "to_dict"):
            payload["cfg"] = policy_cfg.to_dict()
        if "model_state_hash" not in payload and len(payload) == 1:
            payload["object_id"] = policy_id
        cached_hash = stable_json_hash(payload)
        policy_hash_cache[policy_id] = cached_hash
        return cached_hash

    def scorer_config_hash(eval_scorer: Optional[AGLLMScorer], eval_cfg: BenchConfig, backend: str) -> str:
        if backend != "llm":
            return ""
        payload: Dict[str, Any] = {"eval_cfg": eval_cfg.to_dict()}
        if eval_scorer is not None:
            payload.update(
                {
                    "mode": eval_scorer.mode,
                    "model_id": eval_scorer.model_id,
                    "watch_template_hash": eval_scorer.watch_template_hash,
                    "continue_template_hash": eval_scorer.continue_template_hash,
                    "scorer_cfg": eval_scorer.cfg.to_dict(),
                }
            )
        return stable_json_hash(payload)

    def evaluate_cached(
        policy: Any,
        eval_cfg: BenchConfig,
        eval_catalog: np.ndarray,
        eval_cards: Sequence[Dict[str, Any]],
        *,
        policy_label: str,
        backend: str = "param",
        scorer: Optional[AGLLMScorer] = None,
        wrappers: Optional[Dict[str, Any]] = None,
        ablation: Optional[str] = None,
        episode_seeds: Optional[Sequence[int]] = None,
        num_episodes: int = 100,
        deterministic: bool = True,
        label: Optional[str] = None,
        log_progress: bool = False,
        log_every_episodes: Optional[int] = None,
        trace_first_episode: bool = False,
        trace_max_steps: int = 25,
        collect_policy_diagnostics: bool = True,
        progress_tracker: Optional[ProgressTracker] = None,
        progress_task: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nonlocal evaluation_cache_hits, evaluation_cache_misses
        resolved_episode_seeds = list(episode_seeds) if episode_seeds is not None else list(range(num_episodes))
        cache_key = (
            str(policy_label),
            policy_checkpoint_hash(policy),
            str(backend),
            config_hash(eval_cfg),
            bool(deterministic),
            stable_json_hash([int(seed) for seed in resolved_episode_seeds]),
            stable_json_hash(wrappers or {}),
            "" if ablation is None else str(ablation),
            bool(collect_policy_diagnostics),
            scorer_config_hash(scorer, eval_cfg, backend),
        )
        if not trace_first_episode and cache_key in evaluation_cache:
            evaluation_cache_hits += 1
            if progress_tracker is not None and progress_task is not None:
                progress_tracker.start_task(
                    str(progress_task["task_id"]),
                    str(progress_task["name"]),
                    str(progress_task["unit_name"]),
                    int(progress_task["total_units"]),
                )
                progress_tracker.update_task(int(progress_task["total_units"]), extra="cache hit")
                progress_tracker.finish_task(extra="cache hit")
            return copy.deepcopy(evaluation_cache[cache_key])
        evaluation_cache_misses += 1
        metrics = evaluate_policy(
            policy,
            eval_cfg,
            eval_catalog,
            eval_cards,
            backend=backend,
            scorer=scorer,
            wrappers=wrappers,
            ablation=ablation,
            episode_seeds=resolved_episode_seeds,
            num_episodes=len(resolved_episode_seeds),
            deterministic=deterministic,
            label=label,
            log_progress=log_progress,
            log_every_episodes=log_every_episodes,
            trace_first_episode=trace_first_episode,
            trace_max_steps=trace_max_steps,
            collect_policy_diagnostics=collect_policy_diagnostics,
            progress_tracker=progress_tracker,
            progress_task=progress_task,
        )
        evaluation_cache[cache_key] = copy.deepcopy(metrics)
        return copy.deepcopy(metrics)

    def evaluate_with_optional_trace(*eval_args: Any, **eval_kwargs: Any) -> Dict[str, Any]:
        nonlocal trace_next_eval
        if trace_next_eval:
            eval_kwargs.setdefault("trace_first_episode", True)
            eval_kwargs.setdefault("trace_max_steps", trace_max_steps)
            trace_next_eval = False
        eval_kwargs.setdefault("collect_policy_diagnostics", run_policy_diagnostics)
        policy_label = str(eval_kwargs.pop("policy_label", eval_kwargs.get("label") or eval_args[0].__class__.__name__))
        return evaluate_cached(
            eval_args[0],
            eval_args[1],
            eval_args[2],
            eval_args[3],
            policy_label=policy_label,
            **eval_kwargs,
        )

    def warm_llm_cache(
        policy: Any,
        eval_cfg: BenchConfig,
        eval_scorer: AGLLMScorer,
        episode_seeds: Sequence[int],
        label: str,
        deterministic: bool = True,
        progress_tracker: Optional[ProgressTracker] = None,
        progress_task: Optional[Dict[str, Any]] = None,
    ) -> None:
        evaluate_policy(
            policy,
            eval_cfg,
            catalog,
            cards,
            backend="llm",
            scorer=eval_scorer,
            episode_seeds=episode_seeds,
            num_episodes=len(episode_seeds),
            deterministic=deterministic,
            label=f"{label} [cache prebuild]",
            log_progress=False,
            collect_policy_diagnostics=False,
            progress_tracker=progress_tracker,
            progress_task=progress_task,
        )

    def derive_binding_cap_grid(reference_session_lengths: Sequence[float]) -> List[float]:
        configured_caps = sorted(set(float(x) for x in cfg.paper_thresholds.get("cap_grid", [])))
        if configured_caps:
            return configured_caps
        requested_caps = sorted(set(float(x) for x in args.cap_grid))
        if not reference_session_lengths:
            return requested_caps if requested_caps else [float(cfg.T_cap)]
        arr = np.asarray(list(reference_session_lengths), dtype=np.float64)
        if arr.size == 0:
            return requested_caps if requested_caps else [float(cfg.T_cap)]
        derived_caps = sorted(
            set(
                max(5.0, round_to_nearest(float(np.quantile(arr, q)), 5.0))
                for q in (0.90, 0.95, 0.99)
            )
        )
        if derived_caps:
            return [float(cap) for cap in derived_caps]
        return requested_caps if requested_caps else [float(cfg.T_cap)]

    def choose_lagrangian_candidate(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        feasible = [candidate for candidate in candidates if bool(candidate["train_summary"].get("selected_feasible", False))]
        if feasible:
            return max(
                feasible,
                key=lambda candidate: (
                    float((candidate.get("selected_validation") or {}).get("validation_CumWatch", float("-inf"))),
                    float((candidate.get("selected_validation") or {}).get("global_step", 0.0)),
                ),
            )
        return min(
            candidates,
            key=lambda candidate: (
                float((candidate.get("selected_validation") or {}).get("total_violation", float("inf"))),
                -float((candidate.get("selected_validation") or {}).get("validation_CumWatch", float("-inf"))),
                -float((candidate.get("selected_validation") or {}).get("global_step", 0.0)),
            ),
        )

    def lagrangian_candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[int, float, float, float]:
        selected_validation = candidate.get("selected_validation") or {}
        validation_cumwatch = float(selected_validation.get("validation_CumWatch", float("-inf")))
        total_violation = float(selected_validation.get("total_violation", float("inf")))
        if bool(candidate["train_summary"].get("selected_feasible", False)):
            return (0, -validation_cumwatch, total_violation, -float(selected_validation.get("global_step", 0.0)))
        return (1, total_violation, -validation_cumwatch, -float(selected_validation.get("global_step", 0.0)))

    def appendix_dashboard_row(category: str, condition: str, policy_name: str, train_seed: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            "category": str(category),
            "condition": str(condition),
            "policy": str(policy_name),
            "train_seed": int(train_seed),
        }
        row.update({key: metrics.get(key, float("nan")) for key in APPENDIX_DASHBOARD_METRIC_KEYS})
        row["BreakAdherence_num"] = int(metrics.get("BreakAdherence_num", 0))
        row["BreakAdherence_den"] = int(metrics.get("BreakAdherence_den", 0))
        return row

    def official_scorecard_row(policy_name: str, eval_mode: str, train_seed: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return build_official_scorecard_metric_row(policy_name, "Param", eval_mode, train_seed, metrics)

    def append_official_scorecard_result(
        policy_name: str,
        eval_mode: str,
        train_seed: int,
        episode_seeds: Sequence[int],
        metrics: Dict[str, Any],
    ) -> None:
        official_scorecard_rows.append(official_scorecard_row(policy_name, eval_mode, train_seed, metrics))
        official_episode_output_rows.extend(
            build_official_episode_output_rows(policy_name, "Param", eval_mode, train_seed, episode_seeds, metrics)
        )

    def frontier_summary_row(method: str, scale: float, eval_mode: str, train_seed: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "method": str(method),
            "scale": float(scale),
            "eval_mode": str(eval_mode),
            "train_seed": int(train_seed),
            "CumWatch": metrics["CumWatch"],
            "NightMinutes": metrics["NightMinutes"],
            "NightFraction": metrics.get("NightFraction", float("nan")),
            "LateNightSessionStartRate": metrics.get("LateNightSessionStartRate", float("nan")),
            "OverCapMinutes": metrics["OverCapMinutes"],
            "CVaR_0.95(L)": metrics["CVaR_0.95(L)"],
            "ReturnRate60": metrics["ReturnRate60"],
        }

    def save_appendix_plots(metrics: Dict[str, Any], prefix: str, title_prefix: str) -> None:
        stop_hazard = list(metrics.get("StopHazard", []))
        stop_positions = list(metrics.get("StopHazardPositions", []))
        if stop_hazard and stop_positions:
            save_lineplot_progress(
                stop_positions,
                stop_hazard,
                None,
                "Within-session position",
                "Stop hazard",
                f"{title_prefix}: stop hazard",
                fig_dir / f"{prefix}_stop_hazard.png",
                label_a="Observed",
            )
        gaps = list(metrics.get("Gaps", []))
        if gaps:
            save_histogram_progress(
                gaps,
                None,
                "Gap (min)",
                f"{title_prefix}: gap histogram",
                fig_dir / f"{prefix}_gap_histogram.png",
                label_a="Observed",
            )
            gap_hazard = empirical_hazard(gaps, [0.0, 5.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 1440.0])
            save_lineplot_progress(
                list(range(1, len(gap_hazard["hazard"]) + 1)),
                gap_hazard["hazard"],
                None,
                "Gap bin index",
                "Gap hazard",
                f"{title_prefix}: gap hazard",
                fig_dir / f"{prefix}_gap_hazard.png",
                label_a="Observed",
            )
        sus_summary = susceptibility_bin_summary(
            metrics.get("EpisodeSusceptibilityValues", []),
            metrics.get("EpisodeCumWatchValues", []),
        )
        if sus_summary["bin_labels"]:
            save_barplot_progress(
                sus_summary["bin_labels"],
                sus_summary["bin_means"],
                "Susceptibility bin",
                "Mean CumWatch",
                f"{title_prefix}: subgroup disparity by susceptibility",
                fig_dir / f"{prefix}_subgroup_disparity_susceptibility.png",
            )

    run_profile = str(getattr(args, "run_profile", "full")).lower()
    run_heuristics = run_profile == "full"
    run_policy_diagnostics = run_profile == "full"
    run_appendix_stress = run_profile == "full"
    run_wrapper_baselines = run_profile == "full"
    run_break_prompt_probe = run_profile in {"main", "full"}
    run_diagnostic_sweeps = run_profile == "full"
    run_backend_sensitivity = bool(args.with_llm) and run_profile in {"main", "full"}
    run_cap_sensitivity = run_profile in {"main", "full"}
    run_frontier_artifacts = run_profile in {"main", "full"}
    write_diagnostic_artifacts = run_profile == "full"
    write_backend_artifacts = run_backend_sensitivity
    write_appendix_artifacts = run_appendix_stress or run_wrapper_baselines
    selected_threshold_source = str(getattr(args, "threshold_source", DEFAULT_THRESHOLD_SOURCE)).strip().lower()
    if selected_threshold_source not in THRESHOLD_SOURCE_CHOICES:
        raise ValueError(f"Unknown threshold_source={selected_threshold_source!r}. Expected one of {THRESHOLD_SOURCE_CHOICES}.")
    lag_search_mode_arg = getattr(args, "lag_search_mode", None)
    lag_search_mode = default_lag_search_mode(run_profile) if lag_search_mode_arg is None else str(lag_search_mode_arg).lower()
    lag_search_steps = max(1, int(getattr(args, "lag_search_steps", 25_000)))
    lag_search_val_episodes = max(1, int(getattr(args, "lag_search_val_episodes", 200)))
    lag_full_steps = getattr(args, "lag_full_steps", None)
    lag_full_steps = max(1, int(args.ppo_steps if lag_full_steps is None else lag_full_steps))
    lag_full_val_episodes = getattr(args, "lag_full_val_episodes", None)
    lag_full_val_episodes = max(1, int(args.val_episodes if lag_full_val_episodes is None else lag_full_val_episodes))
    lag_topk_candidates = max(1, int(getattr(args, "lag_topk_candidates", 1)))
    lag_dual_lr_grid_search_arg = getattr(args, "lag_dual_lr_grid_search", None)
    if lag_dual_lr_grid_search_arg is None:
        lag_dual_lr_grid_search_arg = [float(args.dual_lr), float(args.dual_lr) * 2.0]
    lag_dual_lr_grid_search = sorted(set(max(1e-4, float(x)) for x in lag_dual_lr_grid_search_arg))
    if run_frontier_artifacts:
        lag_dual_lr_grid_full = sorted(
            set(
                [
                    max(1e-4, float(args.dual_lr) * 0.5),
                    max(1e-4, float(args.dual_lr)),
                    max(1e-4, float(args.dual_lr) * 2.0),
                ]
            )
        )
        lag_scales = [float(x) for x in args.constraint_scales]
    else:
        lag_dual_lr_grid_full = [max(1e-4, float(args.dual_lr))]
        lag_scales = [float(args.constraint_scales[0])]
    extended_lag_steps = int(max(lag_full_steps + args.rollout_steps, round(lag_full_steps * 1.5)))
    official_deterministic_baseline_names = ("RoundRobinPolicy", "LeastRecentPolicy", "NoveltyGreedyPolicy")
    args.lag_search_mode = lag_search_mode
    args.lag_search_steps = lag_search_steps
    args.lag_search_val_episodes = lag_search_val_episodes
    args.lag_full_steps = lag_full_steps
    args.lag_full_val_episodes = lag_full_val_episodes
    args.lag_topk_candidates = lag_topk_candidates
    args.lag_dual_lr_grid_search = list(lag_dual_lr_grid_search)
    set_global_seed(args.seed)
    outdir = ensure_dir(args.outdir)
    llm_summary = "disabled"
    if args.with_llm:
        if run_backend_sensitivity:
            llm_summary = f"enabled(mode={args.llm_mode}, model_id={args.llm_model_id})"
        else:
            llm_summary = f"requested-but-skipped(run_profile={run_profile})"
    LOGGER.info(
        "Run summary | outdir=%s | seed=%s | device=%s | run_profile=%s | num_train_seeds=%s | eval_episodes=%s | ppo_steps=%s | rollout_steps=%s | constraint_scales=%s | cap_grid=%s | llm=%s | calibrate_from_logs=%s | paper_mode=%s | threshold_source=%s",
        outdir,
        args.seed,
        args.device,
        run_profile,
        args.num_train_seeds,
        args.eval_episodes,
        args.ppo_steps,
        args.rollout_steps,
        list(args.constraint_scales),
        list(args.cap_grid),
        llm_summary,
        bool(args.log_csv),
        bool(getattr(args, "paper_mode", False)),
        selected_threshold_source,
    )
    LOGGER.info(
        "Run profile gates | heuristics=%s | policy_diagnostics=%s | appendix_stress=%s | wrapper_baselines=%s | diagnostic_sweeps=%s | backend_sensitivity=%s | cap_sensitivity=%s | frontier_artifacts=%s",
        run_heuristics,
        run_policy_diagnostics,
        run_appendix_stress,
        run_wrapper_baselines,
        run_diagnostic_sweeps,
        run_backend_sensitivity,
        run_cap_sensitivity,
        run_frontier_artifacts,
    )
    LOGGER.info(
        "Lagrangian search config | mode=%s | search_steps=%s | search_val_episodes=%s | full_steps=%s | full_val_episodes=%s | search_dual_lr_grid=%s | full_dual_lr_grid=%s | topk_candidates=%s | scales=%s",
        lag_search_mode,
        lag_search_steps,
        lag_search_val_episodes,
        lag_full_steps,
        lag_full_val_episodes,
        lag_dual_lr_grid_search,
        lag_dual_lr_grid_full,
        lag_topk_candidates,
        lag_scales,
    )

    train_seed_list = [args.seed + i for i in range(args.num_train_seeds)]
    planned_diagnostic_first_seed = train_seed_list[0] if train_seed_list else None
    planned_cap_task_count = max(3, len(args.cap_grid)) if run_cap_sensitivity else 0

    def train_task_id(train_seed: int, label: str) -> str:
        return f"seed{int(train_seed)}_train_{filename_slug(label)}"

    def eval_task_id(train_seed: int, label: str) -> str:
        return f"seed{int(train_seed)}_eval_{filename_slug(label)}"

    def file_task_id(label: str) -> str:
        return f"run_{filename_slug(label)}"

    def steps_task(task_id: str, name: str, total_steps: int) -> Dict[str, Any]:
        return make_progress_task(task_id, name, "steps", total_steps)

    def episodes_task(task_id: str, name: str, total_episodes: int) -> Dict[str, Any]:
        return make_progress_task(task_id, name, "episodes", total_episodes)

    def transitions_task(task_id: str, name: str, total_transitions: int) -> Dict[str, Any]:
        return make_progress_task(task_id, name, "transitions", total_transitions)

    def epochs_task(task_id: str, name: str, total_epochs: int) -> Dict[str, Any]:
        return make_progress_task(task_id, name, "epochs", total_epochs)

    def files_task(task_id: str, name: str, total_files: int) -> Dict[str, Any]:
        return make_progress_task(task_id, name, "files", total_files)

    def planned_table_file_count() -> int:
        count = 8  # scorecards + nohabit table + threshold table
        if write_diagnostic_artifacts:
            count += 14  # diagnostic tables plus return-rate/nohabit notes
        if run_cap_sensitivity:
            count += 4
        if run_frontier_artifacts:
            count += 4
        if run_policy_diagnostics:
            count += 4
        if write_backend_artifacts:
            count += 10
        if write_appendix_artifacts:
            count += 8
        return count

    def planned_figure_file_count() -> int:
        count = 0
        if run_frontier_artifacts:
            count += 4
        if write_backend_artifacts:
            count += 1
        if run_policy_diagnostics:
            lag_policy_count = min(lag_topk_candidates, len(lag_scales)) if lag_search_mode == "two_stage" else len(lag_scales)
            count += 8 + (2 * lag_policy_count) + (10 if run_diagnostic_sweeps else 0)
        if run_appendix_stress:
            count += 4
        return count

    def build_progress_plan() -> List[Dict[str, Any]]:
        plan: List[Dict[str, Any]] = []
        plan.append(files_task(file_task_id("setup_artifacts"), "Setup run artifacts", 5))
        if args.log_csv:
            plan.append(files_task(file_task_id("calibration"), "Calibration against logs", 9))
        if run_backend_sensitivity and bool(args.fit_llm_fusion):
            plan.append(files_task(file_task_id("llm_fusion_fit"), "LLM fusion fit", 1))

        for train_seed in train_seed_list:
            plan.append(episodes_task(eval_task_id(train_seed, "random_test"), f"Random test evaluation (seed {train_seed})", args.eval_episodes))
            if run_policy_diagnostics:
                plan.append(episodes_task(eval_task_id(train_seed, "random_validation_stochastic"), f"Random validation stochastic evaluation (seed {train_seed})", int(args.val_episodes)))

            plan.append(transitions_task(train_task_id(train_seed, "logged_dataset"), f"Logged dataset collection (seed {train_seed})", args.random_log_steps))
            plan.append(epochs_task(train_task_id(train_seed, "myopic"), f"Myopic training (seed {train_seed})", args.myopic_epochs))
            plan.append(episodes_task(eval_task_id(train_seed, "myopic_test_deterministic"), f"Myopic test deterministic evaluation (seed {train_seed})", args.eval_episodes))
            plan.append(episodes_task(eval_task_id(train_seed, "myopic_test_stochastic"), f"Myopic test stochastic evaluation (seed {train_seed})", args.eval_episodes))
            for heuristic_name in official_deterministic_baseline_names:
                plan.append(
                    episodes_task(
                        eval_task_id(train_seed, f"{heuristic_name}_official_test_deterministic"),
                        f"{heuristic_name} official deterministic evaluation (seed {train_seed})",
                        args.eval_episodes,
                    )
                )
                if run_policy_diagnostics:
                    plan.append(
                        episodes_task(
                            eval_task_id(train_seed, f"{heuristic_name}_validation_deterministic"),
                            f"{heuristic_name} validation deterministic evaluation (seed {train_seed})",
                            int(args.val_episodes),
                        )
                    )
            plan.append(episodes_task(eval_task_id(train_seed, "ppo_cap_default_deterministic"), f"PPO default session-cap deterministic evaluation (seed {train_seed})", args.eval_episodes))
            plan.append(episodes_task(eval_task_id(train_seed, "ppo_cap_default_stochastic"), f"PPO default session-cap stochastic evaluation (seed {train_seed})", args.eval_episodes))
            if run_policy_diagnostics:
                plan.append(episodes_task(eval_task_id(train_seed, "myopic_validation_deterministic"), f"Myopic validation deterministic evaluation (seed {train_seed})", int(args.val_episodes)))
                plan.append(episodes_task(eval_task_id(train_seed, "myopic_validation_stochastic"), f"Myopic validation stochastic evaluation (seed {train_seed})", int(args.val_episodes)))

            if run_heuristics:
                heuristic_specs = [
                    ("RandomZ-LowLambda-HighVar", "stochastic"),
                    ("CyclicZ-LowLambda-HighVar", "deterministic"),
                ]
                for heuristic_name, heuristic_eval_mode in heuristic_specs:
                    plan.append(episodes_task(eval_task_id(train_seed, f"{heuristic_name}_test_{heuristic_eval_mode}"), f"{heuristic_name} test {heuristic_eval_mode} evaluation (seed {train_seed})", args.eval_episodes))
                    plan.append(episodes_task(eval_task_id(train_seed, f"{heuristic_name}_validation_{heuristic_eval_mode}"), f"{heuristic_name} validation {heuristic_eval_mode} evaluation (seed {train_seed})", int(args.val_episodes)))

            plan.append(steps_task(train_task_id(train_seed, "ppo"), f"PPO training (seed {train_seed})", args.ppo_steps))
            plan.append(episodes_task(eval_task_id(train_seed, "ppo_validation_deterministic"), f"PPO validation deterministic evaluation (seed {train_seed})", int(args.val_episodes)))
            plan.append(episodes_task(eval_task_id(train_seed, "ppo_test_deterministic"), f"PPO test deterministic evaluation (seed {train_seed})", args.eval_episodes))
            plan.append(episodes_task(eval_task_id(train_seed, "ppo_test_stochastic"), f"PPO test stochastic evaluation (seed {train_seed})", args.eval_episodes))
            plan.append(episodes_task(eval_task_id(train_seed, "ppo_autoplay_off_mitigation"), f"PPO autoplay-off mitigation evaluation (seed {train_seed})", args.eval_episodes))
            if run_policy_diagnostics:
                plan.append(episodes_task(eval_task_id(train_seed, "ppo_validation_stochastic"), f"PPO validation stochastic evaluation (seed {train_seed})", int(args.val_episodes)))

            if lag_search_mode == "two_stage":
                for scale in lag_scales:
                    scale_tag = filename_slug(f"{scale:.4f}")
                    for dual_lr_value in lag_dual_lr_grid_search:
                        dual_tag = filename_slug(f"{dual_lr_value:.4f}")
                        plan.append(
                            steps_task(
                                train_task_id(train_seed, f"lagppo_scale_{scale_tag}_search_dual_{dual_tag}"),
                                f"LagPPO scale={scale:.2f} search dual_lr={dual_lr_value:.4f} (seed {train_seed})",
                                lag_search_steps,
                            )
                        )
                confirm_count = min(lag_topk_candidates, len(lag_scales))
                for confirm_index in range(1, confirm_count + 1):
                    plan.append(steps_task(train_task_id(train_seed, f"lagppo_confirm_{confirm_index}"), f"LagPPO confirm candidate {confirm_index} (seed {train_seed})", lag_full_steps))
                    plan.append(episodes_task(eval_task_id(train_seed, f"lagppo_confirm_{confirm_index}_test_deterministic"), f"LagPPO confirm candidate {confirm_index} test deterministic evaluation (seed {train_seed})", args.eval_episodes))
                    if run_policy_diagnostics:
                        plan.append(episodes_task(eval_task_id(train_seed, f"lagppo_confirm_{confirm_index}_test_stochastic"), f"LagPPO confirm candidate {confirm_index} test stochastic evaluation (seed {train_seed})", args.eval_episodes))
                        plan.append(episodes_task(eval_task_id(train_seed, f"lagppo_confirm_{confirm_index}_validation_deterministic"), f"LagPPO confirm candidate {confirm_index} validation deterministic evaluation (seed {train_seed})", int(args.val_episodes)))
                        plan.append(episodes_task(eval_task_id(train_seed, f"lagppo_confirm_{confirm_index}_validation_stochastic"), f"LagPPO confirm candidate {confirm_index} validation stochastic evaluation (seed {train_seed})", int(args.val_episodes)))
            else:
                for scale in lag_scales:
                    scale_tag = filename_slug(f"{scale:.4f}")
                    for dual_lr_value in lag_dual_lr_grid_full:
                        dual_tag = filename_slug(f"{dual_lr_value:.4f}")
                        plan.append(
                            steps_task(
                                train_task_id(train_seed, f"lagppo_scale_{scale_tag}_dual_{dual_tag}"),
                                f"LagPPO scale={scale:.2f} dual_lr={dual_lr_value:.4f} (seed {train_seed})",
                                lag_full_steps,
                            )
                        )
                        if run_frontier_artifacts:
                            plan.append(
                                steps_task(
                                    train_task_id(train_seed, f"lagppo_scale_{scale_tag}_extend_dual_{dual_tag}"),
                                    f"LagPPO scale={scale:.2f} extend dual_lr={dual_lr_value:.4f} (seed {train_seed})",
                                    extended_lag_steps,
                                )
                            )
                    plan.append(episodes_task(eval_task_id(train_seed, f"lagppo_scale_{scale_tag}_test_deterministic"), f"LagPPO scale={scale:.2f} test deterministic evaluation (seed {train_seed})", args.eval_episodes))
                    if run_policy_diagnostics:
                        plan.append(episodes_task(eval_task_id(train_seed, f"lagppo_scale_{scale_tag}_test_stochastic"), f"LagPPO scale={scale:.2f} test stochastic evaluation (seed {train_seed})", args.eval_episodes))
                        plan.append(episodes_task(eval_task_id(train_seed, f"lagppo_scale_{scale_tag}_validation_deterministic"), f"LagPPO scale={scale:.2f} validation deterministic evaluation (seed {train_seed})", int(args.val_episodes)))
                        plan.append(episodes_task(eval_task_id(train_seed, f"lagppo_scale_{scale_tag}_validation_stochastic"), f"LagPPO scale={scale:.2f} validation stochastic evaluation (seed {train_seed})", int(args.val_episodes)))

            plan.append(episodes_task(eval_task_id(train_seed, "lagppo_main_test_stochastic"), f"Lagrangian PPO test stochastic evaluation (seed {train_seed})", args.eval_episodes))
            if run_policy_diagnostics:
                plan.append(episodes_task(eval_task_id(train_seed, "lagppo_main_validation_deterministic"), f"Lagrangian PPO validation deterministic evaluation (seed {train_seed})", int(args.val_episodes)))
                plan.append(episodes_task(eval_task_id(train_seed, "lagppo_main_validation_stochastic"), f"Lagrangian PPO validation stochastic evaluation (seed {train_seed})", int(args.val_episodes)))

            if run_diagnostic_sweeps and planned_diagnostic_first_seed is not None and int(train_seed) == int(planned_diagnostic_first_seed):
                for sweep_name in ["PPO-LowLR", "PPO-HighEntropy", "PPO-Wide", "PPO-Longer", "FactorizedPPO"]:
                    total_steps_value = int(args.ppo_steps * 1.5) if sweep_name == "PPO-Longer" else int(args.ppo_steps)
                    plan.append(steps_task(train_task_id(train_seed, sweep_name), f"{sweep_name} training (seed {train_seed})", total_steps_value))
                    plan.append(episodes_task(eval_task_id(train_seed, f"{sweep_name}_test_deterministic"), f"{sweep_name} test deterministic evaluation (seed {train_seed})", args.eval_episodes))
                    if run_policy_diagnostics:
                        plan.append(episodes_task(eval_task_id(train_seed, f"{sweep_name}_test_stochastic"), f"{sweep_name} test stochastic evaluation (seed {train_seed})", args.eval_episodes))
                        plan.append(episodes_task(eval_task_id(train_seed, f"{sweep_name}_validation_deterministic"), f"{sweep_name} validation deterministic evaluation (seed {train_seed})", int(args.val_episodes)))
                        plan.append(episodes_task(eval_task_id(train_seed, f"{sweep_name}_validation_stochastic"), f"{sweep_name} validation stochastic evaluation (seed {train_seed})", int(args.val_episodes)))

            if run_cap_sensitivity:
                for cap_index in range(1, planned_cap_task_count + 1):
                    plan.append(episodes_task(eval_task_id(train_seed, f"ppo_cap_{cap_index}"), f"PPO session-cap evaluation #{cap_index} (seed {train_seed})", args.eval_episodes))

            plan.append(episodes_task(eval_task_id(train_seed, "ppo_nohabit"), f"PPO NoHabit evaluation (seed {train_seed})", args.eval_episodes))
            plan.append(episodes_task(eval_task_id(train_seed, "lagppo_nohabit"), f"LagPPO NoHabit evaluation (seed {train_seed})", args.eval_episodes))

            if run_appendix_stress:
                for stress_name in APPENDIX_STRESS_TEST_NAMES:
                    for policy_name in ["Random", "PPO", "Lagrangian PPO"]:
                        plan.append(episodes_task(eval_task_id(train_seed, f"{policy_name}_{stress_name}"), f"{policy_name} stress={stress_name} evaluation (seed {train_seed})", min(args.eval_episodes, 2_000)))

            if run_break_prompt_probe:
                plan.append(
                    episodes_task(
                        eval_task_id(train_seed, "ppo_break_prompt_probe"),
                        f"PPO break-prompt probe evaluation (seed {train_seed})",
                        min(args.eval_episodes, 2_000),
                    )
                )

            if run_wrapper_baselines:
                for wrapper_name in ["throttle_personalization"]:
                    plan.append(episodes_task(eval_task_id(train_seed, f"ppo_wrapper_{wrapper_name}"), f"PPO wrapper={wrapper_name} evaluation (seed {train_seed})", min(args.eval_episodes, 2_000)))

            if run_backend_sensitivity:
                warm_labels = [
                    "ppo_llm_zero_fusion_validation_warm",
                    "ppo_llm_validation_warm",
                    "lagppo_llm_validation_warm",
                    "myopic_llm_validation_warm",
                    "ppo_llm_watch_only_validation_warm",
                    "ppo_llm_continue_only_validation_warm",
                    "ppo_semantic_validation_warm",
                    "ppo_llm_zero_fusion_test_warm",
                    "ppo_llm_test_warm",
                    "lagppo_llm_test_warm",
                    "myopic_llm_test_warm",
                    "ppo_llm_watch_only_test_warm",
                    "ppo_llm_continue_only_test_warm",
                    "ppo_semantic_test_warm",
                ]
                warm_episode_count = int(args.val_episodes)
                for warm_label in warm_labels:
                    total_warm_episodes = warm_episode_count if "validation" in warm_label else args.eval_episodes
                    plan.append(episodes_task(eval_task_id(train_seed, warm_label), f"{warm_label.replace('_', ' ')} (seed {train_seed})", total_warm_episodes))
                for backend_label in [
                    "ppo_llm_zero_fusion",
                    "ppo_llm",
                    "lagppo_llm",
                    "myopic_llm",
                    "ppo_llm_watch_only",
                    "ppo_llm_continue_only",
                    "ppo_semantic_residual",
                ]:
                    plan.append(episodes_task(eval_task_id(train_seed, backend_label), f"{backend_label.replace('_', ' ')} evaluation (seed {train_seed})", args.eval_episodes))

        plan.append(files_task(file_task_id("table_aggregation"), "Table aggregation", planned_table_file_count()))
        if write_diagnostic_artifacts:
            plan.append(files_task(file_task_id("threshold_diagnostics"), "Threshold diagnostics", 4))
        plan.append(files_task(file_task_id("figure_generation"), "Figure generation", planned_figure_file_count()))
        plan.append(files_task(file_task_id("manifest_writing"), "Manifest writing", 1 + (1 if args.log_csv else 0)))
        return plan

    progress_plan = build_progress_plan()
    progress_plan_path = outdir / "progress_plan.json"
    json_dumps({"total_tasks": len(progress_plan), "tasks": progress_plan}, progress_plan_path)
    log_artifact_written(progress_plan_path, "progress_plan_json")
    progress_tracker = ProgressTracker(progress_path=outdir / "progress.json", plan=progress_plan)
    progress_tracker.start_run(len(progress_plan))

    file_task_state: Dict[str, int] = {"done": 0}

    def run_dataset_task(task: Dict[str, Any], **dataset_kwargs: Any) -> Dict[str, np.ndarray]:
        return collect_logged_dataset(**dataset_kwargs, progress_tracker=progress_tracker, progress_task=task)

    def run_myopic_train_task(task: Dict[str, Any], **train_kwargs: Any) -> RewardModel:
        return train_myopic_reward_model(**train_kwargs, progress_tracker=progress_tracker, progress_task=task)

    def run_ppo_train_task(task: Dict[str, Any], **train_kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        return train_ppo(**train_kwargs, progress_tracker=progress_tracker, progress_task=task)

    def run_eval_task(task: Dict[str, Any], *eval_args: Any, use_trace: bool = True, **eval_kwargs: Any) -> Dict[str, Any]:
        eval_kwargs["progress_tracker"] = progress_tracker
        eval_kwargs["progress_task"] = task
        if use_trace:
            return evaluate_with_optional_trace(*eval_args, **eval_kwargs)
        return evaluate_cached(*eval_args, **eval_kwargs)

    def skip_task(task: Dict[str, Any], reason: str) -> None:
        progress_tracker.start_task(
            str(task["task_id"]),
            str(task["name"]),
            str(task["unit_name"]),
            int(task["total_units"]),
        )
        progress_tracker.finish_task(extra=reason)

    def start_file_task(task: Dict[str, Any]) -> None:
        file_task_state["done"] = 0
        progress_tracker.start_task(
            str(task["task_id"]),
            str(task["name"]),
            str(task["unit_name"]),
            int(task["total_units"]),
        )

    def advance_file_task(file_count: int, extra: Optional[str] = None) -> None:
        file_task_state["done"] += int(file_count)
        progress_tracker.update_task(file_task_state["done"], extra=extra)

    def finish_file_task(extra: Optional[str] = None) -> None:
        progress_tracker.finish_task(extra=extra)

    def json_dumps_progress(obj: Any, path: Path, artifact_type: str, file_count: int = 1) -> None:
        json_dumps(obj, path)
        log_artifact_written(path, artifact_type)
        advance_file_task(file_count, extra=path.name)

    def write_text_progress(path: Path, text: str, artifact_type: str, file_count: int = 1) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        log_artifact_written(path, artifact_type)
        advance_file_task(file_count, extra=path.name)

    def save_table_progress(df: pd.DataFrame, path_stem: Path) -> None:
        save_table(df, path_stem)
        advance_file_task(2, extra=path_stem.name)

    def save_csv_progress(df: pd.DataFrame, path: Path) -> None:
        save_csv(df, path)
        advance_file_task(1, extra=path.name)

    def save_histogram_progress(*args: Any, **kwargs: Any) -> None:
        outpath = kwargs.get("outpath")
        if outpath is None and len(args) >= 5:
            outpath = args[4]
        save_histogram(*args, **kwargs)
        advance_file_task(1, extra=Path(outpath).name if outpath is not None else "histogram")

    def save_lineplot_progress(*args: Any, **kwargs: Any) -> None:
        outpath = kwargs.get("outpath")
        if outpath is None and len(args) >= 6:
            outpath = args[6]
        save_lineplot(*args, **kwargs)
        advance_file_task(1, extra=Path(outpath).name if outpath is not None else "lineplot")

    def save_scatter_progress(*args: Any, **kwargs: Any) -> None:
        outpath = kwargs.get("outpath")
        if outpath is None and len(args) >= 6:
            outpath = args[6]
        save_scatter(*args, **kwargs)
        advance_file_task(1, extra=Path(outpath).name if outpath is not None else "scatter")

    def save_barplot_progress(*args: Any, **kwargs: Any) -> None:
        outpath = kwargs.get("outpath")
        if outpath is None and len(args) >= 5:
            outpath = args[5]
        save_barplot(*args, **kwargs)
        advance_file_task(1, extra=Path(outpath).name if outpath is not None else "barplot")

    def save_action_marginals_figure_progress(metrics: Dict[str, Any], title: str, outpath: Path) -> None:
        save_action_marginals_figure(metrics, title, outpath)
        advance_file_task(1, extra=outpath.name)

    model_dir = prepare_logged_dir(outdir / "models", "models")
    fig_dir = prepare_logged_dir(outdir / "figures", "figures")
    table_dir = prepare_logged_dir(outdir / "tables", "tables")
    cache_dir = prepare_logged_dir(outdir / "cache", "cache")

    max_val_episode_count = max(int(args.val_episodes), lag_search_val_episodes, lag_full_val_episodes)
    all_val_episode_seeds = list(range(args.seed + 20_000, args.seed + 20_000 + max_val_episode_count))
    val_episode_seeds = all_val_episode_seeds[: int(args.val_episodes)]
    lag_search_val_episode_seeds = all_val_episode_seeds[:lag_search_val_episodes]
    lag_full_val_episode_seeds = all_val_episode_seeds[:lag_full_val_episodes]
    test_episode_seeds = list(range(args.seed + 30_000, args.seed + 30_000 + args.eval_episodes))
    appendix_episode_seeds = test_episode_seeds[: min(len(test_episode_seeds), 2_000)]
    train_seed_list_path = outdir / "train_seed_list.json"
    val_episode_seeds_path = outdir / "val_episode_seeds.json"
    test_episode_seeds_path = outdir / "test_episode_seeds.json"
    start_file_task(files_task(file_task_id("setup_artifacts"), "Setup run artifacts", 5))
    json_dumps_progress(train_seed_list, train_seed_list_path, "seed_list_json")
    json_dumps_progress(val_episode_seeds, val_episode_seeds_path, "seed_list_json")
    json_dumps_progress(test_episode_seeds, test_episode_seeds_path, "seed_list_json")

    if args.config_json:
        cfg = BenchConfig.from_path(args.config_json)
    else:
        cfg = BenchConfig(device=args.device)
    cfg.threshold_source = selected_threshold_source
    cfg.device = args.device

    catalog = build_catalog(cfg)
    cards_phase_start = time.perf_counter()
    card_source = f"metadata_csv={args.metadata_csv}" if args.metadata_csv else "synthetic fallback"
    if args.with_llm and run_backend_sensitivity and not getattr(args, "paper_mode", False):
        LOGGER.warning(
            "AGLLM is running without paper_mode=True; treat AGLLM results as appendix-only until the strict release protocol is used."
        )
    LOGGER.info("Card build start | source=%s", card_source)
    cards = build_default_cards(cfg, args.metadata_csv, random_seed=args.seed, paper_mode=bool(getattr(args, "paper_mode", False)))
    log_phase_complete("Card build", cards_phase_start, extra=f"source={card_source} | num_cards={len(cards)}")
    config_used_path = outdir / "config_used.json"
    cards_used_path = outdir / "cards_used.json"
    cfg.save(config_used_path)
    log_artifact_written(config_used_path, "config_json")
    advance_file_task(1, extra=config_used_path.name)
    json_dumps_progress(cards, cards_used_path, "cards_json")
    finish_file_task(extra="setup artifacts ready")

    scorer = None
    llm_cfg = copy.deepcopy(cfg)
    llm_fit: Dict[str, Any] = {}
    llm_fit_pending = False
    llm_context_seeds = list(range(args.seed + 2000, args.seed + 2010))
    llm_target_loss_seeds = list(range(args.seed + 2100, args.seed + 2130))
    reference_targets: Optional[Dict[str, Any]] = None
    sim_targets: Optional[Dict[str, Any]] = None
    if run_backend_sensitivity:
        scorer = AGLLMScorer(
            llm_cfg,
            cards,
            mode=args.llm_mode,
            model_id=args.llm_model_id,
            cache_path=str(cache_dir / "agllm_cache.json"),
            device=args.device,
        )
        llm_fit_pending = bool(args.fit_llm_fusion)
    elif args.with_llm:
        LOGGER.info(
            "AGLLM scorer setup skipped | run_profile=%s | backend_sensitivity=%s",
            run_profile,
            run_backend_sensitivity,
        )

    # Optional calibration against user-provided logs.
    calibration_payload = None
    if args.log_csv:
        start_file_task(files_task(file_task_id("calibration"), "Calibration against logs", 9))
        calibration_start = time.perf_counter()
        delta_grid = sorted(set([float(x) for x in DEFAULT_DELTA_SWEEP] + [float(args.delta_sess)]))
        LOGGER.info(
            "Calibration against logs start | log_csv=%s | trials=%s | episodes_per_trial=%s | calibration_policy=%s | delta_grid=%s",
            args.log_csv,
            args.calibration_trials,
            args.calibration_episodes,
            str(args.calibration_policy),
            delta_grid,
        )
        targets_by_delta = {
            float(delta): extract_targets_from_logs(
                args.log_csv,
                float(delta),
                Z=cfg.Z,
                metadata_csv=args.metadata_csv,
                random_seed=args.seed,
                bootstrap_samples=32,
                bootstrap_seed=args.seed + int(delta * 100),
            )
            for delta in delta_grid
        }
        reference_targets = targets_by_delta[float(args.delta_sess)]
        calibrated_cfg, cal_history = random_search_calibration(
            cfg,
            catalog,
            cards,
            targets_by_delta,
            n_trials=args.calibration_trials,
            episodes_per_trial=args.calibration_episodes,
            seed=args.seed,
            selection_delta=float(args.delta_sess),
            calibration_policy=str(args.calibration_policy),
        )
        config_calibrated_path = outdir / "config_calibrated.json"
        calibration_history_path = outdir / "calibration_history.csv"
        pd.DataFrame(cal_history).to_csv(calibration_history_path, index=False)
        log_artifact_written(calibration_history_path, "calibration_history_csv")
        advance_file_task(1, extra=calibration_history_path.name)
        sim_targets = simulate_targets(
            calibrated_cfg,
            catalog,
            cards,
            seeds=list(range(args.seed + 5000, args.seed + 5000 + args.calibration_episodes)),
            gap_bucket_edges=reference_targets.get("gap_bucket_edges"),
            calibration_policy=str(args.calibration_policy),
            calibration_policy_context=resolve_calibration_rollout_policy_context(
                reference_targets,
                str(args.calibration_policy),
            ),
        )
        calibration_status = build_calibration_audit(
            targets_by_delta,
            sim_targets,
            float(args.delta_sess),
        )["status"]
        thresholds = resolve_thresholds_for_source(
            selected_threshold_source,
            selected_delta_sess=float(args.delta_sess),
            reference_targets=reference_targets,
            sim_targets=sim_targets,
            calibration_status=calibration_status,
        )
        apply_paper_thresholds(calibrated_cfg, thresholds)
        calibrated_cfg.save(config_calibrated_path)
        log_artifact_written(config_calibrated_path, "config_json")
        advance_file_task(1, extra=config_calibrated_path.name)
        calibration_payload = build_calibration_payload(
            targets_by_delta,
            sim_targets,
            selected_delta_sess=float(args.delta_sess),
            log_csv=args.log_csv,
            n_trials=args.calibration_trials,
            episodes_per_trial=args.calibration_episodes,
            seed=args.seed,
            calibration_policy=str(args.calibration_policy),
            thresholds=thresholds,
            calibrated_cfg=calibrated_cfg,
        )
        calibration_audit_path = outdir / "table_calibration_audit.md"
        calibration_audit_path.write_text(render_calibration_audit_markdown(calibration_payload["audit"]), encoding="utf-8")
        log_artifact_written(calibration_audit_path, "calibration_audit_markdown")
        advance_file_task(1, extra=calibration_audit_path.name)
        gap_diagnostic_path = outdir / "table_gap_extraction_diagnostic.md"
        write_gap_extraction_diagnostic_artifact(
            reference_targets,
            gap_diagnostic_path,
            delta_sess=float(args.delta_sess),
        )
        advance_file_task(1, extra=gap_diagnostic_path.name)

        save_histogram_progress(reference_targets["session_lengths"], sim_targets["session_lengths"], "Session length (min)", "Calibration: session lengths", fig_dir / "fig_calibration_session_lengths.png")
        save_histogram_progress(reference_targets["gaps"], sim_targets["gaps"], "Gap (min)", "Calibration: inter-session gaps", fig_dir / "fig_calibration_gaps.png")
        if reference_targets.get("stop_hazard") or sim_targets.get("stop_hazard"):
            save_stop_hazard_support_plot(
                reference_targets.get("stop_hazard", []),
                sim_targets.get("stop_hazard", []),
                fig_dir / "fig_calibration_stop_hazard.png",
            )
        common_clusters = sorted(set(reference_targets["cluster_mean_watch"]).intersection(sim_targets["cluster_mean_watch"]))
        if common_clusters:
            save_lineplot_progress(
                common_clusters,
                [reference_targets["cluster_mean_watch"][k] for k in common_clusters],
                [sim_targets["cluster_mean_watch"][k] for k in common_clusters],
                "Cluster",
                "Mean watch time",
                "Calibration: watch time by cluster",
                fig_dir / "fig_calibration_cluster_watch.png",
            )
        if reference_targets.get("session_start_histogram") and sim_targets.get("session_start_histogram"):
            save_lineplot_progress(
                list(range(24)),
                reference_targets["session_start_histogram"],
                sim_targets["session_start_histogram"],
                "Hour of day",
                "Mass",
                "Calibration: session-start time-of-day",
                fig_dir / "fig_calibration_session_start_time_of_day.png",
            )
        cfg = calibrated_cfg
        log_phase_complete(
            "Calibration against logs",
            calibration_start,
            extra=f"config={config_calibrated_path} | history_rows={len(cal_history)} | reference_delta={args.delta_sess}",
        )
        finish_file_task(extra="calibration complete")

    selected_delta_sess = float(args.delta_sess) if args.log_csv else None
    threshold_payload_input = calibration_payload
    if threshold_payload_input is None and selected_threshold_source != "fixed_paper":
        discovered_payload_path = discover_calibration_payload_path(
            config_json=getattr(args, "config_json", None),
            calibration_payload_json=getattr(args, "calibration_payload_json", None),
        )
        if discovered_payload_path is not None:
            threshold_payload_input = json_load(discovered_payload_path)
            LOGGER.info(
                "Loaded calibration payload for threshold resolution | path=%s | threshold_source=%s",
                discovered_payload_path,
                selected_threshold_source,
            )
            if selected_delta_sess is None:
                selected_delta_sess = resolve_selected_delta_sess(threshold_payload_input)
        elif selected_threshold_source != "fixed_paper":
            raise ValueError(
                "threshold_source requires calibration target statistics, but no calibration payload was available. "
                "Pass --calibration_payload_json or run from a config directory that contains calibration_payload.json."
            )
    if threshold_payload_input is not None:
        if reference_targets is None:
            reference_targets = resolve_threshold_target_from_payload(
                threshold_payload_input,
                selected_delta_sess=selected_delta_sess,
            )
        if sim_targets is None:
            sim_targets = copy.deepcopy(threshold_payload_input.get("sim", {}))
        if selected_delta_sess is None:
            selected_delta_sess = resolve_selected_delta_sess(threshold_payload_input)

    threshold_calibration_status = None
    if calibration_payload is not None:
        threshold_calibration_status = str(
            calibration_payload.get("audit", {}).get("status", calibration_payload.get("manifest", {}).get("status"))
        )
    elif threshold_payload_input is not None:
        threshold_calibration_status = str(
            build_calibration_audit_from_payload(
                threshold_payload_input,
                selected_delta_sess=selected_delta_sess,
            )["status"]
        )
    threshold_payload = resolve_thresholds_for_source(
        selected_threshold_source,
        selected_delta_sess=selected_delta_sess,
        reference_targets=reference_targets,
        sim_targets=sim_targets,
        calibration_status=threshold_calibration_status,
    )
    apply_paper_thresholds(cfg, threshold_payload)
    cfg.save(config_used_path)
    if args.with_llm and scorer is not None:
        llm_cfg = copy.deepcopy(cfg)
        scorer.cfg = llm_cfg

    main_rows: List[Dict[str, Any]] = []
    official_scorecard_rows: List[Dict[str, Any]] = []
    official_episode_output_rows: List[Dict[str, Any]] = []
    nohabit_rows: List[Dict[str, Any]] = []
    cap_rows: List[Dict[str, Any]] = []
    frontier_rows: List[Dict[str, Any]] = []
    budget_attainment_rows: List[Dict[str, Any]] = []
    backend_rows: List[Dict[str, Any]] = []
    llm_recovery_rows: List[Dict[str, Any]] = []
    llm_ablation_rows: List[Dict[str, Any]] = []
    appendix_stress_rows: List[Dict[str, Any]] = []
    appendix_wrapper_rows: List[Dict[str, Any]] = []
    break_prompt_probe_rows: List[Dict[str, Any]] = []
    appendix_subgroup_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    frontier_validation_rows: List[Dict[str, Any]] = []
    policy_diagnostic_rows: List[Dict[str, Any]] = []
    baseline_expansion_rows: List[Dict[str, Any]] = []
    policy_diagnostics: Dict[str, List[Dict[str, Any]]] = {
        "Random": [],
        "Myopic": [],
        "PPO": [],
        "Lagrangian PPO": [],
    }
    cap_diagnostics: Dict[float, List[Dict[str, Any]]] = {}
    diagnostic_first_seed = train_seed_list[0] if train_seed_list else None
    binding_cap_grid = derive_binding_cap_grid(reference_targets.get("session_lengths", []) if reference_targets is not None else [])
    appendix_reference_metrics: Optional[Dict[str, Any]] = None
    mitigation_proxy_results: Dict[str, List[Dict[str, Any]]] = {}
    constraint_track_summaries: List[Dict[str, Any]] = []
    autoplay_off_results: List[Dict[str, Any]] = []

    def record_policy_diagnostic(
        policy_name: str,
        category: str,
        policy_obj: Any,
        split: str,
        eval_mode: str,
        seeds: Sequence[int],
        train_seed: int,
        precomputed_metrics: Optional[Dict[str, Any]] = None,
        progress_task: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not run_policy_diagnostics:
            return precomputed_metrics or {}
        deterministic_flag = eval_mode == "deterministic"
        metrics = precomputed_metrics
        if metrics is None:
            metrics = evaluate_cached(
                policy_obj,
                cfg,
                catalog,
                cards,
                policy_label=policy_name,
                episode_seeds=seeds,
                num_episodes=len(seeds),
                deterministic=deterministic_flag,
                label=f"{policy_name} [{split} {eval_mode}]",
                collect_policy_diagnostics=True,
                progress_tracker=progress_tracker,
                progress_task=progress_task,
            )
        row = action_diagnostics_to_row(policy_name, split, eval_mode, train_seed, metrics, category)
        policy_diagnostic_rows.append(row)
        baseline_expansion_rows.append(
            {
                "policy": policy_name,
                "category": category,
                "eval_split": split,
                "eval_mode": eval_mode,
                "train_seed": int(train_seed),
                "CumWatch": float(metrics["CumWatch"]),
                "NightMinutes": float(metrics["NightMinutes"]),
                "NightFraction": float(metrics.get("NightFraction", float("nan"))),
                "LateNightSessionStartRate": float(metrics.get("LateNightSessionStartRate", float("nan"))),
                "OverCapMinutes": float(metrics["OverCapMinutes"]),
                "CVaR_0.95(L)": float(metrics["CVaR_0.95(L)"]),
                "ReturnRate5": float(metrics.get("ReturnRate5", float("nan"))),
                "ReturnRate15": float(metrics.get("ReturnRate15", float("nan"))),
                "ReturnRate30": float(metrics.get("ReturnRate30", float("nan"))),
                "ReturnRate60": float(metrics["ReturnRate60"]),
                "RepeatRate": float(metrics.get("RepeatRate", float("nan"))),
                "ActionEmpiricalEntropy": float(metrics.get("ActionEmpiricalEntropy", float("nan"))),
                "UniqueClusterCount": float(metrics.get("UniqueClusterCount", float("nan"))),
                "FractionNu1": float(metrics.get("FractionNu1", float("nan"))),
            }
        )
        return metrics

    for seed_index, train_seed in enumerate(train_seed_list, start=1):
        train_seed_start = time.perf_counter()
        LOGGER.info("Train seed start | index=%s/%s | train_seed=%s", seed_index, len(train_seed_list), train_seed)

        # Random baseline
        random_start = time.perf_counter()
        LOGGER.info("train_seed=%s | Random baseline start", train_seed)
        rand_policy = RandomPolicy(cfg.num_actions(), seed=train_seed)
        rand_metrics = run_eval_task(
            episodes_task(eval_task_id(train_seed, "random_test"), f"Random test evaluation (seed {train_seed})", args.eval_episodes),
            rand_policy,
            cfg,
            catalog,
            cards,
            policy_label="Random",
            episode_seeds=test_episode_seeds,
            num_episodes=args.eval_episodes,
            label="Random",
            log_progress=True,
        )
        main_rows.append({"policy": "Random", "backend": "Param", "train_seed": train_seed, **{k: rand_metrics[k] for k in MAIN_POLICY_METRIC_KEYS}})
        append_official_scorecard_result("Random", "stochastic", train_seed, test_episode_seeds, rand_metrics)
        policy_diagnostics["Random"].append(rand_metrics)
        log_block_metrics("Random baseline", random_start, rand_metrics, train_seed)
        record_policy_diagnostic("Random", "baseline", rand_policy, "test", "stochastic", test_episode_seeds, train_seed, precomputed_metrics=rand_metrics)
        record_policy_diagnostic(
            "Random",
            "baseline",
            rand_policy,
            "validation",
            "stochastic",
            val_episode_seeds,
            train_seed,
            progress_task=episodes_task(eval_task_id(train_seed, "random_validation_stochastic"), f"Random validation stochastic evaluation (seed {train_seed})", len(val_episode_seeds)),
        )

        # Myopic baseline
        myopic_start = time.perf_counter()
        LOGGER.info("train_seed=%s | Myopic baseline start", train_seed)
        logged = run_dataset_task(
            transitions_task(train_task_id(train_seed, "logged_dataset"), f"Logged dataset collection (seed {train_seed})", args.random_log_steps),
            cfg=cfg,
            catalog=catalog,
            cards=cards,
            num_transitions=args.random_log_steps,
            seed=train_seed,
        )
        myopic_model = run_myopic_train_task(
            epochs_task(train_task_id(train_seed, "myopic"), f"Myopic training (seed {train_seed})", args.myopic_epochs),
            dataset=logged,
            obs_dim=cfg.obs_dim(),
            num_actions=cfg.num_actions(),
            seed=train_seed,
            epochs=args.myopic_epochs,
            batch_size=args.myopic_batch_size,
            lr=args.myopic_lr,
            device=args.device,
        )
        myopic_checkpoint_path = model_dir / f"myopic_seed{train_seed}.pt"
        torch.save({"state_dict": myopic_model.state_dict()}, myopic_checkpoint_path)
        log_artifact_written(myopic_checkpoint_path, "model_checkpoint")
        myopic_policy = MyopicPolicy(myopic_model)
        myopic_metrics = run_eval_task(
            episodes_task(eval_task_id(train_seed, "myopic_test_deterministic"), f"Myopic test deterministic evaluation (seed {train_seed})", args.eval_episodes),
            myopic_policy,
            cfg,
            catalog,
            cards,
            policy_label="Myopic",
            episode_seeds=test_episode_seeds,
            num_episodes=args.eval_episodes,
            label="Myopic",
            log_progress=True,
        )
        main_rows.append({"policy": "Myopic", "backend": "Param", "train_seed": train_seed, **{k: myopic_metrics[k] for k in MAIN_POLICY_METRIC_KEYS}})
        append_official_scorecard_result("Myopic", "deterministic", train_seed, test_episode_seeds, myopic_metrics)
        mitigation_proxy_results.setdefault("Myopic", []).append(copy.deepcopy(myopic_metrics))
        myopic_metrics_stochastic = run_eval_task(
            episodes_task(eval_task_id(train_seed, "myopic_test_stochastic"), f"Myopic test stochastic evaluation (seed {train_seed})", len(test_episode_seeds)),
            myopic_policy,
            cfg,
            catalog,
            cards,
            policy_label="Myopic",
            episode_seeds=test_episode_seeds,
            num_episodes=len(test_episode_seeds),
            deterministic=False,
            label="Myopic [stochastic]",
            log_progress=not run_policy_diagnostics,
            use_trace=False,
        )
        append_official_scorecard_result("Myopic", "stochastic", train_seed, test_episode_seeds, myopic_metrics_stochastic)
        policy_diagnostics["Myopic"].append(myopic_metrics)
        log_block_metrics("Myopic baseline", myopic_start, myopic_metrics, train_seed)
        record_policy_diagnostic("Myopic", "learned", myopic_policy, "test", "deterministic", test_episode_seeds, train_seed, precomputed_metrics=myopic_metrics)
        record_policy_diagnostic(
            "Myopic",
            "learned",
            myopic_policy,
            "test",
            "stochastic",
            test_episode_seeds,
            train_seed,
            precomputed_metrics=myopic_metrics_stochastic,
        )
        record_policy_diagnostic(
            "Myopic",
            "learned",
            myopic_policy,
            "validation",
            "deterministic",
            val_episode_seeds,
            train_seed,
            progress_task=episodes_task(eval_task_id(train_seed, "myopic_validation_deterministic"), f"Myopic validation deterministic evaluation (seed {train_seed})", len(val_episode_seeds)),
        )
        record_policy_diagnostic(
            "Myopic",
            "learned",
            myopic_policy,
            "validation",
            "stochastic",
            val_episode_seeds,
            train_seed,
            progress_task=episodes_task(eval_task_id(train_seed, "myopic_validation_stochastic"), f"Myopic validation stochastic evaluation (seed {train_seed})", len(val_episode_seeds)),
        )

        official_deterministic_baseline_specs = [
            ("RoundRobinPolicy", lambda: RoundRobinPolicy(cfg)),
            ("LeastRecentPolicy", lambda: LeastRecentPolicy(cfg, seed=train_seed)),
            ("NoveltyGreedyPolicy", lambda: NoveltyGreedyPolicy(cfg, seed=train_seed)),
        ]
        for heuristic_name, heuristic_factory in official_deterministic_baseline_specs:
            heuristic_start = time.perf_counter()
            LOGGER.info("train_seed=%s | Official deterministic baseline start | policy=%s", train_seed, heuristic_name)
            heuristic_policy = heuristic_factory()
            heuristic_metrics = run_eval_task(
                episodes_task(
                    eval_task_id(train_seed, f"{heuristic_name}_official_test_deterministic"),
                    f"{heuristic_name} official deterministic evaluation (seed {train_seed})",
                    args.eval_episodes,
                ),
                heuristic_policy,
                cfg,
                catalog,
                cards,
                policy_label=heuristic_name,
                episode_seeds=test_episode_seeds,
                num_episodes=args.eval_episodes,
                deterministic=True,
                label=f"{heuristic_name} [deterministic]",
                log_progress=not run_heuristics,
                use_trace=False,
            )
            append_official_scorecard_result(heuristic_name, "deterministic", train_seed, test_episode_seeds, heuristic_metrics)
            mitigation_proxy_results.setdefault(heuristic_name, []).append(copy.deepcopy(heuristic_metrics))
            record_policy_diagnostic(
                heuristic_name,
                "heuristic",
                heuristic_policy,
                "test",
                "deterministic",
                test_episode_seeds,
                train_seed,
                precomputed_metrics=heuristic_metrics,
            )
            if run_heuristics:
                record_policy_diagnostic(
                    heuristic_name,
                    "heuristic",
                    heuristic_factory(),
                    "validation",
                    "deterministic",
                    val_episode_seeds,
                    train_seed,
                    progress_task=episodes_task(eval_task_id(train_seed, f"{heuristic_name}_validation_deterministic"), f"{heuristic_name} validation deterministic evaluation (seed {train_seed})", len(val_episode_seeds)),
                )
            log_block_metrics(f"Official deterministic baseline {heuristic_name}", heuristic_start, heuristic_metrics, train_seed)

        if run_heuristics:
            heuristic_specs = [
                ("RandomZ-LowLambda-HighVar", "stochastic", lambda: RandomZLowLambdaHighVarPolicy(cfg, seed=train_seed)),
                ("CyclicZ-LowLambda-HighVar", "deterministic", lambda: CyclicZLowLambdaHighVarPolicy(cfg)),
            ]
            for heuristic_name, heuristic_eval_mode, heuristic_factory in heuristic_specs:
                heuristic_start = time.perf_counter()
                LOGGER.info("train_seed=%s | Heuristic baseline start | policy=%s", train_seed, heuristic_name)
                heuristic_policy = heuristic_factory()
                heuristic_metrics = run_eval_task(
                    episodes_task(eval_task_id(train_seed, f"{heuristic_name}_test_{heuristic_eval_mode}"), f"{heuristic_name} test {heuristic_eval_mode} evaluation (seed {train_seed})", args.eval_episodes),
                    heuristic_policy,
                    cfg,
                    catalog,
                    cards,
                    policy_label=heuristic_name,
                    episode_seeds=test_episode_seeds,
                    num_episodes=args.eval_episodes,
                    deterministic=(heuristic_eval_mode == "deterministic"),
                    label=f"{heuristic_name} [{heuristic_eval_mode}]",
                    collect_policy_diagnostics=True,
                )
                record_policy_diagnostic(
                    heuristic_name,
                    "heuristic",
                    heuristic_policy,
                    "test",
                    heuristic_eval_mode,
                    test_episode_seeds,
                    train_seed,
                    precomputed_metrics=heuristic_metrics,
                )
                record_policy_diagnostic(
                    heuristic_name,
                    "heuristic",
                    heuristic_factory(),
                    "validation",
                    heuristic_eval_mode,
                    val_episode_seeds,
                    train_seed,
                    progress_task=episodes_task(eval_task_id(train_seed, f"{heuristic_name}_validation_{heuristic_eval_mode}"), f"{heuristic_name} validation {heuristic_eval_mode} evaluation (seed {train_seed})", len(val_episode_seeds)),
                )
                log_block_metrics(f"Heuristic baseline {heuristic_name}", heuristic_start, heuristic_metrics, train_seed)

        # PPO baseline
        ppo_start = time.perf_counter()
        LOGGER.info("train_seed=%s | PPO baseline start", train_seed)
        ppo_model, ppo_train = run_ppo_train_task(
            steps_task(train_task_id(train_seed, "ppo"), f"PPO training (seed {train_seed})", args.ppo_steps),
            cfg=cfg,
            catalog=catalog,
            cards=cards,
            seed=train_seed,
            total_steps=args.ppo_steps,
            rollout_steps=args.rollout_steps,
            minibatch_size=args.minibatch_size,
            update_epochs=args.update_epochs,
            lr=args.ppo_lr,
            ent_coef=args.ppo_ent_coef,
            hidden_size=args.ppo_hidden,
            policy_arch="flat",
            device=args.device,
            lagrangian=False,
            validate_every=args.validate_every,
            val_episodes=args.val_episodes,
            val_episode_seeds=val_episode_seeds,
        )
        ppo_checkpoint_path = model_dir / f"ppo_seed{train_seed}.pt"
        torch.save({"state_dict": ppo_model.state_dict(), "train_summary": ppo_train, "config": cfg.to_dict()}, ppo_checkpoint_path)
        log_artifact_written(ppo_checkpoint_path, "model_checkpoint")
        save_training_history(ppo_train.get("rollout_history", []), outdir / f"training_history_ppo_seed{train_seed}.csv")
        ppo_policy = PPOPolicy(ppo_model)
        validation_rows.extend(
            [
                {
                    "method": "PPO",
                    "scale": 1.0,
                    "train_seed": train_seed,
                    **row,
                }
                for row in ppo_train["history"]
            ]
        )
        ppo_val_metrics = run_eval_task(
            episodes_task(eval_task_id(train_seed, "ppo_validation_deterministic"), f"PPO validation deterministic evaluation (seed {train_seed})", len(val_episode_seeds)),
            ppo_policy,
            cfg,
            catalog,
            cards,
            policy_label="PPO",
            episode_seeds=val_episode_seeds,
            num_episodes=len(val_episode_seeds),
            label="PPO validation baseline",
            deterministic=True,
            collect_policy_diagnostics=run_policy_diagnostics,
        )
        ppo_val_status = validation_status(ppo_val_metrics)
        frontier_validation_rows.append(
            {
                "method": "PPO",
                "scale": 1.0,
                "train_seed": train_seed,
                "global_step": int(ppo_train.get("selected_global_step") or 0),
                "validation_CumWatch": float(ppo_val_metrics["CumWatch"]),
                "validation_NightMinutes": float(ppo_val_metrics["NightMinutes"]),
                "validation_OverCapMinutes": float(ppo_val_metrics["OverCapMinutes"]),
                "validation_CVaR_0.95(L)": float(ppo_val_metrics["CVaR_0.95(L)"]),
                "feasible": True,
                "selected": True,
                "selected_operating_point": False,
                "selected_source": str(ppo_train.get("selected_source", "reward_best")),
                "night_budget": float("nan"),
                "over_budget": float("nan"),
                "total_violation": float(ppo_val_status["total_violation"]),
            }
        )
        LOGGER.info(
            "train_seed=%s | PPO validation reference | CumWatch=%.3f | NightMinutes=%.3f | OverCapMinutes=%.3f",
            train_seed,
            float(ppo_val_metrics["CumWatch"]),
            float(ppo_val_metrics["NightMinutes"]),
            float(ppo_val_metrics["OverCapMinutes"]),
            )

        if abs(float(ppo_val_metrics["OverCapMinutes"])) <= 1e-12:
            LOGGER.warning(
                "train_seed=%s | PPO validation OverCapMinutes is zero; constrained over-budget channel may be dead",
                train_seed,
            )

        ppo_test_eval_start = time.perf_counter()
        ppo_metrics = run_eval_task(
            episodes_task(eval_task_id(train_seed, "ppo_test_deterministic"), f"PPO test deterministic evaluation (seed {train_seed})", args.eval_episodes),
            ppo_policy,
            cfg,
            catalog,
            cards,
            policy_label="PPO",
            episode_seeds=test_episode_seeds,
            num_episodes=args.eval_episodes,
            label="PPO",
            log_progress=True,
        )
        main_rows.append({"policy": "PPO", "backend": "Param", "train_seed": train_seed, **{k: ppo_metrics[k] for k in MAIN_POLICY_METRIC_KEYS}})
        append_official_scorecard_result("PPO", "deterministic", train_seed, test_episode_seeds, ppo_metrics)
        frontier_rows.append(frontier_summary_row("PPO", 1.0, "deterministic", train_seed, ppo_metrics))
        mitigation_proxy_results.setdefault("PPO", []).append(copy.deepcopy(ppo_metrics))
        ppo_metrics_stochastic = run_eval_task(
            episodes_task(eval_task_id(train_seed, "ppo_test_stochastic"), f"PPO test stochastic evaluation (seed {train_seed})", len(test_episode_seeds)),
            ppo_policy,
            cfg,
            catalog,
            cards,
            policy_label="PPO",
            episode_seeds=test_episode_seeds,
            num_episodes=len(test_episode_seeds),
            deterministic=False,
            label="PPO [stochastic]",
            log_progress=not run_policy_diagnostics,
            use_trace=False,
        )
        append_official_scorecard_result("PPO", "stochastic", train_seed, test_episode_seeds, ppo_metrics_stochastic)
        autoplay_off_metrics = run_eval_task(
            episodes_task(eval_task_id(train_seed, "ppo_autoplay_off_mitigation"), f"PPO autoplay-off mitigation evaluation (seed {train_seed})", args.eval_episodes),
            ppo_policy,
            cfg,
            catalog,
            cards,
            policy_label="PPO [autoplay_off]",
            wrappers={"autoplay_off": True},
            episode_seeds=test_episode_seeds,
            num_episodes=args.eval_episodes,
            deterministic=True,
            label="PPO [autoplay_off]",
            log_progress=not run_wrapper_baselines,
            use_trace=False,
        )
        frontier_rows.append(frontier_summary_row("PPO+AutoplayOff", float(cfg.autoplay_friction), "deterministic", train_seed, autoplay_off_metrics))
        autoplay_off_results.append(copy.deepcopy(autoplay_off_metrics))
        mitigation_proxy_results.setdefault("PPO+AutoplayOff", []).append(copy.deepcopy(autoplay_off_metrics))
        default_cap_label = f"PPO+Cap({int(cfg.T_cap)})" if float(cfg.T_cap).is_integer() else f"PPO+Cap({cfg.T_cap})"
        default_cap_metrics_det = run_eval_task(
            episodes_task(eval_task_id(train_seed, "ppo_cap_default_deterministic"), f"PPO default session-cap deterministic evaluation (seed {train_seed})", args.eval_episodes),
            ppo_policy,
            cfg,
            catalog,
            cards,
            policy_label=default_cap_label,
            wrappers={"session_cap": True, "T_cap": float(cfg.T_cap)},
            episode_seeds=test_episode_seeds,
            num_episodes=args.eval_episodes,
            deterministic=True,
            label=default_cap_label,
            log_progress=not run_cap_sensitivity,
            use_trace=False,
        )
        append_official_scorecard_result(
            f"PPO + SessionCap({int(cfg.T_cap)})" if float(cfg.T_cap).is_integer() else f"PPO + SessionCap({cfg.T_cap})",
            "deterministic",
            train_seed,
            test_episode_seeds,
            default_cap_metrics_det,
        )
        default_cap_metrics_stochastic = run_eval_task(
            episodes_task(eval_task_id(train_seed, "ppo_cap_default_stochastic"), f"PPO default session-cap stochastic evaluation (seed {train_seed})", args.eval_episodes),
            ppo_policy,
            cfg,
            catalog,
            cards,
            policy_label=default_cap_label,
            wrappers={"session_cap": True, "T_cap": float(cfg.T_cap)},
            episode_seeds=test_episode_seeds,
            num_episodes=args.eval_episodes,
            deterministic=False,
            label=f"{default_cap_label} [stochastic]",
            log_progress=False,
            use_trace=False,
        )
        append_official_scorecard_result(
            f"PPO + SessionCap({int(cfg.T_cap)})" if float(cfg.T_cap).is_integer() else f"PPO + SessionCap({cfg.T_cap})",
            "stochastic",
            train_seed,
            test_episode_seeds,
            default_cap_metrics_stochastic,
        )
        mitigation_proxy_results.setdefault(
            f"PPO + SessionCap({int(cfg.T_cap)})" if float(cfg.T_cap).is_integer() else f"PPO + SessionCap({cfg.T_cap})",
            [],
        ).append(copy.deepcopy(default_cap_metrics_det))
        policy_diagnostics["PPO"].append(ppo_metrics)
        log_phase_complete(
            "PPO training + validation reference",
            ppo_start,
            extra=f"train_seed={train_seed} | selected_global_step={ppo_train.get('selected_global_step')}",
        )
        record_policy_diagnostic("PPO", "ppo_main", ppo_policy, "test", "deterministic", test_episode_seeds, train_seed, precomputed_metrics=ppo_metrics)
        record_policy_diagnostic(
            "PPO",
            "ppo_main",
            ppo_policy,
            "test",
            "stochastic",
            test_episode_seeds,
            train_seed,
            precomputed_metrics=ppo_metrics_stochastic,
        )
        record_policy_diagnostic("PPO", "ppo_main", ppo_policy, "validation", "deterministic", val_episode_seeds, train_seed, precomputed_metrics=ppo_val_metrics)
        record_policy_diagnostic("PPO+AutoplayOff", "wrapper_baseline", ppo_policy, "test", "deterministic", test_episode_seeds, train_seed, precomputed_metrics=autoplay_off_metrics)
        record_policy_diagnostic(
            "PPO",
            "ppo_main",
            ppo_policy,
            "validation",
            "stochastic",
            val_episode_seeds,
            train_seed,
            progress_task=episodes_task(eval_task_id(train_seed, "ppo_validation_stochastic"), f"PPO validation stochastic evaluation (seed {train_seed})", len(val_episode_seeds)),
        )
        log_block_metrics("PPO baseline test evaluation", ppo_test_eval_start, ppo_metrics, train_seed)
        if run_appendix_stress and appendix_reference_metrics is None and diagnostic_first_seed is not None and int(train_seed) == int(diagnostic_first_seed):
            appendix_reference_metrics = copy.deepcopy(ppo_metrics)
        if run_wrapper_baselines:
            appendix_wrapper_rows.append(
                appendix_dashboard_row("wrapper_baseline", "autoplay_off", "PPO", train_seed, autoplay_off_metrics)
            )
            autoplay_subgroup = susceptibility_bin_summary(
                autoplay_off_metrics.get("EpisodeSusceptibilityValues", []),
                autoplay_off_metrics.get("EpisodeCumWatchValues", []),
            )
            appendix_subgroup_rows.append(
                {
                    "category": "wrapper_baseline",
                    "condition": "autoplay_off",
                    "policy": "PPO",
                    "train_seed": int(train_seed),
                    "metric": "CumWatch",
                    "top_bottom_ratio": float(autoplay_subgroup.get("top_bottom_ratio", float("nan"))),
                    "top_mean": float(autoplay_subgroup.get("top_mean", float("nan"))),
                    "bottom_mean": float(autoplay_subgroup.get("bottom_mean", float("nan"))),
                }
            )
        if run_break_prompt_probe:
            break_prompt_start = time.perf_counter()
            break_prompt_metrics = run_eval_task(
                episodes_task(
                    eval_task_id(train_seed, "ppo_break_prompt_probe"),
                    f"PPO break-prompt probe evaluation (seed {train_seed})",
                    len(appendix_episode_seeds),
                ),
                ppo_policy,
                cfg,
                catalog,
                cards,
                policy_label="PPO",
                wrappers={"break_prompt": True},
                episode_seeds=appendix_episode_seeds,
                num_episodes=len(appendix_episode_seeds),
                deterministic=True,
                label="PPO [break_prompt probe]",
                log_progress=True,
                use_trace=False,
            )
            break_prompt_probe_rows.append(
                appendix_dashboard_row("threshold_probe", "break_prompt", "PPO", train_seed, break_prompt_metrics)
            )
            log_block_metrics("PPO break-prompt probe", break_prompt_start, break_prompt_metrics, train_seed)

        constraint_track_spec = build_constraint_track_spec(
            ppo_val_metrics,
            build_night_proxy_orthogonality_audit(mitigation_proxy_results),
        )
        constraint_track_spec.update(
            {
                "train_seed": int(train_seed),
                "corr_threshold": float(NIGHT_CUMWATCH_PROXY_CORR_THRESHOLD),
            }
        )
        constraint_track_summaries.append(copy.deepcopy(constraint_track_spec))
        LOGGER.info(
            "train_seed=%s | Constraint-track screening | channels=%s | policy_mean_corr=%.3f | episode_corr=%.3f | reason=%s",
            train_seed,
            constraint_track_spec["channels_label"],
            float(constraint_track_spec["night_minutes_cumwatch_corr_policy_means"]),
            float(constraint_track_spec["night_minutes_cumwatch_corr_episodes"]),
            constraint_track_spec["reason"],
        )

        if llm_fit_pending and scorer is not None:
            fusion_start = time.perf_counter()
            LOGGER.info(
                "LLM fusion fitting start | train_seed=%s | context_policies=%s | heldout_seed_count=%s",
                train_seed,
                ["random", "ppo"],
                len(llm_target_loss_seeds),
            )
            llm_cfg, llm_fit = fit_llm_fusion(
                llm_cfg,
                catalog,
                cards,
                scorer,
                context_seeds=llm_context_seeds,
                target_loss_seeds=llm_target_loss_seeds,
                ppo_policy=ppo_policy,
                ppo_deterministic=True,
            )
            llm_fusion_fit_path = outdir / "llm_fusion_fit.json"
            start_file_task(files_task(file_task_id("llm_fusion_fit"), "LLM fusion fit", 1))
            json_dumps_progress({**llm_cfg.to_dict(), **llm_fit}, llm_fusion_fit_path, "llm_fusion_fit_json")
            finish_file_task(extra="llm fusion fit complete")
            scorer.cfg = llm_cfg
            llm_fit_pending = False
            log_phase_complete(
                "LLM fusion fitting",
                fusion_start,
                extra=f"omega_r_llm={llm_cfg.omega_r_llm:.3f} | omega_c_llm={llm_cfg.omega_c_llm:.3f}",
            )

        # Lagrangian PPO sweep for the frontier.
        lag_runs: List[Dict[str, Any]] = []
        def run_lagrangian_candidate(
            *,
            scale_index: int,
            scale: float,
            night_budget: Optional[float],
            over_budget: Optional[float],
            dual_lr_value: float,
            total_steps_value: int,
            candidate_index: int,
            val_episode_count: int,
            val_episode_seed_subset: Sequence[int],
            stage_name: str,
            record_artifacts: bool,
            progress_task: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            stage_seed_offset = 0 if stage_name == "confirm" else 100_000
            candidate_seed = (
                train_seed
                + stage_seed_offset
                + scale_index * 10_000
                + candidate_index * 137
                + int(round(scale * 100))
            )
            LOGGER.info(
                "train_seed=%s | Lagrangian PPO %s candidate start | scale=%s | dual_lr=%s | total_steps=%s | val_episodes=%s | night_budget=%.3f | over_budget=%.3f",
                train_seed,
                stage_name,
                scale,
                dual_lr_value,
                total_steps_value,
                val_episode_count,
                float("nan") if night_budget is None else float(night_budget),
                float("nan") if over_budget is None else float(over_budget),
            )
            lag_model_candidate, lag_train_candidate = run_ppo_train_task(
                progress_task
                if progress_task is not None
                else steps_task(
                    train_task_id(train_seed, f"lagppo_scale_{filename_slug(f'{scale:.4f}')}_{stage_name}_dual_{filename_slug(f'{dual_lr_value:.4f}')}"),
                    f"LagPPO scale={scale:.2f} {stage_name} dual_lr={dual_lr_value:.4f} (seed {train_seed})",
                    int(total_steps_value),
                ),
                cfg=cfg,
                catalog=catalog,
                cards=cards,
                seed=candidate_seed,
                total_steps=int(total_steps_value),
                rollout_steps=args.rollout_steps,
                minibatch_size=args.minibatch_size,
                update_epochs=args.update_epochs,
                lr=args.ppo_lr,
                ent_coef=args.ppo_ent_coef,
                hidden_size=args.ppo_hidden,
                policy_arch="flat",
                device=args.device,
                lagrangian=True,
                night_budget=night_budget,
                over_budget=over_budget,
                dual_lr=float(dual_lr_value),
                validate_every=args.validate_every,
                val_episodes=int(val_episode_count),
                val_episode_seeds=val_episode_seed_subset,
            )
            selected_val_candidate = copy.deepcopy(lag_train_candidate.get("selected_validation") or {})
            budget_row: Optional[Dict[str, Any]] = None
            if record_artifacts:
                dual_tag = filename_slug(f"{dual_lr_value:.4f}")
                scale_tag = filename_slug(f"{scale:.4f}")
                history_path = outdir / f"training_history_lagppo_seed{train_seed}_scale{scale_tag}_dual{dual_tag}_steps{int(total_steps_value)}.csv"
                save_training_history(lag_train_candidate.get("rollout_history", []), history_path)
                budget_row = {
                    "method": "LagrangianPPO",
                    "scale": float(scale),
                    "train_seed": int(train_seed),
                    "dual_lr": float(dual_lr_value),
                    "total_steps": int(total_steps_value),
                    "target_night_budget": float("nan") if night_budget is None else float(night_budget),
                    "achieved_NightMinutes": float(selected_val_candidate.get("validation_NightMinutes", float("nan"))),
                    "target_over_budget": float("nan") if over_budget is None else float(over_budget),
                    "achieved_OverCapMinutes": float(selected_val_candidate.get("validation_OverCapMinutes", float("nan"))),
                    "validation_CumWatch": float(selected_val_candidate.get("validation_CumWatch", float("nan"))),
                    "validation_CVaR_0.95(L)": float(selected_val_candidate.get("validation_CVaR_0.95(L)", float("nan"))),
                    "feasible": bool(lag_train_candidate.get("selected_feasible", False)),
                    "selected_candidate": False,
                    "selected_operating_point": False,
                    "selected_checkpoint_step": int(lag_train_candidate.get("selected_global_step") or 0),
                    "selected_source": str(lag_train_candidate.get("selected_source")),
                    "final_lambda1": float(lag_train_candidate.get("lambda1", 0.0)),
                    "final_lambda2": float(lag_train_candidate.get("lambda2", 0.0)),
                    "total_violation": float(selected_val_candidate.get("total_violation", float("nan"))),
                    "constraint_channels": str(constraint_track_spec["channels_label"]),
                    "night_minutes_cumwatch_corr_policy_means": float(constraint_track_spec["night_minutes_cumwatch_corr_policy_means"]),
                    "night_minutes_cumwatch_corr_episodes": float(constraint_track_spec["night_minutes_cumwatch_corr_episodes"]),
                    "night_minutes_is_scalar_proxy": bool(constraint_track_spec["night_minutes_is_scalar_proxy"]),
                }
                budget_attainment_rows.append(budget_row)
            return {
                "model": lag_model_candidate,
                "policy": PPOPolicy(lag_model_candidate),
                "train_summary": lag_train_candidate,
                "selected_validation": selected_val_candidate,
                "budget_row": budget_row,
                "dual_lr": float(dual_lr_value),
                "total_steps": int(total_steps_value),
                "night_budget": None if night_budget is None else float(night_budget),
                "over_budget": None if over_budget is None else float(over_budget),
                "scale": float(scale),
                "scale_index": int(scale_index),
                "stage": str(stage_name),
            }

        def finalize_lagrangian_candidate(
            selected_candidate: Dict[str, Any],
            lag_start: float,
            test_task: Dict[str, Any],
            test_stochastic_task: Optional[Dict[str, Any]] = None,
            validation_deterministic_task: Optional[Dict[str, Any]] = None,
            validation_stochastic_task: Optional[Dict[str, Any]] = None,
        ) -> None:
            scale = float(selected_candidate["scale"])
            night_budget = selected_candidate["night_budget"]
            over_budget = selected_candidate["over_budget"]
            if selected_candidate.get("budget_row") is not None:
                selected_candidate["budget_row"]["selected_candidate"] = True
            selected_lag_model = selected_candidate["model"]
            selected_lag_train = selected_candidate["train_summary"]
            validation_rows.extend(
                [
                    {
                        "method": "LagrangianPPO",
                        "scale": float(scale),
                        "train_seed": train_seed,
                        "dual_lr": float(selected_candidate["dual_lr"]),
                        "total_steps": int(selected_candidate["total_steps"]),
                        **row,
                    }
                    for row in selected_lag_train["history"]
                ]
            )
            selected_val = copy.deepcopy(selected_candidate.get("selected_validation") or {})
            frontier_validation_rows.append(
                {
                    "method": "LagrangianPPO",
                    "scale": float(scale),
                    "train_seed": train_seed,
                    "dual_lr": float(selected_candidate["dual_lr"]),
                    "total_steps": int(selected_candidate["total_steps"]),
                    "global_step": int(selected_val.get("global_step", 0)),
                    "validation_CumWatch": float(selected_val.get("validation_CumWatch", float("nan"))),
                    "validation_NightMinutes": float(selected_val.get("validation_NightMinutes", float("nan"))),
                    "validation_OverCapMinutes": float(selected_val.get("validation_OverCapMinutes", float("nan"))),
                    "validation_CVaR_0.95(L)": float(selected_val.get("validation_CVaR_0.95(L)", float("nan"))),
                    "feasible": bool(selected_lag_train.get("selected_feasible", False)),
                    "selected": True,
                    "selected_operating_point": False,
                    "selected_source": str(selected_lag_train.get("selected_source")),
                    "night_budget": float("nan") if night_budget is None else float(night_budget),
                    "over_budget": float("nan") if over_budget is None else float(over_budget),
                    "total_violation": float(selected_val.get("total_violation", float("nan"))),
                    "constraint_channels": str(constraint_track_spec["channels_label"]),
                    "night_minutes_cumwatch_corr_policy_means": float(constraint_track_spec["night_minutes_cumwatch_corr_policy_means"]),
                    "night_minutes_cumwatch_corr_episodes": float(constraint_track_spec["night_minutes_cumwatch_corr_episodes"]),
                    "night_minutes_is_scalar_proxy": bool(constraint_track_spec["night_minutes_is_scalar_proxy"]),
                }
            )
            lag_checkpoint_path = model_dir / f"lagppo_scale{scale}_seed{train_seed}.pt"
            torch.save(
                {
                    "state_dict": selected_lag_model.state_dict(),
                    "train_summary": selected_lag_train,
                    "config": cfg.to_dict(),
                    "night_budget": night_budget,
                    "over_budget": over_budget,
                    "dual_lr": float(selected_candidate["dual_lr"]),
                    "total_steps": int(selected_candidate["total_steps"]),
                },
                lag_checkpoint_path,
            )
            log_artifact_written(lag_checkpoint_path, "model_checkpoint")
            lag_policy = selected_candidate["policy"]
            lag_metrics = run_eval_task(
                test_task,
                lag_policy,
                cfg,
                catalog,
                cards,
                policy_label=f"LagPPO(scale={scale})",
                episode_seeds=test_episode_seeds,
                num_episodes=args.eval_episodes,
                label=f"LagPPO(scale={scale},dual_lr={selected_candidate['dual_lr']},steps={selected_candidate['total_steps']})",
                log_progress=True,
            )
            frontier_rows.append(frontier_summary_row("LagrangianPPO", scale, "deterministic", train_seed, lag_metrics))
            log_block_metrics(f"Lagrangian PPO scale={scale}", lag_start, lag_metrics, train_seed)
            lag_runs.append(
                {
                    "scale": float(scale),
                    "policy": lag_policy,
                    "test_metrics": lag_metrics,
                    "train_summary": selected_lag_train,
                    "selected_validation": selected_val,
                    "night_budget": None if night_budget is None else float(night_budget),
                    "over_budget": None if over_budget is None else float(over_budget),
                    "dual_lr": float(selected_candidate["dual_lr"]),
                    "total_steps": int(selected_candidate["total_steps"]),
                    "budget_row": selected_candidate.get("budget_row"),
                }
            )
            lag_policy_name = f"LagPPO(scale={scale})"
            record_policy_diagnostic(
                lag_policy_name,
                "lagrangian_sweep",
                lag_policy,
                "test",
                "deterministic",
                test_episode_seeds,
                train_seed,
                precomputed_metrics=lag_metrics,
            )
            record_policy_diagnostic(
                lag_policy_name,
                "lagrangian_sweep",
                lag_policy,
                "test",
                "stochastic",
                test_episode_seeds,
                train_seed,
                progress_task=test_stochastic_task,
            )
            record_policy_diagnostic(
                lag_policy_name,
                "lagrangian_sweep",
                lag_policy,
                "validation",
                "deterministic",
                val_episode_seeds,
                train_seed,
                progress_task=validation_deterministic_task,
            )
            record_policy_diagnostic(
                lag_policy_name,
                "lagrangian_sweep",
                lag_policy,
                "validation",
                "stochastic",
                val_episode_seeds,
                train_seed,
                progress_task=validation_stochastic_task,
            )

        if lag_search_mode == "two_stage":
            lag_search_winners: List[Dict[str, Any]] = []
            for scale_index, scale in enumerate(lag_scales, start=1):
                scale = float(scale)
                lag_start = time.perf_counter()
                night_budget, over_budget = scaled_constraint_budgets(scale, ppo_val_metrics, constraint_track_spec)
                search_candidates: List[Dict[str, Any]] = []
                for candidate_index, dual_lr_value in enumerate(lag_dual_lr_grid_search, start=1):
                    scale_tag = filename_slug(f"{scale:.4f}")
                    dual_tag = filename_slug(f"{dual_lr_value:.4f}")
                    search_candidates.append(
                        run_lagrangian_candidate(
                            scale_index=scale_index,
                            scale=scale,
                            night_budget=night_budget,
                            over_budget=over_budget,
                            dual_lr_value=dual_lr_value,
                            total_steps_value=lag_search_steps,
                            candidate_index=candidate_index,
                            val_episode_count=lag_search_val_episodes,
                            val_episode_seed_subset=lag_search_val_episode_seeds,
                            stage_name="search",
                            record_artifacts=False,
                            progress_task=steps_task(
                                train_task_id(train_seed, f"lagppo_scale_{scale_tag}_search_dual_{dual_tag}"),
                                f"LagPPO scale={scale:.2f} search dual_lr={dual_lr_value:.4f} (seed {train_seed})",
                                lag_search_steps,
                            ),
                        )
                    )
                selected_search_candidate = choose_lagrangian_candidate(search_candidates)
                selected_search_candidate["lag_start"] = float(lag_start)
                lag_search_winners.append(selected_search_candidate)
            promoted_search_candidates = sorted(lag_search_winners, key=lagrangian_candidate_sort_key)[: min(lag_topk_candidates, len(lag_search_winners))]
            LOGGER.info(
                "train_seed=%s | Lagrangian PPO two-stage promotion | promoted=%s",
                train_seed,
                [
                    {
                        "scale": float(candidate["scale"]),
                        "dual_lr": float(candidate["dual_lr"]),
                        "validation_CumWatch": float((candidate.get("selected_validation") or {}).get("validation_CumWatch", float("nan"))),
                        "total_violation": float((candidate.get("selected_validation") or {}).get("total_violation", float("nan"))),
                        "feasible": bool(candidate["train_summary"].get("selected_feasible", False)),
                    }
                    for candidate in promoted_search_candidates
                ],
            )
            for confirm_index, promoted_candidate in enumerate(promoted_search_candidates, start=1):
                confirmed_candidate = run_lagrangian_candidate(
                    scale_index=int(promoted_candidate["scale_index"]),
                    scale=float(promoted_candidate["scale"]),
                    night_budget=None if promoted_candidate["night_budget"] is None else float(promoted_candidate["night_budget"]),
                    over_budget=None if promoted_candidate["over_budget"] is None else float(promoted_candidate["over_budget"]),
                    dual_lr_value=float(promoted_candidate["dual_lr"]),
                    total_steps_value=lag_full_steps,
                    candidate_index=1,
                    val_episode_count=lag_full_val_episodes,
                    val_episode_seed_subset=lag_full_val_episode_seeds,
                    stage_name="confirm",
                    record_artifacts=True,
                    progress_task=steps_task(
                        train_task_id(train_seed, f"lagppo_confirm_{confirm_index}"),
                        f"LagPPO confirm candidate {confirm_index} (seed {train_seed})",
                        lag_full_steps,
                    ),
                )
                finalize_lagrangian_candidate(
                    confirmed_candidate,
                    float(promoted_candidate["lag_start"]),
                    episodes_task(
                        eval_task_id(train_seed, f"lagppo_confirm_{confirm_index}_test_deterministic"),
                        f"LagPPO confirm candidate {confirm_index} test deterministic evaluation (seed {train_seed})",
                        args.eval_episodes,
                    ),
                    episodes_task(
                        eval_task_id(train_seed, f"lagppo_confirm_{confirm_index}_test_stochastic"),
                        f"LagPPO confirm candidate {confirm_index} test stochastic evaluation (seed {train_seed})",
                        len(test_episode_seeds),
                    ) if run_policy_diagnostics else None,
                    episodes_task(
                        eval_task_id(train_seed, f"lagppo_confirm_{confirm_index}_validation_deterministic"),
                        f"LagPPO confirm candidate {confirm_index} validation deterministic evaluation (seed {train_seed})",
                        len(val_episode_seeds),
                    ) if run_policy_diagnostics else None,
                    episodes_task(
                        eval_task_id(train_seed, f"lagppo_confirm_{confirm_index}_validation_stochastic"),
                        f"LagPPO confirm candidate {confirm_index} validation stochastic evaluation (seed {train_seed})",
                        len(val_episode_seeds),
                    ) if run_policy_diagnostics else None,
                )
        else:
            for scale_index, scale in enumerate(lag_scales, start=1):
                scale = float(scale)
                lag_start = time.perf_counter()
                night_budget, over_budget = scaled_constraint_budgets(scale, ppo_val_metrics, constraint_track_spec)
                lag_candidates: List[Dict[str, Any]] = []
                for candidate_index, dual_lr_value in enumerate(lag_dual_lr_grid_full, start=1):
                    scale_tag = filename_slug(f"{scale:.4f}")
                    dual_tag = filename_slug(f"{dual_lr_value:.4f}")
                    lag_candidates.append(
                        run_lagrangian_candidate(
                            scale_index=scale_index,
                            scale=scale,
                            night_budget=night_budget,
                            over_budget=over_budget,
                            dual_lr_value=dual_lr_value,
                            total_steps_value=lag_full_steps,
                            candidate_index=candidate_index,
                            val_episode_count=lag_full_val_episodes,
                            val_episode_seed_subset=lag_full_val_episode_seeds,
                            stage_name="confirm",
                            record_artifacts=True,
                            progress_task=steps_task(
                                train_task_id(train_seed, f"lagppo_scale_{scale_tag}_dual_{dual_tag}"),
                                f"LagPPO scale={scale:.2f} dual_lr={dual_lr_value:.4f} (seed {train_seed})",
                                lag_full_steps,
                            ),
                        )
                    )
                extension_tasks_executed = False
                if run_frontier_artifacts and not any(bool(candidate["train_summary"].get("selected_feasible", False)) for candidate in lag_candidates):
                    if extended_lag_steps > lag_full_steps:
                        extension_tasks_executed = True
                        LOGGER.info(
                            "train_seed=%s | scale=%s | no feasible constrained candidate after dual_lr sweep; extending training budget to %s",
                            train_seed,
                            scale,
                            extended_lag_steps,
                        )
                        for extra_index, dual_lr_value in enumerate(lag_dual_lr_grid_full, start=len(lag_candidates) + 1):
                            dual_tag = filename_slug(f"{dual_lr_value:.4f}")
                            lag_candidates.append(
                                run_lagrangian_candidate(
                                    scale_index=scale_index,
                                    scale=scale,
                                    night_budget=night_budget,
                                    over_budget=over_budget,
                                    dual_lr_value=dual_lr_value,
                                    total_steps_value=extended_lag_steps,
                                    candidate_index=extra_index,
                                    val_episode_count=lag_full_val_episodes,
                                    val_episode_seed_subset=lag_full_val_episode_seeds,
                                    stage_name="confirm",
                                    record_artifacts=True,
                                    progress_task=steps_task(
                                        train_task_id(train_seed, f"lagppo_scale_{scale_tag}_extend_dual_{dual_tag}"),
                                        f"LagPPO scale={scale:.2f} extend dual_lr={dual_lr_value:.4f} (seed {train_seed})",
                                        extended_lag_steps,
                                    ),
                                )
                            )
                elif run_frontier_artifacts:
                    extension_tasks_executed = False
                if run_frontier_artifacts and not extension_tasks_executed:
                    for dual_lr_value in lag_dual_lr_grid_full:
                        dual_tag = filename_slug(f"{dual_lr_value:.4f}")
                        skip_task(
                            steps_task(
                                train_task_id(train_seed, f"lagppo_scale_{scale_tag}_extend_dual_{dual_tag}"),
                                f"LagPPO scale={scale:.2f} extend dual_lr={dual_lr_value:.4f} (seed {train_seed})",
                                extended_lag_steps,
                            ),
                            "extension not needed",
                        )
                selected_candidate = choose_lagrangian_candidate(lag_candidates)
                finalize_lagrangian_candidate(
                    selected_candidate,
                    lag_start,
                    episodes_task(
                        eval_task_id(train_seed, f"lagppo_scale_{filename_slug(f'{scale:.4f}')}_test_deterministic"),
                        f"LagPPO scale={scale:.2f} test deterministic evaluation (seed {train_seed})",
                        args.eval_episodes,
                    ),
                    episodes_task(
                        eval_task_id(train_seed, f"lagppo_scale_{filename_slug(f'{scale:.4f}')}_test_stochastic"),
                        f"LagPPO scale={scale:.2f} test stochastic evaluation (seed {train_seed})",
                        len(test_episode_seeds),
                    ) if run_policy_diagnostics else None,
                    episodes_task(
                        eval_task_id(train_seed, f"lagppo_scale_{filename_slug(f'{scale:.4f}')}_validation_deterministic"),
                        f"LagPPO scale={scale:.2f} validation deterministic evaluation (seed {train_seed})",
                        len(val_episode_seeds),
                    ) if run_policy_diagnostics else None,
                    episodes_task(
                        eval_task_id(train_seed, f"lagppo_scale_{filename_slug(f'{scale:.4f}')}_validation_stochastic"),
                        f"LagPPO scale={scale:.2f} validation stochastic evaluation (seed {train_seed})",
                        len(val_episode_seeds),
                    ) if run_policy_diagnostics else None,
                )

        feasible_lag_runs = [run for run in lag_runs if bool(run["train_summary"].get("selected_feasible", False))]
        if not feasible_lag_runs:
            LOGGER.warning("train_seed=%s | all constrained scales are infeasible on validation", train_seed)
            chosen = choose_lagrangian_candidate(lag_runs)
        else:
            feasible_lag_runs = sorted(
                feasible_lag_runs,
                key=lambda run: (
                    float(run["scale"]),
                    float(run["night_budget"]) if run["night_budget"] is not None else float("inf"),
                    float(run["over_budget"]) if run["over_budget"] is not None else float("inf"),
                ),
            )
            chosen = feasible_lag_runs[len(feasible_lag_runs) // 2]

        selected_validation_rows = [
            row for row in frontier_validation_rows
                    if row["method"] == "LagrangianPPO" and int(row["train_seed"]) == int(train_seed)
        ]
        for row in selected_validation_rows:
            row["selected_operating_point"] = bool(abs(float(row["scale"]) - float(chosen["scale"])) < 1e-12)
        if chosen.get("budget_row") is not None:
            chosen["budget_row"]["selected_operating_point"] = True

        if selected_validation_rows and all(bool(row["feasible"]) for row in selected_validation_rows):
            if max(abs(float(row["validation_NightMinutes"])) for row in selected_validation_rows) <= 1e-12 and max(abs(float(row["validation_OverCapMinutes"])) for row in selected_validation_rows) <= 1e-12:
                LOGGER.warning(
                    "train_seed=%s | all constrained scales are trivially feasible because the validation cost channels are inactive",
                    train_seed,
                )

        chosen_scale = float(chosen["scale"])
        chosen_lag_policy = chosen["policy"]
        chosen_lag_metrics = chosen["test_metrics"]
        if not bool(chosen["train_summary"].get("selected_feasible", False)):
            LOGGER.warning(
                "train_seed=%s | selected operating point scale=%s is infeasible on validation and was chosen via least violation",
                train_seed,
                chosen_scale,
            )
        main_rows.append({"policy": "Lagrangian PPO", "backend": "Param", "train_seed": train_seed, **{k: chosen_lag_metrics[k] for k in MAIN_POLICY_METRIC_KEYS}})
        append_official_scorecard_result("Lagrangian PPO", "deterministic", train_seed, test_episode_seeds, chosen_lag_metrics)
        mitigation_proxy_results.setdefault("Lagrangian PPO", []).append(copy.deepcopy(chosen_lag_metrics))
        chosen_lag_metrics_stochastic = run_eval_task(
            episodes_task(eval_task_id(train_seed, "lagppo_main_test_stochastic"), f"Lagrangian PPO test stochastic evaluation (seed {train_seed})", len(test_episode_seeds)),
            chosen_lag_policy,
            cfg,
            catalog,
            cards,
            policy_label="Lagrangian PPO",
            episode_seeds=test_episode_seeds,
            num_episodes=len(test_episode_seeds),
            deterministic=False,
            label="Lagrangian PPO [stochastic]",
            log_progress=not run_policy_diagnostics,
            use_trace=False,
        )
        append_official_scorecard_result("Lagrangian PPO", "stochastic", train_seed, test_episode_seeds, chosen_lag_metrics_stochastic)
        policy_diagnostics["Lagrangian PPO"].append(chosen_lag_metrics)
        chosen_lag_val_det = (
            run_eval_task(
                episodes_task(
                    eval_task_id(train_seed, "lagppo_main_validation_deterministic"),
                    f"Lagrangian PPO validation deterministic evaluation (seed {train_seed})",
                    len(val_episode_seeds),
                ),
                chosen_lag_policy,
                cfg,
                catalog,
                cards,
                policy_label=f"LagPPO(scale={chosen_scale})",
                episode_seeds=val_episode_seeds,
                num_episodes=len(val_episode_seeds),
                deterministic=True,
                label=f"LagPPO(scale={chosen_scale}) [validation deterministic]",
                collect_policy_diagnostics=True,
            )
            if run_policy_diagnostics
            else {}
        )
        record_policy_diagnostic(
            "Lagrangian PPO",
            "lagrangian",
            chosen_lag_policy,
            "test",
            "deterministic",
            test_episode_seeds,
            train_seed,
            precomputed_metrics=chosen_lag_metrics,
        )
        record_policy_diagnostic(
            "Lagrangian PPO",
            "lagrangian",
            chosen_lag_policy,
            "test",
            "stochastic",
            test_episode_seeds,
            train_seed,
            precomputed_metrics=chosen_lag_metrics_stochastic,
        )
        record_policy_diagnostic(
            "Lagrangian PPO",
            "lagrangian",
            chosen_lag_policy,
            "validation",
            "deterministic",
            val_episode_seeds,
            train_seed,
            precomputed_metrics=chosen_lag_val_det,
        )
        record_policy_diagnostic(
            "Lagrangian PPO",
            "lagrangian",
            chosen_lag_policy,
            "validation",
            "stochastic",
            val_episode_seeds,
            train_seed,
            progress_task=episodes_task(eval_task_id(train_seed, "lagppo_main_validation_stochastic"), f"Lagrangian PPO validation stochastic evaluation (seed {train_seed})", len(val_episode_seeds)),
        )

        if run_diagnostic_sweeps and diagnostic_first_seed is not None and int(train_seed) == int(diagnostic_first_seed):
            sweep_specs = [
                ("PPO-LowLR", {"lr": args.ppo_lr * 0.5, "ent_coef": args.ppo_ent_coef, "hidden_size": args.ppo_hidden, "total_steps": args.ppo_steps, "policy_arch": "flat", "category": "ppo_sweep"}),
                ("PPO-HighEntropy", {"lr": args.ppo_lr, "ent_coef": args.ppo_ent_coef * 2.0, "hidden_size": args.ppo_hidden, "total_steps": args.ppo_steps, "policy_arch": "flat", "category": "ppo_sweep"}),
                ("PPO-Wide", {"lr": args.ppo_lr, "ent_coef": args.ppo_ent_coef, "hidden_size": args.ppo_hidden * 2, "total_steps": args.ppo_steps, "policy_arch": "flat", "category": "ppo_sweep"}),
                ("PPO-Longer", {"lr": args.ppo_lr, "ent_coef": args.ppo_ent_coef, "hidden_size": args.ppo_hidden, "total_steps": int(args.ppo_steps * 1.5), "policy_arch": "flat", "category": "ppo_sweep"}),
                ("FactorizedPPO", {"lr": args.ppo_lr, "ent_coef": args.ppo_ent_coef, "hidden_size": args.ppo_hidden, "total_steps": args.ppo_steps, "policy_arch": "factorized", "category": "factorized_ppo"}),
            ]
            for sweep_name, sweep_cfg in sweep_specs:
                sweep_start = time.perf_counter()
                LOGGER.info(
                    "train_seed=%s | PPO diagnostic sweep start | variant=%s | lr=%s | ent_coef=%s | hidden_size=%s | total_steps=%s | policy_arch=%s",
                    train_seed,
                    sweep_name,
                    sweep_cfg["lr"],
                    sweep_cfg["ent_coef"],
                    sweep_cfg["hidden_size"],
                    sweep_cfg["total_steps"],
                    sweep_cfg["policy_arch"],
                )
                sweep_model, sweep_train = run_ppo_train_task(
                    steps_task(
                        train_task_id(train_seed, sweep_name),
                        f"{sweep_name} training (seed {train_seed})",
                        int(sweep_cfg["total_steps"]),
                    ),
                    cfg=cfg,
                    catalog=catalog,
                    cards=cards,
                    seed=train_seed + 50_000 + len(baseline_expansion_rows),
                    total_steps=int(sweep_cfg["total_steps"]),
                    rollout_steps=args.rollout_steps,
                    minibatch_size=args.minibatch_size,
                    update_epochs=args.update_epochs,
                    lr=float(sweep_cfg["lr"]),
                    ent_coef=float(sweep_cfg["ent_coef"]),
                    hidden_size=int(sweep_cfg["hidden_size"]),
                    policy_arch=str(sweep_cfg["policy_arch"]),
                    device=args.device,
                    lagrangian=False,
                    validate_every=args.validate_every,
                    val_episodes=args.val_episodes,
                    val_episode_seeds=val_episode_seeds,
                )
                save_training_history(
                    sweep_train.get("rollout_history", []),
                    outdir / f"training_history_ppo_seed{train_seed}_{filename_slug(sweep_name)}.csv",
                )
                sweep_policy = PPOPolicy(sweep_model)
                sweep_test_det = run_eval_task(
                    episodes_task(
                        eval_task_id(train_seed, f"{sweep_name}_test_deterministic"),
                        f"{sweep_name} test deterministic evaluation (seed {train_seed})",
                        len(test_episode_seeds),
                    ),
                    sweep_policy,
                    cfg,
                    catalog,
                    cards,
                    policy_label=sweep_name,
                    episode_seeds=test_episode_seeds,
                    num_episodes=len(test_episode_seeds),
                    deterministic=True,
                    label=f"{sweep_name} [test deterministic]",
                    use_trace=False,
                )
                record_policy_diagnostic(sweep_name, str(sweep_cfg["category"]), sweep_policy, "test", "deterministic", test_episode_seeds, train_seed, precomputed_metrics=sweep_test_det)
                record_policy_diagnostic(
                    sweep_name,
                    str(sweep_cfg["category"]),
                    sweep_policy,
                    "test",
                    "stochastic",
                    test_episode_seeds,
                    train_seed,
                    progress_task=episodes_task(eval_task_id(train_seed, f"{sweep_name}_test_stochastic"), f"{sweep_name} test stochastic evaluation (seed {train_seed})", len(test_episode_seeds)),
                )
                record_policy_diagnostic(
                    sweep_name,
                    str(sweep_cfg["category"]),
                    sweep_policy,
                    "validation",
                    "deterministic",
                    val_episode_seeds,
                    train_seed,
                    progress_task=episodes_task(eval_task_id(train_seed, f"{sweep_name}_validation_deterministic"), f"{sweep_name} validation deterministic evaluation (seed {train_seed})", len(val_episode_seeds)),
                )
                record_policy_diagnostic(
                    sweep_name,
                    str(sweep_cfg["category"]),
                    sweep_policy,
                    "validation",
                    "stochastic",
                    val_episode_seeds,
                    train_seed,
                    progress_task=episodes_task(eval_task_id(train_seed, f"{sweep_name}_validation_stochastic"), f"{sweep_name} validation stochastic evaluation (seed {train_seed})", len(val_episode_seeds)),
                )
                log_block_metrics(f"PPO diagnostic sweep {sweep_name}", sweep_start, sweep_test_det, train_seed)

        # Evaluation-time session cap on the PPO checkpoint.
        if run_cap_sensitivity:
            for cap_index, T_cap in enumerate(binding_cap_grid, start=1):
                cap_start = time.perf_counter()
                LOGGER.info("train_seed=%s | Session-cap evaluation start | T_cap=%s", train_seed, T_cap)
                cap_metrics = run_eval_task(
                    episodes_task(
                        eval_task_id(train_seed, f"ppo_cap_{cap_index}"),
                        f"PPO session-cap evaluation #{cap_index} (seed {train_seed})",
                        args.eval_episodes,
                    ),
                    ppo_policy,
                    cfg,
                    catalog,
                    cards,
                    wrappers={"session_cap": True, "T_cap": float(T_cap)},
                    episode_seeds=test_episode_seeds,
                    num_episodes=args.eval_episodes,
                    label=f"PPO+Cap({int(T_cap)})" if float(T_cap).is_integer() else f"PPO+Cap({T_cap})",
                    log_progress=True,
                )
                cap_rows.append(
                    {
                        "T_cap": float(T_cap),
                        "train_seed": train_seed,
                        "CumWatch": cap_metrics["CumWatch"],
                        "NightMinutes": cap_metrics["NightMinutes"],
                        "NightFraction": cap_metrics.get("NightFraction", float("nan")),
                        "LateNightSessionStartRate": cap_metrics.get("LateNightSessionStartRate", float("nan")),
                        "CVaR_0.95(L)": cap_metrics["CVaR_0.95(L)"],
                        "ReturnRate60": cap_metrics["ReturnRate60"],
                        "OverCapMinutes": cap_metrics["OverCapMinutes"],
                        "SessionCapTriggerRate": cap_metrics["SessionCapTriggerRate"],
                        "EpisodeSessionCapTriggerRate": cap_metrics["EpisodeSessionCapTriggerRate"],
                        "EpisodeOverCapPositiveRate": cap_metrics["EpisodeOverCapPositiveRate"],
                    }
                )
                frontier_rows.append(frontier_summary_row("PPO+Cap", float(T_cap), "deterministic", train_seed, cap_metrics))
                if abs(float(T_cap) - cfg.T_cap) < 1e-8:
                    main_rows.append({"policy": f"PPO + SessionCap({int(T_cap)})", "backend": "Param", "train_seed": train_seed, **{k: cap_metrics[k] for k in MAIN_POLICY_METRIC_KEYS}})
                cap_diagnostics.setdefault(float(T_cap), []).append(cap_metrics)
                log_block_metrics(f"Session-cap evaluation T_cap={T_cap}", cap_start, cap_metrics, train_seed)
            for cap_index in range(len(binding_cap_grid) + 1, planned_cap_task_count + 1):
                skip_task(
                    episodes_task(
                        eval_task_id(train_seed, f"ppo_cap_{cap_index}"),
                        f"PPO session-cap evaluation #{cap_index} (seed {train_seed})",
                        args.eval_episodes,
                    ),
                    "binding cap grid shorter than planned",
                )

        # NoHabit mechanism check.
        nohabit_block_start = time.perf_counter()
        LOGGER.info("train_seed=%s | NoHabit evaluation start", train_seed)
        nohabit_ppo_start = time.perf_counter()
        nohabit_ppo = run_eval_task(
            episodes_task(eval_task_id(train_seed, "ppo_nohabit"), f"PPO NoHabit evaluation (seed {train_seed})", args.eval_episodes),
            ppo_policy,
            cfg,
            catalog,
            cards,
            ablation="NoHabit",
            episode_seeds=test_episode_seeds,
            num_episodes=args.eval_episodes,
            label="PPO [NoHabit]",
            log_progress=True,
        )
        log_block_metrics("PPO NoHabit evaluation", nohabit_ppo_start, nohabit_ppo, train_seed)
        nohabit_lag_start = time.perf_counter()
        nohabit_lag = run_eval_task(
            episodes_task(eval_task_id(train_seed, "lagppo_nohabit"), f"LagPPO NoHabit evaluation (seed {train_seed})", args.eval_episodes),
            chosen_lag_policy,
            cfg,
            catalog,
            cards,
            ablation="NoHabit",
            episode_seeds=test_episode_seeds,
            num_episodes=args.eval_episodes,
            label=f"LagPPO(scale={chosen_scale}) [NoHabit]",
            log_progress=True,
        )
        log_block_metrics("Lagrangian PPO NoHabit evaluation", nohabit_lag_start, nohabit_lag, train_seed)
        nohabit_rows.extend(
            [
                {"condition": "PPO (Default)", "train_seed": train_seed, "CumWatch": ppo_metrics["CumWatch"], "CVaR_0.95(L)": ppo_metrics["CVaR_0.95(L)"], **{k: ppo_metrics[k] for k in RETURN_RATE_METRIC_KEYS}},
                {"condition": "PPO (NoHabit)", "train_seed": train_seed, "CumWatch": nohabit_ppo["CumWatch"], "CVaR_0.95(L)": nohabit_ppo["CVaR_0.95(L)"], **{k: nohabit_ppo[k] for k in RETURN_RATE_METRIC_KEYS}},
                {"condition": "Lagrangian PPO (Default)", "train_seed": train_seed, "CumWatch": chosen_lag_metrics["CumWatch"], "CVaR_0.95(L)": chosen_lag_metrics["CVaR_0.95(L)"], **{k: chosen_lag_metrics[k] for k in RETURN_RATE_METRIC_KEYS}},
                {"condition": "Lagrangian PPO (NoHabit)", "train_seed": train_seed, "CumWatch": nohabit_lag["CumWatch"], "CVaR_0.95(L)": nohabit_lag["CVaR_0.95(L)"], **{k: nohabit_lag[k] for k in RETURN_RATE_METRIC_KEYS}},
            ]
        )
        log_phase_complete("NoHabit evaluation", nohabit_block_start, extra=f"train_seed={train_seed}")

        if run_appendix_stress:
            appendix_policy_specs = [
                ("Random", rand_policy, False),
                ("PPO", ppo_policy, True),
                ("Lagrangian PPO", chosen_lag_policy, True),
            ]
            for stress_name in APPENDIX_STRESS_TEST_NAMES:
                stress_cfg = cfg.stress_test(stress_name)
                for policy_name, policy_obj, deterministic_flag in appendix_policy_specs:
                    stress_start = time.perf_counter()
                    LOGGER.info(
                        "train_seed=%s | Appendix stress evaluation start | stress=%s | policy=%s",
                        train_seed,
                        stress_name,
                        policy_name,
                    )
                    stress_metrics = run_eval_task(
                        episodes_task(
                            eval_task_id(train_seed, f"{policy_name}_{stress_name}"),
                            f"{policy_name} stress={stress_name} evaluation (seed {train_seed})",
                            len(appendix_episode_seeds),
                        ),
                        policy_obj,
                        stress_cfg,
                        catalog,
                        cards,
                        policy_label=policy_name,
                        episode_seeds=appendix_episode_seeds,
                        num_episodes=len(appendix_episode_seeds),
                        deterministic=deterministic_flag,
                        label=f"{policy_name} [{stress_name}]",
                        log_progress=True,
                        use_trace=False,
                    )
                    appendix_stress_rows.append(
                        appendix_dashboard_row("stress_test", stress_name, policy_name, train_seed, stress_metrics)
                    )
                    stress_subgroup = susceptibility_bin_summary(
                        stress_metrics.get("EpisodeSusceptibilityValues", []),
                        stress_metrics.get("EpisodeCumWatchValues", []),
                    )
                    appendix_subgroup_rows.append(
                        {
                            "category": "stress_test",
                            "condition": stress_name,
                            "policy": policy_name,
                            "train_seed": int(train_seed),
                            "metric": "CumWatch",
                            "top_bottom_ratio": float(stress_subgroup.get("top_bottom_ratio", float("nan"))),
                            "top_mean": float(stress_subgroup.get("top_mean", float("nan"))),
                            "bottom_mean": float(stress_subgroup.get("bottom_mean", float("nan"))),
                        }
                    )
                    log_block_metrics(f"Appendix stress {stress_name} | {policy_name}", stress_start, stress_metrics, train_seed)

        if run_wrapper_baselines:
            wrapper_specs = [
                ("throttle_personalization", {"throttle_personalization": True, "lambda_max": float(cfg.lambda_max)}),
            ]
            for wrapper_name, wrapper_kwargs in wrapper_specs:
                wrapper_start = time.perf_counter()
                LOGGER.info(
                    "train_seed=%s | Appendix wrapper evaluation start | wrapper=%s",
                    train_seed,
                    wrapper_name,
                )
                wrapper_metrics = run_eval_task(
                    episodes_task(
                        eval_task_id(train_seed, f"ppo_wrapper_{wrapper_name}"),
                        f"PPO wrapper={wrapper_name} evaluation (seed {train_seed})",
                        len(appendix_episode_seeds),
                    ),
                    ppo_policy,
                    cfg,
                    catalog,
                    cards,
                    policy_label="PPO",
                    wrappers=wrapper_kwargs,
                    episode_seeds=appendix_episode_seeds,
                    num_episodes=len(appendix_episode_seeds),
                    deterministic=True,
                    label=f"PPO [{wrapper_name}]",
                    log_progress=True,
                    use_trace=False,
                )
                appendix_wrapper_rows.append(
                    appendix_dashboard_row("wrapper_baseline", wrapper_name, "PPO", train_seed, wrapper_metrics)
                )
                wrapper_subgroup = susceptibility_bin_summary(
                    wrapper_metrics.get("EpisodeSusceptibilityValues", []),
                    wrapper_metrics.get("EpisodeCumWatchValues", []),
                )
                appendix_subgroup_rows.append(
                    {
                        "category": "wrapper_baseline",
                        "condition": wrapper_name,
                        "policy": "PPO",
                        "train_seed": int(train_seed),
                        "metric": "CumWatch",
                        "top_bottom_ratio": float(wrapper_subgroup.get("top_bottom_ratio", float("nan"))),
                        "top_mean": float(wrapper_subgroup.get("top_mean", float("nan"))),
                        "bottom_mean": float(wrapper_subgroup.get("bottom_mean", float("nan"))),
                    }
                )
                log_block_metrics(f"Appendix wrapper {wrapper_name}", wrapper_start, wrapper_metrics, train_seed)

        # Optional backend sensitivity.
        if run_backend_sensitivity and scorer is not None:
            backend_start = time.perf_counter()
            LOGGER.info("train_seed=%s | Backend sensitivity block start", train_seed)
            llm_zero = copy.deepcopy(llm_cfg)
            llm_zero.omega_r_llm = 0.0
            llm_zero.omega_c_llm = 0.0
            llm_watch_only = copy.deepcopy(llm_cfg)
            llm_watch_only.omega_c_llm = 0.0
            llm_continue_only = copy.deepcopy(llm_cfg)
            llm_continue_only.omega_r_llm = 0.0
            semantic_scorer = AGLLMScorer(
                llm_cfg,
                cards,
                mode="surrogate",
                model_id="semantic_residual_surrogate",
                cache_path=str(cache_dir / f"agllm_semantic_residual_cache_seed{train_seed}.json"),
                device=args.device,
            )

            for warm_seeds, split_name in ((val_episode_seeds, "validation"), (test_episode_seeds, "test")):
                split_slug = "validation" if split_name == "validation" else "test"
                warm_llm_cache(
                    ppo_policy,
                    llm_zero,
                    scorer,
                    warm_seeds,
                    f"PPO [LLM zero-fusion] {split_name}",
                    deterministic=True,
                    progress_tracker=progress_tracker,
                    progress_task=episodes_task(eval_task_id(train_seed, f"ppo_llm_zero_fusion_{split_slug}_warm"), f"ppo llm zero fusion {split_slug} warm (seed {train_seed})", len(warm_seeds)),
                )
                warm_llm_cache(
                    ppo_policy,
                    llm_cfg,
                    scorer,
                    warm_seeds,
                    f"PPO [LLM] {split_name}",
                    deterministic=True,
                    progress_tracker=progress_tracker,
                    progress_task=episodes_task(eval_task_id(train_seed, f"ppo_llm_{split_slug}_warm"), f"ppo llm {split_slug} warm (seed {train_seed})", len(warm_seeds)),
                )
                warm_llm_cache(
                    chosen_lag_policy,
                    llm_cfg,
                    scorer,
                    warm_seeds,
                    f"LagPPO(scale={chosen_scale}) [LLM] {split_name}",
                    deterministic=True,
                    progress_tracker=progress_tracker,
                    progress_task=episodes_task(eval_task_id(train_seed, f"lagppo_llm_{split_slug}_warm"), f"lagppo llm {split_slug} warm (seed {train_seed})", len(warm_seeds)),
                )
                warm_llm_cache(
                    myopic_policy,
                    llm_cfg,
                    scorer,
                    warm_seeds,
                    f"Myopic [LLM] {split_name}",
                    deterministic=True,
                    progress_tracker=progress_tracker,
                    progress_task=episodes_task(eval_task_id(train_seed, f"myopic_llm_{split_slug}_warm"), f"myopic llm {split_slug} warm (seed {train_seed})", len(warm_seeds)),
                )
                warm_llm_cache(
                    ppo_policy,
                    llm_watch_only,
                    scorer,
                    warm_seeds,
                    f"PPO [LLM watch-only] {split_name}",
                    deterministic=True,
                    progress_tracker=progress_tracker,
                    progress_task=episodes_task(eval_task_id(train_seed, f"ppo_llm_watch_only_{split_slug}_warm"), f"ppo llm watch only {split_slug} warm (seed {train_seed})", len(warm_seeds)),
                )
                warm_llm_cache(
                    ppo_policy,
                    llm_continue_only,
                    scorer,
                    warm_seeds,
                    f"PPO [LLM continue-only] {split_name}",
                    deterministic=True,
                    progress_tracker=progress_tracker,
                    progress_task=episodes_task(eval_task_id(train_seed, f"ppo_llm_continue_only_{split_slug}_warm"), f"ppo llm continue only {split_slug} warm (seed {train_seed})", len(warm_seeds)),
                )
                warm_llm_cache(
                    ppo_policy,
                    llm_cfg,
                    semantic_scorer,
                    warm_seeds,
                    f"PPO [semantic residual] {split_name}",
                    deterministic=True,
                    progress_tracker=progress_tracker,
                    progress_task=episodes_task(eval_task_id(train_seed, f"ppo_semantic_{split_slug}_warm"), f"ppo semantic {split_slug} warm (seed {train_seed})", len(warm_seeds)),
                )

            scorer.reset_stats()
            semantic_scorer.reset_stats()

            scorer.cfg = llm_zero
            zero_start = time.perf_counter()
            zero_metrics = run_eval_task(
                episodes_task(eval_task_id(train_seed, "ppo_llm_zero_fusion"), f"ppo llm zero fusion evaluation (seed {train_seed})", args.eval_episodes),
                ppo_policy,
                llm_zero,
                catalog,
                cards,
                backend="llm",
                scorer=scorer,
                episode_seeds=test_episode_seeds,
                num_episodes=args.eval_episodes,
                label="PPO [LLM zero-fusion]",
                log_progress=True,
            )
            log_block_metrics("PPO zero-fusion recovery", zero_start, zero_metrics, train_seed)

            scorer.cfg = llm_cfg
            llm_ppo_start = time.perf_counter()
            llm_metrics = run_eval_task(
                episodes_task(eval_task_id(train_seed, "ppo_llm"), f"ppo llm evaluation (seed {train_seed})", args.eval_episodes),
                ppo_policy,
                llm_cfg,
                catalog,
                cards,
                backend="llm",
                scorer=scorer,
                episode_seeds=test_episode_seeds,
                num_episodes=args.eval_episodes,
                label="PPO [LLM]",
                log_progress=True,
            )
            log_block_metrics("PPO backend sensitivity (LLM)", llm_ppo_start, llm_metrics, train_seed)
            llm_lag_start = time.perf_counter()
            lag_llm_metrics = run_eval_task(
                episodes_task(eval_task_id(train_seed, "lagppo_llm"), f"lagppo llm evaluation (seed {train_seed})", args.eval_episodes),
                chosen_lag_policy,
                llm_cfg,
                catalog,
                cards,
                backend="llm",
                scorer=scorer,
                episode_seeds=test_episode_seeds,
                num_episodes=args.eval_episodes,
                label=f"LagPPO(scale={chosen_scale}) [LLM]",
                log_progress=True,
            )
            log_block_metrics("Lagrangian PPO backend sensitivity (LLM)", llm_lag_start, lag_llm_metrics, train_seed)
            llm_myopic_start = time.perf_counter()
            myopic_llm_metrics = run_eval_task(
                episodes_task(eval_task_id(train_seed, "myopic_llm"), f"myopic llm evaluation (seed {train_seed})", args.eval_episodes),
                myopic_policy,
                llm_cfg,
                catalog,
                cards,
                backend="llm",
                scorer=scorer,
                episode_seeds=test_episode_seeds,
                num_episodes=args.eval_episodes,
                label="Myopic [LLM]",
                log_progress=True,
            )
            log_block_metrics("Myopic backend sensitivity (LLM)", llm_myopic_start, myopic_llm_metrics, train_seed)

            scorer.cfg = llm_watch_only
            llm_watch_only_start = time.perf_counter()
            llm_watch_only_metrics = run_eval_task(
                episodes_task(eval_task_id(train_seed, "ppo_llm_watch_only"), f"ppo llm watch only evaluation (seed {train_seed})", args.eval_episodes),
                ppo_policy,
                llm_watch_only,
                catalog,
                cards,
                policy_label="PPO",
                backend="llm",
                scorer=scorer,
                episode_seeds=test_episode_seeds,
                num_episodes=args.eval_episodes,
                deterministic=True,
                label="PPO [LLM watch-only fusion]",
                log_progress=True,
                use_trace=False,
            )
            log_block_metrics("PPO backend ablation (watch-only fusion)", llm_watch_only_start, llm_watch_only_metrics, train_seed)

            scorer.cfg = llm_continue_only
            llm_continue_only_start = time.perf_counter()
            llm_continue_only_metrics = run_eval_task(
                episodes_task(eval_task_id(train_seed, "ppo_llm_continue_only"), f"ppo llm continue only evaluation (seed {train_seed})", args.eval_episodes),
                ppo_policy,
                llm_continue_only,
                catalog,
                cards,
                policy_label="PPO",
                backend="llm",
                scorer=scorer,
                episode_seeds=test_episode_seeds,
                num_episodes=args.eval_episodes,
                deterministic=True,
                label="PPO [LLM continue-only fusion]",
                log_progress=True,
                use_trace=False,
            )
            log_block_metrics("PPO backend ablation (continue-only fusion)", llm_continue_only_start, llm_continue_only_metrics, train_seed)

            semantic_start = time.perf_counter()
            semantic_metrics = run_eval_task(
                episodes_task(eval_task_id(train_seed, "ppo_semantic_residual"), f"ppo semantic residual evaluation (seed {train_seed})", args.eval_episodes),
                ppo_policy,
                llm_cfg,
                catalog,
                cards,
                policy_label="PPO",
                backend="llm",
                scorer=semantic_scorer,
                episode_seeds=test_episode_seeds,
                num_episodes=args.eval_episodes,
                deterministic=True,
                label="PPO [semantic residual baseline]",
                log_progress=True,
                use_trace=False,
            )
            log_block_metrics("PPO backend ablation (semantic residual baseline)", semantic_start, semantic_metrics, train_seed)

            backend_rows.extend(
                [
                    {"policy": "Myopic", "backend": "LLM", "train_seed": train_seed, **{k: myopic_llm_metrics[k] for k in MAIN_POLICY_METRIC_KEYS}},
                    {"policy": "PPO", "backend": "LLM", "train_seed": train_seed, **{k: llm_metrics[k] for k in MAIN_POLICY_METRIC_KEYS}},
                    {"policy": "Lagrangian PPO", "backend": "LLM", "train_seed": train_seed, **{k: lag_llm_metrics[k] for k in MAIN_POLICY_METRIC_KEYS}},
                ]
            )
            llm_ablation_rows.extend(
                [
                    {"policy": "PPO", "ablation": "watch_only_fusion", "train_seed": train_seed, **{k: llm_watch_only_metrics[k] for k in MAIN_POLICY_METRIC_KEYS}},
                    {"policy": "PPO", "ablation": "continue_only_fusion", "train_seed": train_seed, **{k: llm_continue_only_metrics[k] for k in MAIN_POLICY_METRIC_KEYS}},
                    {"policy": "PPO", "ablation": "semantic_residual_baseline", "train_seed": train_seed, **{k: semantic_metrics[k] for k in MAIN_POLICY_METRIC_KEYS}},
                ]
            )

            recovery_metrics = ["CumWatch", "CVaR_0.95(L)", "NightMinutes", "OverCapMinutes"]
            zero_max_abs_delta = max(abs(float(zero_metrics[k]) - float(ppo_metrics[k])) for k in recovery_metrics)
            if zero_max_abs_delta > 1e-8:
                LOGGER.warning(
                    "train_seed=%s | zero-fusion backend did not reproduce AGParam exactly | max_abs_delta=%.6g",
                    train_seed,
                    zero_max_abs_delta,
                )
            llm_recovery_rows.append(
                {
                    "setting": "Recovery check",
                    "omega_r_llm": 0.0,
                    "omega_c_llm": 0.0,
                    "train_seed": train_seed,
                    "abs_delta_CumWatch": abs(zero_metrics["CumWatch"] - ppo_metrics["CumWatch"]),
                    "abs_delta_CVaR_0.95(L)": abs(zero_metrics["CVaR_0.95(L)"] - ppo_metrics["CVaR_0.95(L)"]),
                    "abs_delta_NightMinutes": abs(zero_metrics["NightMinutes"] - ppo_metrics["NightMinutes"]),
                    "abs_delta_OverCapMinutes": abs(zero_metrics["OverCapMinutes"] - ppo_metrics["OverCapMinutes"]),
                    "max_abs_delta": float(zero_max_abs_delta),
                    "seed_matched_equivalent": bool(zero_max_abs_delta <= 1e-8),
                }
            )
            llm_recovery_rows.append(
                {
                    "setting": "Calibrated AGLLM",
                    "omega_r_llm": llm_cfg.omega_r_llm,
                    "omega_c_llm": llm_cfg.omega_c_llm,
                    "train_seed": train_seed,
                    "abs_delta_CumWatch": abs(llm_metrics["CumWatch"] - ppo_metrics["CumWatch"]),
                    "abs_delta_CVaR_0.95(L)": abs(llm_metrics["CVaR_0.95(L)"] - ppo_metrics["CVaR_0.95(L)"]),
                    "abs_delta_NightMinutes": abs(llm_metrics["NightMinutes"] - ppo_metrics["NightMinutes"]),
                    "abs_delta_OverCapMinutes": abs(llm_metrics["OverCapMinutes"] - ppo_metrics["OverCapMinutes"]),
                    "max_abs_delta": float(
                        max(abs(float(llm_metrics[k]) - float(ppo_metrics[k])) for k in recovery_metrics)
                    ),
                    "seed_matched_equivalent": False,
                }
            )
            scorer.cfg = llm_cfg
            if float(scorer.stats().get("cache_hit_rate", 0.0)) < 0.95:
                LOGGER.warning(
                    "train_seed=%s | AGLLM cache hit rate is below the 95%% release threshold | stats=%s",
                    train_seed,
                    scorer.stats(),
                )
            log_phase_complete("Backend sensitivity block", backend_start, extra=f"train_seed={train_seed}")

        log_phase_complete("Train seed", train_seed_start, extra=f"index={seed_index}/{len(train_seed_list)} | train_seed={train_seed}")

    # Aggregate tables.
    table_phase_start = time.perf_counter()
    LOGGER.info("Table aggregation start")
    table_output_file_count = 11
    table_output_file_count += 1
    if write_diagnostic_artifacts:
        table_output_file_count += 14
    if run_cap_sensitivity:
        table_output_file_count += 4
    if run_frontier_artifacts:
        table_output_file_count += 4
    if run_policy_diagnostics:
        table_output_file_count += 6
    if write_backend_artifacts:
        table_output_file_count += 10
    if write_appendix_artifacts:
        table_output_file_count += 8
    start_file_task(files_task(file_task_id("table_aggregation"), "Table aggregation", table_output_file_count))
    main_df = aggregate_across_train_seeds(main_rows, ["policy", "backend"])
    official_scorecard_df = aggregate_across_train_seeds(official_scorecard_rows, ["policy", "backend", "eval_mode"]) if official_scorecard_rows else pd.DataFrame()
    official_episode_output_df = pd.DataFrame(official_episode_output_rows)
    if not official_episode_output_df.empty:
        official_episode_output_df = official_episode_output_df.sort_values(["policy", "eval_mode", "train_seed", "episode_index"]).reset_index(drop=True)
        save_csv_progress(official_episode_output_df, table_dir / "table_official_policy_episode_outputs.csv")
        if not official_scorecard_df.empty:
            official_scorecard_df = augment_official_scorecard_with_episode_uncertainty(official_scorecard_df, official_episode_output_df, cfg)
    overall_constraint_proxy = build_night_proxy_orthogonality_audit(mitigation_proxy_results)
    first_validation_cutoff = int(args.validate_every + args.rollout_steps)
    threshold_payload = resolved_paper_thresholds(
        cfg,
        fallback_cap_grid=binding_cap_grid,
        selected_delta_sess=float(args.delta_sess) if args.log_csv else cfg.paper_thresholds.get("selected_delta_sess"),
    )
    save_table_progress(threshold_table_dataframe(threshold_payload), outdir / "table_thresholds")
    save_table_progress(proxy_orthogonality_audit_dataframe(overall_constraint_proxy), table_dir / "table_proxy_orthogonality_audit")
    proxy_audit_note_path = outdir / "proxy_orthogonality_audit.md"
    write_text_progress(proxy_audit_note_path, render_proxy_orthogonality_audit_markdown(overall_constraint_proxy), "diagnostic_memo_markdown")
    returnrate_sensitivity_df = build_returnrate_sensitivity_table(main_df)
    returnrate_assessment = assess_returnrate_main_text_worthiness(main_df)
    if write_diagnostic_artifacts:
        save_table_progress(returnrate_sensitivity_df, table_dir / "table_returnrate_sensitivity")
        returnrate_note_path = outdir / "is_returnrate_main_text_worthy.md"
        write_text_progress(returnrate_note_path, render_returnrate_main_text_note(returnrate_assessment), "diagnostic_memo_markdown")
    main_text_night_proxy = overall_constraint_proxy.get("promoted_main_text_proxy")
    if main_text_night_proxy is None:
        LOGGER.warning(
            "Night-family proxies failed the orthogonality promotion audit; keeping them appendix-only and using OverCapMinutes as the only main-text constrained channel."
        )
    include_returnrate60 = bool(returnrate_assessment.get("returnrate60_main_text_worthy", False))
    if not include_returnrate60:
        LOGGER.warning("ReturnRate60 is not materially policy-sensitive; removing it from the main scorecard and keeping appendix-only sensitivity reporting.")
    scorecard_cols = build_official_scorecard_column_order(
        official_scorecard_df.columns,
        main_text_night_proxy=main_text_night_proxy if isinstance(main_text_night_proxy, str) else None,
        include_returnrate60=include_returnrate60,
    )
    deterministic_scorecard_df = official_scorecard_df[official_scorecard_df["eval_mode"] == "deterministic"].copy() if not official_scorecard_df.empty else pd.DataFrame(columns=scorecard_cols)
    stochastic_scorecard_df = official_scorecard_df[official_scorecard_df["eval_mode"] == "stochastic"].copy() if not official_scorecard_df.empty else pd.DataFrame(columns=scorecard_cols)
    save_table_progress(deterministic_scorecard_df[scorecard_cols].copy(), table_dir / "table_scorecard_deterministic")
    save_table_progress(stochastic_scorecard_df[scorecard_cols].copy(), table_dir / "table_scorecard_stochastic")

    nohabit_df = aggregate_across_train_seeds(nohabit_rows, ["condition"])
    save_table_progress(nohabit_df, table_dir / "table_nohabit_main")
    if write_diagnostic_artifacts:
        nohabit_note_path = outdir / "old_vs_corrected_nohabit.md"
        write_text_progress(nohabit_note_path, render_old_vs_corrected_nohabit_note(nohabit_df), "diagnostic_memo_markdown")

    budget_attainment_df = pd.DataFrame(budget_attainment_rows)
    if not budget_attainment_df.empty:
        budget_attainment_df["selected_checkpoint_is_first_validation"] = (
            pd.to_numeric(budget_attainment_df["selected_checkpoint_step"], errors="coerce").fillna(0.0) <= float(first_validation_cutoff)
        )
        selected_candidate_summary = (
            budget_attainment_df[budget_attainment_df["selected_candidate"]]
            .groupby(["method", "scale"], dropna=False)
            .agg(
                feasible_rate=("feasible", lambda s: float(np.mean(np.asarray(list(s), dtype=np.float64)))),
                selected_checkpoint_first_validation_rate=("selected_checkpoint_is_first_validation", lambda s: float(np.mean(np.asarray(list(s), dtype=np.float64)))),
            )
            .reset_index()
        )
        if not selected_candidate_summary.empty:
            budget_attainment_df = budget_attainment_df.merge(
                selected_candidate_summary,
                on=["method", "scale"],
                how="left",
            )
        budget_attainment_df = budget_attainment_df[
            [
                "method",
                "scale",
                "train_seed",
                "dual_lr",
                "total_steps",
                "constraint_channels",
                "night_minutes_cumwatch_corr_policy_means",
                "night_minutes_cumwatch_corr_episodes",
                "night_minutes_is_scalar_proxy",
                "target_night_budget",
                "achieved_NightMinutes",
                "target_over_budget",
                "achieved_OverCapMinutes",
                "validation_CumWatch",
                "validation_CVaR_0.95(L)",
                "feasible",
                "feasible_rate",
                "selected_candidate",
                "selected_operating_point",
                "selected_checkpoint_step",
                "selected_checkpoint_is_first_validation",
                "selected_checkpoint_first_validation_rate",
                "selected_source",
                "final_lambda1",
                "final_lambda2",
                "total_violation",
            ]
        ].sort_values(["scale", "train_seed", "dual_lr", "total_steps"])
    if write_diagnostic_artifacts:
        save_table_progress(budget_attainment_df, table_dir / "table_budget_attainment")

    cap_df = aggregate_across_train_seeds(cap_rows, ["T_cap"]) if cap_rows else pd.DataFrame()
    if run_cap_sensitivity:
        save_table_progress(cap_df, table_dir / "table_cap_sensitivity")
        save_table_progress(cap_df, table_dir / "table_cap_sensitivity_rebuilt")
    if run_cap_sensitivity and not cap_df.empty and len(cap_df) > 1:
        cap_metric_spreads = []
        cap_spread_metrics = ["CumWatch", "OverCapMinutes", "SessionCapTriggerRate", "EpisodeOverCapPositiveRate"]
        if isinstance(main_text_night_proxy, str) and main_text_night_proxy:
            cap_spread_metrics.append(main_text_night_proxy)
        for metric in cap_spread_metrics:
            if metric in cap_df.columns:
                cap_metric_spreads.append(float(cap_df[metric].max() - cap_df[metric].min()))
        if cap_metric_spreads and max(cap_metric_spreads) <= 1e-6:
            LOGGER.warning(
                "Rebuilt session-cap frontier remains degenerate across the binding cap grid %s; demote the cap-frontier claim from the main text.",
                binding_cap_grid,
            )

    frontier_df = aggregate_across_train_seeds(frontier_rows, ["method", "scale", "eval_mode"]) if frontier_rows else pd.DataFrame()
    if not cap_df.empty and len(cap_df) > 1:
        cap_active_metrics = ["CumWatch", "OverCapMinutes", "SessionCapTriggerRate", "EpisodeOverCapPositiveRate"]
        if isinstance(main_text_night_proxy, str) and main_text_night_proxy:
            cap_active_metrics.append(main_text_night_proxy)
        cap_active = any(
            (
                metric in cap_df.columns
                and float(cap_df[metric].max() - cap_df[metric].min()) > 1e-6
            )
            for metric in cap_active_metrics
        )
        if not cap_active and not frontier_df.empty:
            frontier_df = frontier_df[frontier_df["method"] != "PPO+Cap"].copy()
    if run_frontier_artifacts:
        save_table_progress(frontier_df, table_dir / "table_frontier")
        save_table_progress(frontier_df, table_dir / "table_frontier_rebuilt")
    ppo_param_block = main_df[(main_df["policy"] == "PPO") & (main_df["backend"] == "Param")] if not main_df.empty else pd.DataFrame()
    mitigation_frontier_block = frontier_df[frontier_df["method"] != "PPO"] if not frontier_df.empty else pd.DataFrame()
    if not ppo_param_block.empty and not mitigation_frontier_block.empty:
        ppo_row = ppo_param_block.iloc[0]
        active_frontier_metrics = ["OverCapMinutes"]
        if isinstance(main_text_night_proxy, str) and main_text_night_proxy and main_text_night_proxy in mitigation_frontier_block.columns:
            active_frontier_metrics.append(main_text_night_proxy)
        tradeoff_exists = any(
            (
                any(
                    metric in mitigation_frontier_block.columns
                    and metric in ppo_row.index
                    and float(row[metric]) < float(ppo_row[metric]) - 1e-6
                    for metric in active_frontier_metrics
                )
                or ("CVaR_0.95(L)" in mitigation_frontier_block.columns and float(row["CVaR_0.95(L)"]) < float(ppo_row["CVaR_0.95(L)"]) - 1e-6)
            )
            and (float(row["CumWatch"]) < float(ppo_row["CumWatch"]) - 1e-6)
            for _, row in mitigation_frontier_block.iterrows()
        )
        if not tradeoff_exists:
            LOGGER.warning(
                "Rebuilt constrained frontier does not show a measurable reward-risk tradeoff relative to PPO; demote the constrained-track claim from the main text."
            )

    validation_selection_df = pd.DataFrame(validation_rows)
    if not validation_selection_df.empty:
        validation_selection_cols = [
            "method",
            "scale",
            "train_seed",
            "global_step",
            "validation_CumWatch",
            "validation_NightMinutes",
            "validation_OverCapMinutes",
            "validation_CVaR_0.95(L)",
            "feasible",
            "selected",
            "night_violation",
            "over_violation",
            "total_violation",
            "lambda1",
            "lambda2",
        ]
        insert_at = 3
        for extra_col in ["dual_lr", "total_steps"]:
            if extra_col in validation_selection_df.columns:
                validation_selection_cols.insert(insert_at, extra_col)
                insert_at += 1
        validation_selection_df = validation_selection_df[
            [col for col in validation_selection_cols if col in validation_selection_df.columns]
        ].sort_values([col for col in ["method", "scale", "train_seed", "dual_lr", "total_steps", "global_step"] if col in validation_selection_df.columns])
    if write_diagnostic_artifacts:
        save_table_progress(validation_selection_df, table_dir / "table_validation_selection")

    frontier_validation_df = pd.DataFrame(frontier_validation_rows)
    if not frontier_validation_df.empty:
        frontier_validation_df["selected_checkpoint_is_first_validation"] = (
            pd.to_numeric(frontier_validation_df["global_step"], errors="coerce").fillna(0.0) <= float(first_validation_cutoff)
        )
        frontier_validation_summary = (
            frontier_validation_df.groupby(["method", "scale"], dropna=False)
            .agg(
                feasible_rate=("feasible", lambda s: float(np.mean(np.asarray(list(s), dtype=np.float64)))),
                selected_checkpoint_first_validation_rate=("selected_checkpoint_is_first_validation", lambda s: float(np.mean(np.asarray(list(s), dtype=np.float64)))),
            )
            .reset_index()
        )
        frontier_validation_df = frontier_validation_df.merge(
            frontier_validation_summary,
            on=["method", "scale"],
            how="left",
        )
        frontier_validation_cols = [
            "method",
            "scale",
            "train_seed",
            "global_step",
            "constraint_channels",
            "night_minutes_cumwatch_corr_policy_means",
            "night_minutes_cumwatch_corr_episodes",
            "night_minutes_is_scalar_proxy",
            "validation_CumWatch",
            "validation_NightMinutes",
            "validation_OverCapMinutes",
            "validation_CVaR_0.95(L)",
            "feasible",
            "feasible_rate",
            "selected",
            "selected_operating_point",
            "selected_checkpoint_is_first_validation",
            "selected_checkpoint_first_validation_rate",
            "selected_source",
            "night_budget",
            "over_budget",
            "total_violation",
        ]
        insert_at = 3
        for extra_col in ["dual_lr", "total_steps"]:
            if extra_col in frontier_validation_df.columns:
                frontier_validation_cols.insert(insert_at, extra_col)
                insert_at += 1
        frontier_validation_df = frontier_validation_df[
            [col for col in frontier_validation_cols if col in frontier_validation_df.columns]
        ].sort_values([col for col in ["method", "scale", "train_seed", "dual_lr", "total_steps"] if col in frontier_validation_df.columns])
    if write_diagnostic_artifacts:
        save_table_progress(frontier_validation_df, table_dir / "table_frontier_validation")

    policy_diagnostics_df = pd.DataFrame(policy_diagnostic_rows)
    if not policy_diagnostics_df.empty:
        policy_diagnostics_df = policy_diagnostics_df.sort_values(["split", "policy", "eval_mode", "train_seed"])
    if run_policy_diagnostics:
        save_table_progress(policy_diagnostics_df, table_dir / "table_policy_diagnostics")

    baseline_expansion_df = pd.DataFrame(baseline_expansion_rows)
    if not baseline_expansion_df.empty:
        baseline_expansion_agg = aggregate_across_train_seeds(
            baseline_expansion_df.to_dict("records"),
            ["policy", "category", "eval_split", "eval_mode"],
        )
    else:
        baseline_expansion_agg = pd.DataFrame()
    diversity_diagnostic_df = build_diversity_diagnostic_table(baseline_expansion_df, frontier_validation_df)
    if run_policy_diagnostics:
        save_table_progress(baseline_expansion_agg, table_dir / "table_baseline_expansion")
        save_table_progress(diversity_diagnostic_df, table_dir / "table_diversity_diagnostic")

    if write_backend_artifacts and backend_rows:
        backend_full = main_rows + backend_rows
        backend_df = aggregate_across_train_seeds(backend_full, ["policy", "backend"])
        save_table_progress(backend_df, table_dir / "table_backend_sensitivity")
    else:
        backend_df = pd.DataFrame()

    if write_backend_artifacts and llm_ablation_rows:
        llm_ablation_df = aggregate_across_train_seeds(llm_ablation_rows, ["policy", "ablation"])
    else:
        llm_ablation_df = pd.DataFrame()
    if write_backend_artifacts:
        save_table_progress(llm_ablation_df, table_dir / "table_llm_ablation")

    appendix_stress_df = aggregate_across_train_seeds(appendix_stress_rows, ["category", "condition", "policy"]) if appendix_stress_rows else pd.DataFrame()
    appendix_wrapper_df = aggregate_across_train_seeds(appendix_wrapper_rows, ["category", "condition", "policy"]) if appendix_wrapper_rows else pd.DataFrame()
    break_prompt_probe_df = (
        aggregate_across_train_seeds(break_prompt_probe_rows, ["category", "condition", "policy"])
        if break_prompt_probe_rows
        else pd.DataFrame()
    )
    appendix_dashboard_df = pd.concat([appendix_stress_df, appendix_wrapper_df], ignore_index=True) if (not appendix_stress_df.empty or not appendix_wrapper_df.empty) else pd.DataFrame()
    appendix_subgroup_df = aggregate_across_train_seeds(appendix_subgroup_rows, ["category", "condition", "policy", "metric"]) if appendix_subgroup_rows else pd.DataFrame()
    if write_appendix_artifacts:
        save_table_progress(appendix_stress_df, table_dir / "table_appendix_stress_tests")
        save_table_progress(appendix_wrapper_df, table_dir / "table_appendix_wrapper_baselines")
        save_table_progress(appendix_dashboard_df, table_dir / "table_appendix_dashboard")
        save_table_progress(appendix_subgroup_df, table_dir / "table_appendix_subgroup_disparity")

    if write_backend_artifacts and scorer is not None:
        llm_cache_stats = scorer.stats()
        llm_cache_stats.update(
            {
                "paper_mode": bool(getattr(args, "paper_mode", False)),
                "release_manifest_path": str(AGLLM_RELEASE_MANIFEST_PATH),
                "watch_template_path": str(scorer.watch_template_path),
                "continue_template_path": str(scorer.continue_template_path),
                "cache_hit_rate_warning_threshold": 0.95,
                "context_seeds": llm_context_seeds,
                "heldout_target_seeds": llm_target_loss_seeds,
                "val_episode_seed_count": len(val_episode_seeds),
                "test_episode_seed_count": len(test_episode_seeds),
            }
        )
        llm_cache_stats_path = outdir / "llm_cache_stats.json"
        json_dumps_progress(llm_cache_stats, llm_cache_stats_path, "llm_cache_stats_json")
        if float(llm_cache_stats.get("cache_hit_rate", 0.0)) < 0.95:
            LOGGER.warning("AGLLM cache hit rate remained below the 95%% release threshold | stats=%s", llm_cache_stats)
        release_manifest_runtime = copy.deepcopy(load_agllm_release_manifest())
        release_manifest_runtime["released_dashboard_stats"] = {
            "invalid_json_rate": float(llm_cache_stats["invalid_json_rate"]),
            "repair_rate": float(llm_cache_stats["repair_rate"]),
            "fallback_rate": float(llm_cache_stats["fallback_rate"]),
            "cache_hit_rate": float(llm_cache_stats["cache_hit_rate"]),
            "note": "Measured on seed-matched AGLLM benchmark evaluation after explicit cache prebuild.",
        }
        release_manifest_runtime["runtime_validation"] = {
            "paper_mode": bool(getattr(args, "paper_mode", False)),
            "watch_template_hash": scorer.watch_template_hash,
            "continue_template_hash": scorer.continue_template_hash,
            "cache_hit_rate_warning_threshold": 0.95,
            "fit_llm_fusion": bool(args.fit_llm_fusion),
            "fit_summary": llm_fit,
        }
        runtime_release_manifest_path = outdir / "agllm_release_manifest.json"
        json_dumps_progress(release_manifest_runtime, AGLLM_RELEASE_MANIFEST_PATH, "release_manifest_json")
        json_dumps_progress(release_manifest_runtime, runtime_release_manifest_path, "release_manifest_json")
        scorer.save()
        if scorer.cache_path:
            advance_file_task(1, extra=Path(scorer.cache_path).name)
        LOGGER.info("AGLLM scorer stats | %s", scorer.stats())

    if write_backend_artifacts and llm_recovery_rows:
        recovery_df = aggregate_across_train_seeds(llm_recovery_rows, ["setting", "omega_r_llm", "omega_c_llm"])
        save_table_progress(recovery_df, table_dir / "table_llm_recovery")
    if run_profile in {"main", "full"}:
        if infer_threshold_source_from_payload(threshold_payload) == "simulator_relative":
            assert_paper_threshold_channels_active(main_df, cap_df, break_prompt_probe_df)
            LOGGER.info(
                "Paper threshold invariants passed | T_ref=%s | cap_grid=%s | break_T=%s | break_J=%s",
                threshold_payload["T_ref"],
                threshold_payload["cap_grid"],
                threshold_payload["break_T"],
                threshold_payload["break_J"],
            )
        else:
            try:
                assert_paper_threshold_channels_active(main_df, cap_df, break_prompt_probe_df)
                LOGGER.info(
                    "Paper threshold invariants passed | T_ref=%s | cap_grid=%s | break_T=%s | break_J=%s",
                    threshold_payload["T_ref"],
                    threshold_payload["cap_grid"],
                    threshold_payload["break_T"],
                    threshold_payload["break_J"],
                )
            except AssertionError as exc:
                LOGGER.warning(
                    "Paper threshold activity invariants are advisory under threshold_source=%s | %s",
                    infer_threshold_source_from_payload(threshold_payload),
                    exc,
                )
    finish_file_task(extra="table aggregation complete")
    table_files = sorted(p.name for p in table_dir.iterdir())
    log_phase_complete("Table aggregation", table_phase_start, extra=f"files={table_files}")

    session_length_quantiles_path: Optional[Path] = None
    threshold_activity_report_path: Optional[Path] = None
    memo_path: Optional[Path] = None
    memo_random_vs_ppo_path: Optional[Path] = None
    diversity_memo_path: Optional[Path] = None
    if write_diagnostic_artifacts:
        threshold_phase_start = time.perf_counter()
        LOGGER.info("Threshold diagnostics start")
        start_file_task(files_task(file_task_id("threshold_diagnostics"), "Threshold diagnostics", 5))
        exceed_thresholds = sorted(set([float(x) for x in DEFAULT_ACTIVITY_THRESHOLDS] + [float(x) for x in binding_cap_grid] + [float(cfg.T_ref)]))
        t_ref_grid = sorted(set([float(x) for x in DEFAULT_T_REF_SWEEP] + [float(cfg.T_ref)]))
        session_length_quantiles, threshold_activity_report = summarize_threshold_activity(
            policy_results=policy_diagnostics,
            cap_results=cap_diagnostics,
            default_t_ref=float(cfg.T_ref),
            cap_grid=[float(x) for x in binding_cap_grid],
            exceed_thresholds=exceed_thresholds,
            t_ref_grid=t_ref_grid,
        )
        session_length_quantiles_path = outdir / "session_length_quantiles.json"
        threshold_activity_report_path = outdir / "threshold_activity_report.json"
        memo_path = outdir / "does_120_minutes_bind.md"
        json_dumps_progress(session_length_quantiles, session_length_quantiles_path, "session_length_quantiles_json")
        json_dumps_progress(threshold_activity_report, threshold_activity_report_path, "threshold_activity_report_json")
        write_text_progress(memo_path, render_120_minutes_memo(threshold_activity_report), "threshold_memo_markdown")
        log_phase_complete(
            "Threshold diagnostics",
            threshold_phase_start,
            extra=f"session_quantiles={session_length_quantiles_path.name} | threshold_report={threshold_activity_report_path.name}",
        )

        memo_random_vs_ppo_path = outdir / "why_random_beats_ppo.md"
        write_text_progress(memo_random_vs_ppo_path, render_random_vs_ppo_memo(baseline_expansion_agg), "diagnostic_memo_markdown")
        diversity_memo_path = outdir / "does_diversity_dominate_personalization.md"
        write_text_progress(
            diversity_memo_path,
            render_diversity_dominate_personalization_memo(diversity_diagnostic_df, threshold_payload),
            "diagnostic_memo_markdown",
        )
        finish_file_task(extra="threshold diagnostics complete")

    # Figures.
    figure_phase_start = time.perf_counter()
    LOGGER.info("Figure generation start")
    marginal_block = pd.DataFrame()
    repeat_block = pd.DataFrame()
    if run_policy_diagnostics and not policy_diagnostics_df.empty:
        marginal_block = policy_diagnostics_df[
            (policy_diagnostics_df["split"] == "test")
            & (policy_diagnostics_df["train_seed"] == diagnostic_first_seed)
        ].copy()
        repeat_block = policy_diagnostics_df[policy_diagnostics_df["split"] == "test"].copy()
    figure_output_file_count = 0
    if run_frontier_artifacts and not frontier_df.empty:
        figure_output_file_count += 2
        if isinstance(main_text_night_proxy, str) and main_text_night_proxy:
            figure_output_file_count += 2
    if write_backend_artifacts and not backend_df.empty:
        figure_output_file_count += 1
    if not marginal_block.empty:
        figure_output_file_count += len(marginal_block)
    if not diversity_diagnostic_df.empty:
        figure_output_file_count += 2
    elif not repeat_block.empty:
        figure_output_file_count += 1
    if run_appendix_stress and appendix_reference_metrics is not None:
        figure_output_file_count += 4
    start_file_task(files_task(file_task_id("figure_generation"), "Figure generation", max(1, figure_output_file_count)))
    # Reward-risk frontier.
    if run_frontier_artifacts and not frontier_df.empty:
        frontier_plot_metrics = ["OverCapMinutes"]
        if isinstance(main_text_night_proxy, str) and main_text_night_proxy and main_text_night_proxy in frontier_df.columns:
            frontier_plot_metrics.append(main_text_night_proxy)
        for metric in frontier_plot_metrics:
            block = frontier_df.copy()
            save_scatter_progress(
                x=block[metric].astype(float).tolist(),
                y=block["CumWatch"].astype(float).tolist(),
                labels=[f"{m}:{s}" for m, s in zip(block["method"], block["scale"])],
                xlabel=metric,
                ylabel="CumWatch",
                title=f"Reward-risk frontier ({metric})",
                outpath=fig_dir / f"fig_frontier_{filename_slug(metric)}.png",
            )
        if isinstance(main_text_night_proxy, str) and main_text_night_proxy and main_text_night_proxy in frontier_df.columns:
            save_scatter_progress(
                x=frontier_df[main_text_night_proxy].astype(float).tolist(),
                y=frontier_df["CumWatch"].astype(float).tolist(),
                labels=[f"{m}:{s}" for m, s in zip(frontier_df["method"], frontier_df["scale"])],
                xlabel=main_text_night_proxy,
                ylabel="CumWatch",
                title=f"Reward-risk frontier ({main_text_night_proxy}, rebuilt)",
                outpath=fig_dir / f"fig_frontier_{filename_slug(main_text_night_proxy)}_rebuilt.png",
            )
        save_scatter_progress(
            x=frontier_df["OverCapMinutes"].astype(float).tolist(),
            y=frontier_df["CumWatch"].astype(float).tolist(),
            labels=[f"{m}:{s}" for m, s in zip(frontier_df["method"], frontier_df["scale"])],
            xlabel="OverCapMinutes",
            ylabel="CumWatch",
            title="Reward-risk frontier (OverCapMinutes, rebuilt)",
            outpath=fig_dir / "fig_frontier_overcap_rebuilt.png",
        )

    # Backend sensitivity (Param vs LLM) for shared policies.
    if write_backend_artifacts and not backend_df.empty:
        merged = backend_df.pivot(index="policy", columns="backend", values="CumWatch").dropna()
        save_scatter_progress(
            x=merged["Param"].astype(float).tolist(),
            y=merged["LLM"].astype(float).tolist(),
            labels=merged.index.tolist(),
            xlabel="Param CumWatch",
            ylabel="LLM CumWatch",
            title="Backend sensitivity: CumWatch",
            outpath=fig_dir / "fig_backend_sensitivity_cumwatch.png",
        )

    if run_policy_diagnostics and not policy_diagnostics_df.empty:
        for row in marginal_block.itertuples(index=False):
            metrics_for_plot = {
                "MarginalZ": json.loads(getattr(row, "MarginalZ")),
                "MarginalLambdaIdx": json.loads(getattr(row, "MarginalLambdaIdx")),
                "MarginalNu": json.loads(getattr(row, "MarginalNu")),
            }
            outname = f"fig_action_marginals_{filename_slug(getattr(row, 'policy'))}_{filename_slug(getattr(row, 'split'))}_{filename_slug(getattr(row, 'eval_mode'))}.png"
            save_action_marginals_figure_progress(
                metrics_for_plot,
                title=f"{getattr(row, 'policy')} ({getattr(row, 'split')}, {getattr(row, 'eval_mode')})",
                outpath=fig_dir / outname,
            )

        if not diversity_diagnostic_df.empty:
            save_scatter_progress(
                x=diversity_diagnostic_df["RepeatRate"].astype(float).tolist(),
                y=diversity_diagnostic_df["CumWatch"].astype(float).tolist(),
                labels=[f"{p} ({m})" for p, m in zip(diversity_diagnostic_df["policy"], diversity_diagnostic_df["eval_mode"])],
                xlabel="RepeatRate",
                ylabel="CumWatch",
                title="CumWatch vs RepeatRate",
                outpath=fig_dir / "fig_cumwatch_vs_repeat_rate.png",
            )
            save_scatter_progress(
                x=diversity_diagnostic_df["UniqueClusterCount"].astype(float).tolist(),
                y=diversity_diagnostic_df["CumWatch"].astype(float).tolist(),
                labels=[f"{p} ({m})" for p, m in zip(diversity_diagnostic_df["policy"], diversity_diagnostic_df["eval_mode"])],
                xlabel="UniqueClusterCount",
                ylabel="CumWatch",
                title="CumWatch vs UniqueClusterCount",
                outpath=fig_dir / "fig_cumwatch_vs_unique_cluster_count.png",
            )
        elif not repeat_block.empty:
            save_scatter_progress(
                x=repeat_block["RepeatRate"].astype(float).tolist(),
                y=repeat_block["CumWatch"].astype(float).tolist(),
                labels=[f"{p}:{m}" for p, m in zip(repeat_block["policy"], repeat_block["eval_mode"])],
                xlabel="RepeatRate",
                ylabel="CumWatch",
                title="CumWatch vs RepeatRate",
                outpath=fig_dir / "fig_cumwatch_vs_repeat_rate.png",
            )
    if run_appendix_stress and appendix_reference_metrics is not None:
        save_appendix_plots(
            appendix_reference_metrics,
            prefix="fig_appendix_ppo_default",
            title_prefix="Appendix PPO default",
        )
    if figure_output_file_count == 0:
        advance_file_task(1, extra="no figure artifacts")
    finish_file_task(extra="figure generation complete")
    figure_files = sorted(p.name for p in fig_dir.iterdir())
    log_phase_complete("Figure generation", figure_phase_start, extra=f"files={figure_files}")

    # Manifest.
    manifest_phase_start = time.perf_counter()
    LOGGER.info("Manifest writing start")
    start_file_task(
        files_task(
            file_task_id("manifest_writing"),
            "Manifest writing",
            2 if calibration_payload is not None else 1,
        )
    )
    manifest = {
        "config_path": str(outdir / "config_used.json"),
        "run_profile": run_profile,
        "threshold_source": str(threshold_payload.get("threshold_source", selected_threshold_source)),
        "num_train_seeds": args.num_train_seeds,
        "eval_episodes": args.eval_episodes,
        "lag_search_mode": lag_search_mode,
        "lag_search_steps": lag_search_steps,
        "lag_search_val_episodes": lag_search_val_episodes,
        "lag_full_steps": lag_full_steps,
        "lag_full_val_episodes": lag_full_val_episodes,
        "lag_dual_lr_grid_search": [float(x) for x in lag_dual_lr_grid_search],
        "lag_topk_candidates": lag_topk_candidates,
        "evaluation_cache_entries": len(evaluation_cache),
        "evaluation_cache_hits": evaluation_cache_hits,
        "evaluation_cache_misses": evaluation_cache_misses,
        "tables": sorted([p.name for p in table_dir.iterdir()]),
        "figures": sorted([p.name for p in fig_dir.iterdir()]),
        "train_seed_list": train_seed_list_path.name,
        "val_episode_seeds": val_episode_seeds_path.name,
        "test_episode_seeds": test_episode_seeds_path.name,
        "binding_cap_grid": [float(x) for x in binding_cap_grid],
        "table_scorecard_deterministic": "tables/table_scorecard_deterministic.csv",
        "table_scorecard_stochastic": "tables/table_scorecard_stochastic.csv",
        "official_policy_episode_outputs": "tables/table_official_policy_episode_outputs.csv",
        "official_main_eval_mode": "deterministic",
        "paper_thresholds": copy.deepcopy(threshold_payload),
        "table_thresholds": "table_thresholds.md",
        "table_proxy_orthogonality_audit": "tables/table_proxy_orthogonality_audit.csv",
        "proxy_orthogonality_audit": proxy_audit_note_path.name,
        "constraint_track_proxy_diagnostics": {
            "night_minutes_cumwatch_corr_policy_means": float(overall_constraint_proxy["policy_mean_corr"]),
            "night_minutes_cumwatch_corr_episodes": float(overall_constraint_proxy["episode_corr"]),
            "night_minutes_is_scalar_proxy": bool(overall_constraint_proxy["night_minutes_is_scalar_proxy"]),
            "policy_count": int(overall_constraint_proxy["num_policies"]),
            "episode_count": int(overall_constraint_proxy["num_episodes"]),
            "promoted_main_text_proxy": overall_constraint_proxy.get("promoted_main_text_proxy"),
            "night_proxy_appendix_only": bool(overall_constraint_proxy.get("night_proxy_appendix_only", False)),
            "main_text_scorecard_risk_metrics": list(overall_constraint_proxy.get("main_text_scorecard_risk_metrics", [])),
            "reason": str(overall_constraint_proxy.get("reason", "")),
            "proxy_candidates": copy.deepcopy(overall_constraint_proxy.get("proxy_candidates", {})),
        },
        "constraint_track_screening": list(constraint_track_summaries),
        "with_llm": bool(args.with_llm),
        "paper_mode": bool(getattr(args, "paper_mode", False)),
        "llm_mode": args.llm_mode,
        "llm_model_id": args.llm_model_id if run_backend_sensitivity else None,
        "calibrated_against_logs": bool(args.log_csv),
    }
    if write_diagnostic_artifacts:
        if session_length_quantiles_path is not None:
            manifest["session_length_quantiles"] = session_length_quantiles_path.name
        if threshold_activity_report_path is not None:
            manifest["threshold_activity_report"] = threshold_activity_report_path.name
        if memo_path is not None:
            manifest["does_120_minutes_bind"] = memo_path.name
        if memo_random_vs_ppo_path is not None:
            manifest["why_random_beats_ppo"] = memo_random_vs_ppo_path.name
        if diversity_memo_path is not None:
            manifest["does_diversity_dominate_personalization"] = diversity_memo_path.name
        manifest["is_returnrate_main_text_worthy"] = "is_returnrate_main_text_worthy.md"
        manifest["old_vs_corrected_nohabit"] = "old_vs_corrected_nohabit.md"
        manifest["table_budget_attainment"] = "tables/table_budget_attainment.csv"
        manifest["table_frontier_validation"] = "tables/table_frontier_validation.csv"
        manifest["table_diversity_diagnostic"] = "tables/table_diversity_diagnostic.csv"
        manifest["fig_cumwatch_vs_unique_cluster_count"] = "figures/fig_cumwatch_vs_unique_cluster_count.png"
    if run_frontier_artifacts:
        manifest["table_frontier_rebuilt"] = "tables/table_frontier_rebuilt.csv"
        manifest["fig_frontier_overcap_rebuilt"] = "figures/fig_frontier_overcap_rebuilt.png"
        if isinstance(main_text_night_proxy, str) and main_text_night_proxy:
            manifest["frontier_secondary_metric"] = str(main_text_night_proxy)
            manifest["fig_frontier_secondary_risk_rebuilt"] = f"figures/fig_frontier_{filename_slug(main_text_night_proxy)}_rebuilt.png"
    if run_cap_sensitivity:
        manifest["table_cap_sensitivity_rebuilt"] = "tables/table_cap_sensitivity_rebuilt.csv"
    if write_appendix_artifacts:
        manifest.update(
            {
                "table_appendix_stress_tests": "tables/table_appendix_stress_tests.csv",
                "table_appendix_wrapper_baselines": "tables/table_appendix_wrapper_baselines.csv",
                "table_appendix_dashboard": "tables/table_appendix_dashboard.csv",
                "table_appendix_subgroup_disparity": "tables/table_appendix_subgroup_disparity.csv",
            }
        )
    if run_appendix_stress:
        manifest.update(
            {
                "fig_appendix_stop_hazard": "figures/fig_appendix_ppo_default_stop_hazard.png",
                "fig_appendix_gap_histogram": "figures/fig_appendix_ppo_default_gap_histogram.png",
                "fig_appendix_gap_hazard": "figures/fig_appendix_ppo_default_gap_hazard.png",
                "fig_appendix_subgroup_disparity_susceptibility": "figures/fig_appendix_ppo_default_subgroup_disparity_susceptibility.png",
            }
        )
    if write_backend_artifacts:
        manifest.update(
            {
                "table_backend_sensitivity": "tables/table_backend_sensitivity.csv",
                "table_llm_recovery": "tables/table_llm_recovery.csv",
                "table_llm_ablation": "tables/table_llm_ablation.csv",
                "llm_cache_stats": "llm_cache_stats.json",
                "agllm_release_manifest": "agllm_release_manifest.json",
            }
        )
    if calibration_payload is not None:
        manifest["table_calibration_audit"] = "table_calibration_audit.md"
        manifest["table_gap_extraction_diagnostic"] = "table_gap_extraction_diagnostic.md"
        manifest["calibration_status"] = threshold_calibration_status
        calibration_payload_path = outdir / "calibration_payload.json"
        json_dumps_progress(calibration_payload, calibration_payload_path, "calibration_payload_json")
    elif threshold_payload_input is not None:
        manifest["calibration_status"] = threshold_calibration_status
    manifest_path = outdir / "manifest.json"
    json_dumps_progress(manifest, manifest_path, "manifest_json")
    finish_file_task(extra=f"path={manifest_path}")
    log_phase_complete("Manifest writing", manifest_phase_start, extra=f"path={manifest_path}")

    model_files = sorted(p.name for p in model_dir.iterdir())
    top_level_files = sorted(
        p.name
        for p in outdir.iterdir()
        if p.is_file()
    )
    LOGGER.info(
        "Evaluation cache stats | entries=%s | hits=%s | misses=%s",
        len(evaluation_cache),
        evaluation_cache_hits,
        evaluation_cache_misses,
    )
    LOGGER.info(
        "run_paper_pipeline complete in %s | model_files=%s | table_files=%s | figure_files=%s | top_level_files=%s",
        format_elapsed(time.perf_counter() - run_start),
        len(model_files),
        len(table_files),
        len(figure_files),
        top_level_files,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def add_logging_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--log_level", type=str, default="INFO")
    subparser.add_argument("--log_file", type=str, default=None)


def add_trace_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--trace_first_episode", action="store_true")
    subparser.add_argument("--trace_max_steps", type=int, default=25)


def add_torch_runtime_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--device", type=str, choices=["auto", "cpu", "mps", "cuda"], default="auto")
    subparser.add_argument("--torch_num_threads", type=int, default=default_torch_num_threads())
    subparser.add_argument("--torch_num_interop_threads", type=int, default=1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CompulsionBench draft benchmark runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser("smoke_test", help="Run a tiny end-to-end smoke test.")
    p_smoke.add_argument("--outdir", type=Path, required=True)
    p_smoke.add_argument("--seed", type=int, default=0)
    add_torch_runtime_args(p_smoke)
    add_logging_args(p_smoke)
    add_trace_args(p_smoke)

    p_cal = sub.add_parser("calibrate", help="Fit the default config to a user-supplied public log CSV.")
    p_cal.add_argument("--log_csv", type=str, required=True)
    p_cal.add_argument("--delta_sess", type=float, default=30.0)
    p_cal.add_argument("--outdir", type=Path, required=True)
    p_cal.add_argument("--metadata_csv", type=str, default=None)
    p_cal.add_argument("--n_trials", type=int, default=50)
    p_cal.add_argument("--episodes_per_trial", type=int, default=100)
    p_cal.add_argument("--calibration_policy", type=str, choices=CALIBRATION_POLICY_CHOICES, default=DEFAULT_CALIBRATION_POLICY)
    p_cal.add_argument("--threshold_source", type=str, choices=THRESHOLD_SOURCE_CHOICES, default=DEFAULT_THRESHOLD_SOURCE)
    p_cal.add_argument("--seed", type=int, default=0)
    add_logging_args(p_cal)

    p_cal_optuna = sub.add_parser("calibrate_optuna", help="Fit the default config to a user-supplied public log CSV with Optuna.")
    p_cal_optuna.add_argument("--log_csv", type=str, required=True)
    p_cal_optuna.add_argument("--delta_sess", type=float, default=30.0)
    p_cal_optuna.add_argument("--outdir", type=Path, required=True)
    p_cal_optuna.add_argument("--metadata_csv", type=str, default=None)
    p_cal_optuna.add_argument("--exploratory_trials", type=int, default=300)
    p_cal_optuna.add_argument("--episodes_per_trial", type=int, default=200)
    p_cal_optuna.add_argument("--topk_trials", type=int, default=20)
    p_cal_optuna.add_argument("--topk_episodes", type=int, default=1000)
    p_cal_optuna.add_argument("--finalists", type=int, default=10)
    p_cal_optuna.add_argument("--final_episodes", type=int, default=5000)
    p_cal_optuna.add_argument("--fixed_seed_list_json", type=Path, default=None)
    p_cal_optuna.add_argument("--calibration_policy", type=str, choices=CALIBRATION_POLICY_CHOICES, default=DEFAULT_CALIBRATION_POLICY)
    p_cal_optuna.add_argument("--threshold_source", type=str, choices=THRESHOLD_SOURCE_CHOICES, default=DEFAULT_THRESHOLD_SOURCE)
    p_cal_optuna.add_argument("--seed", type=int, default=0)
    add_logging_args(p_cal_optuna)

    p_feas = sub.add_parser(
        "audit_calibration_feasibility",
        help="Audit whether the current simulator family can structurally match a calibration payload tail.",
    )
    p_feas.add_argument("--payload_json", type=Path, required=True)
    p_feas.add_argument("--outdir", type=Path, required=True)
    add_logging_args(p_feas)

    p_frag = sub.add_parser(
        "rebuild_cap_fragmentation",
        help="Re-evaluate saved PPO checkpoints and write session-cap fragmentation diagnostics.",
    )
    p_frag.add_argument("--bundle_dir", type=Path, required=True)
    p_frag.add_argument("--calibration_audit_md", type=Path, default=None)
    p_frag.add_argument("--cap_grid", type=float, nargs="+", default=[90.0, 120.0, 150.0])
    p_frag.add_argument("--max_workers", type=int, default=4)
    add_torch_runtime_args(p_frag)
    add_logging_args(p_frag)

    p_score = sub.add_parser(
        "rebuild_official_scorecards",
        help="Re-evaluate saved official policies and rewrite the main scorecard tables with episode-bootstrap uncertainty.",
    )
    p_score.add_argument("--bundle_dir", type=Path, required=True)
    p_score.add_argument("--max_workers", type=int, default=4)
    p_score.add_argument("--episode_limit", type=int, default=None)
    add_torch_runtime_args(p_score)
    add_logging_args(p_score)

    p_mech = sub.add_parser(
        "rebuild_mechanism_ablations",
        help="Re-evaluate saved PPO checkpoints under the mechanism ablations and write a compact table plus summary figure.",
    )
    p_mech.add_argument("--bundle_dir", type=Path, required=True)
    p_mech.add_argument("--max_workers", type=int, default=4)
    p_mech.add_argument("--episode_limit", type=int, default=None)
    add_torch_runtime_args(p_mech)
    add_logging_args(p_mech)

    p_lagppo = sub.add_parser(
        "rebuild_lagppo_rescue",
        help="Summarize saved LagPPO feasibility from run.log and run a bounded rescue sweep on selected seeds.",
    )
    p_lagppo.add_argument("--bundle_dir", type=Path, required=True)
    p_lagppo.add_argument("--run_log", type=Path, default=None)
    p_lagppo.add_argument("--train_seeds", type=int, nargs="+", default=None)
    p_lagppo.add_argument("--rescue_scales", type=float, nargs="+", default=[0.90, 0.95])
    p_lagppo.add_argument("--rescue_dual_lrs", type=float, nargs="+", default=[0.005, 0.01])
    p_lagppo.add_argument("--rescue_steps", type=int, default=25_000)
    p_lagppo.add_argument("--rescue_validate_every", type=int, default=12_500)
    p_lagppo.add_argument("--rescue_val_episodes", type=int, default=500)
    p_lagppo.add_argument("--confirm_val_episodes", type=int, default=None)
    p_lagppo.add_argument("--cost_normalization", type=str, choices=["none", "budget"], default="budget")
    add_torch_runtime_args(p_lagppo)
    add_logging_args(p_lagppo)

    p_run = sub.add_parser("run_paper", help="Train baselines, evaluate them, and write paper-style tables and figures.")
    p_run.add_argument("--outdir", type=Path, required=True)
    p_run.add_argument("--config_json", type=str, default=None)
    p_run.add_argument("--metadata_csv", type=str, default=None)
    p_run.add_argument("--log_csv", type=str, default=None)
    p_run.add_argument("--calibration_payload_json", type=Path, default=None)
    p_run.add_argument("--delta_sess", type=float, default=30.0)
    p_run.add_argument("--threshold_source", type=str, choices=THRESHOLD_SOURCE_CHOICES, default=DEFAULT_THRESHOLD_SOURCE)
    p_run.add_argument("--calibration_trials", type=int, default=50)
    p_run.add_argument("--calibration_episodes", type=int, default=100)
    p_run.add_argument("--calibration_policy", type=str, choices=CALIBRATION_POLICY_CHOICES, default=DEFAULT_CALIBRATION_POLICY)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--run_profile", type=str, choices=RUN_PROFILE_CHOICES, default="main")
    add_torch_runtime_args(p_run)
    p_run.add_argument("--num_train_seeds", type=int, default=5)
    p_run.add_argument("--eval_episodes", type=int, default=5_000)
    p_run.add_argument("--random_log_steps", type=int, default=20_000)
    p_run.add_argument("--myopic_epochs", type=int, default=8)
    p_run.add_argument("--myopic_batch_size", type=int, default=512)
    p_run.add_argument("--myopic_lr", type=float, default=1e-3)
    p_run.add_argument("--ppo_steps", type=int, default=1_000_000)
    p_run.add_argument("--rollout_steps", type=int, default=2048)
    p_run.add_argument("--minibatch_size", type=int, default=256)
    p_run.add_argument("--update_epochs", type=int, default=10)
    p_run.add_argument("--ppo_lr", type=float, default=3e-4)
    p_run.add_argument("--ppo_ent_coef", type=float, default=0.01)
    p_run.add_argument("--ppo_hidden", type=int, default=128)
    p_run.add_argument("--validate_every", type=int, default=20_000)
    p_run.add_argument("--val_episodes", type=int, default=1000)
    p_run.add_argument("--dual_lr", type=float, default=0.05)
    p_run.add_argument("--lag_search_steps", type=int, default=25_000)
    p_run.add_argument("--lag_search_val_episodes", type=int, default=200)
    p_run.add_argument("--lag_full_steps", type=int, default=None)
    p_run.add_argument("--lag_full_val_episodes", type=int, default=None)
    p_run.add_argument("--lag_dual_lr_grid_search", type=float, nargs="+", default=None)
    p_run.add_argument("--lag_topk_candidates", type=int, default=1)
    p_run.add_argument("--lag_search_mode", type=str, choices=["full", "two_stage"], default=None)
    p_run.add_argument("--constraint_scales", type=float, nargs="+", default=list(DEFAULT_CONSTRAINT_SCALES))
    p_run.add_argument("--cap_grid", type=float, nargs="+", default=[])
    p_run.add_argument("--with_llm", action="store_true")
    p_run.add_argument("--llm_mode", type=str, default="surrogate", choices=["surrogate", "hf"])
    p_run.add_argument("--llm_model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p_run.add_argument("--fit_llm_fusion", action="store_true")
    p_run.add_argument("--paper_mode", action="store_true", help="Enforce the strict paper protocol for AGLLM artifacts and cards.")
    add_logging_args(p_run)
    add_trace_args(p_run)

    p_cards = sub.add_parser("build_cards", help="Build deterministic archetype cards from metadata.")
    p_cards.add_argument("--metadata_csv", type=str, required=True)
    p_cards.add_argument("--out_json", type=Path, required=True)
    p_cards.add_argument("--seed", type=int, default=0)
    add_logging_args(p_cards)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        setup_logging(args.log_level, args.log_file)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "run_paper":
        apply_run_profile_defaults(args)
    if args.command in {
        "smoke_test",
        "run_paper",
        "rebuild_cap_fragmentation",
        "rebuild_official_scorecards",
        "rebuild_mechanism_ablations",
        "rebuild_lagppo_rescue",
    }:
        try:
            args.device = configure_torch_runtime(
                args.device,
                args.torch_num_threads,
                args.torch_num_interop_threads,
            )
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))

    if args.command == "smoke_test":
        args = argparse.Namespace(
            outdir=args.outdir,
            config_json=None,
            metadata_csv=None,
            log_csv=None,
            calibration_payload_json=None,
            delta_sess=30.0,
            threshold_source=DEFAULT_THRESHOLD_SOURCE,
            calibration_trials=5,
            calibration_episodes=20,
            calibration_policy=DEFAULT_CALIBRATION_POLICY,
            seed=args.seed,
            device=args.device,
            num_train_seeds=1,
            eval_episodes=50,
            random_log_steps=2_000,
            myopic_epochs=2,
            myopic_batch_size=256,
            myopic_lr=1e-3,
            ppo_steps=4_000,
            rollout_steps=512,
            minibatch_size=128,
            update_epochs=4,
            ppo_lr=3e-4,
            ppo_ent_coef=0.01,
            ppo_hidden=128,
            validate_every=2_000,
            val_episodes=16,
            dual_lr=0.05,
            lag_search_steps=2_000,
            lag_search_val_episodes=16,
            lag_full_steps=4_000,
            lag_full_val_episodes=16,
            lag_dual_lr_grid_search=[0.05, 0.1],
            lag_topk_candidates=1,
            lag_search_mode="full",
            constraint_scales=[0.95, 0.9],
            cap_grid=[],
            with_llm=True,
            llm_mode="surrogate",
            llm_model_id="Qwen/Qwen2.5-7B-Instruct",
            fit_llm_fusion=False,
            paper_mode=False,
            run_profile="full",
            trace_first_episode=args.trace_first_episode,
            trace_max_steps=args.trace_max_steps,
            log_level=args.log_level,
            log_file=args.log_file,
            torch_num_threads=args.torch_num_threads,
            torch_num_interop_threads=args.torch_num_interop_threads,
        )
        run_paper_pipeline(args)
        invariant_results = run_invariant_smoke_tests()
        LOGGER.info("Invariant smoke tests passed | %s", invariant_results)
        LOGGER.info("Smoke test complete. Outputs written to %s", args.outdir)
        return

    if args.command == "calibrate":
        outdir = ensure_dir(args.outdir)
        cfg = BenchConfig()
        catalog = build_catalog(cfg)
        cards = build_default_cards(cfg, args.metadata_csv, random_seed=args.seed)
        delta_grid = [float(args.delta_sess)]
        targets = {
            float(delta): extract_targets_from_logs(
                args.log_csv,
                float(delta),
                Z=cfg.Z,
                metadata_csv=args.metadata_csv,
                random_seed=args.seed,
                bootstrap_samples=32,
                bootstrap_seed=args.seed + int(delta * 100),
            )
            for delta in delta_grid
        }
        best_cfg, history = random_search_calibration(
            cfg,
            catalog,
            cards,
            targets,
            n_trials=args.n_trials,
            episodes_per_trial=args.episodes_per_trial,
            seed=args.seed,
            selection_delta=float(args.delta_sess),
            calibration_policy=str(args.calibration_policy),
        )
        sim = simulate_targets(
            best_cfg,
            catalog,
            cards,
            seeds=list(range(args.seed, args.seed + args.episodes_per_trial)),
            gap_bucket_edges=targets[float(args.delta_sess)].get("gap_bucket_edges"),
            calibration_policy=str(args.calibration_policy),
            calibration_policy_context=resolve_calibration_rollout_policy_context(
                targets[float(args.delta_sess)],
                str(args.calibration_policy),
            ),
        )
        calibration_status = build_calibration_audit(targets, sim, float(args.delta_sess))["status"]
        thresholds = resolve_thresholds_for_source(
            str(args.threshold_source),
            selected_delta_sess=float(args.delta_sess),
            reference_targets=targets[float(args.delta_sess)],
            sim_targets=sim,
            calibration_status=calibration_status,
        )
        apply_paper_thresholds(best_cfg, thresholds)
        best_cfg.save(outdir / "config_calibrated.json")
        pd.DataFrame(history).to_csv(outdir / "calibration_history.csv", index=False)
        save_table(threshold_table_dataframe(thresholds), outdir / "table_thresholds")
        calibration_payload = build_calibration_payload(
            targets,
            sim,
            selected_delta_sess=float(args.delta_sess),
            log_csv=args.log_csv,
            n_trials=args.n_trials,
            episodes_per_trial=args.episodes_per_trial,
            seed=args.seed,
            calibration_policy=str(args.calibration_policy),
            thresholds=thresholds,
            calibrated_cfg=best_cfg,
        )
        json_dumps(calibration_payload, outdir / "calibration_payload.json")
        (outdir / "table_calibration_audit.md").write_text(
            render_calibration_audit_markdown(calibration_payload["audit"]),
            encoding="utf-8",
        )
        write_gap_extraction_diagnostic_artifact(
            targets[float(args.delta_sess)],
            outdir / "table_gap_extraction_diagnostic.md",
            delta_sess=float(args.delta_sess),
        )
        json_dumps(
            {
                **calibration_payload["manifest"],
                "artifacts": {
                    "config_calibrated": "config_calibrated.json",
                    "calibration_history": "calibration_history.csv",
                    "calibration_payload": "calibration_payload.json",
                    "table_calibration_audit": "table_calibration_audit.md",
                    "table_gap_extraction_diagnostic": "table_gap_extraction_diagnostic.md",
                    "table_thresholds": "table_thresholds.md",
                },
                "paper_thresholds": thresholds,
            },
            outdir / "manifest.json",
        )
        LOGGER.info(
            "Calibration complete | status=%s | selected_delta_sess=%s | best_config=%s",
            calibration_payload["manifest"]["status"],
            calibration_payload["manifest"]["selected_delta_sess"],
            outdir / "config_calibrated.json",
        )
        return

    if args.command == "calibrate_optuna":
        outdir = ensure_dir(args.outdir)
        cfg = BenchConfig()
        catalog = build_catalog(cfg)
        cards = build_default_cards(cfg, args.metadata_csv, random_seed=args.seed)
        delta_grid = [float(args.delta_sess)]
        targets = {
            float(delta): extract_targets_from_logs(
                args.log_csv,
                float(delta),
                Z=cfg.Z,
                metadata_csv=args.metadata_csv,
                random_seed=args.seed,
                bootstrap_samples=32,
                bootstrap_seed=args.seed + int(delta * 100),
            )
            for delta in delta_grid
        }
        fixed_seed_list = None
        if args.fixed_seed_list_json is not None:
            fixed_seed_blob = json_load(args.fixed_seed_list_json)
            if isinstance(fixed_seed_blob, dict):
                fixed_seed_blob = fixed_seed_blob.get("fixed_seed_list", [])
            fixed_seed_list = [int(value) for value in list(fixed_seed_blob)]
        optuna_result = calibrate_with_optuna(
            cfg,
            catalog,
            cards,
            targets,
            selection_delta=float(args.delta_sess),
            seed=args.seed,
            fixed_seed_list=fixed_seed_list,
            exploratory_trials=int(args.exploratory_trials),
            exploratory_episodes=int(args.episodes_per_trial),
            topk_trials=int(args.topk_trials),
            topk_episodes=int(args.topk_episodes),
            finalists=int(args.finalists),
            final_episodes=int(args.final_episodes),
            storage_path=outdir / "study.sqlite3",
            calibration_policy=str(args.calibration_policy),
        )

        history_df = pd.DataFrame(optuna_result["history_rows"])
        history_df.to_csv(outdir / "calibration_history.csv", index=False)
        top_trials_df = pd.DataFrame(optuna_result["top_trial_rows"])
        top_trials_df.to_csv(outdir / "top_trials.csv", index=False)
        selection_df = build_calibration_selection_table_dataframe(optuna_result["final_rows"])
        save_table(selection_df, outdir / "table_calibration_selection")
        json_dumps(optuna_result["fixed_seed_list"], outdir / "fixed_seed_list.json")
        (outdir / "table_calibration_confirmation.md").write_text(
            render_calibration_confirmation_markdown(
                optuna_result["confirmation_manifest"],
                optuna_result["confirmation_audit"],
            ),
            encoding="utf-8",
        )
        json_dumps(optuna_result["confirmation_manifest"], outdir / "manifest_confirmation.json")

        best_cfg = copy.deepcopy(optuna_result["best_cfg"])
        sim = copy.deepcopy(optuna_result["best_sim"])
        selected_config_confirmed = str(optuna_result["confirmation_status"]) == "confirmed"
        thresholds = resolve_thresholds_for_source(
            str(args.threshold_source),
            selected_delta_sess=float(args.delta_sess),
            reference_targets=targets[float(args.delta_sess)],
            sim_targets=sim,
            calibration_status="passed" if selected_config_confirmed else "failed",
        )
        apply_paper_thresholds(best_cfg, thresholds)
        best_cfg.save(outdir / "config_calibrated.json")
        save_table(threshold_table_dataframe(thresholds), outdir / "table_thresholds")
        calibration_payload = build_calibration_payload(
            targets,
            sim,
            selected_delta_sess=float(args.delta_sess),
            log_csv=args.log_csv,
            n_trials=int(args.exploratory_trials),
            episodes_per_trial=int(args.episodes_per_trial),
            seed=args.seed,
            calibration_policy=str(args.calibration_policy),
            thresholds=thresholds,
            calibrated_cfg=best_cfg,
        )
        calibration_payload["confirmation"] = {
            "manifest": copy.deepcopy(optuna_result["confirmation_manifest"]),
            "audit": copy.deepcopy(optuna_result["confirmation_audit"]),
        }
        calibration_payload["manifest"].update(
            {
                "optimization_backend": "optuna",
                "study_name": str(optuna_result["study_name"]),
                "study_storage": "study.sqlite3",
                "selected_trial_number": int(optuna_result["selected_trial_number"]),
                "selected_reason": str(optuna_result["selected_reason"]),
                "selection_warning": optuna_result["selection_warning"],
                "exploratory_trials": int(args.exploratory_trials),
                "exploratory_episodes": int(args.episodes_per_trial),
                "topk_trials": int(args.topk_trials),
                "topk_episodes": int(args.topk_episodes),
                "requested_finalists": int(args.finalists),
                "finalists": int(optuna_result["finalist_count"]),
                "final_episodes": int(args.final_episodes),
                "fixed_seed_list_artifact": "fixed_seed_list.json",
                "table_calibration_selection": "table_calibration_selection.md",
                "table_calibration_confirmation": "table_calibration_confirmation.md",
                "manifest_confirmation": "manifest_confirmation.json",
                "selected_failed_checks": int(optuna_result["selected_final_row"].get("acceptance_failed_checks", 0)),
                "selected_worst_margin_violation": float(
                    optuna_result["selected_final_row"].get("acceptance_worst_margin_violation", 0.0)
                ),
                "confirmation_seed_count": int(optuna_result["confirmation_manifest"]["confirmation_seed_count"]),
                "confirmation_audit_status": str(optuna_result["confirmation_manifest"]["confirmation_audit_status"]),
                "confirmation_passed": bool(optuna_result["confirmation_manifest"]["confirmation_acceptance_passed"]),
                "confirmation_status": str(optuna_result["confirmation_manifest"]["confirmation_status"]),
                "selected_config_status": str(optuna_result["confirmation_manifest"]["confirmation_status"]),
            }
        )
        if calibration_payload["manifest"]["status"] != "passed":
            calibration_payload["manifest"]["inadequacy_note"] = (
                "Optuna calibration still fails the acceptance thresholds; treat the current model family as inadequate rather than continuing blind tuning."
            )
        json_dumps(calibration_payload, outdir / "calibration_payload.json")
        (outdir / "table_calibration_audit.md").write_text(
            render_calibration_audit_markdown(calibration_payload["audit"]),
            encoding="utf-8",
        )
        write_gap_extraction_diagnostic_artifact(
            targets[float(args.delta_sess)],
            outdir / "table_gap_extraction_diagnostic.md",
            delta_sess=float(args.delta_sess),
        )
        json_dumps(
            {
                **calibration_payload["manifest"],
                "artifacts": {
                    "study": "study.sqlite3",
                    "calibration_history": "calibration_history.csv",
                    "top_trials": "top_trials.csv",
                    "fixed_seed_list": "fixed_seed_list.json",
                    "config_calibrated": "config_calibrated.json",
                    "calibration_payload": "calibration_payload.json",
                    "table_calibration_audit": "table_calibration_audit.md",
                    "table_calibration_selection": "table_calibration_selection.md",
                    "table_calibration_confirmation": "table_calibration_confirmation.md",
                    "manifest_confirmation": "manifest_confirmation.json",
                    "table_gap_extraction_diagnostic": "table_gap_extraction_diagnostic.md",
                    "table_thresholds": "table_thresholds.md",
                },
                "paper_thresholds": thresholds,
            },
            outdir / "manifest.json",
        )
        if calibration_payload["manifest"]["status"] != "passed":
            LOGGER.warning(
                "Optuna calibration failed acceptance | model_family_assessment=%s | config=%s",
                calibration_payload["manifest"]["model_family_assessment"],
                outdir / "config_calibrated.json",
            )
        LOGGER.info(
            "Optuna calibration complete | status=%s | selected_delta_sess=%s | best_config=%s | selected_trial=%s",
            calibration_payload["manifest"]["status"],
            calibration_payload["manifest"]["selected_delta_sess"],
            outdir / "config_calibrated.json",
            optuna_result["selected_trial_number"],
        )
        return

    if args.command == "audit_calibration_feasibility":
        payload = json_load(args.payload_json)
        if isinstance(payload, dict) and payload.get("fitted_config") is None:
            sibling_config_path = args.payload_json.resolve().with_name("config_calibrated.json")
            if sibling_config_path.exists():
                payload = copy.deepcopy(payload)
                payload["fitted_config"] = json_load(sibling_config_path)
        write_calibration_feasibility_artifacts(payload, args.outdir)
        LOGGER.info("Calibration feasibility audit complete. Outputs written to %s", args.outdir)
        return

    if args.command == "rebuild_cap_fragmentation":
        write_cap_fragmentation_artifacts(
            args.bundle_dir,
            device=args.device,
            cap_grid=args.cap_grid,
            calibration_audit_md=args.calibration_audit_md,
            max_workers=args.max_workers,
        )
        LOGGER.info("Cap-fragmentation rebuild complete. Outputs written to %s", args.bundle_dir)
        return

    if args.command == "rebuild_official_scorecards":
        write_official_scorecard_artifacts(
            args.bundle_dir,
            device=args.device,
            max_workers=args.max_workers,
            episode_limit=args.episode_limit,
        )
        LOGGER.info("Official scorecard rebuild complete. Outputs written to %s", args.bundle_dir)
        return

    if args.command == "rebuild_mechanism_ablations":
        write_mechanism_ablation_artifacts(
            args.bundle_dir,
            device=args.device,
            max_workers=args.max_workers,
            episode_limit=args.episode_limit,
        )
        LOGGER.info("Mechanism ablation rebuild complete. Outputs written to %s", args.bundle_dir)
        return

    if args.command == "rebuild_lagppo_rescue":
        write_lagppo_rescue_artifacts(
            args.bundle_dir,
            device=args.device,
            run_log_path=args.run_log,
            train_seeds=args.train_seeds,
            rescue_scales=args.rescue_scales,
            rescue_dual_lrs=args.rescue_dual_lrs,
            rescue_steps=args.rescue_steps,
            rescue_validate_every=args.rescue_validate_every,
            rescue_val_episodes=args.rescue_val_episodes,
            confirm_val_episodes=args.confirm_val_episodes,
            cost_normalization=args.cost_normalization,
        )
        LOGGER.info("LagPPO rescue rebuild complete. Outputs written to %s", args.bundle_dir)
        return

    if args.command == "run_paper":
        run_paper_pipeline(args)
        LOGGER.info("Paper pipeline complete. Outputs written to %s", args.outdir)
        return

    if args.command == "build_cards":
        cfg = BenchConfig()
        cards = build_default_cards(cfg, args.metadata_csv, random_seed=args.seed, paper_mode=True)
        json_dumps(cards, args.out_json)
        LOGGER.info("Wrote %s archetype cards to %s", len(cards), args.out_json)
        return


if __name__ == "__main__":
    main()
