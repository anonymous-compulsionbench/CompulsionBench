import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytest.importorskip("matplotlib")
pytest.importorskip("pandas")


def load_cb():
    path = Path(__file__).resolve().parents[1] / "compulsionbench" / "compulsionbench.py"
    spec = importlib.util.spec_from_file_location("compulsionbench_test_optuna_calibration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cb = load_cb()


def make_cfg():
    cfg = cb.BenchConfig(H=240.0, T_ref=1.0, T_cap=1.0, device="cpu")
    cfg.session_start_hour_probs = [1.0 / 24.0] * 24
    return cfg


def make_uniform_logged_cluster_policy_context(Z):
    uniform = [1.0 / float(Z)] * int(Z)
    return {
        "policy": "logged_cluster_marginal",
        "conditioning": "hour_of_day",
        "cluster_source": "synthetic_uniform",
        "cluster_projection_source": "synthetic_uniform",
        "cluster_probabilities": list(uniform),
        "hourly_cluster_probabilities": [[] for _ in range(24)],
        "fixed_lambda_idx": 0,
        "fixed_nu": 0,
        "fallback_when_hour_empty": "overall_cluster_marginal",
        "rows_used": int(Z),
        "hourly_rows_with_support": 0,
    }


def build_synthetic_problem(seed_list):
    base_cfg = make_cfg()
    catalog = cb.build_catalog(base_cfg)
    cards = cb.build_default_cards(base_cfg, metadata_csv=None, random_seed=0, paper_mode=False)
    target_cfg = cb.replace(base_cfg, continue_logit_bias=1.5, continue_logit_temp=3.0)
    targets = {
        30.0: {
            **cb.simulate_targets(target_cfg, catalog, cards, seed_list),
            "calibration_weights": {
                "session": 1.0,
                "session_item_counts": 1.0,
                "hazard": 0.1,
                "gaps": 0.1,
                "cluster": 0.0,
                "diurnal": 0.0,
                "return_conditional": 0.0,
            },
            "calibration_rollout_policies": {
                "logged_cluster_marginal": make_uniform_logged_cluster_policy_context(base_cfg.Z),
            },
        }
    }
    return base_cfg, catalog, cards, target_cfg, targets


def calibration_snapshot(cfg):
    return cb.calibration_param_snapshot(cfg, cb.CALIBRATION_HISTORY_PARAM_KEYS)


def calibration_row_snapshot(row):
    return {key: float(row[key]) for key in cb.CALIBRATION_HISTORY_PARAM_KEYS if key in row}


def rounded(value):
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {key: rounded(val) for key, val in sorted(value.items())}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def test_optuna_fixed_seed_search_beats_legacy_random_search(tmp_path):
    pytest.importorskip("optuna")
    fixed_seed_list = [10_000 + idx for idx in range(24)]
    base_cfg, catalog, cards, target_cfg, targets = build_synthetic_problem(fixed_seed_list)

    legacy_cfg, _ = cb.random_search_calibration(
        base_cfg,
        catalog,
        cards,
        targets,
        n_trials=24,
        episodes_per_trial=len(fixed_seed_list),
        seed=7,
        selection_delta=30.0,
        scoring_seeds=fixed_seed_list,
        calibration_policy="random",
    )
    _, _, legacy_row = cb.evaluate_calibration_candidate(
        legacy_cfg,
        catalog,
        cards,
        targets,
        seeds=fixed_seed_list,
        selection_delta=30.0,
        trial_id=900,
        stage="legacy_verify",
    )

    optuna_result = cb.calibrate_with_optuna(
        base_cfg,
        catalog,
        cards,
        targets,
        selection_delta=30.0,
        seed=7,
        fixed_seed_list=fixed_seed_list,
        exploratory_trials=6,
        exploratory_episodes=len(fixed_seed_list),
        topk_trials=3,
        topk_episodes=48,
        finalists=2,
        final_episodes=72,
        storage_path=tmp_path / "study.sqlite3",
        initial_trial_params=[calibration_snapshot(target_cfg)],
        calibration_policy="random",
    )
    _, _, optuna_row = cb.evaluate_calibration_candidate(
        optuna_result["best_cfg"],
        catalog,
        cards,
        targets,
        seeds=fixed_seed_list,
        selection_delta=30.0,
        trial_id=901,
        stage="optuna_verify",
    )

    assert (tmp_path / "study.sqlite3").exists()
    assert optuna_result["fixed_seed_list"] == fixed_seed_list
    assert float(legacy_row["loss"]) > 0.0
    assert float(optuna_row["loss"]) < float(legacy_row["loss"])


def test_optuna_search_is_deterministic_for_same_fixed_seed_list():
    pytest.importorskip("optuna")
    fixed_seed_list = [20_000 + idx for idx in range(16)]
    base_cfg, catalog, cards, target_cfg, targets = build_synthetic_problem(fixed_seed_list)
    initial_params = [calibration_snapshot(target_cfg)]

    result_a = cb.calibrate_with_optuna(
        base_cfg,
        catalog,
        cards,
        targets,
        selection_delta=30.0,
        seed=11,
        fixed_seed_list=fixed_seed_list,
        exploratory_trials=5,
        exploratory_episodes=len(fixed_seed_list),
        topk_trials=2,
        topk_episodes=24,
        finalists=2,
        final_episodes=32,
        initial_trial_params=initial_params,
        calibration_policy="logged_cluster_marginal",
    )
    result_b = cb.calibrate_with_optuna(
        base_cfg,
        catalog,
        cards,
        targets,
        selection_delta=30.0,
        seed=11,
        fixed_seed_list=fixed_seed_list,
        exploratory_trials=5,
        exploratory_episodes=len(fixed_seed_list),
        topk_trials=2,
        topk_episodes=24,
        finalists=2,
        final_episodes=32,
        initial_trial_params=initial_params,
        calibration_policy="logged_cluster_marginal",
    )

    assert result_a["fixed_seed_list"] == result_b["fixed_seed_list"] == fixed_seed_list
    assert result_a["calibration_policy"] == result_b["calibration_policy"] == "logged_cluster_marginal"
    assert result_a["selected_trial_number"] == result_b["selected_trial_number"]
    assert int(result_a["selected_final_row"]["trial"]) == result_a["selected_trial_number"]
    assert result_a["selected_final_row"]["calibration_policy"] == "logged_cluster_marginal"
    assert set(result_a["fixed_seed_list"]).isdisjoint(result_a["topk_seed_list"])
    assert set(result_a["fixed_seed_list"]).isdisjoint(result_a["final_seed_list"])
    assert set(result_a["fixed_seed_list"]).isdisjoint(result_a["confirmation_seed_list"])
    assert set(result_a["topk_seed_list"]).isdisjoint(result_a["final_seed_list"])
    assert set(result_a["topk_seed_list"]).isdisjoint(result_a["confirmation_seed_list"])
    assert set(result_a["final_seed_list"]).isdisjoint(result_a["confirmation_seed_list"])
    assert calibration_snapshot(result_a["best_cfg"]) == pytest.approx(
        calibration_row_snapshot(result_a["selected_final_row"])
    )
    final_rows = [row for row in result_a["top_trial_rows"] if row.get("stage") == "final_reeval"]
    if any(bool(row.get("acceptance_passed")) for row in final_rows):
        assert bool(result_a["selected_final_row"]["acceptance_passed"])
        assert result_a["selected_reason"] == "accepted_lowest_loss"
        assert result_a["selection_warning"] is None
    else:
        assert result_a["selected_reason"] == "no_accepted_candidate_fallback"
        assert isinstance(result_a["selection_warning"], str)
        assert "No final_reeval candidate passed acceptance" in result_a["selection_warning"]
    assert calibration_snapshot(result_a["best_cfg"]) == pytest.approx(calibration_snapshot(result_b["best_cfg"]))
    assert rounded(result_a["history_rows"]) == rounded(result_b["history_rows"])
    assert rounded(result_a["top_trial_rows"]) == rounded(result_b["top_trial_rows"])
    assert rounded(result_a["selected_final_row"]) == rounded(result_b["selected_final_row"])
    assert rounded(result_a["confirmation_row"]) == rounded(result_b["confirmation_row"])
    assert rounded(result_a["confirmation_manifest"]) == rounded(result_b["confirmation_manifest"])
    assert result_a["selected_reason"] == result_b["selected_reason"]
    assert result_a["selection_warning"] == result_b["selection_warning"]
    assert result_a["confirmation_status"] == result_b["confirmation_status"]
    assert result_a["confirmation_manifest"]["confirmation_status"] == result_a["confirmation_status"]
    assert result_a["confirmation_manifest"]["confirmation_seed_count"] == len(result_a["confirmation_seed_list"])
    final_passed = bool(result_a["selected_final_row"]["acceptance_passed"])
    confirmation_passed = bool(result_a["confirmation_manifest"]["confirmation_acceptance_passed"])
    expected_confirmation_status = "confirmed" if (final_passed and confirmation_passed) else "unconfirmed"
    assert result_a["confirmation_status"] == expected_confirmation_status

    payload = cb.build_calibration_payload(
        targets,
        result_a["best_sim"],
        selected_delta_sess=30.0,
        log_csv="synthetic.csv",
        n_trials=5,
        episodes_per_trial=len(fixed_seed_list),
        seed=11,
        calibration_policy=result_a["calibration_policy"],
        calibrated_cfg=result_a["best_cfg"],
    )
    assert payload["manifest"]["status"] == result_a["selected_final_row"]["status"]


def test_logged_cluster_marginal_policy_context_tracks_hourly_cluster_frequencies():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        [
            {"timestamp_min": 5.0, "cluster_id": 0},
            {"timestamp_min": 35.0, "cluster_id": 1},
            {"timestamp_min": 61.0, "cluster_id": 1},
            {"timestamp_min": 125.0, "cluster_id": 2},
        ]
    )

    context = cb.build_logged_cluster_marginal_policy_context(df, Z=3, cluster_source="provided")

    assert context["policy"] == "logged_cluster_marginal"
    assert context["conditioning"] == "hour_of_day"
    assert context["cluster_probabilities"] == pytest.approx([0.25, 0.5, 0.25])
    assert context["hourly_cluster_probabilities"][0] == pytest.approx([0.5, 0.5, 0.0])
    assert context["hourly_cluster_probabilities"][1] == pytest.approx([0.0, 1.0, 0.0])
    assert context["hourly_cluster_probabilities"][2] == pytest.approx([0.0, 0.0, 1.0])


def test_logged_cluster_marginal_sampler_uses_neutral_action_tuple():
    cfg = make_cfg()
    context = {
        "cluster_probabilities": [0.0, 1.0] + [0.0] * (cfg.Z - 2),
        "hourly_cluster_probabilities": [[] for _ in range(24)],
        "fixed_lambda_idx": 0,
        "fixed_nu": 0,
    }

    class DummyEnv:
        def __init__(self):
            self.rng = cb.np.random.default_rng(123)
            self.state = SimpleNamespace(tau=0.0, tau_start=0.0)

        def _wall_clock(self, tau, tau_start):
            _ = tau, tau_start
            return 0.0

    action = cb.sample_calibration_rollout_action(
        cfg,
        DummyEnv(),
        "logged_cluster_marginal",
        calibration_policy_context=context,
    )
    z, lam_idx, nu = cb.action_id_to_tuple(action, cfg.P)

    assert (z, lam_idx, nu) == (1, 0, 0)


def test_select_final_calibration_row_prefers_accepted_rows():
    selection = cb.select_final_calibration_row(
        [
            {"trial": 153, "loss_primary": 0.10, "loss_secondary": 0.50, "loss": 100.50, "acceptance_passed": False},
            {"trial": 187, "loss_primary": 0.20, "loss_secondary": 0.20, "loss": 200.20, "acceptance_passed": True},
            {"trial": 206, "loss_primary": 0.25, "loss_secondary": 0.10, "loss": 250.10, "acceptance_passed": True},
        ]
    )

    assert selection["selected_row"]["trial"] == 187
    assert selection["selected_reason"] == "accepted_lowest_loss"
    assert selection["selection_warning"] is None


def test_select_final_calibration_row_warns_on_no_accepted_candidates(caplog):
    with caplog.at_level("WARNING"):
        selection = cb.select_final_calibration_row(
            [
                {"trial": 153, "loss_primary": 0.10, "loss_secondary": 0.50, "loss": 100.50, "acceptance_passed": False},
                {"trial": 187, "loss_primary": 0.20, "loss_secondary": 0.20, "loss": 200.20, "acceptance_passed": False},
            ]
        )

    assert selection["selected_row"]["trial"] == 153
    assert selection["selected_reason"] == "no_accepted_candidate_fallback"
    assert isinstance(selection["selection_warning"], str)
    assert "No final_reeval candidate passed acceptance" in selection["selection_warning"]
    assert "No final_reeval candidate passed acceptance" in caplog.text
