import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest


def load_cb():
    path = Path(__file__).resolve().parents[1] / "compulsionbench" / "compulsionbench.py"
    spec = importlib.util.spec_from_file_location("compulsionbench_test_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cb = load_cb()


def make_cfg():
    return cb.BenchConfig(T_ref=1.0, T_cap=1.0, device="cpu")


def deterministic_action_trace(policy, cfg, catalog, cards, *, seed: int, steps: int) -> list[int]:
    env = cb.make_env(cfg, "param", catalog, cards, scorer=None, wrappers=None, seed=seed)
    obs = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    trace: list[int] = []
    for _ in range(steps):
        action, _ = policy.act_with_info(obs, deterministic=True, env=env, need_info=False)
        trace.append(int(action))
        obs, _, done, _ = env.step(int(action))
        if done:
            break
    return trace


def make_active_threshold_frames():
    main_df = pd.DataFrame(
        [
            {"policy": "Random", "OverCapMinutes": 3.0, "CVaR95_L": 6.0},
            {"policy": "Myopic", "OverCapMinutes": 2.0, "CVaR95_L": 5.0},
            {"policy": "PPO", "OverCapMinutes": 1.5, "CVaR95_L": 4.5},
            {"policy": "Lagrangian PPO", "OverCapMinutes": 0.9, "CVaR95_L": 4.0},
        ]
    )
    cap_df = pd.DataFrame(
        [
            {"T_cap": 5.0, "CumWatch": 8.0, "CVaR95_L": 3.5, "OverCapMinutes": 0.8, "SessionCapTriggerRate": 0.30},
            {"T_cap": 10.0, "CumWatch": 9.0, "CVaR95_L": 4.0, "OverCapMinutes": 1.1, "SessionCapTriggerRate": 0.20},
            {"T_cap": 15.0, "CumWatch": 9.5, "CVaR95_L": 4.5, "OverCapMinutes": 1.4, "SessionCapTriggerRate": 0.10},
        ]
    )
    break_prompt_df = pd.DataFrame(
        [
            {"condition": "break_prompt", "policy": "PPO", "BreakAdherence": 0.5, "BreakAdherence_den": 12.0},
        ]
    )
    return main_df, cap_df, break_prompt_df


def test_low_threshold_overcap_becomes_nonzero():
    cfg = make_cfg()
    catalog = cb.build_catalog(cfg)
    cards = cb.build_default_cards(cfg, metadata_csv=None, random_seed=0, paper_mode=False)
    policy = cb.RandomPolicy(cfg.num_actions(), seed=0)
    metrics = cb.evaluate_policy(
        policy,
        cfg,
        catalog,
        cards,
        episode_seeds=list(range(24)),
        num_episodes=24,
        deterministic=False,
        collect_policy_diagnostics=False,
        label="test_low_threshold_overcap",
    )
    assert float(metrics["OverCapMinutes"]) > 0.0


def test_session_cap_reduces_tail_when_thresholds_bind():
    cfg = make_cfg()
    catalog = cb.build_catalog(cfg)
    cards = cb.build_default_cards(cfg, metadata_csv=None, random_seed=0, paper_mode=False)
    policy = cb.RandomPolicy(cfg.num_actions(), seed=1)
    seeds = list(range(24))
    base = cb.evaluate_policy(
        policy,
        cfg,
        catalog,
        cards,
        episode_seeds=seeds,
        num_episodes=len(seeds),
        deterministic=False,
        collect_policy_diagnostics=False,
        label="test_session_cap_base",
    )
    capped = cb.evaluate_policy(
        policy,
        cfg,
        catalog,
        cards,
        wrappers={"session_cap": True, "T_cap": float(cfg.T_cap)},
        episode_seeds=seeds,
        num_episodes=len(seeds),
        deterministic=False,
        collect_policy_diagnostics=False,
        label="test_session_cap_capped",
    )
    assert float(capped["SessionCapTriggerRate"]) > 0.0
    assert float(capped["CVaR95_L"]) <= float(base["CVaR95_L"]) + 1e-8


def test_evaluate_policy_exports_episode_fragmentation_rows():
    cfg = make_cfg()
    catalog = cb.build_catalog(cfg)
    cards = cb.build_default_cards(cfg, metadata_csv=None, random_seed=0, paper_mode=False)
    policy = cb.RandomPolicy(cfg.num_actions(), seed=7)
    episode_seeds = list(range(8))

    metrics = cb.evaluate_policy(
        policy,
        cfg,
        catalog,
        cards,
        episode_seeds=episode_seeds,
        num_episodes=len(episode_seeds),
        deterministic=False,
        collect_policy_diagnostics=False,
        label="fragmentation_export_test",
        train_seed=11,
    )

    rows = metrics["EpisodeFragmentationRows"]
    assert len(rows) == len(episode_seeds)
    assert math.isfinite(float(metrics["SessionsPerEpisode"]))
    assert math.isclose(
        float(metrics["FractionReturnsWithin5Min"]),
        float(metrics["ReturnRate5"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    sample_row = rows[0]
    assert sample_row["policy"] == "fragmentation_export_test"
    assert sample_row["train_seed"] == 11
    assert sample_row["episode_seed"] == episode_seeds[0]
    assert "fraction_returns_within_1_min" in sample_row
    assert "late_night_session_start_count" in sample_row


def test_official_diversity_baselines_have_distinct_deterministic_traces():
    cfg = cb.BenchConfig(T_ref=1.0, T_cap=1.0, Z=8, P=2, device="cpu")
    catalog = cb.build_catalog(cfg)
    cards = cb.build_default_cards(cfg, metadata_csv=None, random_seed=0, paper_mode=False)
    trace_len = 12

    traces = {
        "RoundRobinPolicy": deterministic_action_trace(cb.RoundRobinPolicy(cfg), cfg, catalog, cards, seed=5, steps=trace_len),
        "LeastRecentPolicy": deterministic_action_trace(cb.LeastRecentPolicy(cfg, seed=5), cfg, catalog, cards, seed=5, steps=trace_len),
        "NoveltyGreedyPolicy": deterministic_action_trace(cb.NoveltyGreedyPolicy(cfg, seed=5), cfg, catalog, cards, seed=5, steps=trace_len),
    }

    assert len(traces["RoundRobinPolicy"]) == trace_len
    assert len(traces["LeastRecentPolicy"]) == trace_len
    assert len(traces["NoveltyGreedyPolicy"]) == trace_len
    assert traces["RoundRobinPolicy"] != traces["LeastRecentPolicy"]
    assert traces["RoundRobinPolicy"] != traces["NoveltyGreedyPolicy"]
    assert traces["LeastRecentPolicy"] != traces["NoveltyGreedyPolicy"]


def test_official_scorecard_episode_bootstrap_ci_stays_nonzero_with_single_train_seed():
    cfg = make_cfg()
    metric_template = {
        "CVaR95_L": 0.0,
        "ReturnRate5": 0.0,
        "ReturnRate15": 0.0,
        "ReturnRate30": 0.0,
        "ReturnRate60": 0.0,
        "NightMinutes": 0.0,
        "NightFraction": 0.0,
        "LateNightSessionStartRate": 0.0,
        "OverCapMinutes": 0.0,
    }
    scorecard_rows = [
        cb.build_official_scorecard_metric_row("PPO", "Param", "deterministic", 0, {"CumWatch": 12.0, **metric_template}),
        cb.build_official_scorecard_metric_row("Myopic", "Param", "deterministic", 0, {"CumWatch": 15.0, **metric_template}),
    ]
    scorecard_df = cb.aggregate_across_train_seeds(scorecard_rows, ["policy", "backend", "eval_mode"])

    def episode_row(policy: str, episode_index: int, cumwatch: float, gaps: list[float], session_lengths: list[float]) -> dict[str, object]:
        return {
            "policy": policy,
            "backend": "Param",
            "split": "test",
            "eval_mode": "deterministic",
            "train_seed": 0,
            "episode_index": episode_index,
            "episode_seed": 30_000 + episode_index,
            "NumEpisodes": 4,
            "CumWatch": cumwatch,
            "NightMinutes": 0.0,
            "NightFraction": 0.0,
            "LateNightSessionStartRate": 0.0,
            "OverCapMinutes": 0.0,
            "SessionLengthsJson": cb.json_compact(session_lengths),
            "GapValuesJson": cb.json_compact(gaps),
            "NumSessionLengths": len(session_lengths),
            "NumGaps": len(gaps),
        }

    official_episode_df = pd.DataFrame(
        [
            episode_row("PPO", 1, 10.0, [2.0, 4.0], [4.0, 6.0]),
            episode_row("PPO", 2, 11.0, [3.0, 5.0], [5.0, 7.0]),
            episode_row("PPO", 3, 12.0, [4.0, 6.0], [6.0, 8.0]),
            episode_row("PPO", 4, 13.0, [5.0, 7.0], [7.0, 9.0]),
            episode_row("Myopic", 1, 14.0, [10.0, 12.0], [8.0, 10.0]),
            episode_row("Myopic", 2, 15.0, [11.0, 13.0], [9.0, 11.0]),
            episode_row("Myopic", 3, 18.0, [12.0, 14.0], [10.0, 12.0]),
            episode_row("Myopic", 4, 21.0, [13.0, 15.0], [11.0, 13.0]),
        ]
    )

    augmented = cb.augment_official_scorecard_with_episode_uncertainty(
        scorecard_df,
        official_episode_df,
        cfg,
        num_resamples=200,
    )
    ppo_row = augmented[augmented["policy"] == "PPO"].iloc[0]
    myopic_row = augmented[augmented["policy"] == "Myopic"].iloc[0]

    assert float(myopic_row["CumWatch_train_seed_ci95"]) == 0.0
    assert float(myopic_row["CumWatch_ci95"]) > 0.0
    assert math.isclose(float(ppo_row["CumWatch_delta_vs_PPO"]), 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(ppo_row["CumWatch_delta_vs_PPO_ci95"]), 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(myopic_row["CumWatch_delta_vs_PPO"]), 5.5, rel_tol=0.0, abs_tol=1e-12)
    assert float(myopic_row["CumWatch_delta_vs_PPO_ci95"]) > 0.0


def test_projected_action_space_policy_drops_personalization_coordinate():
    class DummyPolicy:
        def reset(self):
            return None

        def act_with_info(self, obs, deterministic=True, env=None, need_info=True):
            _ = obs, deterministic, env
            action = cb.tuple_to_action_id(3, 4, 1, 5)
            return action, ({"policy_entropy": 0.0} if need_info else {})

    wrapped = cb.ProjectedActionSpacePolicy(DummyPolicy(), source_P=5, target_P=0)
    action, info = wrapped.act_with_info(obs=None, deterministic=True, need_info=True)
    z, lam_idx, nu = cb.action_id_to_tuple(int(action), 0)

    assert (z, lam_idx, nu) == (3, 0, 1)
    assert math.isclose(float(info["policy_entropy"]), 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_lambda_values_handles_nopers_degenerate_grid():
    cfg = cb.BenchConfig(P=0, device="cpu")
    assert cfg.lambda_values() == [0.0]


def test_build_mechanism_ablation_table_computes_relative_change_vs_default():
    records = [
        {"policy": "PPO", "ablation": "Default", "train_seed": 0, "CumWatch": 100.0, "CVaR95_L": 50.0, "OverCapMinutes": 20.0, "LateNightSessionStartRate": 0.10},
        {"policy": "PPO", "ablation": "NoHabit", "train_seed": 0, "CumWatch": 90.0, "CVaR95_L": 40.0, "OverCapMinutes": 10.0, "LateNightSessionStartRate": 0.08},
        {"policy": "PPO", "ablation": "Default", "train_seed": 1, "CumWatch": 120.0, "CVaR95_L": 60.0, "OverCapMinutes": 25.0, "LateNightSessionStartRate": 0.12},
        {"policy": "PPO", "ablation": "NoHabit", "train_seed": 1, "CumWatch": 108.0, "CVaR95_L": 48.0, "OverCapMinutes": 15.0, "LateNightSessionStartRate": 0.09},
    ]

    table = cb.build_mechanism_ablation_table(records, "LateNightSessionStartRate")
    nohabit_row = table[(table["policy"] == "PPO") & (table["ablation"] == "NoHabit")].iloc[0]

    assert math.isclose(float(nohabit_row["CumWatch"]), 99.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(nohabit_row["CumWatch_pct_change_vs_default"]), -10.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(nohabit_row["CVaR95_L_pct_change_vs_default"]), -20.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(nohabit_row["OverCapMinutes_pct_change_vs_default"]), -45.0, rel_tol=0.0, abs_tol=1e-12)
    assert float(nohabit_row["CumWatch_pct_change_vs_default_ci95"]) == 0.0


def test_nohabit_enforces_zero_habit_state_exactly():
    cfg = make_cfg()
    catalog = cb.build_catalog(cfg)
    cards = cb.build_default_cards(cfg, metadata_csv=None, random_seed=0, paper_mode=False)
    env = cb.make_env(cfg, "param", catalog, cards, scorer=None, wrappers=None, seed=0, ablation="NoHabit")
    env.reset(seed=0)
    assert env.state is not None
    assert abs(float(env.state.h)) <= 1e-12
    for _ in range(8):
        action = int(env.rng.integers(0, cfg.num_actions()))
        _, _, done, _ = env.step(action)
        if env.state is not None:
            assert abs(float(env.state.h)) <= 1e-12
        if done:
            break


def test_break_prompt_wrapper_produces_finite_adherence():
    cfg = cb.BenchConfig(T_ref=1.0, T_cap=1.0, break_T=1.0, break_J=1, device="cpu")
    catalog = cb.build_catalog(cfg)
    cards = cb.build_default_cards(cfg, metadata_csv=None, random_seed=0, paper_mode=False)
    policy = cb.RandomPolicy(cfg.num_actions(), seed=2)
    metrics = cb.evaluate_policy(
        policy,
        cfg,
        catalog,
        cards,
        wrappers={"break_prompt": True},
        episode_seeds=list(range(24)),
        num_episodes=24,
        deterministic=False,
        collect_policy_diagnostics=False,
        label="test_break_prompt_active",
    )
    assert int(metrics["BreakAdherence_den"]) > 0
    assert math.isfinite(float(metrics["BreakAdherence"]))


def test_derive_paper_thresholds_rounds_reference_quantiles():
    thresholds = cb.derive_paper_thresholds_from_reference_distribution(
        {
            "session_lengths": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "session_item_counts": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        },
        threshold_source="empirical_target",
        source_label="unit_test",
        selected_delta_sess=30.0,
    )
    assert thresholds["threshold_source"] == "empirical_target"
    assert thresholds["T_ref"] == 95.0
    assert thresholds["cap_grid"] == [90.0, 95.0, 100.0]
    assert thresholds["break_T"] == 80.0
    assert thresholds["break_J"] == 10


def test_fixed_paper_threshold_source_preserves_prespecified_defaults():
    thresholds = cb.resolve_thresholds_for_source("fixed_paper", selected_delta_sess=30.0)

    assert thresholds["threshold_source"] == "fixed_paper"
    assert thresholds["source"] == "fixed_paper"
    assert thresholds["T_ref"] == 120.0
    assert thresholds["T_cap"] == 120.0
    assert thresholds["cap_grid"] == [90.0, 120.0, 150.0]
    assert thresholds["break_T"] == 30.0
    assert thresholds["break_J"] == 20


def test_empirical_target_thresholds_come_from_public_log_targets():
    thresholds = cb.resolve_thresholds_for_source(
        "empirical_target",
        selected_delta_sess=30.0,
        reference_targets={
            "session_lengths": [100.0, 120.0, 140.0, 160.0, 180.0],
            "session_item_counts": [10.0, 15.0, 20.0, 25.0, 30.0],
        },
    )

    assert thresholds["threshold_source"] == "empirical_target"
    assert thresholds["source"] == "public_log_target_statistics"
    assert thresholds["T_ref"] == 175.0
    assert thresholds["T_cap"] == 175.0
    assert thresholds["cap_grid"] == [165.0, 170.0, 180.0]
    assert thresholds["break_T"] == 165.0
    assert thresholds["break_J"] == 25


def test_simulator_relative_thresholds_require_passed_calibration():
    with pytest.raises(ValueError, match="requires calibration status 'passed'"):
        cb.resolve_thresholds_for_source(
            "simulator_relative",
            selected_delta_sess=30.0,
            sim_targets={
                "session_lengths": [100.0, 120.0, 140.0, 160.0, 180.0],
                "session_item_counts": [10.0, 15.0, 20.0, 25.0, 30.0],
            },
            calibration_status="failed",
        )


def test_threshold_table_includes_threshold_source():
    df = cb.threshold_table_dataframe(cb.fixed_paper_thresholds(selected_delta_sess=30.0))

    assert "threshold_source" in df.columns
    assert "threshold_source" in df["parameter"].tolist()


def test_paper_threshold_invariants_accept_active_frames():
    main_df, cap_df, break_prompt_df = make_active_threshold_frames()
    cb.assert_paper_threshold_channels_active(main_df, cap_df, break_prompt_df)


def test_paper_threshold_invariants_fail_when_main_overcap_is_dead():
    main_df, cap_df, break_prompt_df = make_active_threshold_frames()
    main_df.loc[:, "OverCapMinutes"] = 0.0
    with pytest.raises(AssertionError, match="OverCapMinutes == 0"):
        cb.assert_paper_threshold_channels_active(main_df, cap_df, break_prompt_df)


def test_paper_threshold_invariants_fail_when_cap_trigger_is_dead():
    main_df, cap_df, break_prompt_df = make_active_threshold_frames()
    cap_df.loc[:, "SessionCapTriggerRate"] = 0.0
    with pytest.raises(AssertionError, match="SessionCapTriggerRate == 0"):
        cb.assert_paper_threshold_channels_active(main_df, cap_df, break_prompt_df)


def test_paper_threshold_invariants_fail_when_break_prompt_denominator_is_zero():
    main_df, cap_df, break_prompt_df = make_active_threshold_frames()
    break_prompt_df.loc[:, "BreakAdherence_den"] = 0.0
    with pytest.raises(AssertionError, match="BreakAdherence_den == 0"):
        cb.assert_paper_threshold_channels_active(main_df, cap_df, break_prompt_df)


def test_paper_threshold_invariants_fail_when_break_prompt_is_nan():
    main_df, cap_df, break_prompt_df = make_active_threshold_frames()
    break_prompt_df.loc[:, "BreakAdherence"] = float("nan")
    with pytest.raises(AssertionError, match="finite BreakAdherence"):
        cb.assert_paper_threshold_channels_active(main_df, cap_df, break_prompt_df)


def test_constraint_track_uses_overcap_when_night_is_scalar_proxy():
    spec = cb.build_constraint_track_spec(
        {"NightMinutes": 5.0, "OverCapMinutes": 2.0},
        {"policy_mean_corr": 0.99, "episode_corr": 0.98, "night_minutes_is_scalar_proxy": True},
    )
    assert spec["use_over_budget"] is True
    assert spec["use_night_budget"] is False
    assert spec["active_channels"] == ["OverCapMinutes"]


def test_constraint_track_allows_night_when_not_proxy_and_overcap_inactive():
    spec = cb.build_constraint_track_spec(
        {"NightMinutes": 5.0, "OverCapMinutes": 0.0},
        {"policy_mean_corr": 0.40, "episode_corr": 0.35, "night_minutes_is_scalar_proxy": False},
    )
    assert spec["use_night_budget"] is True
    assert spec["use_over_budget"] is False
    assert spec["active_channels"] == ["NightMinutes"]


def test_scaled_constraint_budgets_skip_inactive_channels():
    budgets = cb.scaled_constraint_budgets(
        0.9,
        {"NightMinutes": 6.0, "OverCapMinutes": 3.0},
        {"use_night_budget": False, "use_over_budget": True},
    )
    assert budgets == (None, 2.7)


def test_negative_night_proxy_correlation_counts_as_proxy_like_for_promotion():
    policy_results = {
        "PolicyA": [
            {
                "CumWatch": 1.0,
                "NightMinutes": 1.0,
                "NightFraction": 3.0,
                "LateNightSessionStartRate": 3.0,
                "EpisodeCumWatchValues": [1.0, 1.1],
                "EpisodeNightValues": [1.0, 1.1],
                "EpisodeNightFractionValues": [3.0, 2.9],
                "EpisodeLateNightSessionStartRateValues": [3.0, 2.9],
            }
        ],
        "PolicyB": [
            {
                "CumWatch": 2.0,
                "NightMinutes": 2.0,
                "NightFraction": 2.0,
                "LateNightSessionStartRate": 2.0,
                "EpisodeCumWatchValues": [2.0, 2.1],
                "EpisodeNightValues": [2.0, 2.1],
                "EpisodeNightFractionValues": [2.0, 1.9],
                "EpisodeLateNightSessionStartRateValues": [2.0, 1.9],
            }
        ],
        "PolicyC": [
            {
                "CumWatch": 3.0,
                "NightMinutes": 3.0,
                "NightFraction": 1.0,
                "LateNightSessionStartRate": 1.0,
                "EpisodeCumWatchValues": [3.0, 3.1],
                "EpisodeNightValues": [3.0, 3.1],
                "EpisodeNightFractionValues": [1.0, 0.9],
                "EpisodeLateNightSessionStartRateValues": [1.0, 0.9],
            }
        ],
    }

    night_fraction_stats = cb.summarize_night_proxy_candidate(policy_results, "NightFraction")
    audit = cb.build_night_proxy_orthogonality_audit(policy_results)

    assert math.isclose(float(night_fraction_stats["policy_mean_corr"]), -1.0, rel_tol=0.0, abs_tol=1e-12)
    assert night_fraction_stats["changed_policy_comparison"] is True
    assert night_fraction_stats["eligible_main_text"] is False
    assert night_fraction_stats["night_minutes_is_scalar_proxy"] is True
    assert audit["promoted_main_text_proxy"] is None
    assert audit["night_proxy_appendix_only"] is True
    assert audit["main_text_scorecard_risk_metrics"] == ["OverCapMinutes"]


def test_calibration_loss_components_include_session_item_counts():
    targets = {
        "session_lengths": [40.0, 45.0, 50.0, 55.0, 60.0],
        "session_item_counts": [8.0, 9.0, 10.0, 11.0, 12.0],
        "gaps": [5.0, 10.0, 15.0, 20.0],
        "stop_hazard": [0.10, 0.12, 0.14, 0.18, 0.22],
    }
    sim = {
        "session_lengths": [40.0, 45.0, 50.0, 55.0, 60.0],
        "session_item_counts": [2.0, 2.0, 3.0, 3.0, 4.0],
        "gaps": [5.0, 10.0, 15.0, 20.0],
        "stop_hazard": [0.25, 0.35],
    }

    components = cb.calibration_loss_components(targets, sim)
    breakdown = cb.calibration_loss_breakdown(targets, sim, components=components)

    assert "session_item_counts" in components
    assert float(components["session_item_counts"]) > 0.50
    assert float(breakdown["primary"]) > 0.0
    assert float(breakdown["lexicographic_scalar"]) >= float(breakdown["primary"]) * cb.CALIBRATION_LOSS_PRIMARY_MULTIPLIER


def test_calibration_hazard_uses_elapsed_session_minutes_when_session_lengths_exist():
    targets = {
        "session_lengths": [10.0, 20.0, 30.0, 40.0, 50.0],
        "session_item_counts": [2.0, 4.0, 6.0, 8.0, 10.0],
        "stop_hazard": [0.05] * 10,
    }
    sim = {
        "session_lengths": [10.0, 20.0, 30.0, 40.0, 50.0],
        "session_item_counts": [2.0, 4.0, 6.0, 8.0, 10.0],
        "stop_hazard": [0.40, 0.60],
    }

    details = cb.calibration_loss_component_details(targets, sim)

    assert details["hazard"]["basis"] == "elapsed_session_minutes"
    assert details["hazard"]["used_raw_item_position_fallback"] is False
    assert details["hazard"]["support_mismatch_addressed"] is True
    assert details["hazard"]["raw_item_position_severe_support_mismatch"] == 1.0
    assert float(details["hazard"]["l2"]) <= 1e-12


def test_calibration_hazard_flags_severe_unaddressed_raw_support_mismatch_on_fallback():
    details = cb.calibration_loss_component_details(
        {"stop_hazard": [0.05] * 10},
        {"stop_hazard": [0.40, 0.60]},
    )

    assert details["hazard"]["basis"] == "raw_item_position_overlap_only"
    assert details["hazard"]["used_raw_item_position_fallback"] is True
    assert details["hazard"]["support_mismatch_addressed"] is False
    assert details["hazard"]["severe_support_mismatch_unaddressed"] == 1.0


def test_calibration_audit_fails_when_session_item_count_p95_is_badly_missed():
    targets_by_delta = {
        30.0: {
            "session_lengths": [50.0, 55.0, 60.0, 65.0, 70.0],
            "session_item_counts": [10.0, 11.0, 12.0, 13.0, 14.0],
            "gaps": [5.0, 10.0, 15.0, 20.0],
            "stop_hazard": [0.10] * 8,
        }
    }
    sim = {
        "session_lengths": [50.0, 55.0, 60.0, 65.0, 70.0],
        "session_item_counts": [2.0, 2.0, 3.0, 3.0, 4.0],
        "gaps": [5.0, 10.0, 15.0, 20.0],
        "stop_hazard": [0.20, 0.30],
    }

    audit = cb.build_calibration_audit(targets_by_delta, sim, 30.0)
    markdown = cb.render_calibration_audit_markdown(audit)

    assert audit["status"] == "failed"
    assert audit["acceptance"]["session_item_count_p95_within_50pct"] is False
    assert "session_item_counts" in audit["calibration_loss_components"]
    assert "## Session item counts" in markdown
    assert "Simulated p95 session item count within 50%" in markdown


def test_calibration_audit_fails_when_gap_distribution_is_badly_missed():
    targets_by_delta = {
        30.0: {
            "session_lengths": [50.0, 55.0, 60.0, 65.0, 70.0],
            "session_item_counts": [10.0, 11.0, 12.0, 13.0, 14.0],
            "gaps": [120.0, 180.0, 240.0, 360.0, 480.0],
            "stop_hazard": [0.10] * 8,
        }
    }
    sim = {
        "session_lengths": [50.0, 55.0, 60.0, 65.0, 70.0],
        "session_item_counts": [10.0, 11.0, 12.0, 13.0, 14.0],
        "gaps": [0.5, 0.75, 1.0, 1.25, 1.5],
        "stop_hazard": [0.10] * 8,
    }

    audit = cb.build_calibration_audit(targets_by_delta, sim, 30.0)

    assert audit["status"] == "failed"
    assert audit["acceptance"]["gap_mean_within_50pct"] is False
    assert audit["acceptance"]["gap_p95_within_50pct"] is False
    assert float(audit["gap_mean_relative_error"]) > 0.9
    assert float(audit["gap_p95_relative_error"]) > 0.9


def test_calibration_audit_fails_when_conditional_return_realism_is_badly_missed():
    targets_by_delta = {
        30.0: {
            "session_lengths": [50.0, 55.0, 60.0, 65.0, 70.0],
            "session_item_counts": [10.0, 11.0, 12.0, 13.0, 14.0],
            "gaps": [5.0, 10.0, 15.0, 20.0],
            "stop_hazard": [0.10] * 8,
            "conditional_return_rates": {
                "short": {"count": 10.0, "ReturnRate5": 0.95, "ReturnRate15": 0.95, "ReturnRate30": 0.95, "ReturnRate60": 0.95},
                "medium": {"count": 10.0, "ReturnRate5": 0.85, "ReturnRate15": 0.85, "ReturnRate30": 0.85, "ReturnRate60": 0.85},
                "long": {"count": 10.0, "ReturnRate5": 0.75, "ReturnRate15": 0.75, "ReturnRate30": 0.75, "ReturnRate60": 0.75},
            },
        }
    }
    sim = {
        "session_lengths": [50.0, 55.0, 60.0, 65.0, 70.0],
        "session_item_counts": [10.0, 11.0, 12.0, 13.0, 14.0],
        "gaps": [5.0, 10.0, 15.0, 20.0],
        "stop_hazard": [0.10] * 8,
        "conditional_return_rates": {
            "short": {"count": 10.0, "ReturnRate5": 0.00, "ReturnRate15": 0.00, "ReturnRate30": 0.00, "ReturnRate60": 0.00},
            "medium": {"count": 10.0, "ReturnRate5": 0.00, "ReturnRate15": 0.00, "ReturnRate30": 0.00, "ReturnRate60": 0.00},
            "long": {"count": 10.0, "ReturnRate5": 0.00, "ReturnRate15": 0.00, "ReturnRate30": 0.00, "ReturnRate60": 0.00},
        },
    }

    audit = cb.build_calibration_audit(targets_by_delta, sim, 30.0)

    assert audit["status"] == "failed"
    assert audit["return_conditional_realism_available"] is True
    assert audit["acceptance"]["return_conditional_realistic_if_available"] is False
    assert float(audit["return_conditional_mse"]) > cb.CALIBRATION_ACCEPTANCE_LIMITS["return_conditional_mse_max"]
