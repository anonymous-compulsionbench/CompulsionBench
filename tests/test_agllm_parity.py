import importlib.util
import sys
import tempfile
from dataclasses import replace
from pathlib import Path


def load_cb():
    path = Path(__file__).resolve().parents[1] / "compulsionbench" / "compulsionbench.py"
    spec = importlib.util.spec_from_file_location("compulsionbench_test_agllm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cb = load_cb()


def test_zero_fusion_agllm_matches_agparam():
    cfg = cb.BenchConfig(T_ref=1.0, T_cap=1.0, device="cpu")
    catalog = cb.build_catalog(cfg)
    cards = cb.build_default_cards(cfg, metadata_csv=None, random_seed=0, paper_mode=False)
    seeds = list(range(20))
    policy = cb.RoundRobinPolicy(cfg)

    param_metrics = cb.evaluate_policy(
        policy,
        cfg,
        catalog,
        cards,
        episode_seeds=seeds,
        num_episodes=len(seeds),
        deterministic=True,
        collect_policy_diagnostics=False,
        label="test_agparam",
    )

    llm_zero = cb.copy.deepcopy(cfg)
    llm_zero.omega_r_llm = 0.0
    llm_zero.omega_c_llm = 0.0
    scorer = cb.AGLLMScorer(llm_zero, cards, mode="surrogate", cache_path=None, device="cpu")
    llm_metrics = cb.evaluate_policy(
        policy,
        llm_zero,
        catalog,
        cards,
        backend="llm",
        scorer=scorer,
        episode_seeds=seeds,
        num_episodes=len(seeds),
        deterministic=True,
        collect_policy_diagnostics=False,
        label="test_agllm_zero_fusion",
    )

    for metric_key in ["CumWatch", "CVaR95_L", "ReturnRate60", "NightMinutes", "OverCapMinutes"]:
        assert abs(float(param_metrics[metric_key]) - float(llm_metrics[metric_key])) <= 1e-8


def test_runtime_prompt_hashes_match_manifest():
    manifest = cb.load_agllm_release_manifest()
    _, watch_hash, _ = cb.load_agllm_prompt_template("watch")
    _, continue_hash, _ = cb.load_agllm_prompt_template("continue")
    assert watch_hash == str(manifest["prompt_templates"]["watch"]["sha256"])
    assert continue_hash == str(manifest["prompt_templates"]["continue"]["sha256"])


def test_card_builder_is_deterministic():
    cfg = cb.BenchConfig(device="cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "items.csv"
        metadata_path.write_text(
            "\n".join(
                [
                    "item_id,caption,category_level1,category_level2,duration_seconds,interaction_count",
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
        cards_a = cb.build_default_cards(small_cfg, str(metadata_path), random_seed=0, paper_mode=True)
        cards_b = cb.build_default_cards(small_cfg, str(metadata_path), random_seed=0, paper_mode=True)
        assert cards_a == cards_b
