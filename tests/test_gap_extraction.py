import importlib.util
import sys
from pathlib import Path

import pytest


pytest.importorskip("matplotlib")
pd = pytest.importorskip("pandas")


def load_cb():
    path = Path(__file__).resolve().parents[1] / "compulsionbench" / "compulsionbench.py"
    spec = importlib.util.spec_from_file_location("compulsionbench_test_gap_extraction", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cb = load_cb()


def test_sessionized_gap_uses_true_last_event_end_timestamp():
    df = pd.DataFrame(
        [
            {"user_id": "u1", "timestamp_min": 0.0, "watch_time_min": 2.0},
            {"user_id": "u1", "timestamp_min": 5.0, "watch_time_min": 1.0},
            {"user_id": "u1", "timestamp_min": 20.0, "watch_time_min": 1.0},
        ]
    )

    sessionized = cb.sessionize_logs(df, delta_sess_minutes=10.0)

    assert sessionized["session_id"].tolist() == [1, 1, 2]
    assert sessionized["event_end_ts"].tolist() == pytest.approx([2.0, 6.0, 21.0])

    targets = cb.summarize_targets_from_sessionized_df(
        sessionized,
        include_gap_extraction_diagnostic=True,
    )

    assert targets["session_lengths"] == pytest.approx([3.0, 1.0])
    assert targets["gaps"] == pytest.approx([14.0])

    diagnostic = targets["gap_extraction_diagnostic"]
    assert diagnostic["legacy"]["gap_summary"]["mean"] == pytest.approx(17.0)
    assert diagnostic["legacy"]["gap_summary"]["p95"] == pytest.approx(17.0)
    assert diagnostic["corrected"]["gap_summary"]["mean"] == pytest.approx(14.0)
    assert diagnostic["corrected"]["gap_summary"]["p95"] == pytest.approx(14.0)
    assert diagnostic["legacy"]["conditional_return_rates"]["short"]["ReturnRate15"] == pytest.approx(0.0)
    assert diagnostic["corrected"]["conditional_return_rates"]["short"]["ReturnRate15"] == pytest.approx(1.0)
