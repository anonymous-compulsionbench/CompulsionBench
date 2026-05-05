import importlib.util
import json
import sys
from pathlib import Path


def load_cb():
    path = Path(__file__).resolve().parents[1] / "compulsionbench" / "compulsionbench.py"
    spec = importlib.util.spec_from_file_location("compulsionbench_test_calibration_feasibility", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cb = load_cb()


def make_payload():
    return {
        "selected_delta_sess": 30.0,
        "targets": {
            30.0: {
                "session_lengths": [55.0, 58.0, 60.0, 62.0, 65.0, 68.0, 70.0, 72.0, 75.0, 80.0],
                "session_item_counts": [11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0, 14.0, 15.0, 16.0],
                "stop_hazard": [0.08, 0.07, 0.06, 0.07, 0.08, 0.10],
                "item_watch_times": [4.6, 4.8, 4.9, 5.0, 4.7, 4.9, 5.0, 4.8, 4.9, 5.0],
            }
        },
        "sim": {
            "session_lengths": [18.0, 20.0, 22.0, 24.0, 27.0, 30.0, 32.0, 35.0, 37.0, 40.0],
            "session_item_counts": [4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0],
            "stop_hazard": [0.18, 0.22, 0.30],
            "item_watch_times": [2.0, 2.1, 2.3, 2.4, 2.5, 2.2, 2.1, 2.4, 2.5, 2.6],
        },
    }


def test_feasibility_report_explicitly_reports_stop_hazard_truncation():
    report = cb.build_calibration_feasibility_report_from_payload(make_payload())
    markdown = cb.render_calibration_feasibility_markdown(report)

    assert report["stop_hazard_support"]["target_len"] == 6
    assert report["stop_hazard_support"]["sim_len"] == 3
    assert report["stop_hazard_support"]["overlap_len"] == 3
    assert report["stop_hazard_support"]["truncated"] is True
    assert report["stop_hazard_support"]["simulator_support_shorter_than_target"] is True
    assert report["structural_feasibility"]["max_p95_session_item_count_under_bounded_continue"] == 10
    assert report["structural_feasibility"]["public_log_p95_session_tail_plausible"] is False
    assert "min(len(target_stop_hazard), len(sim_stop_hazard))" in markdown
    assert "`len(target_stop_hazard) = 6`" in markdown
    assert "`len(sim_stop_hazard) = 3`" in markdown
    assert "truncated" in markdown.lower()
    assert "structurally incapable" in markdown.lower()


def test_write_calibration_feasibility_artifacts_creates_expected_outputs(tmp_path):
    payload_path = tmp_path / "calibration_payload.json"
    payload_path.write_text(json.dumps(make_payload()), encoding="utf-8")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    outdir = tmp_path / "calibration_feasibility"
    report = cb.write_calibration_feasibility_artifacts(payload, outdir)

    assert report["selected_delta_sess"] == 30.0
    assert (outdir / "feasibility_report.md").exists()
    assert (outdir / "feasibility_report.json").exists()
    assert (outdir / "fig_session_item_count_cdf.png").exists()
    assert (outdir / "fig_stop_hazard_support.png").exists()
    assert (outdir / "fig_per_item_watch_cdf.png").exists()

    markdown = (outdir / "feasibility_report.md").read_text(encoding="utf-8")
    json_report = json.loads((outdir / "feasibility_report.json").read_text(encoding="utf-8"))
    assert "Calibration feasibility audit" in markdown
    assert json_report["stop_hazard_support"]["truncated"] is True


def test_feasibility_report_uses_fitted_continue_parameters_when_available():
    payload = make_payload()
    payload["fitted_config"] = cb.replace(
        cb.BenchConfig(),
        continue_logit_bias=4.0,
        continue_logit_temp=10.0,
    ).to_dict()

    report = cb.build_calibration_feasibility_report_from_payload(payload)

    assert report["structural_feasibility"]["continue_parameter_source"] == "payload_fitted_config"
    assert report["structural_feasibility"]["max_parametric_p_continue"] > 0.99
    assert report["structural_feasibility"]["max_parametric_p_continue"] > 0.7311
