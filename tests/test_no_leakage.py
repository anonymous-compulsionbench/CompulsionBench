import importlib.util
import inspect
import sys
from pathlib import Path


def load_cb():
    path = Path(__file__).resolve().parents[1] / "compulsionbench" / "compulsionbench.py"
    spec = importlib.util.spec_from_file_location("compulsionbench_test_no_leakage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cb = load_cb()


def test_no_test_set_metrics_are_used_for_budget_derivation():
    cb.no_test_leakage_invariant()


def test_smoke_target_runs_invariant_suite():
    source = inspect.getsource(cb.main)
    assert "run_invariant_smoke_tests()" in source
