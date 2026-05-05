#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a calibration audit from a calibration_payload.json artifact.")
    parser.add_argument("payload_json", type=Path, help="Path to calibration_payload.json")
    parser.add_argument("--delta_sess", type=float, default=None, help="Override the selected delta_sess if the payload is ambiguous.")
    parser.add_argument("--out", type=Path, default=None, help="Optional markdown output path.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import compulsionbench as cb  # pylint: disable=import-error

    payload = json.loads(args.payload_json.read_text(encoding="utf-8"))
    audit = cb.build_calibration_audit_from_payload(payload, selected_delta_sess=args.delta_sess)
    markdown = cb.render_calibration_audit_markdown(audit)
    if args.out is not None:
        args.out.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
