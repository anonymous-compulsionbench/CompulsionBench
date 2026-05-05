#!/usr/bin/env python3
"""Strict parse/repair path for AGLLM watch/continue JSON outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


_FLOAT_PATTERN = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?")


@dataclass(frozen=True)
class ParseResult:
    field: str
    score: float
    repaired: bool
    fallback: bool


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def parse_response(raw_text: str, field: str) -> ParseResult:
    payload = raw_text.strip()
    try:
        parsed = json.loads(payload)
        return ParseResult(field=field, score=_clamp(float(parsed[field])), repaired=False, fallback=False)
    except Exception:
        match = _FLOAT_PATTERN.search(payload)
        if match is not None:
            repaired_payload = {field: float(match.group(0))}
            return ParseResult(
                field=field,
                score=_clamp(float(repaired_payload[field])),
                repaired=True,
                fallback=False,
            )
        return ParseResult(field=field, score=0.5, repaired=False, fallback=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=("watch_score", "continue_score"), required=True)
    parser.add_argument("raw_text")
    args = parser.parse_args()
    result = parse_response(args.raw_text, args.field)
    print(json.dumps(result.__dict__, ensure_ascii=False))
