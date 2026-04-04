"""Check the Python encoder path against the shared encoder golden vectors."""

from __future__ import annotations

import json
from pathlib import Path

from train.datasets.artifacts import pack_position_features
from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import PositionEncoding, RawPositionRecord


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    golden_path = repo_root / "artifacts" / "golden" / "encoder_golden_v1.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    examples = list(golden["examples"])

    records = [
        RawPositionRecord(
            sample_id=f"golden_{index}",
            fen=str(example["fen"]),
            source="golden",
        )
        for index, example in enumerate(examples)
    ]
    payloads = label_records_with_oracle(records, repo_root=repo_root)
    mismatches: list[str] = []
    for example, payload in zip(examples, payloads, strict=True):
        encoding = PositionEncoding.from_oracle_dict(dict(payload["position_encoding"]))
        actual = pack_position_features(encoding)
        expected = [float(value) for value in list(example["features"])]
        if actual != expected:
            mismatches.append(str(example["fen"]))

    if mismatches:
        print("encoder golden drift detected:")
        for fen in mismatches:
            print(f"  {fen}")
        return 1
    print(f"checked {len(examples)} encoder goldens: exact match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
