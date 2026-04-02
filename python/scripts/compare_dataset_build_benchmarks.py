"""Compare dataset-build benchmark artifacts across hosts or configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Comparison series in the form label:path_10k:path_20k",
    )
    parser.add_argument("--artifact-out", type=Path)
    args = parser.parse_args(argv)

    compared = [_parse_series(value) for value in args.series]
    record_counts = _shared_record_counts(compared)
    summary = {
        _record_count_label(record_count): _summarize_record_count(
            compared,
            expected_count=record_count,
        )
        for record_count in record_counts
    }
    notes = _notes(summary)
    if notes:
        summary["notes"] = notes

    rendered = json.dumps(summary, indent=2)
    if args.artifact_out is not None:
        args.artifact_out.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _parse_series(value: str) -> tuple[str, list[dict[str, Any]]]:
    label, *paths = value.split(":")
    if len(paths) < 2:
        raise ValueError("--series expects at least two artifact paths per label")
    return (
        label,
        [_load_artifact(Path(path)) for path in paths],
    )


def _load_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _shared_record_counts(compared: Sequence[tuple[str, list[dict[str, Any]]]]) -> list[int]:
    counts_per_label = []
    for label, artifacts in compared:
        counts = {int(artifact["record_count"]) for artifact in artifacts}
        if len(counts) != len(artifacts):
            raise ValueError(f"duplicate record_count values in series {label!r}")
        counts_per_label.append(counts)
    shared = sorted(set.intersection(*counts_per_label))
    if len(shared) < 2:
        raise ValueError("comparison needs at least two shared record counts")
    return shared


def _summarize_record_count(
    compared: Sequence[tuple[str, list[dict[str, Any]]]],
    *,
    expected_count: int,
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    reference_digests: dict[str, str] | None = None
    for label, artifacts in compared:
        artifact = next(
            (
                candidate
                for candidate in artifacts
                if int(candidate["record_count"]) == expected_count
            ),
            None,
        )
        if artifact is None:
            raise ValueError(f"missing record_count {expected_count} in series {label!r}")
        if int(artifact["record_count"]) != expected_count:
            raise ValueError(
                f"artifact for {label!r} expected {expected_count} records, "
                f"got {artifact['record_count']}"
            )
        results_by_name = {result["name"]: result for result in artifact["results"]}
        digests = results_by_name["auto_w4"]["digests"]
        if reference_digests is None:
            reference_digests = digests
        elif digests != reference_digests:
            raise ValueError(f"digest mismatch for {label!r} at {expected_count} records")

        auto = results_by_name["auto_w4"]
        explicit = results_by_name["w4_b500"]
        per_label[label] = {
            "runtime": artifact.get("runtime"),
            "input": artifact["input"],
            "repeats": artifact["repeats"],
            "auto_w4_seconds": auto["seconds"],
            "w4_b500_seconds": explicit["seconds"],
            "delta_seconds": round(float(auto["seconds"]) - float(explicit["seconds"]), 6),
            "delta_ratio": round(float(auto["seconds"]) / float(explicit["seconds"]), 3),
            "auto_w4_schedule": auto["oracle_schedule"],
            "w4_b500_schedule": explicit["oracle_schedule"],
            "fastest": artifact["fastest"],
            "digests": digests,
        }
    return per_label


def _record_count_label(record_count: int) -> str:
    if record_count % 1000 == 0:
        return f"{record_count // 1000}k"
    if record_count % 1024 == 0:
        return f"{record_count // 1024}k"
    return str(record_count)


def _notes(summary: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for record_count_label, summary_by_label in summary.items():
        for label, compared in summary_by_label.items():
            if compared["auto_w4_schedule"]["effective_batch_size"] == compared[
                "w4_b500_schedule"
            ]["effective_batch_size"]:
                notes.append(
                    f"At {record_count_label} on {label}, auto_w4 resolves to the same "
                    "effective batch size as w4_b500."
                )
    return notes


if __name__ == "__main__":
    raise SystemExit(main())
