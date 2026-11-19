"""Convert SFT synthesis candidates.jsonl into GRPO train and validation Parquet files."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from synthesize_sft import batches, read_jsonl


def gradeable_candidates(path: Path):
    for problem in read_jsonl(path):
        for candidate in problem["candidates"]:
            if (
                candidate.get("finish_reason") == "stop"
                and candidate.get("parsed_pred")
                and candidate.get("parsed_gt")
                and isinstance(candidate.get("correct"), bool)
                and candidate.get("solution", "").strip()
            ):
                yield problem, candidate


def split_problems(path: Path, validation_fraction: float, seed: int) -> set[str]:
    problem_ids = {problem["problem_id"] for problem, _ in gradeable_candidates(path)}
    if len(problem_ids) < 2:
        raise ValueError("Need at least two distinct problems with gradeable candidates to split the data")

    ordered_ids = sorted(problem_ids, key=lambda problem_id: sha256(f"{seed}:{problem_id}".encode()).digest())
    validation_count = max(1, min(len(problem_ids) - 1, round(len(problem_ids) * validation_fraction)))
    return set(ordered_ids[:validation_count])


def build_row(index: int, problem: dict, candidate: dict) -> dict:
    question = problem["question"]
    solution = candidate["solution"]
    return {
        "idx": index,
        "data_source": "rm_bool",
        "prompt": [
            {"role": "system", "content": "Verify the candidate solution to the math problem."},
            {"role": "user", "content": f"## Problem\n{question}\n\n## Candidate solution\n{solution}"},
        ],
        "reward_model": {"ground_truth": candidate["correct"], "style": "rule"},
        "extra_info": {
            "index": index,
            "problem_id": problem["problem_id"],
            "candidate_id": candidate["candidate_id"],
            "source": problem["source"],
            "problem": question,
            "solution": solution,
        },
    }


def write_parquet(input_path: Path, output_dir: Path, validation_fraction: float, seed: int) -> None:
    import pyarrow as pa
    from pyarrow import parquet as pq

    validation_ids = split_problems(input_path, validation_fraction, seed)
    schema = pa.schema([
        ("idx", pa.int64()),
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("reward_model", pa.struct([("ground_truth", pa.bool_()), ("style", pa.string())])),
        ("extra_info", pa.struct([
            ("index", pa.int64()),
            ("problem_id", pa.string()),
            ("candidate_id", pa.int32()),
            ("source", pa.string()),
            ("problem", pa.string()),
            ("solution", pa.string()),
        ])),
    ])

    counts = {"train": Counter(), "test": Counter()}
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temp_dir:
        paths = {split: Path(temp_dir) / f"{split}.parquet" for split in counts}
        with pq.ParquetWriter(paths["train"], schema) as train, pq.ParquetWriter(paths["test"], schema) as test:
            writers = {"train": train, "test": test}
            for batch in batches(enumerate(gradeable_candidates(input_path)), 1024):
                rows = {"train": [], "test": []}
                for index, (problem, candidate) in batch:
                    split = "test" if problem["problem_id"] in validation_ids else "train"
                    rows[split].append(build_row(index, problem, candidate))
                    counts[split][candidate["correct"]] += 1
                for split, writer in writers.items():
                    if rows[split]:
                        writer.write_table(pa.Table.from_pylist(rows[split], schema=schema))

        if not all(counts.values()):
            raise ValueError("Both GRPO splits must contain gradeable candidates")
        for split, path in paths.items():
            path.replace(output_dir / path.name)

    for split, counts_by_label in counts.items():
        print(f"{output_dir / (split + '.parquet')}: {sum(counts_by_label.values())} rows "
              f"({counts_by_label[True]} correct, {counts_by_label[False]} incorrect)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Labeled candidates.jsonl from synthesize_sft.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Fraction of distinct problems for test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    write_parquet(args.input, args.output_dir, args.validation_fraction, args.seed)


if __name__ == "__main__":
    main()
