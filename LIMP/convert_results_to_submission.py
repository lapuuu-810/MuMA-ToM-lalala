#!/usr/bin/env python3
"""Convert MuMA-ToM result JSON into submission-style records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path(
    "/data/LPP/cvpr/muti_agent/MuMA-ToM_my/local_runs/qwen3_5_27b/results_fix_2.json"
)
DEFAULT_OUTPUT = Path(
    "/data/LPP/cvpr/muti_agent/MuMA-ToM_my/LIMP/results_fix_2_submission.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a results JSON file into "
            '[{"scenario_id": 1, "question_id": 1, "answer": "B"}] format.'
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source results JSON path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Converted JSON path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def convert_results(payload: dict) -> list[dict]:
    episodes = payload.get("episodes")
    if not isinstance(episodes, dict):
        raise ValueError("Input JSON must contain an 'episodes' object.")

    records: list[dict] = []
    for episode_key, episode in episodes.items():
        if not isinstance(episode, dict):
            raise ValueError(f"Episode '{episode_key}' is not a JSON object.")

        scenario_id = episode.get("episode_id", episode_key)
        try:
            scenario_id = int(scenario_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Episode '{episode_key}' has a non-integer episode_id: {scenario_id!r}"
            ) from exc

        predictions = episode.get("predictions")
        if not isinstance(predictions, dict):
            raise ValueError(
                f"Episode '{episode_key}' must contain a 'predictions' object."
            )

        for question_key, prediction_info in predictions.items():
            if not isinstance(prediction_info, dict):
                raise ValueError(
                    f"Prediction '{question_key}' in episode '{episode_key}' is invalid."
                )

            answer = prediction_info.get("prediction")
            if not isinstance(answer, str) or not answer:
                raise ValueError(
                    f"Prediction '{question_key}' in episode '{episode_key}' "
                    "is missing a valid 'prediction' value."
                )

            try:
                question_id = int(question_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Episode '{episode_key}' has a non-integer question id: {question_key!r}"
                ) from exc

            records.append(
                {
                    "scenario_id": scenario_id,
                    "question_id": question_id,
                    "answer": answer,
                }
            )

    records.sort(key=lambda item: (item["scenario_id"], item["question_id"]))
    return records


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    records = convert_results(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
