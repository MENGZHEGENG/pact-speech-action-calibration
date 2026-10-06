#!/usr/bin/env python3
"""Independently recompute PACT-SLM test metrics from saved predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPOSED_ACTIONS = {"ASK_CLARIFY", "CALL_TOOL", "INTERRUPT_SAFE"}
INTERNAL_STATES = {"THINK", "ABSTAIN"}
METRICS = (
    "pre_onset_exposed_action_rate",
    "post_onset_label_accuracy",
    "post_onset_exposed_action_accuracy",
    "post_onset_internal_state_accuracy",
    "trajectory_exact_rate",
    "first_non_wait_state_transition_exact_rate",
    "first_exposed_action_onset_exact_rate",
    "pre_onset_non_wait_rate",
    "post_onset_missed_action_rate",
    "final_action_accuracy",
    "first_non_wait_state_transition_mae_fraction",
    "first_exposed_action_onset_mae_fraction",
    "pair_exact_rate",
    "accuracy_all_prefixes",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                required = {
                    "model", "utterance_id", "contrast_group_id", "trajectory_id",
                    "rendering", "prefix_index", "prefix_fraction",
                    "valid_action_onset_fraction", "gold_action", "predicted_action",
                }
                missing = required - item.keys()
                if missing:
                    raise ValueError(f"missing fields {sorted(missing)}")
                by_model[str(item["model"])].append(item)
            except Exception as error:
                raise ValueError(f"invalid prediction at line {line_no}: {error}") from error
    if not by_model:
        raise ValueError("empty prediction file")
    expected_ids: set[str] | None = None
    for model, rows in by_model.items():
        ids = [str(row["utterance_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate utterance identity for {model}")
        current = set(ids)
        if expected_ids is None:
            expected_ids = current
        elif current != expected_ids:
            raise ValueError(f"prediction identities differ for {model}")
    return dict(by_model)


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for group_totals in _group_metric_totals(rows).values():
        for name, (numerator, denominator) in group_totals.items():
            totals[name][0] += numerator
            totals[name][1] += denominator
    return {name: numerator / denominator if denominator else 0.0
            for name, (numerator, denominator) in totals.items()}


def _group_metric_totals(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Precompute metric numerators and denominators by independent group."""
    if not rows:
        raise ValueError("cannot score an empty row set")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["contrast_group_id"])].append(row)

    result: dict[str, dict[str, tuple[float, float]]] = {}
    for group_id, group_rows in groups.items():
        group_result: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        gold = np.asarray([str(row["gold_action"]) for row in group_rows])
        pred = np.asarray([str(row["predicted_action"]) for row in group_rows])
        fractions = np.asarray([float(row["prefix_fraction"]) for row in group_rows])
        onsets = np.asarray([float(row["valid_action_onset_fraction"]) for row in group_rows])
        pre = fractions < onsets
        post = ~pre
        final = fractions == 1.0
        exposed = np.isin(pred, sorted(EXPOSED_ACTIONS))
        target_exposed = np.isin(gold, sorted(EXPOSED_ACTIONS))
        target_internal = np.isin(gold, sorted(INTERNAL_STATES))
        raw_metrics = {
            "pre_onset_exposed_action_rate": (float(np.sum(exposed & pre)), int(np.sum(pre))),
            "post_onset_label_accuracy": (
                float(np.sum((pred == gold) & post)), int(np.sum(post))
            ),
            "post_onset_exposed_action_accuracy": (
                float(np.sum((pred == gold) & post & target_exposed)),
                int(np.sum(post & target_exposed)),
            ),
            "post_onset_internal_state_accuracy": (
                float(np.sum((pred == gold) & post & target_internal)),
                int(np.sum(post & target_internal)),
            ),
            "pre_onset_non_wait_rate": (float(np.sum((pred != "WAIT") & pre)), int(np.sum(pre))),
            "post_onset_missed_action_rate": (float(np.sum((pred == "WAIT") & post)), int(np.sum(post))),
            "final_action_accuracy": (float(np.sum((pred == gold) & final)), int(np.sum(final))),
            "accuracy_all_prefixes": (float(np.sum(pred == gold)), len(group_rows)),
        }
        for name, (numerator, denominator) in raw_metrics.items():
            group_result[name][0] += numerator
            group_result[name][1] += denominator

        trajectories: dict[tuple[str, str], list[int]] = defaultdict(list)
        pairs: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(group_rows):
            trajectories[(str(row["trajectory_id"]), str(row["rendering"]))].append(index)
            pairs[str(row["rendering"])].append(index)

        for indices in trajectories.values():
            ordered = sorted(indices, key=lambda index: int(group_rows[index]["prefix_index"]))
            exact = float(np.all(pred[ordered] == gold[ordered]))
            true_onset = float(group_rows[ordered[0]]["valid_action_onset_fraction"])
            state_rows = [i for i in ordered if pred[i] != "WAIT"]
            state_onset = (float(group_rows[state_rows[0]]["prefix_fraction"])
                           if state_rows else 1.2)
            action_rows = [i for i in ordered if pred[i] in EXPOSED_ACTIONS]
            action_onset = (float(group_rows[action_rows[0]]["prefix_fraction"])
                            if action_rows else 1.2)
            for name, numerator in (
                ("trajectory_exact_rate", exact),
                ("first_non_wait_state_transition_exact_rate",
                 float(state_onset == true_onset)),
                ("first_non_wait_state_transition_mae_fraction",
                 abs(state_onset - true_onset)),
            ):
                group_result[name][0] += numerator
                group_result[name][1] += 1
            target_actions = {str(group_rows[index]["gold_action"]) for index in ordered
                              if str(group_rows[index]["gold_action"]) != "WAIT"}
            if len(target_actions) != 1:
                raise ValueError(
                    "each trajectory-rendering unit must have exactly one non-wait target action"
                )
            if next(iter(target_actions)) in EXPOSED_ACTIONS:
                group_result["first_exposed_action_onset_exact_rate"][0] += float(
                    action_onset == true_onset
                )
                group_result["first_exposed_action_onset_exact_rate"][1] += 1
                group_result["first_exposed_action_onset_mae_fraction"][0] += abs(
                    action_onset - true_onset
                )
                group_result["first_exposed_action_onset_mae_fraction"][1] += 1

        for indices in pairs.values():
            group_result["pair_exact_rate"][0] += float(np.all(pred[indices] == gold[indices]))
            group_result["pair_exact_rate"][1] += 1
        result[group_id] = {name: (values[0], values[1])
                            for name, values in group_result.items()}
    return result


def metric_denominators(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return the scored unit count for each rate or trajectory-level metric."""
    if not rows:
        raise ValueError("cannot count metric units for an empty row set")
    fractions = np.asarray([float(row["prefix_fraction"]) for row in rows])
    onsets = np.asarray([float(row["valid_action_onset_fraction"]) for row in rows])
    gold = np.asarray([str(row["gold_action"]) for row in rows])
    post = fractions >= onsets
    exposed_onset_units = 0
    internal_onset_units = 0
    trajectories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trajectories[(str(row["trajectory_id"]), str(row["rendering"]))].append(row)
    for unit_rows in trajectories.values():
        targets = {str(row["gold_action"]) for row in unit_rows
                   if str(row["gold_action"]) != "WAIT"}
        if len(targets) != 1:
            raise ValueError(
                "each trajectory-rendering unit must have exactly one non-wait target action"
            )
        if next(iter(targets)) in EXPOSED_ACTIONS:
            exposed_onset_units += 1
        elif next(iter(targets)) in INTERNAL_STATES:
            internal_onset_units += 1

    n_trajectory_units = len(trajectories)
    return {
        "n_predictions": len(rows),
        "n_pre_onset_prefixes": int(np.sum(fractions < onsets)),
        "n_post_onset_prefixes": int(np.sum(fractions >= onsets)),
        "n_post_onset_exposed_action_labels": int(
            np.sum(post & np.isin(gold, sorted(EXPOSED_ACTIONS)))
        ),
        "n_post_onset_internal_state_labels": int(
            np.sum(post & np.isin(gold, sorted(INTERNAL_STATES)))
        ),
        "n_final_prefixes": int(np.sum(fractions == 1.0)),
        "n_trajectory_rendering_units": n_trajectory_units,
        "n_exposed_action_onset_eligible_units": exposed_onset_units,
        "n_internal_state_trajectory_rendering_units": internal_onset_units,
        "n_contrast_group_rendering_units": len({
            (str(row["contrast_group_id"]), str(row["rendering"])) for row in rows
        }),
    }


def metric_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Return exact pooled numerators and denominators for all reported rates."""
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for group_totals in _group_metric_totals(rows).values():
        for name, (numerator, denominator) in group_totals.items():
            totals[name][0] += numerator
            totals[name][1] += denominator
    return {
        name: {"numerator": int(round(numerator)), "denominator": int(round(denominator))}
        for name, (numerator, denominator) in totals.items()
    }


def bootstrap(
    rows: list[dict[str, Any]],
    *,
    repeats: int,
    seed: int,
    right: list[dict[str, Any]] | None = None,
) -> dict[str, list[float]]:
    """Group percentile intervals; paired contrasts reuse sampled group IDs."""
    if right is not None and [r["utterance_id"] for r in rows] != [
        r["utterance_id"] for r in right
    ]:
        raise ValueError("paired models are not aligned by utterance identity")
    left_by_group = _group_metric_totals(rows)
    right_by_group = _group_metric_totals(right) if right is not None else None
    groups = sorted(left_by_group)
    if right_by_group is not None and set(groups) != set(right_by_group):
        raise ValueError("paired models have different contrast-group identities")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(repeats, len(groups)))
    samples: dict[str, list[float]] = {}
    for name in METRICS:
        left_values = [left_by_group[group].get(name, (0.0, 0.0))
                       for group in groups]
        left_num = np.asarray([value[0] for value in left_values], dtype=float)
        left_den = np.asarray([value[1] for value in left_values], dtype=float)
        denominator = left_den[draws].sum(axis=1)
        estimate = np.divide(
            left_num[draws].sum(axis=1), denominator,
            out=np.zeros(repeats, dtype=float), where=denominator > 0,
        )
        if right_by_group is not None:
            right_values = [right_by_group[group].get(name, (0.0, 0.0))
                            for group in groups]
            right_num = np.asarray([value[0] for value in right_values], dtype=float)
            right_den = np.asarray([value[1] for value in right_values], dtype=float)
            right_denominator = right_den[draws].sum(axis=1)
            right_estimate = np.divide(
                right_num[draws].sum(axis=1), right_denominator,
                out=np.zeros(repeats, dtype=float), where=right_denominator > 0,
            )
            estimate -= right_estimate
        samples[name] = estimate.tolist()
    return {
        name: [float(np.quantile(values, 0.025)),
               float(np.quantile(values, 0.975))]
        for name, values in samples.items()
    }


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row[field])].append(row)
    return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("reference_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--expected-prediction-sha256",
        help="independently recorded digest for the prediction file being replayed",
    )
    args = parser.parse_args()

    models = load_predictions(args.predictions)
    report = json.loads(args.reference_report.read_text(encoding="utf-8"))
    prediction_sha256 = sha256(args.predictions)
    expected_prediction_sha256 = args.expected_prediction_sha256
    if expected_prediction_sha256 is not None and (
        len(expected_prediction_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_prediction_sha256)
    ):
        parser.error("--expected-prediction-sha256 must be 64 lowercase hexadecimal characters")
    scored: dict[str, Any] = {}
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n_predictions": len(rows),
            "n_groups": len({str(r["contrast_group_id"]) for r in rows}),
            "metric_denominators": metric_denominators(rows),
            "metric_counts": metric_counts(rows),
            "metrics": compute_metrics(rows),
            "group_bootstrap_95_ci": bootstrap(
                rows, repeats=args.repeats, seed=args.seed
            ),
        }

    for model, rows in sorted(models.items()):
        model_out: dict[str, Any] = {
            "all": summarize(rows),
            "by_rendering": {},
            "by_family": {},
        }
        for rendering, subset in sorted(grouped(rows, "rendering").items()):
            model_out["by_rendering"][rendering] = summarize(subset)
        for family, subset in sorted(grouped(rows, "family").items()):
            model_out["by_family"][family] = summarize(subset)
        scored[model] = model_out

    contrasts: dict[str, Any] = {}
    compare = [("audio_only", name) for name in sorted(models) if name != "audio_only"]
    compare.extend([("audio_text", "audio_only")])
    for left_name, right_name in compare:
        left, right = models[left_name], models[right_name]
        left_metrics, right_metrics = compute_metrics(left), compute_metrics(right)
        contrasts[f"{left_name}_minus_{right_name}"] = {
            "estimate": {name: left_metrics[name] - right_metrics[name]
                         for name in METRICS},
            "group_bootstrap_95_ci": bootstrap(
                left, right=right, repeats=args.repeats, seed=args.seed
            ),
        }

    result = {
        "schema": "pact-slm-independent-prediction-replay/v3",
        "prediction_sha256": sha256(args.predictions),
        "reference_report_sha256": sha256(args.reference_report),
        "bootstrap": {
            "unit": "contrast_group_id",
            "repeats": args.repeats,
            "seed": args.seed,
            "quantiles": [0.025, 0.975],
        },
        "models": scored,
        "paired_differences": contrasts,
        "expected_prediction_sha256": expected_prediction_sha256,
        "expected_prediction_sha256_matches": (
            prediction_sha256 == expected_prediction_sha256
            if expected_prediction_sha256 is not None else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "models": len(models),
        "records_per_model": len(next(iter(models.values()))),
        "prediction_sha256": result["prediction_sha256"],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "expected_prediction_hash_matches": result[
            "expected_prediction_sha256_matches"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
