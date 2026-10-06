from replay import bootstrap, compute_metrics, metric_counts, metric_denominators


def _group_rows(group_id: str, post_onset_prediction: str) -> list[dict[str, object]]:
    common = {
        "contrast_group_id": group_id,
        "trajectory_id": f"{group_id}-trajectory",
        "rendering": "clean",
        "valid_action_onset_fraction": 0.6,
    }
    return [
        {
            **common,
            "utterance_id": f"{group_id}-pre",
            "prefix_index": 0,
            "prefix_fraction": 0.2,
            "gold_action": "WAIT",
            "predicted_action": "WAIT",
        },
        {
            **common,
            "utterance_id": f"{group_id}-post",
            "prefix_index": 1,
            "prefix_fraction": 0.6,
            "gold_action": "CALL_TOOL",
            "predicted_action": post_onset_prediction,
        },
        {
            **common,
            "utterance_id": f"{group_id}-final",
            "prefix_index": 2,
            "prefix_fraction": 1.0,
            "gold_action": "CALL_TOOL",
            "predicted_action": post_onset_prediction,
        },
    ]


def test_group_bootstrap_counts_repeated_group_draws_as_repeated_units() -> None:
    rows = (
        _group_rows("group-0", "CALL_TOOL")
        + _group_rows("group-1", "WAIT")
        + _group_rows("group-2", "WAIT")
    )

    # NumPy seed 60 samples sorted group indices [0, 0, 1] for n=3.
    interval = bootstrap(rows, repeats=1, seed=60)["trajectory_exact_rate"]

    assert interval == [2 / 3, 2 / 3]


def test_internal_state_transition_does_not_count_as_exposed_action_onset() -> None:
    rows = []
    for trajectory_id, predictions in {
        "internal_then_action": ["WAIT", "THINK", "CALL_TOOL", "CALL_TOOL", "CALL_TOOL"],
        "early_action": ["WAIT", "ASK_CLARIFY", "CALL_TOOL", "CALL_TOOL", "CALL_TOOL"],
        "internal_target": ["WAIT", "WAIT", "THINK", "THINK", "THINK"],
    }.items():
        for index, (fraction, predicted) in enumerate(
            zip((0.2, 0.4, 0.6, 0.8, 1.0), predictions)
        ):
            rows.append({
                "contrast_group_id": trajectory_id,
                "trajectory_id": trajectory_id,
                "rendering": "clean",
                "utterance_id": f"{trajectory_id}-{index}",
                "prefix_index": index,
                "prefix_fraction": fraction,
                "valid_action_onset_fraction": 0.6,
                "gold_action": (
                    "WAIT" if fraction < 0.6 else
                    ("THINK" if trajectory_id == "internal_target" else "CALL_TOOL")
                ),
                "predicted_action": predicted,
            })

    metrics = compute_metrics(rows)
    denominators = metric_denominators(rows)

    assert metrics["first_non_wait_state_transition_exact_rate"] == 1 / 3
    assert metrics["first_exposed_action_onset_exact_rate"] == 1 / 2
    assert metrics["pre_onset_exposed_action_rate"] == 1 / 6
    assert denominators["n_trajectory_rendering_units"] == 3
    assert denominators["n_exposed_action_onset_eligible_units"] == 2


def test_family_subset_replay_retains_its_actual_counts() -> None:
    rows = _group_rows("family-group", "CALL_TOOL")
    rows = [{**row, "family": "book_uncertainty"} for row in rows]

    from replay import grouped, metric_denominators

    family_rows = grouped(rows, "family")["book_uncertainty"]

    assert metric_denominators(family_rows)["n_predictions"] == 3
    assert metric_denominators(family_rows)["n_trajectory_rendering_units"] == 1


def test_post_onset_external_actions_and_internal_states_are_scored_separately() -> None:
    rows = []
    cases = {
        "external_internal_before_action": (
            "CALL_TOOL", ["WAIT", "THINK", "CALL_TOOL", "CALL_TOOL", "CALL_TOOL"]
        ),
        "external_early_then_error": (
            "CALL_TOOL", ["WAIT", "ASK_CLARIFY", "ABSTAIN", "CALL_TOOL", "CALL_TOOL"]
        ),
        "internal_target": ("THINK", ["WAIT", "WAIT", "THINK", "THINK", "THINK"]),
    }
    for trajectory_id, (target, predictions) in cases.items():
        for index, (fraction, predicted) in enumerate(
            zip((0.2, 0.4, 0.6, 0.8, 1.0), predictions)
        ):
            rows.append({
                "contrast_group_id": trajectory_id,
                "trajectory_id": trajectory_id,
                "rendering": "clean",
                "utterance_id": f"{trajectory_id}-{index}",
                "prefix_index": index,
                "prefix_fraction": fraction,
                "valid_action_onset_fraction": 0.6,
                "gold_action": "WAIT" if fraction < 0.6 else target,
                "predicted_action": predicted,
            })

    metrics = compute_metrics(rows)
    denominators = metric_denominators(rows)

    assert metrics["post_onset_label_accuracy"] == 8 / 9
    assert metrics["post_onset_exposed_action_accuracy"] == 5 / 6
    assert metrics["post_onset_internal_state_accuracy"] == 1.0
    assert denominators["n_post_onset_exposed_action_labels"] == 6
    assert denominators["n_post_onset_internal_state_labels"] == 3
    assert metric_counts(rows)["post_onset_exposed_action_accuracy"] == {
        "numerator": 5, "denominator": 6
    }
