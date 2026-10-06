# PACT-SLM reproducibility package

The licensed prediction-replay package is in the prediction_replay directory. It recomputes prediction-level metrics and group-bootstrap intervals for the variable-onset diagnostic from saved held-out predictions.

The replay package does not regenerate speech, extract embeddings, fit the WavLM probe, or evaluate a deployed speech-language model. Generated speech and stored embeddings are outside the package.

## Reproduction

From the repository root, run:

    python3 prediction_replay/replay.py prediction_replay/test_predictions.jsonl prediction_replay/reference_report.json --output replay_output.json --expected-prediction-sha256 b4170a945c938c1c8b9ed047f5f7f3e1cf176302f25628a2e4728cd415593ef9

The command recomputes summaries, semantic-label accuracy, exposed-action and internal-state accuracy, onset measures, trajectory exactness, paired differences, and 1,000 group-bootstrap intervals.

## Tests

    python3 -m pip install -r prediction_replay/requirements.txt
    python3 -m pytest -q prediction_replay

The MIT license in prediction_replay/LICENSE applies to the files in prediction_replay/.
