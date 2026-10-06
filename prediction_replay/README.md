# PACT-SLM prediction replay

This MIT-licensed package recomputes the reported prediction-level metrics for the variable-onset diagnostic. It uses saved prediction records, checks their identities, separates exposed-action and internal-state labels, and computes the pooled metrics and group-bootstrap intervals reported in the paper.

The supplement does not regenerate speech, extract embeddings, fit the WavLM probe, or evaluate a deployed speech-language model. The generated-speech files and stored embeddings are outside this package. The replay therefore validates the saved predictions and the metric contract.

## Requirements

Use Python 3.10 or newer. Install the packages listed in `requirements.txt`.

## Replay

From this directory, run:

```bash
python3 replay.py test_predictions.jsonl reference_report.json \
  --output replay_output.json \
  --expected-prediction-sha256 b4170a945c938c1c8b9ed047f5f7f3e1cf176302f25628a2e4728cd415593ef9
```

The command recomputes the model summaries, pooled semantic-label accuracy, exposed-action accuracy, internal-state accuracy, onset measures, trajectory exactness, paired differences, and 1,000 group-bootstrap intervals. The expected prediction digest must match before the output is treated as the saved replay.

Run the contract tests with:

```bash
python3 -m pytest -q
```

`reference_report.json` is a public-safe description of the replay scope. Its hash is retained in the output so a later run can identify the exact report used.
