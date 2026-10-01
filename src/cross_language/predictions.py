"""Write per-pair predictions with pair_id for any matcher.

Ditto, HierGAT, WordCooc and Magellan only ever persisted aggregate F1, so a cell
scored against the wrong label vector could not be corrected without retraining --
which is exactly what happened to the July Ditto/HierGAT numbers. Every evaluation
now writes pair_id, label, probability and prediction next to its metrics.

The serialized Ditto/HierGAT inputs carry no pair_id, so it is read back from the
pair file that produced them. That is only valid if row order is preserved, so the
row count and the full label vector are asserted against the source before writing.
"""

from pathlib import Path

import pandas as pd


def _read_source(source_file):
    path = Path(source_file)
    if path.suffix == ".gz" and ".pkl" in path.name:
        return pd.read_pickle(path)
    if path.suffix == ".gz" and ".json" in path.name:
        return pd.read_json(path, lines=True)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported pair source: {source_file}")


def write_per_pair_predictions(output_path, source_file, labels, probabilities, predictions):
    """Write one prediction row per pair, keyed by pair_id.

    labels/probabilities/predictions must be in the row order of source_file.
    Raises if the lengths or the label vector disagree, because a silent
    misalignment would attach the right predictions to the wrong pairs.
    """
    source = _read_source(source_file)
    labels = [int(value) for value in labels]
    predictions = [int(value) for value in predictions]
    # HierGAT's evaluator returns hard predictions only; keep the column so every
    # matcher writes the same schema, and leave it empty when there is no score.
    probabilities = ([float(value) for value in probabilities]
                     if probabilities is not None else [None] * len(labels))

    if not (len(source) == len(labels) == len(probabilities) == len(predictions)):
        raise ValueError(
            f"{output_path}: row mismatch source={len(source)} labels={len(labels)} "
            f"probs={len(probabilities)} preds={len(predictions)}"
        )
    source_labels = [int(value) for value in source["label"]]
    if source_labels != labels:
        differing = sum(1 for a, b in zip(source_labels, labels) if a != b)
        raise ValueError(
            f"{output_path}: evaluated labels differ from {source_file} in {differing} rows; "
            "the evaluation ran against a different label vector"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "pair_id": source["pair_id"].astype(str).tolist(),
            "label": labels,
            "probability": probabilities,
            "prediction": predictions,
        }
    ).to_csv(output_path, index=False)
    return output_path
