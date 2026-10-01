"""Record and read the protocol identity of one cross-language experiment.

Every cross-language job writes a ``provenance.json`` into its own output
directory before training starts.  ``src/summarize_cross_language_results.py``
joins that file back onto the result rows so that ``metrics.csv`` and
``summary.csv`` carry the backbone and the validation-file identity per row
instead of leaving them to be reconstructed from job scripts afterwards.
"""

import argparse
import hashlib
import json
from pathlib import Path


FILE_NAME = "provenance.json"
FIELDS = (
    "model",
    "backbone",
    "validation_file",
    "validation_sha256",
    "validation_rows",
    "train_file",
    "seeds",
    "batch_size",
)


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count_rows(path):
    path = Path(path)
    if path.suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if path.name.endswith(".pkl.gz"):
        import pandas as pd

        return len(pd.read_pickle(path))
    return ""


def write(output_dir, model, backbone, validation_file, train_file, seeds, batch_size):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Zero-shot matchers select nothing, so they legitimately have no
    # validation file; everything else must point at one that exists.
    record = {
        "model": model,
        "backbone": backbone,
        "validation_file": Path(validation_file).as_posix() if validation_file else "",
        "validation_sha256": file_digest(validation_file) if validation_file else "",
        "validation_rows": count_rows(validation_file) if validation_file else "",
        "train_file": Path(train_file).as_posix() if train_file else "",
        "seeds": seeds,
        "batch_size": batch_size,
    }
    (output_dir / FILE_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def find(path, root):
    """Return the provenance record of the nearest enclosing directory."""
    path = Path(path).resolve()
    root = Path(root).resolve()
    directory = path.parent if path.is_file() else path
    while True:
        candidate = directory / FILE_NAME
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        if directory == root or directory == directory.parent:
            return None
        directory = directory.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backbone",
        default="",
        help="Pretrained checkpoint the matcher fine-tunes, or 'n/a' for the "
        "classical matchers.",
    )
    parser.add_argument("--validation-file", default="")
    parser.add_argument("--train-file", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--batch-size", default="")
    args = parser.parse_args()

    record = write(
        args.output_dir,
        args.model,
        args.backbone,
        args.validation_file,
        args.train_file,
        args.seeds,
        args.batch_size,
    )
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
