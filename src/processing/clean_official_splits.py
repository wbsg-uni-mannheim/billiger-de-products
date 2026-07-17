"""Remove leakage and label conflicts from the official benchmark splits."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


DATA_ROOT = Path("data")
LANGUAGE_ROOTS = {
    "de": DATA_ROOT / "solute_de",
    "en": DATA_ROOT / "solute_en",
}
REPORT_ROOT = Path("reports/split_cleaning")
RATIOS = ("20cc80rnd", "50cc50rnd", "80cc20rnd")
UNSEEN_SHARES = ("000un", "050un", "100un")
SIZES = ("small", "medium", "large")
SPLIT_DIRS = {
    "train": "training-sets",
    "validation": "validation-sets",
    "test": "gold-standards_adjusted",
}
CORE_FIELDS = (
    "pair_id",
    "id_left",
    "id_right",
    "product_id_left",
    "product_id_right",
    "label",
    "is_hard_negative",
)


def canonical_pair(record):
    return tuple(sorted((str(record["id_left"]), str(record["id_right"]))))


def product_ids(records):
    return {
        str(record[column])
        for record in records
        for column in ("product_id_left", "product_id_right")
    }


def offer_ids(records):
    return {
        str(record[column])
        for record in records
        for column in ("id_left", "id_right")
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with gzip.open(path, "rb") as source:
        lines = source.readlines()
    return lines, [json.loads(line) for line in lines]


def write_jsonl(path, lines, keep_indices):
    with path.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            compresslevel=9,
            mtime=0,
        ) as output:
            for index in keep_indices:
                output.write(lines[index])


def relative_path(split, filename):
    return Path(SPLIT_DIRS[split]) / filename


def train_name(ratio, size):
    return f"products{ratio}000un_train_{size}.json.gz"


def validation_name(ratio, unseen, size):
    return f"products{ratio}{unseen}_valid_{size}.json.gz"


def test_name(ratio, unseen):
    return f"products{ratio}{unseen}_gs.json.gz"


def load_language(language):
    root = LANGUAGE_ROOTS[language]
    loaded = {}
    for split, directory in SPLIT_DIRS.items():
        for path in sorted((root / directory).glob("*.json.gz")):
            lines, records = read_jsonl(path)
            loaded[relative_path(split, path.name)] = {
                "split": split,
                "path": path,
                "lines": lines,
                "records": records,
            }
    return loaded


def verify_language_alignment(german, english):
    if set(german) != set(english):
        raise ValueError("German and English file lists differ")

    for relative in sorted(german):
        de_records = german[relative]["records"]
        en_records = english[relative]["records"]
        if len(de_records) != len(en_records):
            raise ValueError(f"Row count differs for {relative}")
        for index, (de_record, en_record) in enumerate(zip(de_records, en_records)):
            if any(
                str(de_record[field]) != str(en_record[field])
                for field in CORE_FIELDS
            ):
                raise ValueError(
                    f"German and English structure differs in {relative}, row {index}"
                )


def find_label_conflicts(loaded):
    labels = defaultdict(set)
    occurrences = defaultdict(list)
    for relative, info in loaded.items():
        for index, record in enumerate(info["records"]):
            pair = canonical_pair(record)
            labels[pair].add(int(record["label"]))
            occurrences[pair].append(
                {
                    "file": relative.as_posix(),
                    "row": index,
                    "label": int(record["label"]),
                }
            )
    conflicts = {pair for pair, values in labels.items() if len(values) > 1}
    return conflicts, occurrences


def create_decisions(loaded, conflicts):
    decisions = {}

    for ratio in RATIOS:
        retained_test_pairs = set()
        for unseen in UNSEEN_SHARES:
            relative = relative_path("test", test_name(ratio, unseen))
            records = loaded[relative]["records"]
            reasons = [
                "label_conflict" if canonical_pair(record) in conflicts else ""
                for record in records
            ]
            decisions[relative] = reasons
            retained_test_pairs.update(
                canonical_pair(record)
                for record, reason in zip(records, reasons)
                if not reason
            )

        retained_validation_pairs = set()
        for unseen in UNSEEN_SHARES:
            for size in SIZES:
                relative = relative_path(
                    "validation",
                    validation_name(ratio, unseen, size),
                )
                records = loaded[relative]["records"]
                reasons = []

                for record in records:
                    pair = canonical_pair(record)
                    if pair in conflicts:
                        reason = "label_conflict"
                    elif pair in retained_test_pairs:
                        reason = "validation_overlaps_test"
                    else:
                        reason = ""
                        retained_validation_pairs.add(pair)
                    reasons.append(reason)
                decisions[relative] = reasons

        for size in SIZES:
            relative = relative_path("train", train_name(ratio, size))
            records = loaded[relative]["records"]
            reasons = []
            for record in records:
                pair = canonical_pair(record)
                if pair in conflicts:
                    reason = "label_conflict"
                elif pair in retained_test_pairs:
                    reason = "train_overlaps_test"
                elif pair in retained_validation_pairs:
                    reason = "train_overlaps_validation"
                else:
                    reason = ""
                reasons.append(reason)
            decisions[relative] = reasons

    if set(decisions) != set(loaded):
        missing = sorted(set(loaded) - set(decisions))
        raise ValueError(f"Missing cleaning decisions for: {missing}")
    return decisions


def retained_records(loaded, decisions, relative):
    return [
        record
        for record, reason in zip(
            loaded[relative]["records"],
            decisions[relative],
        )
        if not reason
    ]


def infer_metadata(relative):
    name = relative.name.removesuffix(".json.gz")
    ratio = next(ratio for ratio in RATIOS if ratio in name)
    unseen = next(unseen for unseen in UNSEEN_SHARES if unseen in name)
    size = next((size for size in SIZES if name.endswith(f"_{size}")), "")
    split = next(
        split for split, directory in SPLIT_DIRS.items() if relative.parts[0] == directory
    )
    return split, ratio, unseen, size


def seen_share(records, train_records):
    products = product_ids(records)
    if not products:
        return None
    return len(products & product_ids(train_records)) / len(products)


def build_statistics(language, loaded, decisions):
    rows = []
    retained = {
        relative: retained_records(loaded, decisions, relative)
        for relative in loaded
    }

    for relative in sorted(loaded):
        split, ratio, unseen, size = infer_metadata(relative)
        before = loaded[relative]["records"]
        after = retained[relative]
        train_after = None
        if split == "validation":
            train_after = retained[
                relative_path("train", train_name(ratio, size))
            ]
        elif split == "test":
            train_after = retained[
                relative_path("train", train_name(ratio, "large"))
            ]

        rows.append(
            {
                "language": language,
                "split": split,
                "ratio": ratio,
                "unseen": unseen,
                "size": size,
                "file": relative.as_posix(),
                "rows_before": len(before),
                "rows_deleted": len(before) - len(after),
                "rows_after": len(after),
                "matches_before": sum(int(record["label"]) for record in before),
                "matches_after": sum(int(record["label"]) for record in after),
                "nonmatches_before": sum(
                    int(record["label"]) == 0 for record in before
                ),
                "nonmatches_after": sum(
                    int(record["label"]) == 0 for record in after
                ),
                "products_before": len(product_ids(before)),
                "products_after": len(product_ids(after)),
                "offers_before": len(offer_ids(before)),
                "offers_after": len(offer_ids(after)),
                "seen_product_share_after": (
                    seen_share(after, train_after) if train_after is not None else ""
                ),
                "sha256_before": sha256(loaded[relative]["path"]),
            }
        )
    return rows, retained


def verify_outputs(retained_by_language):
    german = retained_by_language["de"]
    english = retained_by_language["en"]
    checks = {
        "files_checked": len(german),
        "language_alignment": True,
        "internal_pair_uniqueness": True,
        "internal_pair_id_uniqueness": True,
        "label_conflicts": 0,
        "family_overlaps": {},
        "seen_product_shares": {},
    }

    for relative in sorted(german):
        de_records = german[relative]
        en_records = english[relative]
        if len(de_records) != len(en_records):
            raise ValueError(f"Cleaned language row count differs for {relative}")
        for index, (de_record, en_record) in enumerate(zip(de_records, en_records)):
            if any(
                str(de_record[field]) != str(en_record[field])
                for field in CORE_FIELDS
            ):
                raise ValueError(
                    f"Cleaned language structure differs in {relative}, row {index}"
                )

        pairs = [canonical_pair(record) for record in de_records]
        pair_ids = [str(record["pair_id"]) for record in de_records]
        if len(pairs) != len(set(pairs)):
            raise ValueError(f"Duplicate offer pairs remain in {relative}")
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError(f"Duplicate pair IDs remain in {relative}")

    labels = defaultdict(set)
    for records in german.values():
        for record in records:
            labels[canonical_pair(record)].add(int(record["label"]))
    remaining_conflicts = {
        pair for pair, pair_labels in labels.items() if len(pair_labels) > 1
    }
    if remaining_conflicts:
        raise ValueError(f"{len(remaining_conflicts)} label conflicts remain")

    expected_shares = {"000un": 1.0, "050un": 0.5, "100un": 0.5}
    for ratio in RATIOS:
        test_pairs = set()
        validation_pairs = set()
        train_pairs = set()

        for unseen in UNSEEN_SHARES:
            test_records = german[
                relative_path("test", test_name(ratio, unseen))
            ]
            test_pairs.update(canonical_pair(record) for record in test_records)

            for size in SIZES:
                train_records = german[
                    relative_path("train", train_name(ratio, size))
                ]
                validation_records = german[
                    relative_path(
                        "validation",
                        validation_name(ratio, unseen, size),
                    )
                ]
                share = seen_share(validation_records, train_records)
                checks["seen_product_shares"][
                    f"{ratio}_{unseen}_{size}_validation"
                ] = share
                if share != expected_shares[unseen]:
                    raise ValueError(
                        f"Unexpected seen share {share} for "
                        f"{ratio} {unseen} {size} validation"
                    )
                validation_pairs.update(
                    canonical_pair(record) for record in validation_records
                )
                train_pairs.update(
                    canonical_pair(record) for record in train_records
                )

        overlaps = {
            "train_test": len(train_pairs & test_pairs),
            "train_validation": len(train_pairs & validation_pairs),
            "validation_test": len(validation_pairs & test_pairs),
        }
        checks["family_overlaps"][ratio] = overlaps
        if any(overlaps.values()):
            raise ValueError(f"Split overlap remains for {ratio}: {overlaps}")

    return checks


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    loaded_by_language,
    decisions,
    conflicts,
    occurrences,
    statistics,
    verification,
):
    removed_rows = []
    for language, loaded in loaded_by_language.items():
        for relative in sorted(loaded):
            for index, (record, reason) in enumerate(
                zip(loaded[relative]["records"], decisions[relative])
            ):
                if not reason:
                    continue
                removed_rows.append(
                    {
                        "language": language,
                        "file": relative.as_posix(),
                        "row": index,
                        "pair_id": record["pair_id"],
                        "id_left": record["id_left"],
                        "id_right": record["id_right"],
                        "product_id_left": record["product_id_left"],
                        "product_id_right": record["product_id_right"],
                        "label": record["label"],
                        "reason": reason,
                    }
                )

    conflict_rows = []
    for left, right in sorted(conflicts):
        pair = (left, right)
        pair_occurrences = occurrences[pair]
        conflict_rows.append(
            {
                "id_a": left,
                "id_b": right,
                "labels": ",".join(
                    str(label)
                    for label in sorted(
                        {occurrence["label"] for occurrence in pair_occurrences}
                    )
                ),
                "occurrences": ";".join(
                    f"{occurrence['file']}:{occurrence['row']}="
                    f"{occurrence['label']}"
                    for occurrence in pair_occurrences
                ),
            }
        )

    write_csv(
        REPORT_ROOT / "removed_rows.csv",
        removed_rows,
        (
            "language",
            "file",
            "row",
            "pair_id",
            "id_left",
            "id_right",
            "product_id_left",
            "product_id_right",
            "label",
            "reason",
        ),
    )
    write_csv(
        REPORT_ROOT / "label_conflicts.csv",
        conflict_rows,
        ("id_a", "id_b", "labels", "occurrences"),
    )
    write_csv(
        REPORT_ROOT / "before_after_statistics.csv",
        statistics,
        tuple(statistics[0]),
    )
    (REPORT_ROOT / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Overwrite the official solute_de and solute_en JSONL files.",
    )
    args = parser.parse_args()

    loaded_by_language = {
        language: load_language(language) for language in LANGUAGE_ROOTS
    }
    verify_language_alignment(
        loaded_by_language["de"],
        loaded_by_language["en"],
    )
    conflicts, occurrences = find_label_conflicts(loaded_by_language["de"])
    decisions = create_decisions(loaded_by_language["de"], conflicts)

    statistics = []
    retained_by_language = {}
    for language, loaded in loaded_by_language.items():
        language_statistics, retained = build_statistics(
            language,
            loaded,
            decisions,
        )
        statistics.extend(language_statistics)
        retained_by_language[language] = retained

    verification = verify_outputs(retained_by_language)
    verification["label_conflicts_removed"] = len(conflicts)
    verification["rows_deleted"] = {
        language: sum(
            int(row["rows_deleted"])
            for row in statistics
            if row["language"] == language
        )
        for language in LANGUAGE_ROOTS
    }

    if args.apply:
        for language, loaded in loaded_by_language.items():
            for relative, info in loaded.items():
                keep_indices = [
                    index
                    for index, reason in enumerate(decisions[relative])
                    if not reason
                ]
                write_jsonl(info["path"], info["lines"], keep_indices)

        for row in statistics:
            row["sha256_after"] = sha256(
                LANGUAGE_ROOTS[row["language"]] / row["file"]
            )
    else:
        for row in statistics:
            row["sha256_after"] = ""

    write_reports(
        loaded_by_language,
        decisions,
        conflicts,
        occurrences,
        statistics,
        verification,
    )

    print(f"Label conflicts removed: {len(conflicts)}")
    for language in LANGUAGE_ROOTS:
        deleted = verification["rows_deleted"][language]
        print(f"{language.upper()} rows deleted: {deleted}")
    print(f"Reports written to {REPORT_ROOT}")
    print("Official files overwritten" if args.apply else "Dry run only")


if __name__ == "__main__":
    main()
