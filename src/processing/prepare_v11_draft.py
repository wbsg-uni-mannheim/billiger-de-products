"""Create an isolated, auditable offer-replacement draft; never edit v1.0.

python -m src.processing.prepare_v11_draft
python -m src.processing.prepare_v11_draft --check-readiness data/releases/v1.1-draft

Unknown changed-pair labels are null, not inferred from source product IDs.
The output is deliberately not a model-input directory or an official release.
"""

import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


FAMILIES = ("20cc80", "50cc50", "80cc20")
LANGUAGES = ("de", "en")
TRAIN = "training-sets"
VALID = "validation-sets"
TEST = "gold-standards_adjusted"
CORE = ("pair_id", "id_left", "id_right", "product_id_left",
        "product_id_right", "label", "is_hard_negative")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, ensure_ascii=False) + "\n").encode())


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pair_key(row):
    return tuple(sorted((int(row["id_left"]), int(row["id_right"]))))


def pair_id(key):
    return "#".join(map(str, key))


def review_id(key, content_hash):
    return pair_id(key) + ":" + content_hash[:16]


def offer_ids(rows):
    return {row[f"id_{side}"] for row in rows for side in ("left", "right")}


def product_offers(rows):
    result = defaultdict(set)
    for row in rows:
        for side in ("left", "right"):
            result[row[f"product_id_{side}"]].add(row[f"id_{side}"])
    return result


def allocate(old, reserved):
    """Keep shared IDs and bijectively map remaining slots, independent of labels."""
    if len(old) != 2 or len(reserved) != 2:
        raise ValueError("Replacement requires exactly two source and reserved offers")
    common = old & reserved
    mapping = {offer: offer for offer in sorted(common)}
    mapping.update(zip(sorted(old - common), sorted(reserved - common)))
    return mapping


def exact_test_evidence(loaded):
    labels, files = defaultdict(set), defaultdict(set)
    for relative, rows in loaded.items():
        if relative.parts[0] == TEST:
            for row in rows:
                key = pair_key(row)
                labels[key].add(row["label"])
                files[key].add(str(relative))
    return labels, files


def resolve_label(key, content_hash, evidence, decisions):
    """Only exact pair evidence or an explicit text-bound human decision counts."""
    known = evidence.get(key, set())
    if len(known) > 1:
        return None, "conflicting_existing_test_labels"
    decision = decisions.get(review_id(key, content_hash))
    if decision:
        if decision["content_sha256"] != content_hash:
            raise ValueError(f"Stale decision for {pair_id(key)}")
        if known and decision["label"] not in known:
            raise ValueError(f"Decision contradicts existing test label: {pair_id(key)}")
        return decision["label"], "human_review"
    if known:
        return next(iter(known)), "existing_exact_test_pair"
    return None, "needs_human_review"


def read_decisions(path):
    result = {}
    if path:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                value = row["decision_label"].strip()
                if not value:
                    continue
                if value not in ("0", "1") or not row["reviewer"].strip() or not row["rationale"].strip():
                    raise ValueError("Decisions require 0/1, reviewer and rationale")
                key = row["review_id"]
                if key in result:
                    raise ValueError(f"Duplicate decision: {key}")
                result[key] = {"label": int(value), "content_sha256": row["content_sha256"],
                               "reviewer": row["reviewer"], "rationale": row["rationale"]}
    return result


def load_source(root):
    loaded, registries, hashes = {}, {}, {}
    text_variants = defaultdict(dict)
    for lang in LANGUAGES:
        loaded[lang], registries[lang] = {}, {}
        base = root / f"solute_{lang}"
        for folder in (TRAIN, VALID, TEST):
            registries[lang][folder] = {}
            for path in sorted((base / folder).glob("*.json.gz")):
                relative = path.relative_to(base)
                rows = read_rows(path)
                loaded[lang][relative] = rows
                hashes[str(path.relative_to(root))] = digest(path)
                for row in rows:
                    for side in ("left", "right"):
                        suffix = "_" + side
                        offer = {k[:-len(suffix)]: v for k, v in row.items() if k.endswith(suffix)}
                        previous = registries[lang][folder].setdefault(offer["id"], offer)
                        if previous != offer:
                            raise ValueError(f"Inconsistent {lang}/{folder} offer {offer['id']}")
                        text_variants[(lang, offer["id"])][folder] = json_hash(offer)
        if len(loaded[lang]) != 45:
            raise ValueError(f"Expected 45 source files for {lang}")
    if loaded["de"].keys() != loaded["en"].keys():
        raise ValueError("DE/EN file lists differ")
    for path, rows in loaded["de"].items():
        english = loaded["en"][path]
        if len(rows) != len(english) or any(
            any(a[k] != b[k] for k in CORE) for a, b in zip(rows, english)
        ):
            raise ValueError(f"DE/EN structure differs: {path}")
    variants = [{"language": lang, "offer_id": offer, "split_content_sha256": versions}
                for (lang, offer), versions in sorted(text_variants.items())
                if len(set(versions.values())) > 1]
    return loaded, registries, hashes, variants


def paths_for(family, folder, condition="000", size="large"):
    ending = "gs" if folder == TEST else f"{'train' if folder == TRAIN else 'valid'}_{size}"
    return Path(folder) / f"products{family}rnd{condition}un_{ending}.json.gz"


def build_maps(loaded):
    mappings, events, shortages = {}, [], []
    for family in FAMILIES:
        train = loaded[paths_for(family, TRAIN)]
        valid = loaded[paths_for(family, VALID)]
        seen_test = loaded[paths_for(family, TEST)]
        seen_products = set(product_offers(train))
        train_ids, valid_ids = offer_ids(train), offer_ids(valid)
        all_rows = [r for p, rows in loaded.items() if family in p.name for r in rows]
        all_products = product_offers(all_rows)
        for product, reserved in sorted(product_offers(seen_test).items()):
            available = all_products[product] - train_ids - valid_ids
            if len(available) < 2:
                shortages.append({"family": family, "product_id": product,
                                  "available_test_offers": len(available),
                                  "available_ids": json.dumps(sorted(available)),
                                  "seen_test_ids": json.dumps(sorted(reserved)),
                                  "reason": "fewer_than_two_offers_outside_frozen_train_and_seen_validation"})
        for folder in (TEST, VALID):
            source = product_offers(loaded[paths_for(family, folder, "050")])
            reserved = product_offers(loaded[paths_for(family, folder)])
            mapping = {}
            for product in sorted(set(source) & seen_products):
                slots = allocate(source[product], reserved[product])
                mapping.update(slots)
                for old, new in slots.items():
                    if old != new:
                        events.append({"family": family, "split": folder,
                                       "product_id": product, "old_offer_id": old,
                                       "new_offer_id": new})
            mappings[(family, folder)] = mapping
    return mappings, events, shortages


def replace_row(row, mapping, registry):
    result = row.copy()
    for side in ("left", "right"):
        old = row[f"id_{side}"]
        new = mapping.get(old, old)
        if old != new:
            offer = registry[new]
            if offer["product_id"] != row[f"product_id_{side}"]:
                raise ValueError("Replacement changed source product")
            result.update({f"{k}_{side}": v for k, v in offer.items()})
    result["pair_id"] = f"{result['id_left']}#{result['id_right']}"
    return result


def collapse_duplicates(rows):
    """Return unique candidates, logging consolidations; conflicting labels block."""
    kept, duplicate_of, conflicts = {}, {}, []
    for index, row in enumerate(rows):
        key = pair_key(row)
        if key in kept:
            first = kept[key]
            if rows[first]["label"] != row["label"]:
                conflicts.append(pair_id(key))
            duplicate_of[index] = first
        else:
            kept[key] = index
    return [rows[i] for i in kept.values()], duplicate_of, conflicts


def inspect_splits(loaded):
    checks = []
    for family in FAMILIES:
        groups = {folder: [r for p, rows in loaded.items()
                           if p.parts[0] == folder and family in p.name for r in rows]
                  for folder in (TRAIN, VALID, TEST)}
        for left, right in ((TRAIN, VALID), (TRAIN, TEST), (VALID, TEST)):
            a, b = groups[left], groups[right]
            overlap = sorted(offer_ids(a) & offer_ids(b))
            keys = {pair_key(r) for r in a} & {pair_key(r) for r in b}
            checks.append({"family": family, "left": left, "right": right,
                           "shared_offers": len(overlap), "offer_ids": overlap,
                           "shared_pairs": len(keys), "pair_ids": [pair_id(k) for k in sorted(keys)]})
    labels = defaultdict(set)
    for rows in loaded.values():
        for row in rows:
            if row["label"] is not None:
                labels[pair_key(row)].add(row["label"])
    return checks, [pair_id(k) for k, values in sorted(labels.items()) if len(values) > 1]


def frozen_offer_ids(loaded, family):
    """Offers reserved for training or the unchanged Seen validation files."""
    return offer_ids([row for path, rows in loaded.items()
                      if family in path.name and (
                          path.parts[0] == TRAIN or
                          (path.parts[0] == VALID and "rnd000un" in path.name))
                      for row in rows])


def prepare(source_root, output_root, decisions_path=None, drop_overlapping_test_offers=False):
    source_root, output_root = source_root.resolve(), output_root.resolve()
    if output_root.exists():
        raise ValueError(f"Output already exists; choose a new draft directory: {output_root}")
    if any(output_root.is_relative_to(source_root / f"solute_{lang}") for lang in LANGUAGES):
        raise ValueError("Output must be outside the official language directories")
    loaded, registries, source_hashes, source_variants = load_source(source_root)
    decisions = read_decisions(decisions_path)
    evidence, evidence_files = exact_test_evidence(loaded["de"])
    mappings, mapping_events, shortages = build_maps(loaded["de"])
    forbidden_test_ids = {family: frozen_offer_ids(loaded["de"], family) for family in FAMILIES}
    candidates = {lang: {} for lang in LANGUAGES}
    ledger, stats, review_items, structural = [], [], {}, []
    applied_decisions = set()
    for relative, originals in sorted(loaded["de"].items()):
        family = re.search(r"products(\d+cc\d+)rnd", relative.name)[1]
        condition = re.search(r"rnd(\d+)un", relative.name)[1]
        folder = relative.parts[0]
        mutable = (folder == TEST and condition == "050") or (folder == VALID and condition in ("050", "100"))
        mapping = mappings[(family, folder)] if mutable else {}
        if drop_overlapping_test_offers and folder == TEST and condition == "000":
            mutable = True
        proposals = {lang: [] for lang in LANGUAGES}
        file_ledger = []
        proposal_source_indices = []
        for index, old in enumerate(originals):
            new = {lang: replace_row(loaded[lang][relative][index], mapping, registries[lang][folder])
                   for lang in LANGUAGES}
            changed = pair_key(old) != pair_key(new["de"])
            key = pair_key(new["de"])
            if (drop_overlapping_test_offers and folder == TEST
                    and set(key) & forbidden_test_ids[family]):
                file_ledger.append({"file": str(relative), "row": index,
                                    "old_pair_id": old["pair_id"], "new_pair_id": new["de"]["pair_id"],
                                    "old_label": old["label"], "new_label": None,
                                    "changed": changed, "label_source": "removed_before_label_review",
                                    "duplicate_of_row": "", "removed": True,
                                    "removal_reason": "test_offer_reserved_for_training_or_seen_validation"})
                continue
            label, origin = old["label"], "unchanged_input"
            if changed:
                payload = {lang: [registries[lang][folder][i] for i in key] for lang in LANGUAGES}
                content_hash = json_hash(payload)
                label, origin = resolve_label(key, content_hash, evidence, decisions)
                if origin == "human_review":
                    applied_decisions.add(review_id(key, content_hash))
                review_key = review_id(key, content_hash)
                item = review_items.setdefault(review_key, {
                    "review_id": review_key, "pair_id": pair_id(key), "content_sha256": content_hash,
                    "resolved_label": label, "label_source": origin,
                    "existing_test_files": sorted(evidence_files.get(key, set())),
                    "new_offers": payload, "original_occurrences": [],
                })
                item["original_occurrences"].append({"file": str(relative), "row": index,
                                                     "old_pair_id": old["pair_id"], "old_label": old["label"]})
            if key[0] == key[1]:
                structural.append({"file": str(relative), "pair_id": pair_id(key), "issue": "self_pair"})
            for lang in LANGUAGES:
                new[lang]["label"] = label
                proposals[lang].append(new[lang])
            proposal_source_indices.append(index)
            file_ledger.append({"file": str(relative), "row": index,
                                "old_pair_id": old["pair_id"], "new_pair_id": new["de"]["pair_id"],
                                "old_label": old["label"], "new_label": label,
                                "changed": changed, "label_source": origin, "duplicate_of_row": "",
                                "removed": False, "removal_reason": ""})
        for lang in LANGUAGES:
            if mutable:
                rows, duplicates, conflicts = collapse_duplicates(proposals[lang])
            else:
                rows, duplicates, conflicts = proposals[lang], {}, []
                if rows != loaded[lang][relative]:
                    raise ValueError(f"Frozen data would change: {relative}")
            candidates[lang][relative] = rows
            if lang == "de":
                for index, original_index in duplicates.items():
                    file_ledger[proposal_source_indices[index]]["duplicate_of_row"] = proposal_source_indices[original_index]
                structural.extend({"file": str(relative), "pair_id": key,
                                   "issue": "duplicate_label_conflict"} for key in conflicts)
                stats.append({"file": str(relative), "mutable": mutable,
                              "source_rows": len(originals), "candidate_rows": len(rows),
                              "changed_source_rows": sum(r["changed"] for r in file_ledger),
                              "removed_rows": sum(r["removed"] for r in file_ledger),
                              "deduplicated_rows": len(duplicates),
                              "old_positives": sum(r["label"] == 1 for r in originals),
                              "new_positives": sum(r["label"] == 1 for r in rows),
                              "unresolved_rows": sum(r["label"] is None for r in rows),
                              "known_label_changes": sum(r["new_label"] is not None and r["new_label"] != r["old_label"] for r in file_ledger)})
        ledger.extend(file_ledger)
    if set(decisions) != applied_decisions:
        raise ValueError(f"Unknown or unused decisions: {sorted(set(decisions) - applied_decisions)}")
    checks, global_conflicts = inspect_splits(candidates["de"])
    for relative, rows in candidates["de"].items():
        en = candidates["en"][relative]
        if len(rows) != len(en) or any(any(a[k] != b[k] for k in CORE) for a, b in zip(rows, en)):
            raise ValueError(f"Candidate DE/EN alignment failed: {relative}")
    output_root.mkdir(parents=True)
    report = output_root / "review"
    report.mkdir()
    output_hashes = {}
    for lang in LANGUAGES:
        for relative, rows in candidates[lang].items():
            dest = output_root / "candidate_sets" / f"solute_{lang}" / relative
            if rows == loaded[lang][relative]:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / f"solute_{lang}" / relative, dest)
            else:
                write_rows(dest, rows)
            output_hashes[str(dest.relative_to(output_root))] = digest(dest)
    write_csv(report / "offer_mapping.csv", mapping_events,
              ["family", "split", "product_id", "old_offer_id", "new_offer_id"])
    write_csv(report / "pair_changes.csv", [r for r in ledger if r["changed"] or r["removed"] or r["duplicate_of_row"] != ""],
              list(ledger[0]))
    write_csv(report / "removed_test_pairs.csv", [r for r in ledger if r["removed"]], list(ledger[0]))
    write_csv(report / "file_statistics.csv", stats, list(stats[0]))
    write_csv(report / "allocation_blockers.csv", shortages,
              ["family", "product_id", "available_test_offers", "available_ids", "seen_test_ids", "reason"])
    pending = [item for _, item in sorted(review_items.items()) if item["resolved_label"] is None]
    fields = ["review_id", "content_sha256", "decision_label", "reviewer", "rationale",
              "old_labels", "occurrences", "label_source", "de_left", "de_right", "en_left", "en_right"]
    review_rows = []
    for item in pending:
        row = {k: "" for k in fields}
        row.update({"review_id": item["review_id"], "content_sha256": item["content_sha256"],
                    "old_labels": json.dumps(sorted({r["old_label"] for r in item["original_occurrences"]})),
                    "occurrences": len(item["original_occurrences"]), "label_source": item["label_source"]})
        for lang in LANGUAGES:
            for i, side in enumerate(("left", "right")):
                row[f"{lang}_{side}"] = json.dumps(item["new_offers"][lang][i], ensure_ascii=False)
        review_rows.append(row)
    write_csv(report / "label_review.csv", review_rows, fields)
    with (report / "label_provenance.jsonl").open("w", encoding="utf-8") as stream:
        for _, item in sorted(review_items.items()):
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (report / "structural_checks.json").write_text(json.dumps({"split_checks": checks,
        "row_issues": structural, "global_label_conflicts": global_conflicts}, indent=2) + "\n")
    (report / "source_text_variants.json").write_text(json.dumps(source_variants, indent=2) + "\n")
    blockers = {
        "unresolved_unique_changed_pairs": len(pending),
        "products_with_insufficient_reserved_capacity": 0 if drop_overlapping_test_offers else len(shortages),
        "split_overlap_checks_failed": sum(bool(r["shared_offers"] or r["shared_pairs"]) for r in checks),
        "row_issues": len(structural), "global_label_conflicts": len(global_conflicts),
    }
    manifest = {"version": "1.1-draft", "status": "DRAFT_NOT_FOR_EVALUATION",
                "ready_for_inference": not any(blockers.values()), "blockers": blockers,
                "source_root": str(source_root), "source_sha256": source_hashes,
                "candidate_sha256": output_hashes, "script_sha256": digest(Path(__file__)),
                "source_offers_with_split_specific_text": len(source_variants),
                "replacement_text_rule": "Reuse the destination split's existing offer text, consistently within each language and split type.",
                "decisions": decisions,
                "test_allocation_policy": "drop_overlapping_offers" if drop_overlapping_test_offers else "unresolved",
                "original_products_with_insufficient_reserved_capacity": len(shortages),
                "label_rule": "Changed pairs use exact existing test labels or explicit human decisions; other labels are null.",
                "hardness_rule": "is_hard_negative is inherited sampling metadata; changed text hardness is not revalidated.",
                "frozen": ("All training, Seen validation and Unseen test files remain byte-identical."
                           if drop_overlapping_test_offers else
                           "All training, Seen validation, Seen test and Unseen test files remain byte-identical."),
                "scope": "Half-Seen test and 050/100 validation aliases, all sizes and both languages."
                         + (" Additionally remove test rows containing offers reserved for training or Seen validation."
                            if drop_overlapping_test_offers else ""),
                "changed_rows_per_language": sum(r["changed_source_rows"] for r in stats),
                "removed_rows_per_language": sum(r["removed_rows"] for r in stats),
                "deduplicated_rows_per_language": sum(r["deduplicated_rows"] for r in stats),
                "known_label_changes_per_language": sum(r["known_label_changes"] for r in stats)}
    if any(digest(source_root / path) != value for path, value in source_hashes.items()):
        raise ValueError("Source changed during generation; discard this draft")
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary = ["# Version 1.1: replacement draft", "", "**Not an official release. Do not train or evaluate on these files.**", "",
               "Original v1.0 files and existing scores are unchanged. All training and Seen validation files are frozen.", "",
               "## Readiness", "", "```json", json.dumps(blockers, indent=2), "```", "",
               "## Modified test candidates", "",
               "| Family | Original pairs | Candidate pairs | Changed inputs | Known label changes | Unresolved labels |",
               "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in stats:
        if row["file"].startswith(TEST) and row["mutable"]:
            family = re.search(r"products(\d+cc\d+rnd\d+un)", row["file"])[1]
            summary.append(f"| {family} | {row['source_rows']} | {row['candidate_rows']} | {row['changed_source_rows']} | {row['known_label_changes']} | {row['unresolved_rows']} |")
    summary += ["", "Numbers are per language. Validation aliases are included in the full CSV, not in this test-only table.", "",
                "`review/offer_mapping.csv` records replacements; `pair_changes.csv` records every changed or consolidated source row.",
                "`label_provenance.jsonl` records original labels, exact-label sources and bilingual new records.",
                "`label_review.csv` contains unresolved unique pairs. Fill decision_label (0/1), reviewer and rationale; keep IDs and hashes.",
                "Run the generator again with --decisions PATH and a NEW --output-root. It refuses stale decisions and overwrites.", "",
                "`allocation_blockers.csv` records the original products with insufficient reserved capacity. Their treatment is recorded in manifest.test_allocation_policy.",
                "`removed_test_pairs.csv` records every excluded test row, including its original label and the reason for exclusion.",
                "`structural_checks.json` includes existing and remaining offer/pair overlaps across all splits within each family.",
                "Labels are never inferred solely from product IDs. Duplicate candidate pairs are consolidated and counted; no replacement pairs are invented.",
                "`source_text_variants.json` records source offer IDs whose existing text differs between split types; replacements reuse the destination split's version.",
                "Retained rows preserve unseen-product endpoints. Explicit exclusions can change product membership; effective hardness and class balance may change.", "",
                "The old results and prediction caches apply to v1.0 only. Reuse a prediction only when the complete serialized input is unchanged.",
                "Before inference: resolve labels and allocation blockers, validate all split/label checks, review hardness and statistics, then create a separately approved release."]
    (output_root / "README.md").write_text("\n".join(summary) + "\n")
    shutil.copy2(Path(__file__), output_root / "generator_snapshot.py")
    print(json.dumps({"output": str(output_root), "blockers": blockers}, indent=2))
    return manifest


def check_readiness(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if any(digest(root / path) != value for path, value in manifest["candidate_sha256"].items()):
        raise ValueError("Candidate files have changed; regenerate the draft")
    if any(digest(Path(manifest["source_root"]) / path) != value for path, value in manifest["source_sha256"].items()):
        raise ValueError("Official source changed since this draft")
    if not manifest["ready_for_inference"] or any(manifest["blockers"].values()):
        raise ValueError(f"Inference blocked: {manifest['blockers']}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("data/releases/v1.1-draft"))
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--drop-overlapping-test-offers", action="store_true",
                        help="Explicitly remove test pairs whose replaced endpoints occur in frozen training or Seen validation")
    parser.add_argument("--check-readiness", type=Path)
    args = parser.parse_args()
    try:
        if args.check_readiness:
            check_readiness(args.check_readiness)
        else:
            prepare(args.source_root, args.output_root, args.decisions, args.drop_overlapping_test_offers)
    except ValueError as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
