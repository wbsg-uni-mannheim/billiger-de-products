"""Checks for label preservation and fail-closed draft generation."""

import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.processing.prepare_v11_draft import (
    FAMILIES,
    TRAIN,
    VALID,
    TEST,
    allocate,
    check_readiness,
    collapse_duplicates,
    digest,
    exact_test_evidence,
    paths_for,
    prepare,
    read_decisions,
    read_rows,
    replace_row,
    resolve_label,
    write_rows,
)


class ReplacementTests(unittest.TestCase):
    def test_explicit_exclusion_removes_mapped_overlap_before_review_and_preserves_frozen_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "draft"
            def pair(a, b, offset):
                def product(offer):
                    return offset + (100 if offer in (1, 2, 3, 5, 6) else 200)
                return {"pair_id": f"{offset+a}#{offset+b}", "id_left": offset+a,
                        "id_right": offset+b, "product_id_left": product(a),
                        "product_id_right": product(b), "label": int(product(a) == product(b)),
                        "is_hard_negative": False}
            for lang in ("de", "en"):
                for number, family in enumerate(FAMILIES):
                    offset = number * 1000
                    base = source / f"solute_{lang}"
                    for size in ("small", "medium", "large"):
                        write_rows(base / paths_for(family, TRAIN, size=size), [pair(1, 6, offset)])
                        write_rows(base / paths_for(family, VALID, size=size), [pair(2, 5, offset)])
                        for condition in ("050", "100"):
                            write_rows(base / paths_for(family, VALID, condition, size),
                                       [pair(1, 9, offset), pair(2, 9, offset)])
                    write_rows(base / paths_for(family, TEST), [pair(2, 3, offset)])
                    write_rows(base / paths_for(family, TEST, "050"),
                               [pair(1, 4, offset), pair(3, 4, offset)])
                    write_rows(base / paths_for(family, TEST, "100"), [pair(4, 10, offset)])
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare(source, output, drop_overlapping_test_offers=True)
                unchanged_policy = prepare(source, Path(directory) / "without_exclusions")
            self.assertEqual(unchanged_policy["removed_rows_per_language"], 0)
            self.assertEqual(unchanged_policy["blockers"]["split_overlap_checks_failed"], 3)
            self.assertEqual(manifest["removed_rows_per_language"], 6)
            self.assertEqual(manifest["blockers"]["split_overlap_checks_failed"], 0)
            self.assertEqual(manifest["blockers"]["products_with_insufficient_reserved_capacity"], 0)
            self.assertFalse(manifest["ready_for_inference"])
            for lang in ("de", "en"):
                for number, family in enumerate(FAMILIES):
                    base = output / "candidate_sets" / f"solute_{lang}"
                    self.assertEqual(read_rows(base / paths_for(family, TEST)), [])
                    self.assertEqual(read_rows(base / paths_for(family, TEST, "050")),
                                     [pair(3, 4, number * 1000)])
                    for relative in [paths_for(family, TEST, "100")] + [
                        paths_for(family, folder, size=size)
                        for folder in (TRAIN, VALID) for size in ("small", "medium", "large")
                    ]:
                        self.assertEqual(digest(base / relative), digest(source / f"solute_{lang}" / relative))
            with (output / "review/removed_test_pairs.csv").open() as stream:
                removed = list(csv.DictReader(stream))
            self.assertEqual(len(removed), 6)
            self.assertTrue(all(r["label_source"] == "removed_before_label_review" for r in removed))
            provenance = [json.loads(line) for line in
                          (output / "review/label_provenance.jsonl").read_text().splitlines()]
            self.assertFalse(any(item["pair_id"] in ("2#4", "1002#1004", "2002#2004") for item in provenance))

    def test_keep_shared_offer_and_replace_other_slot(self):
        self.assertEqual(allocate({10, 20}, {20, 30}), {20: 20, 10: 30})
        self.assertEqual(allocate({10, 20}, {30, 40}), {10: 30, 20: 40})

    def test_insufficient_reserved_offers_cannot_collapse_a_pair(self):
        with self.assertRaises(ValueError):
            allocate({10, 20}, {30})

    def test_replacement_uses_all_new_attributes_and_preserves_other_endpoint(self):
        old = {"id_left": 10, "product_id_left": 100, "name_left": "two tires",
               "price_left": 20, "id_right": 20, "product_id_right": 200,
               "name_right": "unseen", "price_right": 30,
               "pair_id": "10#20", "label": 0}
        registry = {30: {"id": 30, "product_id": 100, "name": "four tires", "price": 40}}
        new = replace_row(old, {10: 30}, registry)
        self.assertEqual(new["pair_id"], "30#20")
        self.assertEqual(new["name_left"], "four tires")
        self.assertEqual(new["price_left"], 40)
        self.assertEqual({k: v for k, v in new.items() if k.endswith("_right")},
                         {k: v for k, v in old.items() if k.endswith("_right")})
        self.assertEqual(old["name_left"], "two tires")
        registry[30]["product_id"] = 999
        with self.assertRaises(ValueError):
            replace_row(old, {10: 30}, registry)

    def test_exact_new_pair_label_overrides_old_product_group_assumption(self):
        loaded = {Path("gold-standards_adjusted/seen.json.gz"):
                  [{"id_left": 40, "id_right": 30, "label": 1}],
                  Path("training-sets/train.json.gz"):
                  [{"id_left": 50, "id_right": 60, "label": 0}]}
        evidence, _ = exact_test_evidence(loaded)
        self.assertEqual(resolve_label((30, 40), "hash", evidence, {}),
                         (1, "existing_exact_test_pair"))
        self.assertEqual(resolve_label((50, 60), "hash", evidence, {}),
                         (None, "needs_human_review"))

    def test_conflicting_existing_labels_are_not_silently_selected(self):
        self.assertEqual(resolve_label((30, 40), "hash", {(30, 40): {0, 1}}, {}),
                         (None, "conflicting_existing_test_labels"))

    def test_human_decisions_are_bound_to_new_bilingual_content(self):
        decisions = {"30#40:hash": {"label": 0, "content_sha256": "hash"}}
        self.assertEqual(resolve_label((30, 40), "hash", {}, decisions), (0, "human_review"))
        with self.assertRaisesRegex(ValueError, "Stale"):
            resolve_label((30, 40), "hash", {},
                          {"30#40:hash": {"label": 0, "content_sha256": "different"}})
        with self.assertRaisesRegex(ValueError, "contradicts"):
            resolve_label((30, 40), "hash", {(30, 40): {1}}, decisions)

    def test_duplicate_reversed_pairs_are_logged_and_conflicts_remain_blockers(self):
        rows = [{"id_left": 1, "id_right": 2, "label": 1},
                {"id_left": 2, "id_right": 1, "label": 1},
                {"id_left": 2, "id_right": 1, "label": 0}]
        kept, duplicates, conflicts = collapse_duplicates(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(duplicates, {1: 0, 2: 0})
        self.assertEqual(conflicts, ["1#2"])

    def test_refuses_to_overwrite_existing_output_before_reading_source(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(Path("/does/not/exist"), Path(root))

    def test_missing_reviewer_is_not_an_approved_label(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "labels.csv"
            with p.open("w", newline="") as stream:
                w = csv.DictWriter(stream, fieldnames=["review_id", "decision_label", "reviewer", "rationale"])
                w.writeheader()
                w.writerow({"review_id": "1#2", "decision_label": "1", "reviewer": "", "rationale": "same item"})
            with self.assertRaises(ValueError):
                read_decisions(p)

    def test_draft_blockers_and_changed_files_prevent_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "pairs.json.gz"
            write_rows(output, [{"label": None}])
            manifest = {"candidate_sha256": {output.name: digest(output)},
                        "source_sha256": {}, "source_root": str(root),
                        "ready_for_inference": False, "blockers": {"pending": 1}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Inference blocked"):
                check_readiness(root)
            write_rows(output, [{"label": 1}])
            with self.assertRaisesRegex(ValueError, "Candidate files have changed"):
                check_readiness(root)


if __name__ == "__main__":
    unittest.main()
