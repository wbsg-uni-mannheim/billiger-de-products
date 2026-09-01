# Paper changes required by the Table 3 / Table 6 realignment

Companion to [`table3_table6_consistency.md`](table3_table6_consistency.md).

The LaTeX sources (`BTW-German-WDC-working-title-/`) are **not present in this
checkout** — this repository contains only code, data and results. The list below
is therefore by section and by claim; the file paths are the ones named in the
task. The project website `website/index.html` carries the same tables and needs
the same edits.

Numbers marked **[pending]** come from jobs that are still running
(§7 of the consistency report); everything else is final.

---

## 1. `sections/06_cross_language.tex`

### 1.1 The "4,436 validation pairs" figure — keep the number, fix the claim

4,436 is correct for the released `products80cc20rnd050un_valid_large` split, and
the deduplication step removes nothing (it is byte-identical to the released
file). What is wrong is the accompanying statement that this is the split the
experiment selects on.

* **Delete** any wording that attributes 4,436 to a deduplication step. It is the
  released split, unmodified.
* **Replace** the selection-split sentence. The cross-language runs now select on
  the German `products80cc20rnd000un` `large` validation split, **4,452 pairs**,
  seen-product share 1.0 — the same file the main benchmark grid validates on.
* If §5.1 quotes 4,436 as *the* validation size for the benchmark generally, that
  is wrong for every reported number in both tables: Table 3 has always used the
  4,452-pair `000un` split.

### 1.2 The three-seeds claim

Section 3 and 5.1 say three seeds. That was true for RoBERTa, R-SupCon and
HierGAT and false for Ditto in **both** tables:

* Table 3's Ditto cell is the mean over run_ids **0, 3, 4**; seeds 1 and 2
  collapsed (32.52 and 20.00 F1) and were replaced without documentation. The
  mean over the documented seeds 0, 1, 2 is 40.53, not 64.10.
* The old Table 6's Ditto column was a **two-run** mean (`runs=2` in
  `summary.csv`) over a stale summary; six Ditto result files with run_ids
  1, 2, 3, 4, 5, 7 existed on disk and no seed 0 ever converged.

The reruns use seeds 0, 1, 2 and report all three, with no replacement of a
collapsed seed. Either:

* the seed claim stands and the Ditto cells are replaced by the seeds-0,1,2 means
  from the rerun **[pending]**; or
* if a seed still collapses and you choose to substitute it, the substitution has
  to be stated in the text, with the run_ids used.

Also worth a sentence: WordCooc and Magellan do not use training seeds at all;
they repeat their `RandomizedSearchCV` three times with `random_state` 1, 2, 3.
"Three seeds" currently reads as if it covers them too.

### 1.3 The selection-split description

The current text (mirrored in `REPRODUCTION.md` §5) says the cross-language
experiment "selects checkpoints or hyperparameters on the German DE-DE
`80cc20rnd050un` large validation split". After the realignment that is
`80cc20rnd000un`. Update it, and see §3 below for the wider inconsistency this
exposes.

### 1.4 Table 6 cells

The whole table is regenerated from
`results/generated/cross_language/summary.csv` once the jobs finish **[pending]**.
Every supervised row changes, not only DE-DE, because the selected checkpoint
changes for all five language variants at once. The GPT-5.2 rows are unaffected
(zero-shot, no selection).

Two rows need a second look beyond the numbers:

* **WordCooc** — Table 3 reports LinearSVC (43.16), the old Table 6 reported
  LogisticRegression (43.61). If the tables report "the best classifier", say so
  and name it per cell; otherwise fix one classifier for the row.
* **Magellan** — both tables report XGBoost; worth naming it explicitly for the
  same reason.

### 1.5 Prose that quotes specific values

`website/index.html` contains at least: *"WordCooc drops from 43.6 F1 on DE-DE to
28.2 on EN-EN"* and *"HierGAT reaches its best score on the German pairs it …
to around 49 F1"*. Any equivalent sentences in the LaTeX have to be recomputed
from the new summary **[pending]**.

---

## 2. `sections/04_matching_systems.tex`

### 2.1 Name the HierGAT backbone

HierGAT is described without naming its pretrained encoder. It fine-tunes
**`roberta-base`** — `--lm roberta` in `src/models/hiergat/all_runs_de.py` and in
`slurm_runs/cross_language_hiergat.sh`, mapped to `roberta-base` at
`src/models/hiergat/model/model.py:23`. The same holds for Ditto
(`src/models/ditto/ditto_light/ditto.py:18`), so the natural fix is one sentence
covering all four neural matchers:

> RoBERTa, R-SupCon, Ditto and HierGAT all fine-tune `roberta-base`.

The `--lm` argparse default in the HierGAT and Ditto trainers used to be `'bert'`
even though every driver overrode it; it is now `'roberta'`, so the code and the
paper agree without relying on a command-line flag.

### 2.2 Batch sizes

If §4 lists hyper-parameters, note that the RoBERTa baseline is
`per_device_train_batch_size=32` after commit `068a3fc`, not 1024. The
cross-language RoBERTa job now uses 32 as well; before this change it still used
1024, which was a second reason its DE-DE number differed from Table 3's cell.

---

## 3. `REPRODUCTION.md` §2 and §5 — and the open methodological question

This is not a cosmetic edit and it needs your decision.

`REPRODUCTION.md` §2 currently says:

> Both `050un` and `100un` experiments use a 50%-seen validation split; `100un`
> refers to the fully unseen test condition, not to model selection.
> […] For supervised matchers, train on the corresponding German or English
> `80cc20rnd000un` training file and validate on the same language's
> `80cc20rnd050un` validation file.

**No benchmark run has ever done this.** The main grid trains one model per
(corner-case ratio, size) cell, selects it on the `000un` validation split, and
reads the Seen, Half-Seen and Unseen test columns off that single checkpoint. The
documented design would need two training runs per cell.

Two ways out:

* **(a) Correct the documentation** — state that model selection uses the
  `000un` validation split for all three test conditions. This is what the code
  does, it is what both tables will now report, and it costs nothing. It does
  mean conceding that the Half-Seen and Unseen columns are selected on a
  100 %-seen validation split.
* **(b) Correct the experiment** — rerun the main grid selecting the Half-Seen
  and Unseen columns on the `050un` split, as the text claims. This is the
  methodologically better choice: option (a) lets model selection see a
  distribution the test condition does not have. It costs 27 German + 27 English
  variants, plus a second training run per cell if the Seen column keeps its own
  `000un`-selected checkpoint.

The realignment in this branch implements **(a)** for the cross-language run,
because that was the instructed direction and the cheaper one. **(b) has not been
started and will not be without your go-ahead.**

---

## 4. Table 3 cells that need attention regardless

Two Table 3 cells do not reproduce from the checked-in outputs. Neither is caused
by the cross-language work, and neither is fixed by this branch.

| Cell | Published | Reproduced from repository | Note |
|---|---:|---:|---|
| HierGAT, 80cc20 / Large / Half-Seen | 65.40 | 66.13 ± 0.56 | *all nine* German HierGAT cells are close to but never equal to the checked-in `.txt` outputs, and `50cc50rnd000un/large` has only 2 of 3 seed files on disk — the reference outputs are a different snapshot from the one Table 3 was built from |
| Ditto, 80cc20 / Large / Half-Seen | 64.10 | 64.11 over run_ids 0, 3, 4; 40.53 over seeds 0, 1, 2 | undocumented seed replacement, see §1.2 |

RoBERTa, R-SupCon, WordCooc and Magellan reproduce their Table 3 cells exactly.
Reproduce with:

```bash
/home/aasteine/miniconda3/envs/ditto-modern/bin/python reports/table3_table6_consistency_checks.py
```

The HierGAT gap needs either the original run artifacts or a rerun of the German
HierGAT grid before Table 3 can be called reproducible.
