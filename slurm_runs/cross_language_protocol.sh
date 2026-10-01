# Single definition of the cross-language selection protocol.
#
# Every cross-language job trains on the German 80cc20rnd000un large training
# split and selects its checkpoint or hyper-parameters on the German
# 80cc20rnd000un large *validation* split -- the very same file the main
# benchmark grid validates on (run_confidence_test.sh,
# run_finetune_siamese.sh, all_runs_de.py, hier_de.sh).  Selecting on the
# 80cc20rnd050un validation split instead is what made the DE-DE column of the
# cross-language table disagree with the 80cc20 / Large / Half-Seen cell of the
# main table; see reports/table3_table6_consistency.md.
#
# All four neural matchers fine-tune roberta-base.

source slurm_runs/env.sh

SELECTION_VARIANT="products80cc20rnd000un"
TRAIN_VARIANT="products80cc20rnd000un"
TRAIN_SIZE="large"
BACKBONE="roberta-base"
# Overridable so a single collapsed seed can be rerun without redoing the cell:
#   SEEDS=0 sbatch slurm_runs/cross_language_ditto.sh
SEEDS="${SEEDS:-0 1 2}"

TRAIN_PKL="data/processed/training-sets/preprocessed_${TRAIN_VARIANT}_train_${TRAIN_SIZE}.pkl.gz"
SELECTION_VALIDATION_PKL="data/processed/validation-sets/preprocessed_${SELECTION_VARIANT}_valid_${TRAIN_SIZE}.pkl.gz"
SELECTION_VALIDATION_DITTO="data/processed/ditto/data/final_output/preprocessed_${SELECTION_VARIANT}_valid_${TRAIN_SIZE}.txt"
SELECTION_VALIDATION_HIERGAT="data/processed/hiergat/data/final_output/preprocessed_${SELECTION_VARIANT}_valid_${TRAIN_SIZE}.txt"
TRAIN_WORDCOOC="data/processed_cross_language/wordcooc/preprocessed_${TRAIN_VARIANT}_train_${TRAIN_SIZE}_wordcooc.pkl.gz"
SELECTION_VALIDATION_WORDCOOC="data/processed_cross_language/wordcooc/preprocessed_${SELECTION_VARIANT}_valid_${TRAIN_SIZE}_wordcooc.pkl.gz"

TEST_PKL="data/processed/gold-standards_adjusted/preprocessed_products80cc20rnd050un_gs.pkl.gz"
CROSS_TEST_PAIR_DIR="data/processed_cross_language/gold-standards_adjusted"
CROSS_TEST_DITTO_DIR="data/processed_cross_language/ditto/data/final_output"
CROSS_TEST_HIERGAT_DIR="data/processed_cross_language/hiergat/data/final_output"

# The main RoBERTa grid was rerun at batch size 32 (commit 068a3fc); the
# cross-language run has to use the same batch size or its DE-DE column is
# again a different experiment from the cell it is compared against.
ROBERTA_TRAIN_BATCH_SIZE=32
RSUPCON_TRAIN_BATCH_SIZE=64
DITTO_TRAIN_BATCH_SIZE=64
HIERGAT_TRAIN_BATCH_SIZE=16

# Record which interpreter produced the results; a Ditto rerun that silently
# used a second conda environment is one of the differences this report had to
# reconstruct after the fact.
echo "[protocol] python=$(command -v python) host=$(hostname) date=$(date)"
echo "[protocol] selection_validation=$SELECTION_VALIDATION_PKL backbone=$BACKBONE seeds=$SEEDS"
