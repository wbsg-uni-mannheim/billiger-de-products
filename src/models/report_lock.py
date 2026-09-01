"""Guard against two jobs appending into the same result table.

``run_wordcooc.run_wordcooc`` and ``run_magellan.run_magellan`` truncate their
report file once and then append one line per (run, classifier).  Two jobs
started against the same output path therefore interleave their rows, and the
second truncation silently drops whatever the first job had already written.
That is how ``results/generated/cross_language/wordcooc`` ended up with six
DE-DE runs per classifier -- five for NaiveBayes -- while the other four
language variants had three.

The lock is advisory and whole-file: a sequential rerun still truncates and
rewrites the table as before, but a concurrent one fails loudly instead of
producing a table whose row count no longer matches the number of runs.
"""

import fcntl
import os


def acquire_report_lock(report_path):
    """Take an exclusive lock for ``report_path``; raise if one is held."""
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    handle = open(f"{report_path}.lock", "w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            f"Another job is already writing {report_path}. Refusing to append "
            "to the same result table; give this run its own output directory."
        ) from error
    return handle


def release_report_lock(handle):
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
