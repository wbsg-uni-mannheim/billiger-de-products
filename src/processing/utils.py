"""Text normalization shared by benchmark preprocessing."""

from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_multiple_whitespaces,
    strip_tags,
)


CUSTOM_FILTERS = (strip_tags, strip_multiple_whitespaces)


def clean_string_2020(value):
    if not value:
        return None
    return " ".join(preprocess_string(value, CUSTOM_FILTERS))
