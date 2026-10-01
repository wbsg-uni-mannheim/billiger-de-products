import gzip
import json
import fasttext
import os
from gensim.parsing.preprocessing import lower_to_unicode, preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric


model = fasttext.load_model("fasttext/lid.176.bin")
LANGUEAGE_FIELDS_left = ["name_left", "desc_left"]
LANGUEAGE_FIELDS_right = ["name_right", "desc_right"]

def assigned_language(text,language):
    
    processed_text = lower_to_unicode(text)
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric]
    
    processed_text = ' '.join(preprocess_string(processed_text, CUSTOM_FILTERS))
    
    if processed_text.strip() == '':
        return False
    
    labels, probs = model.predict(processed_text, k=3)
    
    langs = [l.replace("__label__", "") for l in labels]
    #map labels and probs
    lang_prob = dict(zip(langs, probs))

    if language in langs:
        if "de" in langs and lang_prob["de"] > 0.5:    
            return True
    return False




non_german_datasets = []
dataset_paths = ["training-sets", "validation-sets", "gold-standards_adjusted", "gold-standards"]
base_path = "data/derived/"
ids_not_translated = set()
for folder in dataset_paths:
    for file in os.listdir(os.path.join(base_path, folder)):
        if file.endswith(".json.gz") and not file.__contains__("multi"):
            print(f"Checking {file} in folder {folder}...")
            with gzip.open(os.path.join(base_path, folder, file), "rt", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    item = json.loads(line)

                    # name und desc left und name und desc right einzeln prüfen, ob sie überwiegend deutsch sind

                    # alle String-Felder zusammenfassen
                    text_left = " ".join(
                        item[k] for k in LANGUEAGE_FIELDS_left
                        if k in item and isinstance(item[k], str)
                    )
                    text_right = " ".join(
                        item[k] for k in LANGUEAGE_FIELDS_right
                        if k in item and isinstance(item[k], str)
                    )

                    if not assigned_language(text_left, language = "de"):
                        if file not in non_german_datasets:
                            if file not in non_german_datasets:
                                non_german_datasets.append(file)
                            ids_not_translated.add(item.get("id_left"))
                    if not assigned_language(text_right, language = "de"):
                        if file not in non_german_datasets:
                            if file not in non_german_datasets:
                                non_german_datasets.append(file)
                            ids_not_translated.add(item.get("id_right"))
                                
print("Datasets that are not mostly German:", non_german_datasets)
print("IDs that were not translated:", ids_not_translated)