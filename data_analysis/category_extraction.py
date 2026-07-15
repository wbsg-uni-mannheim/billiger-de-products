import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from sklearn.utils import resample
from collections import Counter
from tqdm import tqdm

df = pd.read_pickle("data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name.pkl.gz")
df = df[["id", "product_id", "name", "desc", "brand", "shop_cat", "price"]]
print("Amount of distinct shop categories:", df["shop_cat"].nunique())


# ------------------------------------------------------------
# 0) VORAUSSETZUNGEN
# ------------------------------------------------------------

TOP_CATEGORIES = [
    "Auto & Motorrad",
    "Möbel & Wohnen",
    "Werkzeug & Baumarkt",
    "Elektronik & Computer",
    "Spielzeug & Baby",
    "Kleidung & Accessoires",
    "Kosmetik & Drogerie",
    "Lebensmittel & Getränke",
    "Gesundheit & Pflege",
    "Bücher, Filme & Musik",
    "Sport & Freizeit",
    "Haustier & Tierbedarf",
    "Bürobedarf"
]
# Heuristische Keyword-Regeln mit direkter, eindeutiger Zuordnung
# (falls ein Wort aus der Liste vorkommt → sofortige Zuordnung)
HEURISTIC_RULES = {
    "Auto & Motorrad": [
        "pkw", "motorradhelm", "neureifen", "autoreifen",
        "sommerreifen", "winterreifen", "allwetterreifen",
        "felge", "reifen", "kfz", "auto", "scheibenwischer",
        "motor", "motorrad", "scooter", "moped", "autoteil",
        "kennzeichen", "dachbox", "dachträger", "wagenheber", "motoröl",
    ],
    "Elektronik & Computer": [
        "gardine", "haushaltsgerät", "technik", "telefon","laptop", "smartphone", "fernseher", "tv", 
        "monitor", "kamera", "router", "konsole", "kühlschrank", "herd", "ofen", "computer", "elektronik", 
        "pc", "handy", "notebook", "handy", "tablet", "smartwatch", "bildschirm", "objektiv", "kopfhörer", 
        "router", "netzwerk", "tastatur", "konsole", "playstation", "xbox", "nintendo", "grafikkarte", "multimedia",
        "trockner", "waschmaschine", "spülmaschine", "küchenkleingerät", "küchengerät", "elektrogerät", 
        "digitalcamera", "digitalkamera"
    ],
    "Möbel & Wohnen": [
        "matratze", "bett", "sofa", "kommode", "lampe", "beleuchtung", "vorhang", "kissen", "teppich",  "wohnzimmer", "schlafzimmer", "zimmer", "wohnen", "möbel", "bad", "tisch", "stuhl", "regal", "schrank", "couch",
        "möbelzubehör", "pfanne", "geschirr", "topf", "töpfe", "esszimmer", "grill", "bettwäsche", "gardine", "decke", "decken", "küchenstuhl", "küchentisch",
        "gartenmöbel", "deko"
    ],
    "Werkzeug & Baumarkt": [
        "werkstatt", "akkuschrauber", "bohrer", "schraube",
        "säge", "schweißgerät", "dübel", "mörtel", "leiter",
        "werkzeug", "baumarkt", "hammer", "zange", "elektrowerkzeug", "gartenwerkzeug", "farbe", "pinsel", "spachtel", "schraubenzieher", "sägen", "bohrmaschine",
        "gartengeräte", "rasenmäher", "heckenschere", "gartenschlauch"
    ],
    "Spielzeug & Baby": [
        "lego", "puppe", "spielzeug", "kinderwagen",
        "wickel", "kuscheltier", "baustein", "puzzle", "spiele", "spielware",
        "toys", "games", "wasserspielzeug", "spielware"
    ],
    "Kleidung & Accessoires": [
        "t-shirt", "hose", "jacke", "kleid", "schuh", "sneaker",
        "unterwäsche", "schmuck", "tasche", "gürtel", "mütze",
        "accessoire", "mode", "kleidung", "rock", "jeans", "uhr", "rucksack", "taschen", "körperpflege", "radbekleidung",
        "sportbekleidung", "anzug", "bluse", "sportschuh", "damenschuh", "herrenschuh", "kinderschuh",
        "rucksäcke", "shoes"
    ],
    "Kosmetik & Drogerie": [
        "parfum", "deo", "shampoo", "seife", "make-up", "nagellack", "makeup", "kosmetik", "drogerie", "dusche", "rasur", "spülung", "seife", "zahnpasta",
        "kontaktlinsen", "creme", "lotion", "beauty", "handpflege", "körperpflege", "foundation", "haare",
        "gesichtspflege"
    ],
    "Lebensmittel & Getränke": [
        "kaffee", "spirituose", "tee", "bier", "wein", "snack", "essen", "getränk", "lebensmittel",
        "nahrung", "brot", "whyski", "haushaltsgerät", "kaffee", "tee"
    ],
    "Gesundheit & Pflege": [
        "vitamin", "arzneimittel", "medikament", "vitamin", "verband", "pflaster", "desinfektion", "gesundheit", "pflege", "apotheke", "thermometer", "gesundheit",
        "gesichtspflege", "körperpflege"
    ],
    "Bücher, Filme & Musik": [
        "buch", "film", "dvd", "blu-ray", "cd", "hörbuch", "zeitschrift", "musik", "schallplatte", "hörbuch", "hörspiel", 
    ],
    "Sport & Freizeit": [
        "fahrrad", "hantel", "yoga", "zelt", "angeln", "ski", "trampolin", "sport", "fitness", "laufen", "joggen", "tennis", "fußball"
    ],
    "Haustier & Tierbedarf": [
        "hundefutter", "katzenstreu", "kratzbaum", "hundespielzeug", "aquarium", "nager", "tier", "haustier", "hund", "katze", "fisch", "vogel", "käfig", "terrarium", "leckerli"
    ],
    "Bürobedarf": [
        "ordner", "hefter", "locher", "druckerpapier", "kugelschreiber", "post-it", "drucker", "scanner", "bürobedarf", "papier", "büro", "stift", "kalender"
    ],
    "Sonstige": [
    ],
    "Undefiniert": [
    ]
}

# Pseudo-Kategorien, die oft als oberste Ebene auftauchen und übersprungen werden sollen
STOPWORDS_FIRST_LEVEL = {
    "damen", "herren", "kinder", "mädchen", "jungen", "baby", "babys",
    "sale", "angebote", "neuheiten", "marke", "brand", "top", "neu"
}

# ------------------------------------------------------------
# 1) HEURISTISCHE ZUORDNUNG
# ------------------------------------------------------------
def normalize_text(s: str) -> str:
    s = s.lower()
    # vereinheitliche Trennzeichen zu ">"
    s = re.sub(r"[|/:\\>]+", ">", s)
    # entferne Sonderzeichen, mehrfachspaces
    s = re.sub(r"[^0-9a-zäöüß><\s\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
    
def tokenize(text: str):
    return set(text.lower().split())

def match_with_suffix(word, tokens):
    suffixes = ["", "e", "en", "n", "s", "er"]
    for suf in suffixes:
        if word + suf in tokens:
            return True
    return False
    
def extract_all_meaningful_segments(cat): 
    """ Gibt eine Liste aller Segmente zurück, überspringt dabei Stopwords. """ 
    if not isinstance(cat, str) or not cat.strip(): 
        return None
        
    s = normalize_text(cat) 
        
    parts = [p.strip() for p in s.split(">") if p.strip()] 
    if not parts: 
        return None

    return [p for p in parts if p not in STOPWORDS_FIRST_LEVEL] or [parts[0]]


def heuristic_match(cat):
    """
    Nimmt alle Segmente einer Kategorie, prüft gegen HEURISTIC_RULES.
    Gibt die Kategorie mit den meisten Treffern zurück.
    """
    segments = extract_all_meaningful_segments(cat)
    if segments is None:
        return "Undefiniert"
    else:
        tokens = tokenize(" ".join(segments))
        
        counts = Counter()
        for cat_name, keys in HEURISTIC_RULES.items():
            for k in keys:
                if match_with_suffix(k, tokens):
                    counts[cat_name] += 1
        
        if not counts:
            return "Sonstige"
        
        return counts.most_common(1)[0][0]

# ------------------------------------------------------------
# 2) HAUPTFUNKTION: MAPPING PIPELINE
# ------------------------------------------------------------
def map_shop_categories(df: pd.DataFrame, col: str = "shop_cat") -> pd.DataFrame:
    out = df.copy()
    tqdm.pandas(desc="Prozessiere Kategorien")
    out["top_category_mapped"] = out["shop_cat"].progress_apply(heuristic_match)
    
    return out

# ------------------------------------------------------------
# 3) REPORTING / EXPORT
# ------------------------------------------------------------
def summarize_and_export(out_df, name, original_col= "shop_cat"):
    # Reduktionsübersicht
    n_original = out_df[original_col].nunique(dropna=True)

    print(f"Einzigartige Original-Kategorien: {n_original}")
    print()
    print("Top 20 Mapped Kategorien (Count):")
    print(out_df["top_category_mapped"].value_counts())
    print(out_df.head())

    # Speichere Mapping (ein Datensatz je Zeile, inkl. Score & Quelle)
    mapping = (
        out_df[[original_col,"top_category_mapped"]] # "tfidf_score", ,"heuristic_hit"
        .copy()
    )
    mapping.to_csv(f"notebooks/Categories/category_mapping_{name}.csv", index=False)

    # Häufigkeiten der Zielkategorien
    counts = out_df["top_category_mapped"].value_counts().rename_axis("top_category").reset_index(name="count")
    counts.to_csv(f"notebooks/Categories/top_category_counts_{name}.csv", index=False)

    return mapping, counts

# ------------------------------------------------------------
# 4) BEISPIEL-DURCHLAUF (auskommentiert, damit du es selbst steuerst)
# ------------------------------------------------------------
out_df = map_shop_categories(df, col="shop_cat")
mapping, counts = summarize_and_export(out_df, name="initial_mapping", original_col="shop_cat")


# ------------------------------------------------------------
# 5) In pickle umwandeln für spätere Sets
# ------------------------------------------------------------
#out_df.to_pickle("../data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_title_only_mainentity_with_new_category.pkl.gz", compression="gzip")

print("Amount of distinct shop categories for top category mapped Snstige:", out_df[out_df["top_category_mapped"]=="Sonstige"]["shop_cat"].nunique())

import openai
import os; API_key = os.environ.get("OPENAI_API_KEY")  # set OPENAI_API_KEY env var (see REPRODUCTION.md)
import time
import re
from tqdm import tqdm

def normalize(text):
    return re.sub(r"[^a-z0-9]", " ", text.lower())
# Function to call OpenAI API for entity matching
def category_matching(api_key, product, shop_cat, category_list):
    if shop_cat == "Unbekannt":
        prompt = (
            f"Ordne das folgende Produkt einer passenden Oberkategorie zu.\n"
            f"Der Produktname: {product}\n"
            f"Verfügbare Kategorien: {category_list}\n"
            f"Antworte **ausschließlich** mit genau einem dieser Begriffe, "
            f"ohne zusätzliche Erklärungen oder Alternativen."
        )
    else:
        prompt = (
            f"Ein Produkt gehört laut Shop zur Kategorie '{shop_cat}'. "
            f"Ordne diese Shop-Kategorie einer passenden Oberkategorie zu.\n"
            f"Verfügbare Oberkategorien: {category_list}\n"
            f"Antworte **nur** mit genau einem dieser Begriffe. Keine Erklärungen oder Begründungen."
        )
    openai.api_key = api_key

    response = openai.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    return response.choices[0].message.content

# Function to safely call the LLM with retries
def safe_category_match(api_key, shop_cat, product, category_list, retries=5, delay=5):
    for attempt in range(1, retries + 1):
        try:
            result = category_matching(api_key, product, shop_cat, category_list)
            # Validate structure
            if result is None and attempt < retries:
                time.sleep(delay)
                continue
            elif result is None and attempt >= retries:
                raise ValueError("Empty result from LLM.")
            
            answer = result.strip().lower() # type: ignore
            # Validate that the answer is one of the expected outputs
            normalized_answer = normalize(answer)
            if any(normalize(category) in normalized_answer for category in category_list):
                return result
            else:
                print(f"[Attempt {attempt}] Invalid answer: '{answer}'")
                if attempt < retries:
                    time.sleep(delay)
                else:
                    raise ValueError("Invalid response from LLM after multiple attempts.")
        except (openai.APIError, openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            print(f"[Attempt {attempt}] OpenAI API error: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                print(" All retries failed. Exiting.")
                raise e  # or handle how you'd like
            
# Run LLM for all items with top mapped category "Sonstige" or "Undefiniert"
try:
    # 1️⃣ Filter: Nur "Sonstige" oder "Undefiniert"
    sonstige_df = out_df[
        (out_df["top_category_mapped"] == "Sonstige") |
        (out_df["top_category_mapped"] == "Undefiniert")
    ].copy()

    # Cache für Ergebnisse
    category_cache = {}

    # Ersetze NaN in shop_cat vorerst durch Platzhalter
    sonstige_df["shop_cat"] = sonstige_df["shop_cat"].fillna("Unbekannt")

    # 2️⃣ Hauptloop: gruppiere nach shop_cat
    for shop_cat, group in tqdm(
        sonstige_df.groupby("shop_cat"),
        desc="LLM Kategorisierung (per shop_cat)"
    ):
        try:
            # Nur ausführen, wenn shop_cat bekannt (nicht Unbekannt)
            if shop_cat != "Unbekannt":
                # Nimm ein Beispielprodukt aus der Gruppe
                product_example = group["name"].iloc[0]
                mapped_category = safe_category_match(API_key, shop_cat, product_example, TOP_CATEGORIES)
                print("Shop-cat:", shop_cat, "→was Mapped to:", mapped_category)
                category_cache[shop_cat] = mapped_category
                sonstige_df.loc[group.index, "top_category_mapped"] = mapped_category
        except Exception as e_inner:
            print(f"[Shop-Cat '{shop_cat}'] Error: {e_inner}")
            sonstige_df.loc[group.index, "top_category_mapped"] = group["top_category_mapped"]

    # Update ins Haupt-DataFrame
    out_df.loc[sonstige_df.index, "top_category_mapped"] = sonstige_df["top_category_mapped"].values

    # 3️⃣ Extra-Loop: Alle Zeilen ohne shop_cat (NaN oder 'Unbekannt')
    nan_df = out_df[out_df["shop_cat"].isna()].copy()
    print(f"\nNaN-Fälle gefunden: {len(nan_df)}")

    if len(nan_df) > 0:
        results = {}
        for idx, row in tqdm(
            nan_df.iterrows(),
            total=len(nan_df),
            desc="LLM Kategorisierung (NaN shop_cat)"
        ):
            try:
                mapped = safe_category_match(API_key, "Unbekannt", row["name"], TOP_CATEGORIES)
            except Exception as e_nan:
                print(f"[Index {idx}] Error bei NaN-Kategorisierung: {e_nan}")
                mapped = "Sonstige"
            results[idx] = mapped

        # Ergebnisse zurück in DataFrame schreiben
        out_df.loc[nan_df.index, "top_category_mapped"] = [results[i] for i in nan_df.index]

    print("\n✅ LLM-Kategorisierung abgeschlossen!")

except Exception as e:
    print(f"\n❌ Script interrupted due to error:\n{e}\n")
    out_df.loc[nan_df.index, "top_category_mapped"] = [results[i] for i in nan_df.index]
    out_df.to_pickle("data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category_partial.pkl.gz", compression="gzip")

# Final summary after LLM mapping
mapping, counts = summarize_and_export(out_df, name="after_llm_mapping", original_col="shop_cat")


out_df.to_pickle("data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz", compression="gzip")