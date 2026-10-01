"""Deterministic check that language-invariant, identity-relevant tokens of every German offer
survive in its English translation.

Three token classes are extracted from the German name and description:
  codes    : tokens that mix letters and digits (model numbers, article codes such as 380738-0, AF1, UniversalLevel2)
             and digit-only tokens of at least four digits
  measures : numbers followed by a unit (1,1 m, 100x145x160 cm, 500 GB, 42,5) with decimal comma normalised
  numbers  : every number in the text, as the most lenient class
A token is preserved if its normalised form occurs anywhere in the English name or description
(units are mapped to their English equivalents, decimal commas to points). Rates are reported per
class and per field, and every miss is written to reports/translation_audit/invariant_misses.csv
for manual inspection.

    python src/translation_audit/invariants.py
"""
import csv, html, re, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from offers import aligned_offers

UNIT_MAP = {"u/min": ["rpm", "min-1", "u/min", "/min", "min⁻¹"], "upm": ["rpm"], "zoll": ["inch", "inches", "in", '"'], "stück": ["pcs", "pieces", "piece", "pc", "units"], "stk": ["pcs", "pieces", "piece", "pc"],
            "st": ["pcs", "pieces", "piece", "pc", "st"], "tlg": ["pcs", "pc", "pc.", "piece", "pieces", "-piece", "-pc", "part", "parts"], "teilig": ["piece", "pieces", "-piece", "part", "parts"],
            "std": ["h", "hours", "hrs", "hour"], "min": ["min", "minutes"], "sek": ["s", "sec", "seconds"], "ltr": ["l", "litre", "liter", "litres", "liters"], "l": ["l", "litre", "liter", "litres", "liters"]}
UNITS = r"(u/min|upm|mm|cm|dm|m|km|g|kg|mg|l|ltr|ml|cl|gb|tb|mb|kb|w|kw|v|mv|ah|mah|wh|kwh|hz|ghz|mhz|khz|bar|zoll|\"|″|%|stück|stk|st|tlg|teilig|std|min|sek|db|lm|k|nm|px|dpi|ppi|fps|mp|cd|lux|ohm|a|ma|ps|kw|rpm|u/min|kcal|kj|°c|°)"
RE_MEASURE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?(?:\s?[x×]\s?\d+(?:[.,]\d+)?)*(?:\s?-\s?\d+(?:[.,]\d+)?)?)\s?-?" + UNITS + r"(?![a-zäöü])", re.I)
RE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9./\-]*[A-Za-z0-9]|[A-Za-z0-9]")
def code_tokens(text):
    """Tokens that mix letters and digits, or digit-only tokens of at least four digits, excluding
    measurements, dimension strings and the literal unicode escapes that occur in some records."""
    out = []
    for m in RE_TOKEN.finditer(unescape(text)):
        tok = m.group(0)
        has_digit, has_alpha = any(c.isdigit() for c in tok), any(c.isalpha() for c in tok)
        tok = re.sub(r"(?<=\d)[a-zäöüß]{3,}$", "", tok)          # 5.0Farbe -> 5.0 (missing space in the source)
        if len(tok) < 3:
            continue
        if re.fullmatch(r"\d+(?:[.,]\d+)?(?:\s?-\s?\d+(?:[.,]\d+)?)?-?(" + UNITS[1:-1] + r"|fach|er|tlg|teilig|pack|paar)\.?", tok, re.I):
            continue                                                 # 3-tlg, 1fach, 15-36Kg, 2er
        if not ((has_digit and has_alpha) or (has_digit and not has_alpha and len(re.sub(r"\D", "", tok)) >= 4)):
            continue
        if re.fullmatch(r"\d+(?:[.,]\d+)?(?:[x×]\d+(?:[.,]\d+)?)*", tok):   # dimension / plain number
            continue
        letters = re.sub(r"[^A-Za-zÄÖÜäöüß]", "", tok)
        if letters and not (letters.isupper() or len(letters) <= 2):       # 5-Punkt-Gurt, 4-seitigen, IP68Kompatibel: words, not identifiers
            continue
        if re.fullmatch(r"\d+(?:[.,]\d+)?" + UNITS, tok, re.I):                 # attached unit, e.g. 500GB, 18cm
            continue
        out.append(tok)
    return out
RE_NUMBER = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(?![\w])")


def unescape(s):
    s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)
    s = re.sub(r"(?<![A-Za-z])u([0-9a-fA-F]{4})(?![0-9a-zA-Z])", lambda m: chr(int(m.group(1), 16)) if m.group(1)[:2] in ("00", "20", "21") else m.group(0), s)
    return html.unescape(s)


def norm_text(s):
    s = unescape(s).lower().replace("×", "x").replace("–", "-").replace("—", "-")
    s = re.sub(r"(\d)\s?-\s?(\d)", r"\1-\2", s)                # 90 - 100 -> 90-100.replace("″", '"').replace("’", "'").replace(" °", "°")
    s = re.sub(r"(\d) (\d{3})(?![\d.,])", r"\1\2", s)      # thousands space: 1 500 -> 1500
    s = re.sub(r"(\d),(\d{3})(?![\d.,])", r"\1\2", s)      # thousands comma (English): 1,500 -> 1500
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)                  # decimal comma: 42,5 -> 42.5
    s = re.sub(r"(\d)\.(\d{3})(?![\d.,])", r"\1\2", s)     # thousands point (German): 8.500 -> 8500
    s = re.sub(r"(\d)\s?x\s?(\d)", r"\1x\2", s)            # 150 x 80 -> 150x80
    return s


def norm_number(n):
    return norm_text(n).replace(" ", "")


def candidates_for_measure(num, unit):
    num = norm_number(num); unit = unit.lower()
    units = UNIT_MAP.get(unit, [unit])
    return [(num, u) for u in units]


def measure_present(num, unit, en):
    for n, u in candidates_for_measure(num, unit):
        if re.search(r"(?<![\d.])" + re.escape(n) + r"\s?-?" + re.escape(u) + r"(?![a-z])", en):
            return True
    if unit.lower() in ("°",):
        return re.search(r"(?<![\d.])" + re.escape(norm_number(num)) + r"\s?°", en) is not None
    return False


def check_offer(de, en):
    en_all = norm_text(en["name"] + " \n " + en["desc"])
    en_compact = re.sub(r"[^a-z0-9]", "", en_all)
    misses, counts = [], {}
    for field in ("name", "desc"):
        text = de[field]
        for m in RE_MEASURE.finditer(norm_text(text)):
            counts[("measures", field)] = counts.get(("measures", field), 0) + 1
            if not measure_present(m.group(1), m.group(2), en_all):
                misses.append(("measures", field, m.group(0)))
        for raw in code_tokens(text):
            tok = norm_text(raw).strip(".-/")
            counts[("codes", field)] = counts.get(("codes", field), 0) + 1
            variants = {tok, tok.replace("-", ""), tok.replace("-", " "), tok.replace("/", " "), tok.replace(".", "")}
            compact = re.sub(r"[^a-z0-9]", "", tok)
            if not any(v in en_all for v in variants) and not (len(compact) >= 4 and compact in en_compact):
                misses.append(("codes", field, raw))
        for m in RE_NUMBER.finditer(norm_text(text)):
            counts[("numbers", field)] = counts.get(("numbers", field), 0) + 1
            n = m.group(1)
            if not re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", en_all) and not re.search(r"(?<![\d.])" + re.escape(n.rstrip("0").rstrip(".")) + r"(?![\d])", en_all):
                misses.append(("numbers", field, m.group(1)))
    return counts, misses


def main():
    offers = aligned_offers()
    totals, missed, offers_with_miss = {}, {}, {}
    rows = []
    for i, o in offers.items():
        counts, misses = check_offer(o["de"], o["en"])
        for k, v in counts.items(): totals[k] = totals.get(k, 0) + v
        for cls, field, tok in misses:
            missed[(cls, field)] = missed.get((cls, field), 0) + 1
            offers_with_miss.setdefault(cls, set()).add(i)
            rows.append({"id": i, "class": cls, "field": field, "token": tok, "de_name": o["de"]["name"], "en_name": o["en"]["name"],
                         "de_desc": o["de"]["desc"][:300], "en_desc": o["en"]["desc"][:300]})
    os.makedirs("reports/translation_audit", exist_ok=True)
    with open("reports/translation_audit/invariant_misses.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()) if rows else ["id"]); w.writeheader(); w.writerows(rows)
    print(f"offers: {len(offers)}")
    print(f"{'class':10s} {'field':6s} {'tokens':>8s} {'missing':>8s} {'preserved':>10s}")
    for cls in ("codes", "measures", "numbers"):
        for field in ("name", "desc"):
            t = totals.get((cls, field), 0); m = missed.get((cls, field), 0)
            print(f"{cls:10s} {field:6s} {t:8d} {m:8d} {100*(t-m)/t if t else float('nan'):9.2f}%")
        t = sum(totals.get((cls, f), 0) for f in ("name", "desc")); m = sum(missed.get((cls, f), 0) for f in ("name", "desc"))
        print(f"{cls:10s} {'all':6s} {t:8d} {m:8d} {100*(t-m)/t if t else float('nan'):9.2f}%   offers with >=1 miss: {len(offers_with_miss.get(cls, ()))} ({100*len(offers_with_miss.get(cls, ()))/len(offers):.2f}%)")
    print("misses written to reports/translation_audit/invariant_misses.csv:", len(rows))


if __name__ == "__main__":
    main()
