"""LLM-assisted translation audit of all offers through OpenRouter, three model families in parallel.

Every offer is judged independently by each model with the prompt in prompt.md (temperature 0,
JSON answer). Results are appended per model to results/generated/translation_audit/<model>.jsonl
and the script resumes from what is already there. Nothing about labels or matcher results is in
the prompt.

    export OPENROUTER_API_KEY=...
    python src/translation_audit/run_openrouter_audit.py --list-models          # see current slugs
    python src/translation_audit/run_openrouter_audit.py --limit 5 --dry-run    # print prompts only
    python src/translation_audit/run_openrouter_audit.py --limit 20             # small live test
    python src/translation_audit/run_openrouter_audit.py                        # all 13,730 offers
    python src/translation_audit/summarize_audit.py                             # aggregate

Check the model slugs on https://openrouter.ai/models before a full run; the defaults below are
placeholders for one Anthropic, one Moonshot (Kimi) and one Google (Gemini) model.
"""
import argparse, glob, gzip, json, os, random, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
sys.path.insert(0, os.path.dirname(__file__))
from offers import aligned_offers

API = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = ["anthropic/claude-sonnet-4.5", "moonshotai/kimi-k2", "google/gemini-2.5-pro"]
OUT = "results/generated/translation_audit"
PROMPT = open(os.path.join(os.path.dirname(__file__), "prompt.md"), encoding="utf-8").read()
REASONING = {}   # per model: the reasoning setting the endpoint accepted, filled in on the first 400
KEYS = ("codes_preserved", "variants_preserved", "quantities_preserved", "omission_or_addition", "could_change_match_decision", "verdict", "issue")


def test_set_offer_ids():
    """Offers that occur in at least one released German test set."""
    import pandas as pd
    ids = set()
    for f in glob.glob("data/solute_de/gold-standards_adjusted/*.json.gz"):
        with gzip.open(f, "rt", encoding="utf-8") as h:
            df = pd.read_json(h, lines=True)
        ids |= set(df["id_left"].astype(str)) | set(df["id_right"].astype(str))
    return ids


def estimate_cost(models, offers):
    r = requests.get(f"{API}/models", timeout=60); r.raise_for_status()
    price = {m["id"]: m.get("pricing", {}) for m in r.json()["data"]}
    n = len(offers); tok_in = sum(len(build_prompt(o)) for o in offers.values()) / 3.5; tok_out = n * 90
    total = 0.0
    print(f"{n} offers, about {tok_in/n:.0f} input and 90 output tokens per offer")
    for m in models:
        p = price.get(m)
        if not p:
            print(f"  {m:45s} not found on OpenRouter"); continue
        c = tok_in * float(p.get("prompt", 0)) + tok_out * float(p.get("completion", 0)); total += c
        print(f"  {m:45s} ${c:6.2f}   (in ${float(p['prompt'])*1e6:.2f}/M, out ${float(p['completion'])*1e6:.2f}/M)")
    print(f"  total about ${total:.2f}")


def build_prompt(o):
    out = PROMPT
    for key, val in (("{brand}", o["de"]["brand"]), ("{de_name}", o["de"]["name"]), ("{de_desc}", o["de"]["desc"]), ("{en_name}", o["en"]["name"]), ("{en_desc}", o["en"]["desc"])):
        out = out.replace(key, val)
    return out


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return {k: j.get(k, "") for k in KEYS}


def call(model, prompt, key, retries=5):
    # Reasoning is switched off through OpenRouter's unified parameter: the audit needs a short JSON
    # answer, and reasoning tokens are billed as completion tokens. max_tokens leaves headroom for
    # models that ignore the switch.
    body = {"model": model, "temperature": 0, "max_tokens": 1000,
            "reasoning": REASONING.get(model, {"enabled": False}),
            "messages": [{"role": "user", "content": prompt}]}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "HTTP-Referer": "https://github.com/wbsg-uni-mannheim/billiger-de-products", "X-Title": "translation audit"}
    for attempt in range(retries):
        try:
            r = requests.post(f"{API}/chat/completions", headers=headers, json=body, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt); continue
            if r.status_code == 400 and "cannot be disabled" in r.text:
                # Some endpoints (GLM, Gemini 3.x) reject reasoning={"enabled": false}. The lowest
                # effort level is accepted there and keeps the answer short.
                body["reasoning"] = REASONING[model] = {"effort": "minimal"}; continue
            r.raise_for_status(); j = r.json()
            content = j["choices"][0]["message"].get("content") or ""
            return content, j.get("usage", {}), j.get("model", model)
        except (requests.RequestException, KeyError, ValueError) as e:
            err = e; time.sleep(2 ** attempt)
    raise RuntimeError(f"{model}: giving up after {retries} attempts: {err}")


def run_model(model, offers, key, workers, log_lock):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{model.replace('/', '__')}.jsonl"
    done = set()
    if os.path.exists(path):                      # keep parsed rows, drop failed ones so they are retried
        with open(path, encoding="utf-8") as h:
            kept = [json.loads(l) for l in h if l.strip()]
        kept = [r for r in kept if r.get("parsed")]
        with open(path, "w", encoding="utf-8") as h:
            for r in kept: h.write(json.dumps(r, ensure_ascii=False) + "\n")
        done = {r["id"] for r in kept}
    todo = [(i, o) for i, o in offers.items() if i not in done]
    print(f"[{model}] {len(done)} done, {len(todo)} to do", flush=True)
    out = open(path, "a", encoding="utf-8"); n_ok = n_bad = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(call, model, build_prompt(o), key): i for i, o in todo}
        for k, f in enumerate(as_completed(futs), 1):
            i = futs[f]
            try:
                text, usage, resolved = f.result(); parsed = parse_json(text)
                rec = {"id": i, "model": resolved, "parsed": parsed, "raw": text if parsed is None else "", "usage": usage, "ts": time.time()}
                n_ok += parsed is not None; n_bad += parsed is None
            except Exception as e:
                rec = {"id": i, "model": model, "parsed": None, "raw": f"ERROR {e}", "usage": {}, "ts": time.time()}; n_bad += 1
            with log_lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            if k % 200 == 0:
                print(f"[{model}] {k}/{len(todo)} ({n_bad} unparsed) {time.time()-t0:.0f}s", flush=True)
    out.close(); print(f"[{model}] finished: {n_ok} parsed, {n_bad} unparsed or failed", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS)); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--sample", type=int, default=0, help="random sample of this many offers drawn from the released test sets")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--estimate", action="store_true", help="print the cost at current OpenRouter prices and exit")
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key and os.path.exists(".env"):                      # repo-root .env (git-ignored), KEY=VALUE lines
        for line in open(".env", encoding="utf-8"):
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if a.list_models:
        r = requests.get(f"{API}/models", timeout=60); r.raise_for_status()
        for m in sorted(r.json()["data"], key=lambda m: m["id"]):
            if any(v in m["id"] for v in ("anthropic/", "moonshotai/", "google/gemini")):
                p = m.get("pricing", {}); print(f"{m['id']:45s} in ${float(p.get('prompt', 0))*1e6:.2f}/M  out ${float(p.get('completion', 0))*1e6:.2f}/M")
        return
    offers = aligned_offers()
    if a.sample:
        pool = sorted(set(offers) & test_set_offer_ids())
        chosen = random.Random(a.seed).sample(pool, min(a.sample, len(pool)))
        offers = {i: offers[i] for i in sorted(chosen)}
        print(f"sampled {len(offers)} of {len(pool)} test-set offers (seed {a.seed})")
    if a.limit:
        offers = dict(list(offers.items())[: a.limit])
    if a.estimate:
        estimate_cost(a.models.split(","), offers); return
    if a.dry_run:
        for i, o in list(offers.items())[:2]:
            print(f"===== offer {i}\n{build_prompt(o)}\n")
        print(f"{len(offers)} offers, {len(a.models.split(','))} models, {sum(len(build_prompt(o)) for o in offers.values())/len(offers):.0f} characters per prompt on average"); return
    assert key, "OPENROUTER_API_KEY is not set"
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=len(a.models.split(","))) as ex:
        list(ex.map(lambda m: run_model(m, offers, key, a.workers, lock), a.models.split(",")))


if __name__ == "__main__":
    main()
