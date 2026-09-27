"""
icefall-style KWS metrics per test set from decode.py --hits-file dumps (decoded at a LOW threshold so
every candidate hit and its acoustic score is kept; thresholds are swept offline).

Per keyword k: positives = utterances containing k (meta.json); a positive is detected at τ if a hit
of k with score ≥ τ occurs in it. False alarm = a hit of k with score ≥ τ in a negative utterance
(contains none of the set's keywords). FA/h = total false alarms (all keywords) / negative hours.

  python local/kws_metrics.py --meta data/kws-test/<set>/meta.json --hits hits.jsonl [--op 0.35] [--at 1.0 --at 0.1]
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--hits", required=True)
ap.add_argument("--op", type=float, default=0.35)
ap.add_argument("--at", type=float, action="append", default=None)
ap.add_argument("--json", default="")
ap.add_argument("--per-keyword", action="store_true")
a = ap.parse_args()
targets = a.at or [1.0, 0.1]
meta = json.load(open(a.meta, encoding="utf-8"))
strip = lambda s: s.replace(" ", "")
hits = {}
for l in open(a.hits, encoding="utf-8"):
    r = json.loads(l); hits[r["cut"]] = [(strip(h["kw"]), h["score"]) for h in r["hits"]]
negset = set(meta["negatives"]); hours = meta["negative_hours"]
best = {k: [max([s for w, s in hits.get(c, []) if w == k] or [-1.0]) for c in meta["positives"][k]] for k in meta["keywords"]}
fa_scores = [s for c in negset for w, s in hits.get(c, [])]
npos = sum(len(v) for v in best.values())
frr = lambda th: sum(1 for v in best.values() for b in v if b < th) / npos
fah = lambda th: sum(1 for s in fa_scores if s >= th) / hours
grid = sorted({round(x / 100, 2) for x in range(1, 101)} | {a.op})
res = {"set": meta["set"], "keywords": len(meta["keywords"]), "positives": npos, "negative_hours": hours,
       "FRR@op": frr(a.op), "FA/h@op": fah(a.op), "op": a.op}
for t in targets:
    ok = [th for th in grid if fah(th) <= t]
    res[f"FRR@{t}FA/h"] = frr(min(ok)) if ok else 1.0
    res[f"thr@{t}FA/h"] = min(ok) if ok else None
print(f"  {meta['set']:17} kw {res['keywords']:2}  pos {npos:4}  neg {hours:5.2f} h | FRR@τ{a.op} {100*res['FRR@op']:5.1f}%  FA/h@τ{a.op} {res['FA/h@op']:6.2f} | "
      + "  ".join(f"FRR@{t}FA/h {100*res[f'FRR@{t}FA/h']:5.1f}% (τ={res[f'thr@{t}FA/h']})" for t in targets))
if a.per_keyword:
    for k in meta["keywords"]:
        v = best[k]; fa = sum(1 for c in negset for w, s in hits.get(c, []) if w == k and s >= a.op)
        print(f"      {k:10} pos {len(v):3}  miss@τ {sum(1 for b in v if b < a.op):3}  FA@τ {fa:3}")
if a.json:
    json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1)
