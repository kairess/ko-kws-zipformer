"""
icefall-style KWS test sets built from held-out ASR test sets.

For each test set: keywords = the K most frequent nouns (Kiwi NNG/NNP, >= 3 syllables) by number of
utterances containing them (ties → alphabetical). No keyword is a substring of another. Positive set of keyword k = utterances with a word that
starts with k; utterances containing k only inside a word are ignored. Negative set = utterances that contain none of the K keywords
(the role GigaSpeech test plays in icefall's KWS results).

  python local/make_kws_testsets.py --fbank data/fbank-k --out data/kws-test --k 20
→ data/kws-test/<set>/{keywords.txt (icefall decode.py format), meta.json}
"""
import argparse, collections, gzip, json
from pathlib import Path
from kiwipiepy import Kiwi  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fbank", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sets", default="kspon_eval_clean,kspon_eval_other,fleurs_test")
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--min-syl", type=int, default=3)
ap.add_argument("--min-pos", type=int, default=3, help="minimum positive utterances per keyword")
a = ap.parse_args()
kiwi = Kiwi()
strip = lambda s: s.replace(" ", "")
for ts in a.sets.split(","):
    cuts = [json.loads(l) for l in gzip.open(Path(a.fbank) / f"kws_cuts_{ts}.jsonl.gz", "rt", encoding="utf-8")]
    utts = [(c["id"], c["supervisions"][0]["text"], c["duration"]) for c in cuts]
    nouns = set()
    for _, t, _ in utts:
        nouns |= {strip(x.form) for x in kiwi.tokenize(t) if x.tag in ("NNG", "NNP") and len(strip(x.form)) >= a.min_syl}
    # rank candidate nouns by the SAME rule used for positives: utterances with a word starting with it
    df = collections.Counter()
    for _, t, _ in utts:
        words = t.split()
        df.update({n for n in nouns if any(w.startswith(n) for w in words)})
    df = collections.Counter({n: c for n, c in df.items() if c >= a.min_pos})
    # No nested keywords (e.g. 라운드 / 그라운드, 아프리카 / 남아프리카): a hit on the longer word would
    # otherwise be scored as a miss of the shorter one. Skip any candidate that contains, or is
    # contained in, an already chosen keyword and move on to the next most frequent noun.
    kws, skipped = [], []
    for w, _ in sorted(df.items(), key=lambda kv: (-kv[1], kv[0])):
        if any(w in k or k in w for k in kws):
            skipped.append(w); continue
        kws.append(w)
        if len(kws) == a.k: break
    # Positive: some word (어절) STARTS with k — Korean particles follow the noun (프랑스군, 대학교에서).
    # A keyword only inside a word (라운드 in 그라운드) cannot be spotted by a keyword that starts
    # with the word-boundary token, so such utterances are neither positive nor negative (ignored).
    starts = lambda w, t: any(x.startswith(w) for x in t.split())
    pos = {w: [cid for cid, t, _ in utts if starts(w, t)] for w in kws}
    neg = [(cid, d) for cid, t, d in utts if not any(w in strip(t) for w in kws)]
    ignored = [cid for cid, t, _ in utts if any(w in strip(t) for w in kws) and not any(starts(w, t) for w in kws)]
    d = Path(a.out) / ts; d.mkdir(parents=True, exist_ok=True)
    (d / "keywords.txt").write_text("\n".join(kws) + "\n", encoding="utf-8")
    meta = {"set": ts, "keywords": kws, "positives": pos, "negatives": [c for c, _ in neg],
            "negative_hours": sum(x for _, x in neg) / 3600, "ignored": ignored, "total_hours": sum(u[2] for u in utts) / 3600, "utts": len(utts)}
    json.dump(meta, open(d / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{ts}: {len(kws)} keywords, {sum(len(v) for v in pos.values())} positive (keyword, utt) pairs, "
          f"negatives {len(neg)} utts / {meta['negative_hours']:.2f} h of {meta['total_hours']:.2f} h, ignored {len(ignored)} (keyword only inside a word)")
    print("   ", ", ".join(f"{w}({len(pos[w])})" for w in kws))
    if skipped: print("    skipped (nested):", ", ".join(skipped))
