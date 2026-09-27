"""FLEURS test: hyp/ref length ratio by utterance duration (detects end-of-utterance deletions).
  python local/fleurs_len.py <recogs-fleurs_test-*.txt>"""
import gzip, json, re, sys
dur = {json.loads(l)["id"]: json.loads(l)["duration"] for l in gzip.open("data/fbank-k/kws_cuts_fleurs_test.jsonl.gz", "rt")}
p = {}
for l in open(sys.argv[1], encoding="utf-8"):
    m = re.match(r"(\S+):\s+(ref|hyp)=\[(.*)\]", l)
    if m: p.setdefault(m.group(1), {})[m.group(2)] = "".join(re.findall(r"'([^']*)'", m.group(3)))
b = {"<10s": [], "10-14s": [], ">=14s": []}
for cid, d in p.items():
    t = dur.get(cid, 0); r = len(d.get("hyp", "")) / max(1, len(d["ref"]))
    b["<10s" if t < 10 else "10-14s" if t < 14 else ">=14s"].append(r)
print("  fleurs_test hyp/ref length: " + ", ".join(f"{k} {sum(v) / len(v):.2f} (n={len(v)})" for k, v in b.items() if v))
