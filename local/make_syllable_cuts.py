"""Rewrite fbank cut manifests with syllable (composed Hangul) text instead of jamo, into a new
manifest dir that symlinks the same feature storage. Features are not recomputed.
  python local/make_syllable_cuts.py --src data/fbank --dst data/fbank-syl
"""
import argparse, gzip, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from text_norm import compose

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
a = ap.parse_args()
src, dst = Path(a.src).resolve(), Path(a.dst)
dst.mkdir(parents=True, exist_ok=True)
for p in sorted(src.iterdir()):
    if p.name.endswith(".jsonl.gz") and p.name.startswith("kws_cuts_") and not p.is_symlink():
        n = 0
        with gzip.open(p, "rt", encoding="utf-8") as fi, gzip.open(dst / p.name, "wt", encoding="utf-8") as fo:
            for line in fi:
                c = json.loads(line)
                for s in c.get("supervisions", []):
                    s["text"] = compose(s["text"])
                fo.write(json.dumps(c, ensure_ascii=False) + "\n"); n += 1
        print(f"{p.name}: {n}")
    elif not (dst / p.name).exists():
        os.symlink(p, dst / p.name)   # feature dirs, musan cuts
