"""Character error rate (Hangul syllables, spaces ignored) from an icefall recogs file.
  python local/cer.py <recogs-*.txt>   → prints 'CER xx.x% (n chars)'"""
import re, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from text_norm import compose


def ed(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


pairs = {}
for l in open(sys.argv[1], encoding="utf-8"):
    m = re.match(r"(\S+):\s+(ref|hyp)=\[(.*)\]", l)
    if m:
        pairs.setdefault(m.group(1), {})[m.group(2)] = compose("".join(re.findall(r"'([^']*)'", m.group(3))))
err = n = 0
for d in pairs.values():
    r, h = d.get("ref", ""), d.get("hyp", "")
    err += ed(r, h); n += len(r)
print(f"CER {100 * err / max(n, 1):.1f}% ({n} chars)")
