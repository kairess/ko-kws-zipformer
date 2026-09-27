"""
Korean text normalization + Hangul ⇄ jamo for a small (500-piece) BPE.

Why jamo: a 3M KWS zipformer keeps its official architecture with a 500-token vocab, but Korean
has ~2,000 common syllables, so syllable-level BPE-500 would be mostly <unk>. Decomposing to
compatibility jamo (51 symbols) lets unigram BPE learn frequent syllable/word pieces instead.
`compose()` restores readable Hangul for logs and for icefall's substring-based KWS scoring.
"""
import re
import unicodedata

_L = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
_V = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
_T = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
_Lset, _Vset, _Tset = set(_L), set(_V), set(_T[1:])
_Lidx = {c: i for i, c in enumerate(_L)}
_Vidx = {c: i for i, c in enumerate(_V)}
_Tidx = {c: i for i, c in enumerate(_T)}

# Keep Hangul syllables/jamo, Latin letters, digits, spaces. Everything else → space.
_KEEP = re.compile(r"[^0-9a-z가-힣ㄱ-ㅣ ]+")


def normalize(text: str) -> str:
    """NFC, lowercase Latin, strip punctuation/symbols, collapse whitespace."""
    t = unicodedata.normalize("NFC", text).lower()
    t = t.replace("—", " ").replace("…", " ")
    t = _KEEP.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def decompose(text: str) -> str:
    """Hangul syllables → compatibility jamo (no separators; syllable boundaries are implicit
    from the L-V-T grammar and restored by compose())."""
    out = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code <= 11171:
            l, v, t = code // 588, (code % 588) // 28, code % 28
            out.append(_L[l]); out.append(_V[v])
            if t: out.append(_T[t])
        else:
            out.append(ch)
    return "".join(out)


def compose(jamo: str) -> str:
    """Greedy L-V(-T) recomposition. A trailing consonant is attached only if the next symbol
    is not a vowel (so ㅂㅏㄴㅏ → 바나, not 반ㅏ)."""
    out = []
    i = 0
    n = len(jamo)
    while i < n:
        c = jamo[i]
        if c in _Lset and i + 1 < n and jamo[i + 1] in _Vset:
            l = _Lidx[c]; v = _Vidx[jamo[i + 1]]; t = 0; i += 2
            if i < n and jamo[i] in _Tset and not (i + 1 < n and jamo[i + 1] in _Vset):
                t = _Tidx[jamo[i]]; i += 1
            out.append(chr(0xAC00 + l * 588 + v * 28 + t))
        else:
            out.append(c); i += 1
    return "".join(out)


def to_train_text(text: str) -> str:
    """Normalized jamo text used for BPE training, supervisions and keywords."""
    return decompose(normalize(text))


if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        j = to_train_text(line)
        print(j, "→", compose(j))
