"""Shared crop-boundary logic (offline local/make_aug_cuts.py and the on-the-fly dataset)."""

START_OFFSET = 0.06   # aligner word starts are this much later than Scribe's (median, local/verify_align_scribe.py)
PAD = 0.08


def crop_bounds(words, i, j, dur, start_offset=START_OFFSET, pad=PAD):
    """words: [[w, start, end], ...] (seconds). Span i..j inclusive → (start, end) in seconds, never inside a
    neighbouring word."""
    st = lambda k: words[k][1] - start_offset
    s = max(words[i - 1][2], st(i) - pad) if i > 0 else max(0.0, st(i) - 0.3)
    e = min(st(j + 1), words[j][2] + pad) if j + 1 < len(words) else dur
    return max(0.0, s), min(dur, e)
