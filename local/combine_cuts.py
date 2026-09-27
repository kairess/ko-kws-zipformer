"""Combine fbank cut sets → kws_cuts_train / kws_cuts_dev.
  python local/combine_cuts.py --fbank $DATA/fbank --extra pretrain_mux=kspon_train,zeroth_train,fleurs_train --mux
"""
import argparse
from pathlib import Path
from lhotse import load_manifest_lazy, combine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbank", required=True)
    ap.add_argument("--train", default="zeroth_train,fleurs_train")
    ap.add_argument("--dev", default="zeroth_test,fleurs_dev")
    ap.add_argument("--extra", action="append", default=[], help="name=part1,part2 (e.g. pretrain=zeroth_train,fleurs_train)")
    ap.add_argument("--mux", action="store_true", help="interleave parts randomly (weights = sizes) instead of appending them")
    a = ap.parse_args()
    f = Path(a.fbank)
    ap_extra = [tuple(e.split("=", 1)) for e in a.extra]
    # with --extra given, only write the extra sets (never rewrite train/dev under a running job)
    jobs = ap_extra if ap_extra else [("train", a.train), ("dev", a.dev)]
    for name, parts in jobs:
        sets = [load_manifest_lazy(f / f"kws_cuts_{p}.jsonl.gz") for p in parts.split(",")]
        if a.mux and len(sets) > 1:
            # random interleave in proportion to each part's size, so every stretch of the file (and thus every
            # shuffle buffer of the lazy sampler) contains all parts; exhausts every part exactly once
            from lhotse import CutSet
            sizes = [sum(1 for _ in s) for s in sets]
            c = CutSet.mux(*sets, weights=sizes, seed=0, stop_early=False)
        else:
            c = combine(*sets) if len(sets) > 1 else sets[0]
        c.to_file(f / f"kws_cuts_{name}.jsonl.gz")
        n = sum(1 for _ in load_manifest_lazy(f / f"kws_cuts_{name}.jsonl.gz"))
        print(f"kws_cuts_{name}: {n} cuts from {parts}")


if __name__ == "__main__":
    main()
