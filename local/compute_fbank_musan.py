"""MUSAN → 10 s windows → fbank cuts for on-the-fly CutMix (icefall convention).
  python local/compute_fbank_musan.py --manifests $DATA/manifests --out $DATA/fbank
"""
import argparse, logging, os
from pathlib import Path
import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, load_manifest, combine

torch.set_num_threads(1); torch.set_num_interop_threads(1)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _long(c):
    return c.duration > 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nj", type=int, default=min(16, os.cpu_count() or 4))
    a = ap.parse_args()
    m, out = Path(a.manifests), Path(a.out)
    if (out / "musan_cuts.jsonl.gz").exists():
        logging.info("musan cuts exist, skip"); return
    parts = [load_manifest(m / f"musan_recordings_{p}.jsonl.gz") for p in ["music", "noise", "speech"] if (m / f"musan_recordings_{p}.jsonl.gz").exists()]
    cuts = CutSet.from_manifests(recordings=combine(*parts)).cut_into_windows(10.0).filter(_long).to_eager()
    cuts = cuts.compute_and_store_features(extractor=Fbank(FbankConfig(num_mel_bins=80)), storage_path=str(out / "musan_feats"),
                                           num_jobs=a.nj, storage_type=LilcomChunkyWriter)
    cuts.to_file(out / "musan_cuts.jsonl.gz")
    logging.info(f"musan: {len(cuts)} cuts")


if __name__ == "__main__":
    main()
