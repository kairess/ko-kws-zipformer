"""
80-dim fbank for one manifest pair → $DATA/fbank/kws_cuts_<name>.jsonl.gz (+ feats).
Train parts get 0.9/1.0/1.1 speed perturbation (icefall convention).

  python local/compute_fbank.py --name zeroth_train --perturb-speed 1 \
      --recordings $DATA/manifests/zeroth_recordings_train.jsonl.gz \
      --supervisions $DATA/manifests/zeroth_supervisions_train.jsonl.gz --out $DATA/fbank
"""
import argparse, logging, os
from pathlib import Path
import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, Recording, load_manifest

torch.set_num_threads(1); torch.set_num_interop_threads(1)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# module-level predicates: lazy CutSets are pickled for the feature-extraction workers
def _dur_ok(c, lo, hi):
    return lo <= c.duration <= hi


def _pick(c, frac):
    import zlib  # deterministic across processes (str hash() is salted per process)
    return (zlib.crc32(c.id.encode()) % 1000) < frac * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--supervisions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perturb-speed", type=int, default=0)
    ap.add_argument("--nj", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--min-dur", type=float, default=0.3)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--rir-dir", default="", help="RIRS_NOISES/simulated_rirs: add a reverberated copy of a fraction of the cuts")
    ap.add_argument("--rir-frac", type=float, default=0.3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cuts_file = out / f"kws_cuts_{a.name}.jsonl.gz"
    if cuts_file.exists():
        logging.info(f"{cuts_file} exists, skip"); return
    cuts = CutSet.from_manifests(recordings=load_manifest(a.recordings), supervisions=load_manifest(a.supervisions))
    cuts = cuts.resample(16000)
    from functools import partial
    cuts = cuts.filter(partial(_dur_ok, lo=a.min_dur, hi=a.max_dur)).to_eager()
    if a.perturb_speed:
        cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
    if a.rir_dir:
        from lhotse import RecordingSet
        import glob, random
        rir_files = sorted(glob.glob(str(Path(a.rir_dir) / "**" / "*.wav"), recursive=True))
        random.Random(0).shuffle(rir_files)
        rirs = RecordingSet.from_recordings(Recording.from_file(f) for f in rir_files[:400])
        frac = a.rir_frac
        base = cuts.filter(partial(_pick, frac=frac)).to_eager()
        reverbed = base.reverb_rir(rir_recordings=rirs, affix_id=True)
        cuts = cuts + reverbed
        logging.info(f"{a.name}: +reverb copies ({frac:.0%} of cuts, {len(rirs)} RIRs)")
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    cuts = cuts.compute_and_store_features(extractor=extractor, storage_path=str(out / f"kws_feats_{a.name}"),
                                           num_jobs=a.nj, storage_type=LilcomChunkyWriter)
    cuts.to_file(cuts_file)
    logging.info(f"{a.name}: {len(cuts)} cuts, {sum(c.duration for c in cuts) / 3600:.2f} h → {cuts_file}")


if __name__ == "__main__":
    main()
