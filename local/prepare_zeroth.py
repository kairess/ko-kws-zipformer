"""
Zeroth-Korean (openslr 40, CC-BY-4.0) → lhotse manifests.
Layout: <root>/{train_data_01,test_data_01}/<scriptid>/<reader>/<reader>_<scriptid>_<n>.flac
        + <reader>_<scriptid>.trans.txt ("uttid text" per line), <root>/AUDIO_INFO (reader|name|gender|...).

  python local/prepare_zeroth.py --root $RAW/zeroth_korean --out $DATA/manifests
→ zeroth_recordings_{train,test}.jsonl.gz, zeroth_supervisions_{train,test}.jsonl.gz
"""
import argparse, logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet
from text_norm import to_train_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def read_gender(root: Path):
    g = {}
    f = root / "AUDIO_INFO"
    if f.exists():
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = [x.strip() for x in line.split("|")]
            if len(p) >= 3 and p[0].isdigit():
                g[p[0]] = p[2].lower()[:1]
    return g


def one(args):
    flac, text, reader, gender = args
    rec = Recording.from_file(flac, recording_id=flac.stem)
    sup = SupervisionSegment(id=flac.stem, recording_id=rec.id, start=0.0, duration=rec.duration,
                             channel=0, language="Korean", speaker=reader, gender=gender or None,
                             text=to_train_text(text), custom={"raw_text": text, "source": "zeroth"})
    return rec, sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nj", type=int, default=16)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gender = read_gender(root)
    for part, name in [("train_data_01", "train"), ("test_data_01", "test")]:
        if (out / f"zeroth_supervisions_{name}.jsonl.gz").exists():
            logging.info(f"{name}: exists, skip"); continue
        jobs = []
        for trans in sorted((root / part).rglob("*.trans.txt")):
            reader_dir = trans.parent
            reader = reader_dir.name
            for line in trans.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                utt, _, text = line.partition(" ")
                flac = reader_dir / f"{utt}.flac"
                if flac.exists(): jobs.append((flac, text, reader, gender.get(reader)))
        logging.info(f"{name}: {len(jobs)} utterances")
        with ProcessPoolExecutor(a.nj) as ex:
            res = list(ex.map(one, jobs, chunksize=64))
        recs = RecordingSet.from_recordings(r for r, _ in res)
        sups = SupervisionSet.from_segments(s for _, s in res)
        recs.to_file(out / f"zeroth_recordings_{name}.jsonl.gz")
        sups.to_file(out / f"zeroth_supervisions_{name}.jsonl.gz")
        logging.info(f"{name}: {sum(r.duration for r in recs) / 3600:.2f} h")


if __name__ == "__main__":
    main()
