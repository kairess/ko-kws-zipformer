"""
FLEURS ko_kr (CC-BY-4.0, HF google/fleurs) → lhotse manifests.
Expects $RAW/fleurs/data/ko_kr/{train,dev,test}.tsv and audio/{split}.tar.gz (extracted to audio/{split}/).
TSV columns: id, file, raw_transcription, transcription, chars, num_samples, gender.

  python local/prepare_fleurs.py --root $RAW/fleurs/data/ko_kr --out $DATA/manifests
"""
import argparse, csv, logging, tarfile
from pathlib import Path
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet
from text_norm import to_train_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        if (out / f"fleurs_supervisions_{split}.jsonl.gz").exists():
            logging.info(f"{split}: exists, skip"); continue
        wav_dir = root / "audio" / split
        if not wav_dir.exists():
            logging.info(f"extracting {split}.tar.gz")
            with tarfile.open(root / "audio" / f"{split}.tar.gz") as tf:
                tf.extractall(root / "audio")
        recs, sups = [], []
        with open(root / f"{split}.tsv", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) < 4: continue
                wav = wav_dir / row[1]
                if not wav.exists(): continue
                rid = f"fleurs_{split}_{wav.stem}"
                rec = Recording.from_file(wav, recording_id=rid)
                gender = (row[6].lower()[:1] if len(row) > 6 else None)
                recs.append(rec)
                sups.append(SupervisionSegment(id=rid, recording_id=rid, start=0.0, duration=rec.duration, channel=0,
                                               language="Korean", speaker=f"fleurs_{row[0]}", gender=gender,
                                               text=to_train_text(row[2]), custom={"raw_text": row[2], "source": "fleurs"}))
        RecordingSet.from_recordings(recs).to_file(out / f"fleurs_recordings_{split}.jsonl.gz")
        SupervisionSet.from_segments(sups).to_file(out / f"fleurs_supervisions_{split}.jsonl.gz")
        logging.info(f"{split}: {len(recs)} utts, {sum(r.duration for r in recs) / 3600:.2f} h")


if __name__ == "__main__":
    main()
