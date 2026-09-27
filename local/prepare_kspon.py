"""
KsponSpeech (AI Hub 123) → lhotse manifests with our syllable text, streaming with bounded memory
(lhotse's own recipe queues all 620k utterances and grew past 18 GB in a minute).

PCM (16 kHz, 16-bit, mono, headerless) → FLAC next to it (skipped if it exists). Text: spelling side
of "(철수)/(칠수)", noise tags (b/ n/ l/ o/ u/) removed, then text_norm.normalize().

  python local/prepare_kspon.py --corpus $RAW/ksponspeech/KsponSpeech --out $DATA/manifests --nj 12
→ kspon_{recordings,supervisions}_{train,dev,eval_clean,eval_other}.jsonl.gz
"""
import argparse, logging, re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import soundfile as sf
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet
from text_norm import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
PARTS = ["dev", "eval_clean", "eval_other", "train"]


def clean(t: str) -> str:
    t = re.sub(r"\((.*?)\)/\((.*?)\)", r"\1", t)      # (spelling)/(pronunciation) → spelling
    t = re.sub(r"\b[a-z]/", " ", t)                     # noise tags
    t = t.replace("*", "").replace("+", "").replace("/", "")
    return normalize(t)


def one(args):
    corpus, part, line = args
    path, _, text = line.rstrip("\n").partition(" :: ")
    if part.startswith("eval"):
        path = path.split("/", 1)[1]                    # "KsponSpeech_eval/eval_clean/…" → "eval_clean/…"
    pcm = Path(corpus) / path
    if not pcm.is_file():
        return None
    flac = pcm.with_suffix(".flac")
    if not flac.exists() or flac.stat().st_size == 0:
        data, _ = sf.read(pcm, channels=1, samplerate=16000, format="RAW", subtype="PCM_16")
        sf.write(flac, data, 16000, format="FLAC")
    text = clean(text)
    if not text:
        return None
    rec = Recording.from_file(flac, recording_id=pcm.stem)
    sup = SupervisionSegment(id=pcm.stem, recording_id=pcm.stem, start=0.0, duration=rec.duration, channel=0,
                             language="Korean", speaker=pcm.parent.name if part == "train" else part, text=text,
                             custom={"source": "kspon"})
    return rec.to_dict(), sup.to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nj", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        rf, sf_ = out / f"kspon_recordings_{part}.jsonl.gz", out / f"kspon_supervisions_{part}.jsonl.gz"
        if rf.exists() and sf_.exists():
            logging.info(f"{part}: exists, skip"); continue
        lines = [l for l in open(Path(a.corpus) / f"{part}.trn", encoding="utf-8") if l.strip()]
        n = hours = 0
        with RecordingSet.open_writer(rf) as rw, SupervisionSet.open_writer(sf_) as sw, ProcessPoolExecutor(a.nj) as ex:
            for i, r in enumerate(ex.map(one, ((a.corpus, part, l) for l in lines), chunksize=256)):
                if r is None: continue
                rw.write(Recording.from_dict(r[0])); sw.write(SupervisionSegment.from_dict(r[1]))
                n += 1; hours += r[1]["duration"] / 3600
                if i % 50000 == 0: logging.info(f"{part}: {i}/{len(lines)}")
        logging.info(f"{part}: {n} utts, {hours:.1f} h")


if __name__ == "__main__":
    main()
