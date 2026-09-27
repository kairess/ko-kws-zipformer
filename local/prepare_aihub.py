#!/usr/bin/env python3
"""Lhotse manifests for AI Hub datasets converted by local/aihub_fetch.py (16 kHz Opus).

Audio index: <root>/<key>/index/<zip stem>.tsv (relpath, orig_sr, channels, seconds) — no audio
is re-read. Labels: <root>/<key>/labels/**. Each dataset has a parser that yields
(audio basename, text, speaker, extra). Split = train/valid from the source zip name.

Truncation: in 명령어 (96/95) many wavs are cut at exactly 2.601 s / 1.3 s although the label's
FileLength (and transcript) is longer; utterances shorter than their labelled length by >0.25 s
go to separate aihub<key>_*_{train,valid}_trunc manifests (not for training as-is).

Text: KsponSpeech-style cleanup + text_norm.normalize(). Utterances whose text still contains
digits or Latin letters are dropped (their reading is ambiguous and the syllable BPE was
trained on spelled-out Korean); counts are logged.

  python local/prepare_aihub.py 485 [109 96 95 71405] --root data/raw/aihub --out data/manifests
Writes data/manifests/aihub<key>_{recordings,supervisions}_{train,valid}.jsonl.gz
"""
import argparse, glob, json, logging, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from text_norm import normalize
from lhotse import AudioSource, Recording, RecordingSet, SupervisionSegment, SupervisionSet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
BAD = re.compile(r"[0-9a-z]")


def clean(t):
    t = re.sub(r"\(([A-Z]{1,3}):([^)]*)\)", r"\2", t)  # (SN:x)/(SP:x)/(NO:) tags → keep spoken x
    t = re.sub(r"\((.*?)\)/\((.*?)\)", r"\1", t)   # (spelling)/(pronunciation) → spelling
    t = re.sub(r"\b[a-zA-Z]{1,2}/", " ", t)          # noise tags like b/ n/
    t = t.replace("*", "").replace("+", "").replace("/", " ")
    return normalize(t)


def split_of(stem):
    s = stem.lower()
    return "valid" if ("validation" in s or s.startswith("vs") or "vs_" in s) else "train"


# ---- per-dataset label parsers: yield (basename, text, speaker, extra) --------------------
def p485(root):
    for f in glob.glob(f"{root}/labels/**/*.json", recursive=True):
        for item in json.load(open(f)):
            spk = f"{item['reciter']['gender'][0]}{item['reciter']['age']}"
            style = ",".join(s["style"] for s in item["recite_src"].get("styles", []))
            genre = item["recite_src"]["literature"]["genre"]
            for s in item["sentences"]:
                vp = s["voice_piece"]
                yield Path(vp["filename"]).stem, vp["tr"], f"485_{item['id'].rsplit('-', 1)[0]}_{spk}", {"genre": genre, "style": style}


def p109(root):
    """자유대화: one JSON per utterance; 발화정보.stt / fileNm, 녹음자정보.recorderId."""
    for f in glob.glob(f"{root}/labels/**/*.json", recursive=True):
        try:
            d = json.load(open(f, encoding="utf-8-sig"))
        except Exception:
            continue
        u, info, spk = d["발화정보"], d["대화정보"], d["녹음자정보"]
        yield Path(u["fileNm"]).stem, u["stt"], f"109_{spk['recorderId']}", {
            "unit": info.get("recrdUnit"), "env": info.get("recrdEnvrn"), "collect": info.get("colctUnitCode"),
            "age": spk.get("age"), "gender": spk.get("gender")}


def p_command(root):
    """명령어 음성 (Mediazen format, 96 일반 / 95 소아·유아): 전사정보.LabelText, 파일정보.FileName."""
    key = Path(root).name
    for f in glob.glob(f"{root}/labels/**/*.json", recursive=True):
        try:
            d = json.load(open(f, encoding="utf-8-sig"))
            fi, sp, env = d["파일정보"], d["화자정보"], d.get("환경정보", {})
            text = d["전사정보"]["LabelText"]
        except Exception:
            continue
        if d.get("기타정보", {}).get("QualityStatus", "Good") not in ("Good", "N/A"):
            continue
        spk = f"{key}_{sp.get('SpeakerName')}_{sp.get('Gender', '')[:1]}_{sp.get('Age')}_{sp.get('Region')}"
        yield Path(fi["FileName"]).stem, text, spk, {
            "category": d.get("기본정보", {}).get("DataCategory"), "device": env.get("RecordingDevice"),
            "noise": env.get("NoiseEnviron"), "distance": fi.get("Distance"), "age": sp.get("Age"), "gender": sp.get("Gender"),
            "label_sec": fi.get("FileLength")}


def p71405(root):
    """명령어 인식용 소음 환경: <name>-J.json labels two wavs, <name>-N (noisier) and <name>-S.
    Both carry the command (checked by ASR on 150 pairs: CER 47% N vs 30% S)."""
    for f in glob.glob(f"{root}/labels/**/*.json", recursive=True):
        try:
            d = json.load(open(f, encoding="utf-8-sig"))
            cmd, spk, noise = d["command"], d["speaker"], d.get("noise", {})
        except Exception:
            continue
        base = Path(f).stem
        base = base[:-2] if base.endswith("-J") else base
        for suf in ("N", "S"):
            yield f"{base}-{suf}", cmd["text"], f"71405_{spk.get('id')}", {
                "mic": suf, "category": cmd.get("category"), "snr": cmd.get("snr"), "level": noise.get("level"),
                "device": spk.get("recordingDevice"), "mask": spk.get("mask"), "age": spk.get("age"), "gender": spk.get("gender"),
                "label_sec": d.get("file", {}).get("length")}


PARSERS = {"485": p485, "109": p109, "96": p_command, "95": p_command, "71405": p71405}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--root", default="data/raw/aihub")
    ap.add_argument("--out", default="data/manifests")
    ap.add_argument("--min-sec", type=float, default=0.3)
    ap.add_argument("--max-sec", type=float, default=20.0)
    a = ap.parse_args()
    for key in a.datasets:
        root = Path(a.root) / key
        audio = {}
        for tsv in sorted((root / "index").glob("*.tsv")):
            if tsv.name.endswith(".bad.tsv"):
                continue
            stem = tsv.stem
            for line in open(tsv).read().splitlines()[1:]:
                rel, sr, ch, sec = line.split("\t")
                b = Path(rel).stem
                if b in audio:
                    logging.warning(f"dup basename {b}")
                audio[b] = (root / "audio" / stem / (os.path.splitext(rel)[0] + ".ogg"), float(sec), split_of(stem))
        stats = Counter()
        recs, sups = defaultdict(list), defaultdict(list)
        seen = set()
        for b, text, spk, extra in PARSERS[key](root):
            stats["labels"] += 1
            if b not in audio:
                stats["no_audio"] += 1; continue
            if b in seen:
                stats["dup_label"] += 1; continue
            seen.add(b)
            path, sec, split = audio[b]
            t = clean(text)
            if not t:
                stats["empty"] += 1; continue
            if BAD.search(t):
                stats["digit_latin"] += 1; continue
            # Some distributed wavs are cut at exactly 2.6 s / 1.3 s while the label (and the
            # transcript) covers the full take: drop audio shorter than the labelled length.
            ls = (extra or {}).pop("label_sec", None)
            try:
                ls = float(ls)
            except (TypeError, ValueError):
                ls = None
            if ls is not None and ls - sec > 0.25:
                # kept aside (…_{split}_trunc) for ASR-based salvage: many lose only trailing
                # silence, most lose words (pt-kspon-small median CER 84% on a 300-clip probe).
                stats["truncated"] += 1
                split = f"{split}_trunc"
            if not a.min_sec <= sec <= a.max_sec:
                stats["duration"] += 1; continue
            rid = f"aihub{key}_{b}"
            recs[split].append(Recording(id=rid, sources=[AudioSource(type="file", channels=[0], source=str(path.resolve()))],
                                         sampling_rate=16000, num_samples=int(round(sec * 16000)), duration=sec))
            sups[split].append(SupervisionSegment(id=rid, recording_id=rid, start=0.0, duration=sec, channel=0,
                                                  language="Korean", speaker=spk, text=t, custom=extra or None))
            stats[f"kept_{split}"] += 1
            stats[f"hours_{split}"] += sec / 3600
        stats["audio_files"] = len(audio)
        stats["audio_unlabeled"] = len(audio) - len(seen)
        for split in recs:
            RecordingSet.from_recordings(recs[split]).to_file(f"{a.out}/aihub{key}_recordings_{split}.jsonl.gz")
            SupervisionSet.from_segments(sups[split]).to_file(f"{a.out}/aihub{key}_supervisions_{split}.jsonl.gz")
        logging.info(f"{key}: " + ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
