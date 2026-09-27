#!/usr/bin/env python3
"""Download AI Hub datasets file-by-file and convert audio to 16 kHz mono Ogg/Opus.

Why not aihubshell -mode d: it stores download.tar, untars the .partN pieces, then cats
them, i.e. up to 3x the file size on disk. Here the tar stream is merged on the fly into
one zip (1x), converted, and the zip deleted. One zip is prefetched while the previous
one converts, so peak disk = 2 zips.

Opus 32 kbps via PyAV/libopus complexity 10 (local/opus_enc.py, ~14 MB/h, 3.4x faster than
libsndfile). KsponSpeech eval_clean CER with pt-kspon-small (2026-09-26): FLAC 10.0%,
libsndfile Opus 0.9 10.1%, PyAV c10/32k 10.1%, c5/32k 10.2%, c5/48k 10.0% (26 MB/h, too big).
485 VS/TS were written with libsndfile 0.9; the rest with PyAV.

Layout: <out>/<datasetkey>/audio/<zip stem>/<member path>.ogg
        <out>/<datasetkey>/labels/<zip stem>/<member path>     (json/xlsx/csv/txt, nested zips opened)
        <out>/<datasetkey>/index/<zip stem>.tsv                 (relpath, orig_sr, channels, seconds)
        <out>/<datasetkey>/done.txt                             (finished filekeys; rerun resumes)

Key: $AIHUB_API_KEY or ~/.config/ko-kws/aihub.key (never in the repo).
  python local/aihub_fetch.py --out data/raw/aihub 485 109 96 95 71405
"""
import argparse, io, os, re, shutil, subprocess, sys, tarfile, threading, queue, time, zipfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

API = "https://api.aihub.or.kr"
TOOL = Path(os.environ.get("AIHUBSHELL", Path(__file__).resolve().parents[1] / "third_party/tools/aihubshell"))
LABEL_RE = re.compile(r"라벨|레이블|^TL|^VL|대본")
AUDIO_EXT = (".wav", ".wavp", ".pcm", ".flac", ".mp3", ".m4a")
KEEP_EXT = (".json", ".xlsx", ".csv", ".txt", ".tsv", ".xml")
UNITS = {"B": 1e-9, "KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TB": 1e3}


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def api_key():
    k = os.environ.get("AIHUB_API_KEY")
    if not k:
        k = (Path.home() / ".config/ko-kws/aihub.key").read_text().strip()
    return k


def file_tree(dkey):
    out = subprocess.run([str(TOOL), "-mode", "l", "-datasetkey", str(dkey)], capture_output=True, text=True, timeout=300).stdout
    files = []
    for l in out.splitlines():
        m = re.search(r"─(.+?) \| ([\d.]+) (KB|MB|GB|TB|B) \| (\d+)", l)
        if m:
            files.append(dict(name=m.group(1).strip(), gb=float(m.group(2)) * UNITS[m.group(3)], fkey=m.group(4)))
    return files


def zname(info):
    n = info.filename
    if not info.flag_bits & 0x800:
        try:
            n = n.encode("cp437").decode("cp949")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return n.lstrip("/")


def download(dkey, fkey, dst_dir, key):
    """Stream the tar from the API, merging .partN members into whole files. Returns written paths."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{API}/down/0.6/{dkey}.do?fileSn={fkey}", headers={"apikey": key})
    written = {}
    with urllib.request.urlopen(req, timeout=120) as r:
        with tarfile.open(fileobj=r, mode="r|") as tf:
            files = {}   # .partN suffix = byte offset of the piece in the merged file
            for m in tf:
                if not m.isfile():
                    continue
                pm = re.match(r"(.*)\.part(\d+)$", m.name)
                base, off = (pm.group(1), int(pm.group(2))) if pm else (m.name, 0)
                if base not in files:
                    path = dst_dir / Path(base).name
                    files[base] = open(path, "wb")
                    written[base] = path
                fh = files[base]
                fh.seek(off)
                shutil.copyfileobj(tf.extractfile(m), fh, 16 << 20)
            for fh in files.values():
                fh.close()
    return list(written.values())


def _convert(args):
    zpath, idxs, out_root = args
    import numpy as np, soundfile as sf, soxr
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from opus_enc import write_opus
    rows, bad = [], []
    with zipfile.ZipFile(zpath) as zf:
        il = zf.infolist()
        for i in idxs:
            info = il[i]
            rel = zname(info)
            try:
                b = zf.read(info)
                if rel.lower().endswith(".pcm"):
                    x, sr = sf.read(io.BytesIO(b), channels=1, samplerate=16000, format="RAW", subtype="PCM_16", dtype="float32")
                else:
                    try:
                        x, sr = sf.read(io.BytesIO(b), dtype="float32")
                    except Exception:   # e.g. .wavp / headerless
                        x, sr = sf.read(io.BytesIO(b), channels=1, samplerate=16000, format="RAW", subtype="PCM_16", dtype="float32")
                ch = 1 if x.ndim == 1 else x.shape[1]
                if ch > 1:
                    x = x.mean(axis=1)
                if sr != 16000:
                    x = soxr.resample(x, sr, 16000, "HQ")
                if len(x) < 1600:
                    bad.append(f"{rel}\ttoo_short"); continue
                dst = out_root / (os.path.splitext(rel)[0] + ".ogg")
                dst.parent.mkdir(parents=True, exist_ok=True)
                write_opus(dst, x, 16000, bit_rate=32000, complexity=10)
                rows.append(f"{rel}\t{sr}\t{ch}\t{len(x) / 16000:.3f}")
            except Exception as e:
                bad.append(f"{rel}\t{type(e).__name__}: {e}")
    return rows, bad


def extract_labels(zpath, out_root):
    n = 0
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = zname(info)
            if rel.lower().endswith(".zip"):
                tmp = out_root / "_nested.zip"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as s, open(tmp, "wb") as d:
                    shutil.copyfileobj(s, d)
                n += extract_labels(tmp, out_root / os.path.splitext(rel)[0])
                tmp.unlink()
            elif rel.lower().endswith(KEEP_EXT):
                dst = out_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as s, open(dst, "wb") as d:
                    shutil.copyfileobj(s, d)
                n += 1
    return n


def process_zip(zpath, droot, pool, nproc):
    stem = zpath.name[:-4] if zpath.name.endswith(".zip") else zpath.name
    stem = stem.replace(".zip", "")
    with zipfile.ZipFile(zpath) as zf:
        il = zf.infolist()
        audio = [i for i, f in enumerate(il) if not f.is_dir() and zname(f).lower().endswith(AUDIO_EXT)]
        other = [i for i, f in enumerate(il) if not f.is_dir() and zname(f).lower().endswith(KEEP_EXT + (".zip",))]
    if other and not audio or LABEL_RE.search(zpath.name):
        n = extract_labels(zpath, droot / "labels" / stem)
        log(f"  labels {zpath.name}: {n} files")
    if audio:
        out_root = droot / "audio" / stem
        chunks = [audio[i::nproc * 4] for i in range(nproc * 4)]
        rows, bad = [], []
        for r, b in pool.map(_convert, [(str(zpath), c, out_root) for c in chunks if c]):
            rows += r; bad += b
        (droot / "index").mkdir(parents=True, exist_ok=True)
        (droot / "index" / f"{stem}.tsv").write_text("relpath\torig_sr\tchannels\tseconds\n" + "\n".join(sorted(rows)) + "\n")
        if bad:
            (droot / "index" / f"{stem}.bad.tsv").write_text("\n".join(bad) + "\n")
        hours = sum(float(r.rsplit("\t", 1)[1]) for r in rows) / 3600
        log(f"  audio {zpath.name}: {len(rows)} files {hours:.1f} h, {len(bad)} bad")
    zpath.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", default="data/raw/aihub")
    ap.add_argument("--nproc", type=int, default=12)
    ap.add_argument("--reserve-gb", type=float, default=15)
    ap.add_argument("--only", default="", help="comma-separated filekeys (testing)")
    a = ap.parse_args()
    key, out = api_key(), Path(a.out)
    jobs = []
    for d in a.datasets:
        droot = out / d
        droot.mkdir(parents=True, exist_ok=True)
        done = set((droot / "done.txt").read_text().split()) if (droot / "done.txt").exists() else set()
        files = file_tree(d)
        # labels first (small), then sources smallest-first
        files.sort(key=lambda f: (not LABEL_RE.search(f["name"]), f["gb"]))
        todo = [f for f in files if f["fkey"] not in done and (not a.only or f["fkey"] in a.only.split(","))]
        log(f"dataset {d}: {len(files)} files, {sum(f['gb'] for f in files):.0f} GB, todo {len(todo)} ({sum(f['gb'] for f in todo):.0f} GB)")
        jobs += [(d, droot, f) for f in todo]

    q = queue.Queue(maxsize=1)
    stage = out / "_stage"
    shutil.rmtree(stage, ignore_errors=True)

    def downloader():
        for d, droot, f in jobs:
            need = f["gb"] * 1.05 + a.reserve_gb
            waited = 0
            while shutil.disk_usage(out).free / 1e9 < need and waited < 3600:
                log(f"  waiting for disk: need {need:.0f} GB"); time.sleep(60); waited += 60
            if waited >= 3600:
                log(f"DISK FULL: stopping before {d}/{f['name']}"); break
            for attempt in range(5):
                try:
                    t = time.time()
                    paths = download(d, f["fkey"], stage / f["fkey"], key)
                    sz = sum(p.stat().st_size for p in paths) / 1e9
                    log(f"downloaded {d}/{f['name']} {sz:.1f} GB in {time.time() - t:.0f}s")
                    break
                except Exception as e:
                    log(f"  download failed {d}/{f['name']} (attempt {attempt + 1}): {e}")
                    shutil.rmtree(stage / f["fkey"], ignore_errors=True)
                    time.sleep(30 * (attempt + 1))
            else:
                q.put((d, droot, f, None)); continue
            q.put((d, droot, f, paths))
        q.put(None)

    threading.Thread(target=downloader, daemon=True).start()
    failed = []
    with ProcessPoolExecutor(a.nproc) as pool:
        while (item := q.get()) is not None:
            d, droot, f, paths = item
            if paths is None:
                failed.append(f"{d}/{f['name']}"); continue
            try:
                for p in paths:
                    if p.name.endswith(".zip"):
                        process_zip(p, droot, pool, a.nproc)
                    else:
                        dst = droot / "labels" / p.name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(p, dst)
                shutil.rmtree(stage / f["fkey"], ignore_errors=True)
                with open(droot / "done.txt", "a") as fh:
                    fh.write(f["fkey"] + "\n")
            except Exception as e:
                log(f"  convert failed {d}/{f['name']}: {type(e).__name__}: {e}")
                failed.append(f"{d}/{f['name']}")
                shutil.rmtree(stage / f["fkey"], ignore_errors=True)
    log(f"finished; failed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
