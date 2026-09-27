#!/usr/bin/env python3
"""Cap how many utterances share the same transcript (scripted corpora repeat sentences many
times, e.g. AI Hub 96 명령어: 52k texts for 2.15M utts). Random choice, fixed seed.
  python local/cap_per_text.py data/manifests/aihub96 train 20   → aihub96_*_train_cap20.jsonl.gz"""
import random, sys
from collections import defaultdict
from lhotse import load_manifest, RecordingSet, SupervisionSet

prefix, split, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
sups = load_manifest(f"{prefix}_supervisions_{split}.jsonl.gz")
recs = load_manifest(f"{prefix}_recordings_{split}.jsonl.gz")
by = defaultdict(list)
for s in sups:
    by[s.text].append(s)
rng = random.Random(0)
keep = []
for t in sorted(by):
    v = by[t]
    keep += rng.sample(v, cap) if len(v) > cap else v
ids = {s.recording_id for s in keep}
SupervisionSet.from_segments(sorted(keep, key=lambda s: s.id)).to_file(f"{prefix}_supervisions_{split}_cap{cap}.jsonl.gz")
RecordingSet.from_recordings(r for r in recs if r.id in ids).to_file(f"{prefix}_recordings_{split}_cap{cap}.jsonl.gz")
print(f"{len(sups)} → {len(keep)} utts, {sum(s.duration for s in keep) / 3600:.1f} h, {len({s.speaker for s in keep})} speakers")
