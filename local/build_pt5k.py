#!/usr/bin/env python3
"""Build the AI Hub–extended pretraining set and the two command test sets.

Train  kws_cuts_pt5k: pretrain_mux (Kspon+Zeroth+FLEURS) + AI Hub 485 train, 109 train,
       96 train_cap20, 71405 train_cap20 — globally shuffled (train_cuts is read lazily).
Tests  kws_cuts_aihub_cmd_test   : 96 명령어 valid, speakers absent from 96 train
       kws_cuts_aihub_noisy_test : 71405 소음 명령어 valid (no speaker overlap with train)
       random 3000 utts each, at most 2 per transcript (seed 0).
Audio-only cuts (features computed on the fly), syllable text as in data/fbank-k.
"""
import gzip, json, random, sys
from collections import Counter
from lhotse import CutSet, load_manifest_lazy

M, FK = "data/manifests", "data/fbank-k"
rng = random.Random(0)


def cuts_of(prefix, split):
    return CutSet.from_manifests(recordings=load_manifest_lazy(f"{M}/{prefix}_recordings_{split}.jsonl.gz"),
                                 supervisions=load_manifest_lazy(f"{M}/{prefix}_supervisions_{split}.jsonl.gz"))


def write(lines, name):
    with gzip.open(f"{FK}/kws_cuts_{name}.jsonl.gz", "wt", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")


# ---- tests (already built → reuse, so KWS sets in data/kws-test stay valid)
import os
REBUILD_TESTS = not os.path.exists(f"{FK}/kws_cuts_aihub_noisy_test.jsonl.gz")
train96_spk = {json.loads(l)["speaker"] for l in gzip.open(f"{M}/aihub96_supervisions_train.jsonl.gz")}
for name, prefix, excl in ([("aihub_cmd_test", "aihub96", train96_spk), ("aihub_noisy_test", "aihub71405", set())] if REBUILD_TESTS else []):
    cs = [c for c in cuts_of(prefix, "valid") if c.supervisions[0].speaker not in excl]
    rng.shuffle(cs)
    per, out = Counter(), []
    for c in cs:
        t = c.supervisions[0].text
        if per[t] >= 2:
            continue
        per[t] += 1
        out.append(c)
        if len(out) == 3000:
            break
    out.sort(key=lambda c: c.id)
    write([json.dumps(c.to_dict(), ensure_ascii=False) for c in out], name)
    print(f"{name}: {len(out)} utts {sum(c.duration for c in out) / 3600:.2f} h, "
          f"{len({c.supervisions[0].speaker for c in out})} speakers (pool {len(cs)})", flush=True)

# ---- train: drop every utterance whose sentence (spaces ignored) occurs in any test set.
# The command corpora are scripted: without this 99% of the command test sentences were in train
# (speakers differed, sentences did not) and CER/FRR were inflated (run exp/pt5k-tiny2-leaky).
TESTS = ["kspon_eval_clean", "kspon_eval_other", "fleurs_test", "aihub_cmd_test", "aihub_noisy_test"]
key = lambda t: t.replace(" ", "")
test_txt = {key(json.loads(l)["supervisions"][0]["text"]) for ts in TESTS
            for l in gzip.open(f"{FK}/kws_cuts_{ts}.jsonl.gz", "rt", encoding="utf-8")}
print("test sentences", len(test_txt), flush=True)
lines = []
for l in gzip.open(f"{FK}/kws_cuts_pretrain_mux.jsonl.gz", "rt", encoding="utf-8"):
    if key(json.loads(l)["supervisions"][0]["text"]) not in test_txt:
        lines.append(l.rstrip("\n"))
print("pretrain_mux", len(lines), flush=True)
for prefix, split in [("aihub485", "train"), ("aihub109", "train"), ("aihub96", "train_cap20"), ("aihub71405", "train_cap20")]:
    n0, drop = len(lines), 0
    for c in cuts_of(prefix, split):
        if key(c.supervisions[0].text) in test_txt:
            drop += 1; continue
        lines.append(json.dumps(c.to_dict(), ensure_ascii=False))
    print(prefix, split, len(lines) - n0, "dropped (test sentence)", drop, flush=True)
rng.shuffle(lines)
write(lines, "pt5k")
print("pt5k", len(lines))
