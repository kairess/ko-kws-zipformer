"""
On-the-fly crop + concat applied to the cut STREAM before the bucketing sampler .

Why here: concatenating inside an already-bucketed batch turns a batch of hundreds of short cuts into a
batch padded to 10–20 s. Those ever-changing batch shapes made the trainer's native memory grow by ~5 GB
per 100 batches (OOM). Augmenting the stream lets the sampler bucket crops and concatenations by their
real length, so batches look like the unaugmented ones. Audio is still read in the dataloader workers.

Output fractions (by count) match crop_frac / concat_frac exactly in expectation: each output is an
untouched cut, a crop (1..4 aligned words; boundaries from aug_utils.crop_bounds), or a lazy
concatenation (MixedCut, 0.1–0.4 s gaps) of 2–3 consecutive cuts, capped at max_concat_s. A fresh random
draw every epoch. ConcatCollapse (dataset) turns a concat MixedCut into a single-supervision MonoCut.
"""
import random, sys
from pathlib import Path
from lhotse import MonoCut, Recording, SupervisionSegment
from lhotse.audio import AudioSource
from lhotse.cut import MixedCut

sys.path.insert(0, str(Path(__file__).parents[1] / "local"))
from aug_utils import crop_bounds  # noqa: E402


class AugStream:
    def __init__(self, cuts, crop_frac=0.0, concat_frac=0.0, max_words=4, max_crop_s=5.0, max_concat_s=20.0, seed=0):
        self.cuts, self.cf, self.kf = cuts, crop_frac, concat_frac
        self.max_words, self.max_crop_s, self.max_concat_s, self.seed, self.epoch = max_words, max_crop_s, max_concat_s, seed, 0

    def _crop(self, c, rng):
        W = (c.supervisions[0].custom or {}).get("words") if c.supervisions else None
        if not W or len(W) < 2: return None
        for _ in range(4):
            k = rng.randint(1, min(self.max_words, len(W) - 1)); i = rng.randint(0, len(W) - k); j = i + k - 1
            s, e = crop_bounds(W, i, j, c.duration)
            if 0.3 <= e - s <= self.max_crop_s: break
        else:
            return None
        t = c.truncate(offset=s, duration=e - s, keep_excessive_supervisions=False)
        sup = c.supervisions[0]
        t.supervisions = [SupervisionSegment(id=f"{c.id}_crop{i}-{j}", recording_id=t.recording_id, start=0.0, duration=t.duration,
                                             channel=sup.channel, language=sup.language, speaker=sup.speaker,
                                             text=" ".join(w[0] for w in W[i : j + 1]), custom={"aug": "crop"})]
        t.id = f"{c.id}_crop{i}-{j}"
        return t

    def __iter__(self):
        """Random decisions whose probabilities are corrected every 1000 outputs by target/actual, so the
        output fractions converge to crop_frac / concat_frac even though only aligned cuts can be cropped
        and some concat groups cannot be filled. Selection stays random → a new draw every epoch."""
        rng = random.Random(self.seed * 1000 + self.epoch); self.epoch += 1
        p_cat, p_crop = self.kf, self.cf          # p_crop applies to cuts that HAVE an alignment
        n = {"orig": 0, "crop": 0, "concat": 0}
        def emit(kind):
            n[kind] += 1
            nonlocal p_cat, p_crop
            tot = sum(n.values())
            if tot % 1000 == 0:
                if self.kf > 0: p_cat = min(0.9, max(0.01, p_cat * self.kf / max(1e-3, n["concat"] / tot)))
                if self.cf > 0: p_crop = min(1.0, max(0.01, p_crop * self.cf / max(1e-3, n["crop"] / tot)))
        it = iter(self.cuts)
        for c in it:
            aligned = bool(c.supervisions and (c.supervisions[0].custom or {}).get("words"))
            r = rng.random()
            if self.kf > 0 and r < p_cat:
                group = [c]; target = rng.choice([2, 2, 3]); dur = c.duration
                while len(group) < target:
                    try: nx = next(it)
                    except StopIteration: break
                    if dur + nx.duration + 0.4 > self.max_concat_s:
                        emit("orig"); yield nx; break          # does not fit: emit untouched, stop filling
                    group.append(nx); dur += nx.duration + 0.25
                if len(group) < 2:
                    emit("orig"); yield c; continue
                m = group[0]
                for x in group[1:]:
                    m = m.pad(duration=m.duration + rng.uniform(0.1, 0.4)).append(x)
                m.id = "concat_" + "+".join(x.id for x in group)[:150]
                emit("concat"); yield m
            elif self.cf > 0 and aligned and rng.random() < p_crop:
                t = self._crop(c, rng)
                if t is None: emit("orig"); yield c
                else: emit("crop"); yield t
            else:
                emit("orig"); yield c


def collapse_concat(cut):
    """In a dataloader worker: MixedCut built by AugStream → MonoCut with in-memory audio and one supervision."""
    if not (isinstance(cut, MixedCut) and cut.id.startswith("concat_")): return cut
    import io
    import numpy as np
    import soundfile as sf
    y = cut.load_audio()[0].astype(np.float32)
    y = y + (np.random.default_rng(abs(hash(cut.id)) % (1 << 30)).standard_normal(len(y)) * 1e-3 * (np.abs(y) < 1e-6)).astype(np.float32)
    buf = io.BytesIO(); sf.write(buf, y, 16000, format="WAV", subtype="FLOAT")
    rec = Recording(id=cut.id, sources=[AudioSource(type="memory", channels=[0], source=buf.getvalue())],
                    sampling_rate=16000, num_samples=len(y), duration=len(y) / 16000)
    text = " ".join(s.text for s in sorted(cut.supervisions, key=lambda s: s.start))
    return MonoCut(id=cut.id, start=0.0, duration=rec.duration, channel=0, recording=rec,
                   supervisions=[SupervisionSegment(id=cut.id, recording_id=cut.id, start=0.0, duration=rec.duration, channel=0,
                                                    language="Korean", text=text, custom={"aug": "concat"})])
