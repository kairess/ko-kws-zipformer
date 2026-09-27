"""Fast Ogg/Opus writer via PyAV (libopus). libsndfile's Opus path is ~12x slower
(complexity fixed at 10). Default: 32 kbps, complexity 5, 16 kHz mono.
Audio is fed in 20 ms frames (one big frame produced files libsndfile rejects)."""
import av, numpy as np

FRAME = 320  # 20 ms at 16 kHz


def write_opus(path, x, sr=16000, bit_rate=32000, complexity=5):
    x = np.ascontiguousarray(np.clip(x, -1, 1), dtype=np.float32)
    with av.open(str(path), "w", format="ogg") as c:
        s = c.add_stream("libopus", rate=sr, options={"compression_level": str(complexity), "application": "audio"})
        s.bit_rate = bit_rate
        s.layout = "mono"
        for i in range(0, len(x), FRAME):
            fr = av.AudioFrame.from_ndarray(x[None, i:i + FRAME], format="flt", layout="mono")
            fr.sample_rate = sr
            fr.pts = i
            for p in s.encode(fr):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)
