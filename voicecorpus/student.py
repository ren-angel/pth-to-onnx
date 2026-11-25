"""Phonetic student: causal spectral context -> PCA(ContentVec unit12). Shared by training, corpus build and evaluation."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from units import H, NB
# Past frames used as context; LOOKAHEAD future frames delay the voice by LOOKAHEAD x 10 ms.
LOOKAHEAD = int(os.environ.get('STUDENT_LOOKAHEAD', 0))
TAPS = tuple(range(-LOOKAHEAD, 0)) + (0, 1, 2, 3, 4, 6, 8, 11, 14)
MEAN_RATE = 1 - np.exp(-0.01 / 3.0)
UP, DOWN = 1 - np.exp(-0.01 / 0.25), 1 - np.exp(-0.01 / 4.0)
D_IN = len(TAPS) * (NB + 2)  # set at import from STUDENT_LOOKAHEAD

def frame_inputs(L, conf, m0):
    """Causal per-frame vectors [normalised band shape (32), rel level, voicing] with a running channel mean."""
    n = len(L); out = np.zeros((n, NB + 2), np.float32); m = m0.astype(np.float64).copy(); r = -45.0
    for t in range(n):
        lev = L[t].mean(); shape = L[t] - lev
        r += (UP if lev > r else DOWN) * (lev - r); r = max(r, -80.0)
        rel = min(max(lev - r, -60.0), 0.0)
        if rel > -20: m += MEAN_RATE * (shape - m)
        out[t, :NB] = (shape - m) / 10.0; out[t, NB] = rel / 20.0; out[t, NB + 1] = conf[t]
    return out

def context(Fr, taps=None):
    """Stack frames at offsets taps (positive = past, negative = future), repeating the edges."""
    taps = TAPS if taps is None else taps
    n = len(Fr)
    def shift(k):
        if k > 0: return np.concatenate([np.repeat(Fr[:1], k, 0), Fr[:n - k]])
        if k < 0: return np.concatenate([Fr[-k:], np.repeat(Fr[-1:], -k, 0)])
        return Fr
    return np.concatenate([shift(k) for k in taps], 1)

def to_bytes(model):
    """Binary student block ("CST1", little-endian) embedded in VCP3 files and read by SourceAnalyzer."""
    import struct
    W = model['W']; taps = model.get('taps', TAPS)
    hid, d_in = W[0][0].shape
    b = bytearray(b'CST1') + struct.pack('<2i', int(model.get('lookahead', 0)), len(taps)) + struct.pack(f'<{len(taps)}i', *taps)
    b += struct.pack('<3i', d_in, hid, W[2][0].shape[0])
    for a in [model['M0']] + [t for wb in W for t in wb]: b += np.ascontiguousarray(a, '<f4').tobytes()
    return bytes(b)

def from_bytes(b):
    import struct
    assert b[:4] == b'CST1'
    la, nt = struct.unpack_from('<2i', b, 4); taps = struct.unpack_from(f'<{nt}i', b, 12); o = 12 + 4 * nt
    d_in, hid, out = struct.unpack_from('<3i', b, o); o += 12
    def take(n):
        nonlocal o
        a = np.frombuffer(b, '<f4', n, o).astype(np.float32); o += 4 * n; return a
    M0 = take(NB)
    W = [(take(hid * d_in).reshape(hid, d_in), take(hid)), (take(hid * hid).reshape(hid, hid), take(hid)), (take(out * hid).reshape(out, hid), take(out))]
    assert o == len(b)
    return dict(lookahead=la, taps=tuple(taps), W=W, M0=M0)

def mlp(X, W):
    h = X
    for i, (w, b) in enumerate(W):
        h = h @ w.T + b
        if i < len(W) - 1: h = np.maximum(h, 0)
    return h

def teacher_at_frames(T20, n):
    """Interpolate 20 ms teacher frames at each 10 ms analysis-window centre (160t - 96 samples)."""
    tau = (160 * np.arange(n) - 96 - 200) / 320.0
    k0 = np.clip(np.floor(tau).astype(int), 0, len(T20) - 1); k1 = np.clip(k0 + 1, 0, len(T20) - 1)
    a = np.clip(tau - np.floor(tau), 0, 1)[:, None]
    return (1 - a) * T20[k0] + a * T20[k1]
