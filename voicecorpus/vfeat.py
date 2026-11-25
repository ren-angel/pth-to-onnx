"""Reference implementation of the live 'TITAN filter' analysis. Must match TitanFilter.kt."""
import numpy as np
NB = 32
EDGES = np.concatenate([np.arange(0, 1001, 100.0), 1000 * 8.0 ** (np.arange(1, 24) / 23)])  # 34 points
assert len(EDGES) == NB + 2
CTX = 4  # current + 3 past frames

def geometry(fs):
    W = int(round(0.032 * fs)); W -= W % 4; H = W // 4
    N = 1 << int(np.ceil(np.log2(W)))
    return W, H, N

def window(W):
    return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(W) / W))

def bank(fs, N):
    f = np.arange(N // 2 + 1) * fs / N
    T = np.zeros((NB, len(f)))
    for b in range(NB):
        lo, c, hi = EDGES[b], EDGES[b + 1], EDGES[b + 2]
        T[b] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
        T[b] /= T[b].sum()
    return T

def stft(x, fs):
    W, H, N = geometry(fs)
    w = window(W)
    xp = np.concatenate([np.zeros(W - H), x])
    n = (len(xp) - W) // H + 1
    idx = np.arange(W)[None] + H * np.arange(n)[:, None]
    return np.fft.rfft(xp[idx] * w, N)  # frame i ends at sample (i+1)*H

def band_db(Z, fs):
    W, H, N = geometry(fs)
    P = np.abs(Z) ** 2 / (window(W) ** 2).sum() / (fs / 16000)
    return 10 * np.log10(P @ bank(fs, N).T + 1e-10)

# Level tracker / channel mean constants (per frame, hop 8 ms)
UP, DOWN = 1 - np.exp(-0.008 / 0.25), 1 - np.exp(-0.008 / 4.0)
MEAN_RATE = 1 - np.exp(-0.008 / 2.0)
REF0 = -45.0

def features(L, m0):
    """L: frames x NB band dB. Returns frames x F feature matrix (causal), plus active mask."""
    n = len(L)
    lev = L.mean(1)
    shape = L - lev[:, None]
    ref = np.empty(n); r = REF0
    mean = np.empty((n, NB)); m = m0.copy()
    for i in range(n):
        r += (UP if lev[i] > r else DOWN) * (lev[i] - r)
        r = max(r, -80.0)
        ref[i] = r
        if lev[i] > r - 20: m += MEAN_RATE * (shape[i] - m)
        mean[i] = m
    rel = np.clip(lev - ref, -60, 20) / 20
    cols = [shape / 20, rel[:, None], mean / 20]
    for k in range(1, CTX):
        sp = np.concatenate([np.repeat(shape[:1] * 0, k, 0), shape[:-k]]) / 20
        rp = np.concatenate([np.full(k, -3.0), rel[:-k]])[:, None]
        cols += [sp, rp]
    return np.concatenate(cols, 1).astype(np.float32), rel
