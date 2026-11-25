"""Compile a unit-selection corpus file (VCP1) and the matching Python twin of CorpusVoice.kt."""
import sys, os, struct, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from units import *
from units_synth import build as build_raw, LAGS, LAGW
PCA_DIM, CLUSTERS, PROBES = 24, 256, 8
D = NC + 8 + 3

def compile_corpus(files, seed=0, max_minutes=None):
    C = build_raw(files)
    w = WEIGHTS / C['sd']
    Z = (C['F'] - C['mu']) * w
    S = np.concatenate([np.concatenate([np.repeat(Z[:1], l, 0), Z[:len(Z) - l]]) * g if l else Z * g for l, g in zip(LAGS, LAGW)], 1)
    cand = np.flatnonzero(C['valid'])
    X = S[cand]; mean = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    P = Vt[:PCA_DIM].T.astype(np.float32)                       # (3D x PCA_DIM)
    E = ((X - mean) @ P).astype(np.float32)
    rng = np.random.default_rng(seed)
    cent = E[rng.choice(len(E), CLUSTERS, replace=False)].copy()
    for it in range(15):
        a = np.argmin(((E[:, None, :] - cent[None]) ** 2).sum(2) if len(E) < 5000 else np.stack([((E - c) ** 2).sum(1) for c in cent], 1), 1)
        for k in range(CLUSTERS):
            m = a == k
            cent[k] = E[m].mean(0) if m.any() else E[rng.integers(len(E))]
    order = np.argsort(a, kind='stable')
    starts = np.searchsorted(a[order], np.arange(CLUSTERS + 1))
    return dict(Y=C['Y'], LY=C['LY'].astype(np.float32), valid=C['valid'], mu=C['mu'].astype(np.float32),
                w=w.astype(np.float32), mean=mean.astype(np.float32), P=P, cent=cent.astype(np.float32),
                cand=cand[order].astype(np.int32), E=E[order], starts=starts.astype(np.int32), cmean0=C['cmean0'].astype(np.float32))

def write(path, K, name='Villager'):
    with open(path, 'wb') as f:
        nm = name.encode('utf-8')
        f.write(b'VCP2'); f.write(struct.pack('<6i', FS, H, len(K['LY']), len(K['Y']), len(K['cand']), CLUSTERS))
        f.write(struct.pack('<3i', D, len(LAGS) * D, PCA_DIM)); f.write(struct.pack('<i', len(nm))); f.write(nm + b'\0' * (-len(nm) % 4))
        for a in ('mu', 'w', 'cmean0', 'mean', 'P', 'cent', 'E', 'LY'):
            f.write(np.ascontiguousarray(K[a], '<f4').tobytes())
        f.write(np.ascontiguousarray(K['cand'], '<i4').tobytes()); f.write(np.ascontiguousarray(K['starts'], '<i4').tobytes())
        f.write((np.clip(K['Y'], -1, 1) * 32767).astype('<i2').tobytes())

import os
STILL, LOOP, SLACK = 3.0, 6, 1.5  # sustain hold (see CorpusVoice.kt)
def synth(x, K, ratio=10.0, tol=80, stats=None, hold=True):
    """Offline twin of CorpusVoice (gate off). x: 16 kHz float."""
    L, c, lev, hz, conf = raw_frames(x)
    F = features(c, lev, hz, conf, K['cmean0'].astype(np.float64).copy(), np.log2(120.0))
    pw = power_level(L)
    Z = (F - K['mu']) * K['w']
    n = len(F); Y = K['Y']; G = 2 * H; han = np.hanning(G + 1)[:G]
    pos = -np.ones(len(K['LY']), int); pos[K['cand']] = np.arange(len(K['cand']))
    out = np.zeros(len(x) + 2 * G); prev = -1; pstart = 0; g = 0.0; runs = []; run = 0
    hist = []; anchor = np.inf; holds = 0
    def dist(f, e):
        k = pos[f] if 0 <= f < len(pos) else -1
        return np.inf if k < 0 else float(((K['E'][k] - e) ** 2).sum())
    for t in range(n):
        if F[t, NC + 8] <= -1.5:
            prev = -1
            if run: runs.append(run); run = 0
            continue
        q = np.concatenate([Z[max(t - l, 0)] * gw for l, gw in zip(LAGS, LAGW)])
        e = (q - K['mean']) @ K['P']
        dc = ((K['cent'] - e) ** 2).sum(1)
        best, j = np.inf, -1
        for k in np.argsort(dc, kind='stable')[:PROBES]:
            a, b = K['starts'][k], K['starts'][k + 1]
            if a == b: continue
            d = ((K['E'][a:b] - e) ** 2).sum(1); m = int(np.argmin(d))
            if d[m] < best: best, j = d[m], int(K['cand'][a + m])
        hist = (hist + [e])[-4:]
        still = hold and len(hist) == 4 and float(((hist[-1] - hist[0]) ** 2).sum()) < STILL
        if not still: anchor = np.inf
        cont = prev >= 0 and pos[prev + 1] >= 0 and ((K['E'][pos[prev + 1]] - e) ** 2).sum() <= ratio * best
        if still and prev >= 0:
            dn = dist(prev + 1, e); anchor = min(anchor, dist(prev, e))
            if dn > SLACK * anchor and best > 0.3 * anchor:
                back = [(dist(f, e), f) for f in range(prev - LOOP, prev) if dist(f, e) < np.inf]
                if back:
                    cont = False; j = min(back)[1]; holds += 1
        if cont:
            j = prev + 1; start = pstart + H; run += 1
        else:
            centre = (j + 1) * H - W // 2; start = centre - H
            if prev >= 0:
                nat = Y[pstart + H:pstart + 2 * H]; nn = float(nat @ nat); bestr = -2.0
                if nn > 1e-9:
                    for dl in range(-tol, tol + 1, 2):
                        s0 = centre - H + dl
                        if s0 < 0 or s0 + G > len(Y): continue
                        seg = Y[s0:s0 + H]; r = float(seg @ nat) / np.sqrt(float(seg @ seg) * nn + 1e-12)
                        if r > bestr: bestr, start = r, s0
            if run: runs.append(run)
            run = 1
        start = int(np.clip(start, 0, len(Y) - G))
        target = 10 ** ((pw[t] - K['LY'][j]) / 20)
        g = target if prev < 0 else g + 0.5 * (target - g)
        out[t * H:t * H + G] += Y[start:start + G] * han * g
        prev = j; pstart = start
    if run: runs.append(run)
    if stats is not None: stats['mean_run'] = float(np.mean(runs)) if runs else 0.0; stats['holds'] = holds
    return out[H:H + len(x)]

def read(path):
    b = open(path, 'rb').read(); assert b[:4] == b'VCP2'
    fs, h, nf, ns, nc, cl, d, d3, pd, nl = struct.unpack_from('<10i', b, 4); o = 44 + nl + (-nl % 4)
    K = dict(name=b[44:44 + nl].decode())
    for k, n in (('mu', d), ('w', d), ('cmean0', NC), ('mean', d3), ('P', d3 * pd), ('cent', cl * pd), ('E', nc * pd), ('LY', nf)):
        K[k] = np.frombuffer(b, '<f4', n, o).astype(np.float64); o += 4 * n
    K['P'] = K['P'].reshape(d3, pd); K['cent'] = K['cent'].reshape(cl, pd); K['E'] = K['E'].reshape(nc, pd)
    K['cand'] = np.frombuffer(b, '<i4', nc, o).astype(int); o += 4 * nc
    K['starts'] = np.frombuffer(b, '<i4', cl + 1, o).astype(int); o += 4 * (cl + 1)
    K['Y'] = np.frombuffer(b, '<i2', ns, o) / 32767.0; o += 2 * ns
    assert o == len(b); return K
