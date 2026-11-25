"""Frame tables of paired clips (source speech, voice-model output) for the corpus compiler."""
import sys, os, numpy as np
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(__file__)); from units import *

PADF = 4
def build(files, max_frames=None):
    Ys, Fs, LY, valid = [], [], [], []
    raws = []
    for f in files:
        d = np.load(f); x = d['x']; y = resample_poly(d['y'], 2, 5)[:len(x)]; y = np.pad(y, (0, len(x) - len(y)))
        L, c, lev, hz, conf = raw_frames(x)
        raws.append((f, x, y, c, lev, hz, conf, float(d['med'])))
    spk = {}
    for f, x, y, c, lev, hz, conf, med in raws:
        s = os.path.basename(f).split('_')[1]; a = lev > lev.max() - 25
        spk.setdefault(s, []).append(c[a])
    spk = {s: np.concatenate(v).mean(0) for s, v in spk.items()}
    off = 0
    for f, x, y, c, lev, hz, conf, med in raws:
        s = os.path.basename(f).split('_')[1]
        F = features(c, lev, hz, conf, spk[s].copy(), np.log2(med))
        Ly = power_level(raw_frames(y)[0])
        n = len(F); ok = (F[:, NC + 8] > -1.75) & (Ly > Ly.max() - 40)
        ok[:6] = False; ok[-6:] = False
        Ys.append(np.concatenate([y[:n * H], np.zeros(PADF * H)])); Fs.append(F); LY.append(Ly); valid.append(ok)
        # frame t of this utterance -> global frame index off+t; audio padded by W per utterance
        Fs.append(np.zeros((PADF, F.shape[1]))); LY.append(np.full(PADF, -100.0)); valid.append(np.zeros(PADF, bool))
    Y = np.concatenate(Ys); F = np.concatenate(Fs); LY = np.concatenate(LY); valid = np.concatenate(valid)
    mu = F[valid].mean(0); sd = F[valid].std(0) + 1e-6
    gm = np.mean([v for v in spk.values()], 0)
    return dict(Y=Y, F=F, LY=LY, valid=valid, mu=mu, sd=sd, cmean0=gm)

LAGS = (0, 3, 6)
LAGW = (1.0, 0.7, 0.5)
def stack(F):
    out = []
    for lag, w in zip(LAGS, LAGW):
        G = np.concatenate([np.repeat(F[:1], lag, 0), F[:len(F) - lag]]) if lag else F
        out.append(G * w)
    return np.concatenate(out, 1)
