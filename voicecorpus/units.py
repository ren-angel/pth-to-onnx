"""Source-speech features of the Timbre Studio corpus voice (16 kHz, 10 ms frames). Mirrors SourceAnalyzer in the app."""
import numpy as np
from scipy.fft import dct
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); from vfeat import bank, EDGES, NB
from pitch16 import frame_pitch
FS, H, W = 16000, 160, 512
WIN = np.hanning(W + 1)[:W]
BANK = bank(FS, W)
NC = 12
UP, DOWN = 1 - np.exp(-0.01 / 0.25), 1 - np.exp(-0.01 / 4.0)
CMN = 1 - np.exp(-0.01 / 3.0)
PMED = 1 - np.exp(-0.01 / 2.0)

def raw_frames(x):
    """Per 10 ms hop t (frame ends at (t+1)*H): band dB L, cepstra c, level, tracker hz/conf."""
    xp = np.concatenate([np.zeros(W - H), x]); n = len(x) // H
    idx = np.arange(W)[None] + H * np.arange(n)[:, None]
    P = np.abs(np.fft.rfft(xp[idx] * WIN, W)) ** 2 / (WIN ** 2).sum()
    L = 10 * np.log10(P @ BANK.T + 1e-10)
    lev = L.mean(1)
    c = dct(L - lev[:, None], type=2, norm='ortho', axis=1)[:, 1:NC + 1] / 10
    est = np.array([frame_pitch(xp[t * H:t * H + W]) for t in range(n)]).reshape(-1, 2)
    return L, c, lev, est[:, 0], est[:, 1]

MULTIPLES = np.log2(np.array([1.0, 2.0, 3.0, 0.5], np.float32)).astype(np.float32)

def octave_fix(log_hz, recent):
    """Snap period-tracker octave/third errors back to the median of recent corrected voiced pitch (log2 Hz)."""
    if len(recent) < 3: return log_hz
    ref = np.float32(np.sort(np.asarray(recent, np.float32))[len(recent) // 2])
    if abs(log_hz - ref) < 0.4: return log_hz
    best = log_hz
    for k in MULTIPLES[1:]:
        cand = np.float32(log_hz + k)
        if abs(cand - ref) < 0.2 and abs(cand - ref) < abs(best - ref): best = cand
    return best

def features(c, lev, hz, conf, cmean0, pmed0):
    """Causal normalised features (as computed live)."""
    n = len(c); F = np.zeros((n, NC + 8 + 3)); m = cmean0.copy(); r = -45.0; pm = pmed0
    hist = np.zeros((3, NC)); recent = []
    for t in range(n):
        r += (UP if lev[t] > r else DOWN) * (lev[t] - r); r = max(r, -80)
        rel = np.clip(lev[t] - r, -60, 0)
        if rel > -20: m += CMN * (c[t] - m)
        cn = c[t] - m
        v = conf[t]
        lh = np.log2(hz[t])
        if v > .5:
            lh = octave_fix(np.float32(lh), recent); recent = (recent + [lh])[-9:]
        if v > .5 and rel > -30: pm += PMED * (lh - pm)
        p = (lh - pm) if v > .5 else 0.0
        F[t, :NC] = cn; F[t, NC:NC + 8] = cn[:8] - hist[1, :8]
        F[t, NC + 8] = rel / 20; F[t, NC + 9] = v; F[t, NC + 10] = np.clip(p, -1, 1)
        hist = np.roll(hist, 1, 0); hist[0] = cn
    return F
BW = (EDGES[2:] - EDGES[:-2]) / 2
def power_level(L):
    """Bandwidth-weighted power in dB (loudness for gain matching)."""
    return 10 * np.log10((10 ** (L / 10) * BW).sum(1) + 1e-10)
WEIGHTS_CEP = np.array([1.0] * NC + [0.5] * 8 + [1.2, 1.2, 2.5])
WEIGHTS = WEIGHTS_CEP

# Feature set: 'cep' = normalised cepstra + deltas (VCP2); 'phon' = phonetic student embedding (VCP3).
MODE = 'cep'
STUDENT = None
PITCH_WEIGHT = 1.5

def use_student(model):
    """model: dict(W=[(w, b)...], M0=...) from train_student.py; switches features to the phonetic set."""
    global MODE, STUDENT, WEIGHTS
    MODE, STUDENT = 'phon', model
    WEIGHTS = np.array([1.0] * 32 + [1.2, 1.2, PITCH_WEIGHT])

def dim(): return NC + 8 + 3 if MODE == 'cep' else 35
def rel_index(): return NC + 8 if MODE == 'cep' else 32

def features_any(L, c, lev, hz, conf, cmean0, pmed0):
    F = features(c, lev, hz, conf, cmean0, pmed0)
    if MODE == 'cep': return F
    from student import frame_inputs, context, mlp
    E = mlp(context(frame_inputs(L, conf, STUDENT['M0']), STUDENT.get('taps')), STUDENT['W'])
    return np.concatenate([E, F[:, NC + 8:NC + 11]], 1)
