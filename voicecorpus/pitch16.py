"""Frame pitch for the corpus voice (twin of SourceAnalyzer pitch): window-corrected autocorrelation of
the 32 ms analysis frame, first peak >= 0.85 * strongest in 60-600 Hz, parabolic refinement."""
import numpy as np
FS, W = 16000, 512
WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(W) / W)).astype(np.float32)
LMIN, LMAX = FS // 600, FS // 60        # 26 .. 266
_rw = np.fft.irfft(np.abs(np.fft.rfft(WIN, 1024)) ** 2, 1024)[:LMAX + 2]
RW = (_rw / _rw[0]).astype(np.float32)

def frame_pitch(frame):
    """frame: W samples (oldest first, unwindowed). Returns (hz, confidence 0..1)."""
    x = frame * WIN
    if float(x @ x) < 1e-7: return 150.0, 0.0
    r = np.fft.irfft(np.abs(np.fft.rfft(x, 1024)) ** 2, 1024)[:LMAX + 2]
    if r[0] <= 0: return 150.0, 0.0
    nr = (r / r[0]) / np.maximum(RW, 1e-3)
    seg = nr[LMIN:LMAX + 1]
    best = float(seg.max())
    if best < 0.3: return 150.0, 0.0
    for lag in range(LMIN + 1, LMAX):
        s = nr[lag]
        if s >= 0.85 * best and s >= nr[lag - 1] and s > nr[lag + 1]:
            curv = 2 * s - nr[lag - 1] - nr[lag + 1]
            dl = 0.5 * (nr[lag + 1] - nr[lag - 1]) / curv if curv > 1e-4 else 0.0
            return FS / (lag + float(np.clip(dl, -0.5, 0.5))), float(np.clip((min(s, 1.0) - 0.5) * 2.4, 0, 1))
    return 150.0, 0.0
