"""Build a live corpus voice (.vcp) for Timbre Studio from an RVC ONNX model (encoder + synthesizer).
Runs the model over the bundled reference speech (data/source_speech.src) and compiles the corpus.

Usage:
  python -m voicecorpus.build ENCODER.onnx SYNTH.onnx "Voice name" out.vcp [--hz 110]
ENCODER: ContentVec/HuBERT ONNX matching the model (e.g. dist/rvc/contentvec.onnx, output unit12 or units9);
SYNTH: RVC ONNX exported with metadata (see RVC_MODELS.md). --hz: median pitch of the voice (110 low, 150 mid, 220 high).
Import the .vcp in the app: Biblioteca > Ao vivo > Importar voz ao vivo.
"""
import argparse, json, os, struct, sys, tempfile
import numpy as np, onnxruntime as ort
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(__file__))
from corpus_file import compile_corpus, write

def feature_name(meta):
    """Encoder output the synthesizer expects (same rule as RvcPipeline.synth in the app)."""
    layer, proj, dim = int(meta.get('embOutputLayer', 12)), bool(meta.get('useFinalProj', False)), int(meta.get('embChannels', 768))
    if layer == 12 and not proj and dim == 768: return 'unit12'
    if layer == 9 and proj and dim == 256: return 'units9'
    if layer == 12 and proj and dim == 256: return 'unit12s'
    raise SystemExit(f'Unsupported layer/projection/dims: {layer}/{proj}/{dim}')


def build_vcp(encoder, synth, name, out, hz=110.0, source=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'source_speech.src'), log=print):
    o = ort.SessionOptions(); o.intra_op_num_threads = os.cpu_count() or 4
    enc = ort.InferenceSession(str(encoder), o, providers=['CPUExecutionProvider'])
    syn = ort.InferenceSession(str(synth), o, providers=['CPUExecutionProvider'])
    meta = json.loads(syn.get_modelmeta().custom_metadata_map.get('metadata', '{}'))
    if not meta: raise SystemExit('The synthesizer ONNX has no "metadata" (export it with tools/rvc_to_phone.py).')
    rate = int(meta['samplingRate']); f0 = bool(meta['f0']); key = feature_name(meta)
    if key not in [x.name for x in enc.get_outputs()]: raise SystemExit(f'Encoder has no output {key}')
    b = open(source, 'rb').read(); assert b[:4] == b'SRC1'
    n = struct.unpack_from('<i', b, 4)[0]; at = 8; work = tempfile.mkdtemp(); files = []
    for i in range(n):
        spk, med, ns, nf = struct.unpack_from('<ifii', b, at); at += 16
        pf = np.frombuffer(b, '<f4', nf, at).copy(); at += 4 * nf
        x = np.frombuffer(b, '<i2', ns, at) / 32767.0; at += 2 * ns
        x = x[:len(x) // 320 * 320].astype(np.float32)
        feats = enc.run([key], {'audio': x[None]})[0].repeat(2, axis=1); T = feats.shape[1]
        p = np.zeros(T, np.float32); m = min(T, len(pf)); p[:m] = pf[:m] * (hz / med)
        inputs = {'feats': feats, 'p_len': np.array([T]), 'sid': np.array([0], np.int64)}
        if f0:
            mel = np.log1p(p / 700)
            inputs['pitch'] = np.where(p > 0, np.rint(1 + 254 * (mel - np.log1p(50 / 700)) / (np.log1p(1100 / 700) - np.log1p(50 / 700))), 1).clip(1, 255).astype(np.int64)[None]
            inputs['pitchf'] = p[None]
        y = syn.run(['audio'], inputs)[0].reshape(-1)
        g = np.gcd(16000, rate); y = resample_poly(y, 16000 // g, rate // g)[:len(x)]; y = np.pad(y, (0, len(x) - len(y)))
        # compile_corpus expects 40 kHz targets (it resamples by 2/5 to 16 kHz).
        path = os.path.join(work, f'train_{spk}_{i}.npz'); np.savez(path, x=x, y=resample_poly(y, 5, 2), f0=p, med=med); files.append(path)
        log(f'  reference speech {i + 1}/{n}')
    # VCP3: match on the bundled phonetic content model (also embedded in the .vcp for the app).
    import units as U
    from student import from_bytes
    U.use_student(from_bytes(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'content_model.bin'), 'rb').read()))
    K = compile_corpus(files); write(out, K, name)
    return os.path.getsize(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('encoder'); ap.add_argument('synth'); ap.add_argument('name'); ap.add_argument('out')
    ap.add_argument('--hz', type=float, default=110.0)
    ap.add_argument('--source', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'source_speech.src'))
    a = ap.parse_args()
    size = build_vcp(a.encoder, a.synth, a.name, a.out, a.hz, a.source)
    print('wrote', a.out, round(size / 2**20, 1), 'MiB')
