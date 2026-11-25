"""pth-to-onnx: turn an RVC voice model (.pth) into everything the Timbre Studio app needs, in one command.

    python pth_to_onnx.py MyVoice.pth --name "My voice" --hz 110

Writes to phone-files/ (or --out):
  <name>.vcp            live voice  -> app: Biblioteca > Ao vivo > Importar voz ao vivo  (the main file)
  <name>.onnx           synthesizer -> app: Biblioteca > RVC · fala > Importar sintetizador (file conversion)
  contentvec-int8.onnx  encoder     -> app: Biblioteca > RVC · fala > Importar encoder (once, shared by all voices)

Supports standard RVC v1 and v2 checkpoints, with or without pitch (f0), 32/40/48 kHz, as saved by the
RVC WebUI, Applio and similar trainers (the small "weights" .pth, ~50-60 MB, not the G_*.pth/D_*.pth training
checkpoints). The .index file is not used. --hz is the voice's typical pitch: ~110 low/male, ~150 middle,
~220 high/female; pick what sounds right for the voice.

Needs Python 3.10-3.12, git, internet on first run (downloads ~380 MB encoder and the pinned exporter code),
and: pip install -r requirements.txt
"""
import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get('PTH2ONNX_CACHE', Path.home() / '.cache' / 'pth-to-onnx'))
EXPORTER_REPO = 'https://github.com/w-okada/voice-changer'
EXPORTER_COMMIT = 'd8ef15799470193f7c8176ef471245753a656626'
ENCODER_URL = 'https://huggingface.co/wok000/vcclient_modules/resolve/main/contentvec/contentvec-f.onnx'
ENCODER_SHA256 = '4b31ed3d95a568fab7952de923ff7f7d3d17128ea6fce69f665509d24c3156db'


def step(msg):
    print(f'\n== {msg}', flush=True)


def exporter_sources():
    """Pinned w-okada voice-changer RVC sources (MIT + LICENSE-NOTICE), fetched once into the cache."""
    repo = CACHE / 'voice-changer'
    marker = repo / 'server' / 'voice_changer' / 'RVC' / 'onnxExporter'
    if not marker.is_dir():
        if shutil.which('git') is None:
            sys.exit('git is required on the first run (https://git-scm.com/downloads).')
        if repo.exists():
            shutil.rmtree(repo)
        CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '-q', '--filter=blob:none', '--no-checkout', EXPORTER_REPO, str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '-q', EXPORTER_COMMIT, '--', 'server/voice_changer/RVC'], check=True)
    return repo / 'server'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def encoder_fp32(explicit):
    if explicit:
        return Path(explicit)
    candidate = CACHE / 'contentvec.onnx'
    if candidate.is_file() and sha256(candidate) == ENCODER_SHA256:
        return candidate
    target = CACHE / 'contentvec.onnx'
    CACHE.mkdir(parents=True, exist_ok=True)
    print('  downloading ContentVec encoder (~380 MB, once)...', flush=True)
    urllib.request.urlretrieve(ENCODER_URL, str(target) + '.part')
    if sha256(str(target) + '.part') != ENCODER_SHA256:
        os.remove(str(target) + '.part')
        sys.exit('Encoder download corrupted (hash mismatch); try again.')
    os.replace(str(target) + '.part', target)
    return target


def encoder_int8(fp32, out_dir):
    """Smaller, faster encoder for the phone (dynamic INT8 on MatMul weights; same outputs)."""
    target = out_dir / 'contentvec-int8.onnx'
    cached = CACHE / 'contentvec-int8.onnx'
    if not cached.is_file():
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(fp32), str(cached) + '.part', op_types_to_quantize=['MatMul'],
                         weight_type=QuantType.QInt8, extra_options={'MatMulConstBOnly': True})
        os.replace(str(cached) + '.part', cached)
    if not target.exists():
        shutil.copyfile(cached, target)
    return target


def export_onnx(pth, out_path):
    import onnx
    import onnxruntime as ort
    import torch
    server = exporter_sources()
    sys.path.insert(0, str(server))
    try:
        cpt = torch.load(pth, map_location='cpu', weights_only=True)
    except Exception as e:
        sys.exit(f'Could not read {pth} as an RVC model: {e}')
    if not isinstance(cpt, dict) or 'weight' not in cpt or 'config' not in cpt:
        sys.exit('This .pth is not an RVC inference model (needs "weight" and "config"). Use the small model '
                 'file from the trainer\'s weights folder, not G_*.pth/D_*.pth checkpoints.')
    version = str(cpt.get('version', 'v1'))
    f0 = int(cpt.get('f0', 1)) == 1
    config = list(cpt['config'])
    rate = int(config[-1])
    if rate not in (32000, 40000, 48000):
        sys.exit(f'Unsupported sample rate {rate}')
    v2 = version == 'v2'
    name = f"SynthesizerTrnMs{'768' if v2 else '256'}NSFsid{'' if f0 else '_nono'}_ONNX"
    cls = getattr(importlib.import_module('voice_changer.RVC.onnxExporter.' + name), name)
    net = cls(*config, is_half=False) if f0 else cls(*config)
    weights = {k: v.float() for k, v in cpt['weight'].items()}
    if not any(k.startswith('enc_q.') for k in weights) and hasattr(net, 'enc_q'):
        del net.enc_q  # training-only encoder, absent from inference files
    missing, unexpected = net.load_state_dict(weights, strict=False)
    if missing or unexpected:
        sys.exit(f'Model layout mismatch (missing {missing[:3]}, unexpected {unexpected[:3]})')
    net.eval()
    dim = 768 if v2 else 256
    t = 64
    inputs = [torch.randn(1, t, dim), torch.tensor([t])]
    names = ['feats', 'p_len']
    if f0:
        inputs += [torch.full((1, t), 100), torch.full((1, t), 150.)]
        names += ['pitch', 'pitchf']
    inputs.append(torch.tensor([0]))
    names.append('sid')
    with torch.inference_mode():
        torch.onnx.export(net, tuple(inputs), str(out_path), input_names=names, output_names=['audio'],
                          dynamic_axes={k: {1: 'frames'} for k in names if k in ('feats', 'pitch', 'pitchf')} | {'audio': {0: 'samples'}},
                          opset_version=17, dynamo=False)
    model = onnx.load(str(out_path))
    meta = dict(samplingRate=rate, f0=f0, embChannels=dim, embedder='contentvec',
                embOutputLayer=12 if v2 else 9, useFinalProj=not v2, version=version)
    onnx.helper.set_model_props(model, {'metadata': json.dumps(meta)})
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    # Quick check: ONNX Runtime runs it and returns the expected duration.
    session = ort.InferenceSession(str(out_path), providers=['CPUExecutionProvider'])
    feed = {n: x.numpy() for n, x in zip(names, inputs)}
    audio = session.run(['audio'], feed)[0].reshape(-1)
    assert np.isfinite(audio).all() and abs(len(audio) - t * rate // 100) <= rate // 20, 'exported model output is invalid'
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pth', type=Path, help='RVC model (.pth)')
    ap.add_argument('--name', help='voice name shown in the app (default: file name)')
    ap.add_argument('--hz', type=float, default=110.0, help='typical pitch of the voice: 110 low, 150 middle, 220 high')
    ap.add_argument('--out', type=Path, default=Path('phone-files'))
    ap.add_argument('--encoder', help='existing contentvec ONNX (default: download)')
    ap.add_argument('--no-vcp', action='store_true', help='only export the ONNX files')
    a = ap.parse_args()
    if not a.pth.is_file():
        sys.exit(f'File not found: {a.pth}')
    name = (a.name or a.pth.stem).strip()[:60]
    safe = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in name).strip() or 'voice'
    a.out.mkdir(parents=True, exist_ok=True)
    step('1/4 Exporting the model to ONNX')
    synth = a.out / f'{safe}.onnx'
    meta = export_onnx(a.pth, synth)
    print(f'  {synth.name}: RVC {meta["version"]}, {meta["samplingRate"]} Hz, pitch {"yes" if meta["f0"] else "no"}')
    step('2/4 Encoder (ContentVec)')
    fp32 = encoder_fp32(a.encoder)
    phone_encoder = encoder_int8(fp32, a.out)
    print(f'  {phone_encoder.name}')
    if not a.no_vcp:
        step('3/4 Building the live voice (runs the model over 5 min of reference speech)')
        sys.path.insert(0, str(ROOT / 'voicecorpus'))
        from build import build_vcp
        vcp = a.out / f'{safe}.vcp'
        size = build_vcp(fp32, synth, name, vcp, a.hz, str(ROOT / 'data' / 'source_speech.src'))
        print(f'  {vcp.name}: {size / 2**20:.1f} MiB')
    step('4/4 Done. Copy to the phone (e.g. Downloads):')
    if not a.no_vcp:
        print(f'  {safe}.vcp -> Biblioteca > Ao vivo > Importar voz ao vivo (use it live right away)')
    print(f'  {safe}.onnx + contentvec-int8.onnx -> Biblioteca > RVC · fala (optional: convert recordings)')
    print(f'  Folder: {a.out}')


if __name__ == '__main__':
    main()
