# pth-to-onnx

Turn an **RVC voice model (`.pth`)** into files for the **Timbre Studio** Android voice changer in one command — no coding, no GPU.

```bash
python pth_to_onnx.py MyVoice.pth --name "My voice" --hz 110
```

Output (in `phone-files/`):

| File | What it is | Where it goes in the app |
|---|---|---|
| `My voice.vcp` | **live voice** — talk through it in real time | Biblioteca → Ao vivo → Importar voz ao vivo |
| `My voice.onnx` | the model as ONNX (optional) — converts recordings | Biblioteca → RVC · fala → Importar sintetizador |
| `contentvec-int8.onnx` | speech encoder (optional, once) | Biblioteca → RVC · fala → Importar encoder |

A `.vcp` is self-contained: share it and anyone can import it without the model.

## Setup (once)

1. Install **Python 3.12** ([python.org/downloads](https://www.python.org/downloads/)). On Windows tick **“Add python.exe to PATH”**.
2. Install **Git** ([git-scm.com/downloads](https://git-scm.com/downloads)).
3. Download this repository (green **Code** button → *Download ZIP*, then unzip; or `git clone`), and open a terminal inside the folder.
4. Create an environment and install the dependencies (~1 GB):

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` first.

**macOS / Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Convert

With the environment active:

```bash
python pth_to_onnx.py "path/to/MyVoice.pth" --name "My voice" --hz 110
```

- `--hz`: the voice's typical pitch — **110** low (male voices), **150** middle, **220** high (female/child voices). If the live voice sounds too deep or too thin, run again with another value.
- First run downloads the ContentVec encoder (~380 MB, hash-checked) and the pinned RVC exporter code; later runs reuse them (cache: `~/.cache/pth-to-onnx`).
- Takes about 3 minutes on a laptop CPU.
- `--no-vcp` exports only the ONNX files.

**Which `.pth`?** The final model file (usually 50–60 MB) that trainers save in their *weights*/*logs* folder under the model's name — not the `G_*.pth` / `D_*.pth` training checkpoints. The `.index` file is not needed. Supported: RVC **v1 and v2**, with or without pitch (f0), **32 / 40 / 48 kHz**, fp32 or fp16 weights.

## Train your own voice

Training happens in an RVC trainer, then you convert the result here:

- **Applio** — [github.com/IAHispano/Applio](https://github.com/IAHispano/Applio) · docs: [docs.applio.org](https://docs.applio.org)
- **RVC WebUI** — [github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

General recipe (follow your trainer's own guide for the buttons):

1. **Audio:** 10–30+ minutes of that voice only — no music, echo or background noise — as WAV/FLAC. Variety helps: sentences, long vowels, questions, low and high tones; for a game character, its lines and sounds. Trim long silences; use a vocal separator first if there is music.
2. **New model:** RVC **v2**, **40 kHz**, pitch extraction **RMVPE** (the best-tested setup for the app).
3. **Preprocess**, **extract features**, then **train**. A few hundred epochs is typical for 10–30 min of audio. Test in the trainer and stop when the voice is clear — over-training sounds harsh/metallic.
4. Take the final `.pth` and run `pth_to_onnx.py`.

Only use voices you have the right to use: your own, voices whose owner agreed, or characters whose licence allows your use. Don't impersonate people.

## Troubleshooting

| Message / symptom | Fix |
|---|---|
| `This .pth is not an RVC inference model` | Use the final model `.pth`, not `G_*.pth`/`D_*.pth`. |
| `git is required` | Install Git and open a new terminal. |
| `python` not found (Windows) | Reinstall Python with “Add to PATH”, or use `py -3.12`. |
| Live voice too deep / too thin | Re-run with another `--hz`. |
| Words unclear | Expected: the live voice is built only from the model's own audio, favouring the character over clarity. |

## How it works

1. **Export** — loads the checkpoint with PyTorch (`weights_only`), rebuilds the RVC synthesizer with the [w-okada voice-changer](https://github.com/w-okada/voice-changer) inference classes (fetched at a pinned commit), exports ONNX opset 17 with the metadata the app validates (`samplingRate`, `f0`, `embChannels`, `embOutputLayer`, `useFinalProj`) and checks it runs in ONNX Runtime.
2. **Encoder** — ContentVec ONNX (`units9` for v1, `unit12` for v2), plus a dynamic INT8 copy for phones.
3. **Live voice** — runs the model over `data/source_speech.src` (5 min: sentences and sustained vowels from 24 speakers, pitch moved to `--hz`), then `voicecorpus/` measures the source speech per 10 ms frame (spectral shape, loudness, voicing, relative pitch), reduces it with PCA and indexes it with k-means. The app plays back the model's audio for the frames that best match your speech. File format: `VCP2`.

`voicecorpus/` must stay in step with the app's `CorpusVoice`/`SourceAnalyzer`; a `.vcp` built by a mismatched version is rejected by the app.

See [NOTICE.md](NOTICE.md) for third-party material.
