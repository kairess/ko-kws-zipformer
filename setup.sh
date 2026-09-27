#!/usr/bin/env bash
# One-time environment: uv venv (Python 3.12) + CUDA 12.8 torch 2.9.1 + prebuilt k2 wheel + icefall.
# Tested on Ubuntu, RTX 5090 (driver with CUDA 12.8). No Docker needed.
set -eou pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"; . ./env.sh
ICEFALL_COMMIT=3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1

[ -x "$PY" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$PY" torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" "k2==1.24.4.dev20260625+cuda12.8.torch2.9.1" -f https://k2-fsa.github.io/k2/cuda.html
uv pip install --python "$PY" lhotse==1.33.0 lilcom kaldialign sentencepiece==0.2.2 kiwipiepy==0.23.2 \
    soundfile soxr av onnx onnxruntime sherpa-onnx==1.13.8 tensorboard

if [ ! -d "$ICEFALL" ]; then
  git clone https://github.com/k2-fsa/icefall.git "$ICEFALL"
  git -C "$ICEFALL" checkout "$ICEFALL_COMMIT"
fi
uv pip install --python "$PY" -r "$ICEFALL/requirements.txt"

# unmodified icefall modules used by the recipe (modified ones live in zipformer/)
Z="$ICEFALL/egs/librispeech/ASR/zipformer"
for f in decoder.py joiner.py model.py optim.py scaling.py subsampling.py zipformer.py export.py export-onnx-streaming.py; do
  ln -sfn "$Z/$f" zipformer/$f
done
ln -sfn "$ICEFALL/egs/librispeech/ASR/transducer_stateless/encoder_interface.py" zipformer/encoder_interface.py
ln -sfn "$ICEFALL/egs/gigaspeech/ASR/conformer_ctc/gigaspeech_scoring.py" zipformer/gigaspeech_scoring.py
"$PY" -c "import torch, k2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
