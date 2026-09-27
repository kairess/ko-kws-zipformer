# Source me: `. ./env.sh`
export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KWS_TRAIN="$ROOT"
export RAW="${RAW:-$ROOT/data/raw}"          # downloads (zeroth, fleurs, musan, ksponspeech, aihub/)
export DATA="${DATA:-$ROOT/data}"            # manifests / fbank / fbank-k / lang / kws-test
export ICEFALL="${ICEFALL:-$ROOT/third_party/icefall}"
export VENV="${VENV:-$ROOT/.venv}"
export PY="$VENV/bin/python"
export PYTHONPATH="$ICEFALL:$ROOT/local:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
