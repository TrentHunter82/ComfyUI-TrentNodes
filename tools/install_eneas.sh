#!/usr/bin/env bash
# Run from WSL/Linux. Optional arguments: model directory, existing Florence directory.
set -euo pipefail
PACK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${1:-$PACK/../../models/eneas}"
FLORENCE="${2:-}"
UV="${UV:-$HOME/.local/bin/uv}"
if ! command -v "$UV" >/dev/null; then
    echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi
mkdir -p "$PACK/.eneas" "$MODEL_ROOT"
if [ ! -d "$PACK/.eneas/source" ]; then
    git clone https://github.com/speridlabs/eneas.git "$PACK/.eneas/source"
    git -C "$PACK/.eneas/source" checkout c59a028f19573720fc1c52fea21874d33e02ed3f
fi
if [ "$(git -C "$PACK/.eneas/source" rev-parse HEAD)" != c59a028f19573720fc1c52fea21874d33e02ed3f ]; then
    echo 'Unexpected Eneas checkout; inspect .eneas/source before continuing.' >&2
    exit 1
fi
(cd "$PACK/.eneas/source" && "$UV" sync --frozen --no-dev)
PYTHON="$PACK/.eneas/source/.venv/bin/python"
"$UV" pip install --python "$PYTHON" zstandard==0.25.0
"$PYTHON" - "$PACK" "$MODEL_ROOT" "$FLORENCE" <<'PY'
import json
from pathlib import Path
import sys
pack = Path(sys.argv[1])
source = pack / '.eneas/source/eneas/segmentation/unique_instance.py'
text = source.read_text()
# Upstream starts at frame_idx-1, but SeC excludes that frame in reverse mode.
text = text.replace('start_frame_idx=frame_idx - 1,', 'start_frame_idx=frame_idx,')
source.write_text(text)
config = {'model_root': str(Path(sys.argv[2]).resolve()), 'florence_path': sys.argv[3] or None}
(pack / '.eneas/config.json').write_text(json.dumps(config, indent=2))
PY
if [ ! -x "$PACK/.eneas/ollama/bin/ollama" ]; then
    curl -fL https://github.com/ollama/ollama/releases/download/v0.33.3/ollama-linux-amd64.tar.zst -o "$PACK/.eneas/ollama.tar.zst"
    "$PYTHON" - "$PACK" <<'PY'
from pathlib import Path
import sys
import tarfile
import zstandard
root = Path(sys.argv[1]) / '.eneas'
with (root / 'ollama.tar.zst').open('rb') as archive:
    with zstandard.ZstdDecompressor().stream_reader(archive) as reader:
        with tarfile.open(fileobj=reader, mode='r|') as tar:
            tar.extractall(root / 'ollama', filter='data')
(root / 'ollama.tar.zst').unlink()
PY
fi
echo 'Eneas ready. Restart ComfyUI; add Eneas Segment & Track (Trent). Weights download on first use.'
