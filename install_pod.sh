#!/usr/bin/env bash
# Installe tout le workflow Genjutsu (mode Advanced depth + SAM3) sur un pod RunPod.
# Usage : bash /workspace/ComfyUI/custom_nodes/ComfyUI-Genjutsu-Tools/install_pod.sh
# Variables optionnelles : COMFY=/chemin/ComfyUI  VENV=/chemin/venv
set -e

COMFY="${COMFY:-/workspace/ComfyUI}"
NODES="$COMFY/custom_nodes"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python du venv ComfyUI (auto-detection)
if [ -z "$VENV" ]; then
  for v in /workspace/my_env /workspace/venv_cu130 /workspace/venv; do
    [ -x "$v/bin/python" ] && VENV="$v" && break
  done
fi
if [ -n "$VENV" ]; then PY="$VENV/bin/python"; else PY="$(command -v python3)"; fi
echo "==> ComfyUI : $COMFY"
echo "==> Python  : $PY"
[ -d "$NODES" ] || { echo "ERREUR : $NODES introuvable. Relance avec COMFY=/chemin/vers/ComfyUI"; exit 1; }

# 1. Custom nodes externes
clone_or_pull () {
  local url="$1" dir="$NODES/$(basename "$1" .git)"
  if [ -d "$dir/.git" ]; then echo "==> Update $(basename "$dir")"; git -C "$dir" checkout -- . 2>/dev/null || true; git -C "$dir" pull --ff-only || true
  else echo "==> Clone $(basename "$dir")"; git clone --depth 1 "$url" "$dir"; fi
  if [ -f "$dir/requirements.txt" ]; then "$PY" -m pip install -q -r "$dir/requirements.txt"; fi
}
clone_or_pull https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
clone_or_pull https://github.com/christian-byrne/audio-separation-nodes-comfyui.git
clone_or_pull https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git
clone_or_pull https://github.com/WaveSpeedAI/wavespeed-comfyui.git
"$PY" -m pip install -q -r "$HERE/requirements.txt"

# Patch Video Depth Anything : matplotlib >= 3.9 n'a plus cm.get_cmap (colormap inferno)
VDA="$NODES/ComfyUI-Video-Depth-Anything"
FILES=$(grep -rl 'cm\.get_cmap(' --include=*.py "$VDA" 2>/dev/null || true)
if [ -n "$FILES" ]; then
  echo "==> Patch Video Depth Anything (matplotlib get_cmap)"
  echo "$FILES" | xargs sed -i 's/cm\.get_cmap(\("[A-Za-z_]*"\))/__import__("matplotlib").colormaps[\1]/g'
fi

# 2. Rubber Band (optionnel, sinon fallback librosa)
if ! command -v rubberband >/dev/null 2>&1; then
  echo "==> Install Rubber Band"
  (apt-get update -qq && apt-get install -y -qq rubberband-cli) || echo "   Rubber Band non installe, le node utilisera librosa"
fi

# 3. Checkpoint SAM 3.1
CKPT="$COMFY/models/checkpoints/sam3.1_multiplex_fp16.safetensors"
URL="https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"
if [ ! -s "$CKPT" ]; then
  echo "==> Download SAM 3.1"
  mkdir -p "$(dirname "$CKPT")"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -d "$(dirname "$CKPT")" -o "$(basename "$CKPT")" "$URL"
  else
    wget -q --show-progress -O "$CKPT" "$URL"
  fi
fi

# 4. Workflow visible dans le menu Workflows de ComfyUI
mkdir -p "$COMFY/user/default/workflows"
cp "$HERE/example_workflows/"*.json "$COMFY/user/default/workflows/"

# 5. Verification version ComfyUI (SAM3 natif depuis v0.20.0)
if [ ! -f "$COMFY/comfy_extras/nodes_sam3.py" ]; then
  echo "ATTENTION : ComfyUI trop ancien pour SAM3 natif (v0.20.0 minimum). Mets ComfyUI a jour."
fi

echo "==> Termine. Redemarre ComfyUI, puis ajoute ta cle dans Settings > WaveSpeed."
echo "    Video Depth Anything et Demucs se telechargent seuls au premier lancement."
