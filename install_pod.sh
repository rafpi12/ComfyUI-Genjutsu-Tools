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

# 3. Modeles
dl () {  # dl <dossier models/...> <url>
  local dest="$COMFY/models/$1/$(basename "$2")"
  [ -s "$dest" ] && return 0
  echo "==> Download $(basename "$2")"
  mkdir -p "$(dirname "$dest")"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -q -x 8 -s 8 -d "$(dirname "$dest")" -o "$(basename "$dest")" "$2"
  else
    wget -q --show-progress -O "$dest" "$2"
  fi
}
dl checkpoints "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"

# Workflow MiniMax H3 (environ 40 Go) : seulement si MINIMAX=1
if [ "${MINIMAX:-0}" = "1" ]; then
  H3="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
  dl diffusion_models "$H3/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
  dl text_encoders "$H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  dl vae "$H3/vae/minimax_h3_video_vae_int8_convrot.safetensors"
  dl vae "$H3/vae/minimax_h3_audio_vae_fp32.safetensors"
  dl model_patches "$H3/model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"
  dl loras "$H3/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
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
