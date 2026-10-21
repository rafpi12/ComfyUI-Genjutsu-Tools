# Genjutsu pour ComfyUI

Deux workflows dans `example_workflows/` :
- `genjutsu_advanced_depth_sam3.json` : Seedance via WaveSpeed (API, payant).
- `genjutsu_minimax_h3_depth_inpaint.json` : MiniMax H3 en local (Fun ControlNet depth + masque SAM3, lip sync sur la voix isolee).

## Workflow Seedance (mode Advanced : depth + SAM3)

Adaptation ComfyUI du workflow open source Genjustsu (TEIN, MIT) :
https://github.com/sirioberati/Genjustsu-Open-Source-Workflow

## Ce que fait le workflow

1. Charge la video source en 24 fps (360 frames max = 15 s, limite Seedance), redimensionne a 960 px max.
2. Isole la voix (Demucs, local) et la pitche de +3 demi-tons sans changer la duree.
3. Calcule une depth coloree (Video Depth Anything, colormap inferno).
4. Segmente le sujet avec SAM 3.1 natif de ComfyUI (prompt texte, tracking video).
5. Colle le sujet en depth sur le fond original, embarque la voix pitchee : c'est la video envoyee a Seedance.
6. Seedance 2.0 Video Edit via WaveSpeed (groupe mute par defaut, payant).
7. Remplace l'audio du resultat par la bande son originale complete.

## Nodes de ce pack

| Node | Role |
|---|---|
| GJ Pitch Shift | Pitch en demi-tons, duree conservee. Rubber Band CLI si installe (memes reglages que l'original), sinon librosa |
| GJ Encode Video | Frames + audio vers MP4 H.264/AAC dans `output/`, renvoye en VIDEO fichier (necessaire pour que WaveSpeed l'uploade) |
| GJ Fit To Grid | Taille multiple de 32 et duree sur la grille 17n+5 frames de MiniMax H3, sort width/height/length |
| GJ Load Saved Video | Recharge un MP4 deja sauvegarde dans `output/genjutsu` (`latest` = le plus recent) pour relancer Seedance sans recalculer la preparation |

## Installation sur le pod

```bash
cd /workspace/ComfyUI/custom_nodes
git clone https://github.com/rafpi12/ComfyUI-Genjutsu-Tools.git
bash /workspace/ComfyUI/custom_nodes/ComfyUI-Genjutsu-Tools/install_pod.sh
```

Le script installe VideoHelperSuite, audio-separation, Video Depth Anything et WaveSpeed, Rubber Band,
telecharge le checkpoint SAM 3.1 et copie le workflow dans le menu Workflows de ComfyUI.
Il detecte le venv (`/workspace/my_env`, `/workspace/venv_cu130`). Sinon : `VENV=/chemin/venv bash .../install_pod.sh`.

Ensuite : redemarre ComfyUI, puis Settings, WaveSpeed, colle ta cle API.

Telechargement automatique au premier lancement : Video Depth Anything (vers `models/videodepthanything`) et Demucs (cache torchaudio).
Prerequis : ComfyUI v0.20.0 ou plus recent (SAM3 natif).

Modeles MiniMax H3 (environ 40 Go, ignores s'ils sont deja la) :

```bash
MINIMAX=1 bash /workspace/ComfyUI/custom_nodes/ComfyUI-Genjutsu-Tools/install_pod.sh
```

Mise a jour du pack :

```bash
cd /workspace/ComfyUI/custom_nodes/ComfyUI-Genjutsu-Tools
git pull
```

## Utilisation

**Passe 1 (gratuite)** : groupe rouge en Mute. Charge ta video, Queue.
- REVIEW 1 : overlay du masque SAM3 sur toute la duree.
- REVIEW 2 : la video Seedance (composite depth + voix pitchee).

**Passe 2 (payante)** : selectionne le groupe rouge, Ctrl+M pour le reactiver, relie REVIEW 2 vers `video` du Predictor et la ref perso vers `reference_images`, verifie le prompt, Queue.

Draft en 480p par defaut. 1080p = nouvelle generation, pas un upscale du draft. Pour garder le draft valide, upscale-le avec SeedVR2.
