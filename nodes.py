"""
Genjutsu Tools for ComfyUI
Helper nodes to run the Genjutsu (TEIN) character replacement workflow in ComfyUI:
  - GJ Pitch Shift: shift isolated vocals by N semitones, duration preserved (Rubber Band CLI or librosa fallback)
  - GJ Encode Video: frames + audio -> H.264 MP4 file, returned as a native VIDEO (VideoFromFile)
    so API nodes such as WaveSpeed Predictor can upload it.
"""

import os
import shutil
import subprocess
import tempfile
from fractions import Fraction

import numpy as np
import torch

import folder_paths


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _write_wav(path, samples, sr):
    """samples: float32 numpy [C, T]"""
    from scipy.io import wavfile
    wavfile.write(path, int(sr), np.ascontiguousarray(samples.T.astype(np.float32)))


def _read_wav(path):
    """returns float32 numpy [C, T], sr"""
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 1:
        data = data[:, None]
    return data.T, sr


def _fit_length(samples, n):
    """pad or trim [C, T] to exactly n samples"""
    if samples.shape[1] < n:
        samples = np.pad(samples, ((0, 0), (0, n - samples.shape[1])))
    return samples[:, :n]


def _pitch_rubberband(samples, sr, semitones, binary):
    """
    Try the original Genjutsu flags first (--fine = R3 engine, Rubber Band v3+),
    then the default engine for older builds (Ubuntu 22.04 ships v2.x without --fine).
    Raises RuntimeError with Rubber Band's own message if every attempt fails.
    """
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        dst = os.path.join(tmp, "out.wav")
        _write_wav(src, samples, sr)
        attempts = [
            ["--fine", "--pitch", str(semitones), "--time", "1"],
            ["--pitch", str(semitones), "--time", "1"],
        ]
        errors = []
        for flags in attempts:
            if os.path.exists(dst):
                os.remove(dst)
            proc = subprocess.run([binary, *flags, src, dst], capture_output=True, text=True)
            if proc.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 44:
                if flags[0] != "--fine":
                    print("[Genjutsu] Rubber Band without --fine (older version, default engine)")
                out, _ = _read_wav(dst)
                return out
            msg = (proc.stderr or proc.stdout or "").strip().splitlines()
            errors.append(f"{' '.join(flags)} -> exit {proc.returncode}: {msg[-1] if msg else 'no output'}")
        raise RuntimeError("Rubber Band failed: " + " | ".join(errors))


def _pitch_librosa(samples, sr, semitones):
    import librosa
    channels = [librosa.effects.pitch_shift(ch, sr=sr, n_steps=semitones) for ch in samples]
    return np.stack(channels, axis=0).astype(np.float32)


class GJ_PitchShift:
    """Shift pitch of an AUDIO by N semitones without changing its duration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "semitones": ("FLOAT", {"default": 3.0, "min": -24.0, "max": 24.0, "step": 0.5}),
                "engine": (["auto", "rubberband", "librosa"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "Genjutsu"

    def run(self, audio, semitones, engine):
        waveform = audio["waveform"]  # [B, C, T]
        sr = int(audio["sample_rate"])
        binary = shutil.which("rubberband")

        if engine == "rubberband" and not binary:
            raise RuntimeError(
                "Rubber Band CLI not found. Install it with: apt-get install -y rubberband-cli "
                "(or pick engine 'librosa')."
            )
        use_rb = binary is not None and engine in ("auto", "rubberband")
        print(f"[Genjutsu] Pitch shift {semitones:+} semitones with {'rubberband' if use_rb else 'librosa'}")

        if semitones == 0:
            return ({"waveform": waveform.clone(), "sample_rate": sr},)

        out_batches = []
        for b in range(waveform.shape[0]):
            samples = waveform[b].detach().cpu().float().numpy()
            n = samples.shape[1]
            if use_rb:
                try:
                    shifted = _pitch_rubberband(samples, sr, semitones, binary)
                except RuntimeError as e:
                    if engine == "rubberband":
                        raise
                    print(f"[Genjutsu] {e}. Falling back to librosa.")
                    shifted = _pitch_librosa(samples, sr, semitones)
            else:
                shifted = _pitch_librosa(samples, sr, semitones)
            if shifted.shape[0] != samples.shape[0]:
                shifted = np.repeat(shifted[:1], samples.shape[0], axis=0)
            out_batches.append(torch.from_numpy(_fit_length(shifted, n)))

        return ({"waveform": torch.stack(out_batches, dim=0), "sample_rate": sr},)


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------

class GJ_EncodeVideo:
    """
    Encode frames (+ optional audio) to an H.264 / AAC MP4 in the output folder and
    return it as a native VIDEO backed by a real file. Dimensions are cropped to even numbers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "filename_prefix": ("STRING", {"default": "genjutsu/clip"}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Genjutsu"

    def run(self, images, fps, filename_prefix, crf, audio=None):
        from comfy_api.latest import InputImpl, Types

        h, w = images.shape[1], images.shape[2]
        images = images[:, : h - (h % 2), : w - (w % 2), :3]

        out_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images.shape[2], images.shape[1]
        )
        file = f"{filename}_{counter:05}_.mp4"
        path = os.path.join(out_dir, file)

        components = Types.VideoComponents(images=images, frame_rate=Fraction(round(fps * 1000), 1000), audio=audio)
        InputImpl.VideoFromComponents(components).save_to(
            path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264, crf=crf
        )
        print(f"[Genjutsu] Encoded {images.shape[0]} frames at {fps} fps -> {path}")

        video = InputImpl.VideoFromFile(path)
        preview = {"filename": file, "subfolder": subfolder, "type": "output"}
        return {"ui": {"images": [preview], "animated": (True,)}, "result": (video,)}


def _saved_videos():
    """MP4 files under output/genjutsu, newest first, as paths relative to the output folder."""
    root = folder_paths.get_output_directory()
    base = os.path.join(root, "genjutsu")
    found = []
    if os.path.isdir(base):
        for dirpath, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(".mp4"):
                    full = os.path.join(dirpath, f)
                    found.append((os.path.getmtime(full), os.path.relpath(full, root)))
    found.sort(reverse=True)
    return [p.replace(os.sep, "/") for _, p in found]


class GJ_LoadSavedVideo:
    """
    Load an MP4 already written by GJ Encode Video (output/genjutsu) as a native VIDEO file,
    so the Seedance pass can run alone without recomputing Demucs, SAM3 and depth.
    'latest' picks the newest file whose name starts with the prefix. Press R in ComfyUI to refresh the list.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (["latest"] + _saved_videos(), {"default": "latest"}),
                "latest_prefix": ("STRING", {"default": "seedance_input"}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "run"
    CATEGORY = "Genjutsu"

    @classmethod
    def _resolve(cls, file, latest_prefix):
        root = folder_paths.get_output_directory()
        if file == "latest":
            for rel in _saved_videos():
                if os.path.basename(rel).startswith(latest_prefix):
                    return os.path.join(root, rel)
            raise FileNotFoundError(f"No MP4 starting with '{latest_prefix}' in {os.path.join(root, 'genjutsu')}")
        path = os.path.join(root, file)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path

    @classmethod
    def IS_CHANGED(cls, file, latest_prefix):
        try:
            path = cls._resolve(file, latest_prefix)
            return f"{path}:{os.path.getmtime(path)}"
        except FileNotFoundError:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, file, latest_prefix):
        return True

    def run(self, file, latest_prefix):
        from comfy_api.latest import InputImpl
        path = self._resolve(file, latest_prefix)
        print(f"[Genjutsu] Loading saved video {path}")
        return (InputImpl.VideoFromFile(path), path)


NODE_CLASS_MAPPINGS = {
    "GJ_PitchShift": GJ_PitchShift,
    "GJ_EncodeVideo": GJ_EncodeVideo,
    "GJ_LoadSavedVideo": GJ_LoadSavedVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GJ_PitchShift": "GJ Pitch Shift (duration preserved)",
    "GJ_EncodeVideo": "GJ Encode Video (MP4 file)",
    "GJ_LoadSavedVideo": "GJ Load Saved Video (skip recompute)",
}
