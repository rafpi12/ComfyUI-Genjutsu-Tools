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
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        dst = os.path.join(tmp, "out.wav")
        _write_wav(src, samples, sr)
        # Same flags as the original Genjutsu workflow
        subprocess.run(
            [binary, "--fine", "--pitch", str(semitones), "--time", "1", src, dst],
            check=True, capture_output=True,
        )
        out, _ = _read_wav(dst)
    return out


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
            shifted = _pitch_rubberband(samples, sr, semitones, binary) if use_rb else _pitch_librosa(samples, sr, semitones)
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


NODE_CLASS_MAPPINGS = {
    "GJ_PitchShift": GJ_PitchShift,
    "GJ_EncodeVideo": GJ_EncodeVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GJ_PitchShift": "GJ Pitch Shift (duration preserved)",
    "GJ_EncodeVideo": "GJ Encode Video (MP4 file)",
}
