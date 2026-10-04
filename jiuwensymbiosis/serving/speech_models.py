# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Optional model adapters for speech serving; no playback on the server."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from ChatTTS import Chat

__all__ = ["ChatTTSSynthesizer"]


class ChatTTSSynthesizer:
    """ChatTTS 0.2.4's public waveform API, with explicitly prepared weights.

    Upstream v0.2.4 ``Chat.load`` / ``Chat.infer`` return mono waveforms at
    24000 Hz. Using source=custom avoids implicit weight downloads at startup.
    This is separate from the legacy external safe_speak_text module.
    """

    sample_rate = 24000
    voices = ("default",)
    model_version = "ChatTTS-0.2.4"

    def __init__(self, model_path: str | Path, *, device: str = "cpu"):
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = device
        self._model: Chat | None = None
        self._speaker: str | None = None
        self._ready = False

    def load(self) -> None:
        if self._ready:
            return
        if self._model is not None:
            raise RuntimeError("ChatTTS partial load requires close before retrying")
        if not self.model_path.is_dir():
            raise ValueError("ChatTTS weights directory does not exist; prepare weights before startup")
        import ChatTTS
        import torch

        model = ChatTTS.Chat()
        # Retain a cleanup handle even if weight or speaker initialization fails.
        self._model = model
        if not model.load(
            source="custom", custom_path=str(self.model_path), device=torch.device(self.device), compile=False
        ):
            raise RuntimeError("ChatTTS model did not become ready")
        self._speaker = model.sample_random_speaker()
        self._ready = True

    def synthesize(self, text: str, voice: str = "default") -> tuple[np.ndarray, int]:
        if voice not in self.voices:
            raise ValueError("Unsupported voice")
        self.load()
        import ChatTTS

        params = ChatTTS.Chat.InferCodeParams(spk_emb=self._speaker, show_tqdm=False)
        waveforms = cast("Chat", self._model).infer([text], stream=False, split_text=False, params_infer_code=params)
        if not waveforms:
            raise RuntimeError("ChatTTS returned no waveform")
        audio = np.asarray(waveforms[0]).reshape(-1)
        if not audio.size or not np.isfinite(audio).all():
            raise RuntimeError("ChatTTS returned an invalid waveform")
        return (np.clip(audio, -1, 1) * 32767).astype(np.int16), self.sample_rate

    def close(self) -> None:
        self._ready = False
        if self._model is not None:
            self._model.unload()
            self._model = None
            self._speaker = None
