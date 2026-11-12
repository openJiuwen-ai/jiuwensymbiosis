"""The portable encoder preserves the immediate H264 / forced-IDR contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings
from jiuwensymbiosis.simulators.genesis.encoding import SoftwareEncoder


def test_software_h264_decodes_and_forced_keyframe_recovers_a_new_decoder():
    av = pytest.importorskip("av")
    pixels = np.zeros((64, 96, 3), dtype=np.uint8)
    pixels[..., 0] = 220
    rgb = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: pixels))
    encoder = SoftwareEncoder(PreviewSettings(fps=30), resolution=(96, 64))
    try:
        decoder = av.CodecContext.create("h264", "r")
        for sequence in range(1, 5):
            forced = sequence in {1, 4}
            (packet,) = encoder.encode(rgb, {"sequence": sequence, "stream_epoch": 2}, force_keyframe=forced)
            assert packet["sequence"] == sequence and packet["stream_epoch"] == 2
            if forced:
                assert packet["keyframe"]
                decoder = av.CodecContext.create("h264", "r")
            (frame,) = decoder.decode(av.Packet(packet["h264"]))
            assert (frame.width, frame.height) == (96, 64)
            decoded = frame.to_ndarray(format="rgb24").astype(float)
            assert np.abs(decoded - pixels).mean() < 5
    finally:
        encoder.close()
        encoder.close()
    with pytest.raises(RuntimeError, match="closed"):
        encoder.encode(rgb, {"sequence": 5}, force_keyframe=False)
