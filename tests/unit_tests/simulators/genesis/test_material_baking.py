"""Asset-independent texture operations."""

import pytest


def test_texture_bake_applies_rotation_and_linear_gloss_inversion(tmp_path):
    import numpy as np
    from PIL import Image

    from jiuwensymbiosis.simulators.genesis.material_baking import bake_texture

    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :4, 0] = 255
    pixels[:, 4:, 1] = 255
    source = tmp_path / "stripes.png"
    Image.fromarray(pixels).save(source)
    path, color, report = bake_texture(
        source, tmp_path / "baked", "diffuseColor", ((1, 1), 90, (1, 0)), True, None, None
    )
    result = np.asarray(Image.open(path))
    assert result[1, 4, 0] > 250 and result[6, 4, 1] > 250
    assert color and report["periodic_exact"]
    Image.new("RGB", (4, 4), (128, 128, 128)).save(source)
    path, color, _ = bake_texture(source, tmp_path / "baked", "roughness", None, True, (-1, -1, -1, 1), (1, 1, 1, 0))
    assert not color
    assert np.asarray(Image.open(path))[0, 0, 0] == pytest.approx(200, abs=1)


def test_hdr_scalar_texture_is_decoded_without_pillow(tmp_path):
    import numpy as np
    from PIL import Image

    from jiuwensymbiosis.simulators.genesis.material_baking import bake_texture

    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "roughness.hdr"
    assert cv2.imwrite(str(source), np.full((4, 4, 3), 0.25, dtype=np.float32))
    path, color, _ = bake_texture(source, tmp_path / "cache", "roughness", None, False, None, None)
    assert not color
    assert np.asarray(Image.open(path))[0, 0, 0] == pytest.approx(64, abs=1)
