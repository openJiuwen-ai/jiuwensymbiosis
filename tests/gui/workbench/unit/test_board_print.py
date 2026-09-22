# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""board_print:标定板参数校验与可打印 PDF 的物理尺寸。

尺寸测试是这里最重要的一条:方格实测边长与配置值不符时,整条标定链会全程静默
通过,只是解出来的外参带比例误差。PDF 的 MediaBox 是这个错误在自动化里唯一能被
拦住的地方。
"""

from __future__ import annotations

import re

import pytest

from jiuwensymbiosis_gui.workbench.board_print import (
    _DPI,
    BoardParams,
    BoardParamsError,
    _compose_sheet,
    _fit_to_page,
    _font,
    _header_lines,
    _mm_to_px,
    _render_board_image,
    _text_size,
    render_preview_data_uri,
    write_board_pdf,
)

pytest.importorskip("cv2", reason="标定板生成依赖 OpenCV([calib] extra)")

_PT_PER_MM = 72.0 / 25.4


def _media_box_mm(pdf_bytes: bytes) -> tuple[float, float]:
    match = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf_bytes)
    assert match is not None, "PDF 缺少 MediaBox,无法判断物理尺寸"
    return float(match.group(1)) / _PT_PER_MM, float(match.group(2)) / _PT_PER_MM


def _px_to_mm(px: int) -> float:
    return px / _DPI * 25.4


class TestValidation:
    @pytest.mark.parametrize(
        ("params", "fragment"),
        [
            (BoardParams(squares_x=6, squares_y=6), "横纵格数不能相同"),
            (BoardParams(squares_x=10, squares_y=14), "超出字典"),
            (BoardParams(square_size_mm=20.0, marker_size_mm=20.0), "marker 边长"),
            (BoardParams(squares_x=2), "至少为 3"),
            (BoardParams(square_size_mm=0.0), "必须为正数"),
            (BoardParams(kind="dots"), "板型"),
        ],
    )
    def test_rejects_unusable_boards(self, params, fragment):
        with pytest.raises(BoardParamsError, match=fragment):
            params.validate()

    def test_oversize_board_warns_but_stays_printable(self):
        # 超 A4 是提醒不是错误:用户可能打在 A3 上或分页拼接。
        params = BoardParams(squares_x=9, squares_y=11, square_size_mm=30.0, marker_size_mm=22.0)
        params.validate()
        assert any("超出 A4" in note for note in params.warnings())

    def test_chessboard_ignores_marker_size(self):
        BoardParams(kind="chessboard", square_size_mm=20.0, marker_size_mm=99.0).validate()


class TestPhysicalSize:
    """按 100% 打印时纸上方格必须等于设定边长 —— 这是唯一会静默出错的输入。"""

    @pytest.mark.parametrize("square_mm", [16.0, 20.0, 24.86])
    def test_board_image_matches_the_requested_millimetres(self, square_mm):
        params = BoardParams(squares_x=5, squares_y=7, square_size_mm=square_mm, marker_size_mm=square_mm * 0.75)
        board = _render_board_image(params)

        assert _px_to_mm(board.width) == pytest.approx(5 * square_mm, abs=0.05)
        assert _px_to_mm(board.height) == pytest.approx(7 * square_mm, abs=0.05)
        assert _px_to_mm(board.width) / 5 == pytest.approx(square_mm, abs=0.01)

    def test_pdf_is_a_standard_a4_page(self, tmp_path):
        """输出标准纸张,「实际大小」与「适应页面」才会得到同一个结果。"""
        pdf = write_board_pdf(BoardParams(), tmp_path / "board.pdf")
        width_mm, height_mm = _media_box_mm(pdf.read_bytes())

        assert width_mm == pytest.approx(210.0, abs=0.3)
        assert height_mm == pytest.approx(297.0, abs=0.3)

    def test_oversize_board_keeps_its_own_size_rather_than_being_scaled(self, tmp_path):
        """塞不进 A4 时保持原尺寸 —— 缩放就等于偷偷改小了方格。"""
        params = BoardParams(squares_x=9, squares_y=11, square_size_mm=30.0, marker_size_mm=22.0)
        pdf = write_board_pdf(params, tmp_path / "big.pdf")
        width_mm, height_mm = _media_box_mm(pdf.read_bytes())

        board_w_mm, board_h_mm = params.board_size_mm()
        assert width_mm > board_w_mm and height_mm > board_h_mm
        assert height_mm > 297.0, "超出 A4 的板不应被压回 A4"

    def test_a4_page_does_not_resize_the_board(self):
        """居中放进 A4 只是加白边,板本身一个像素都不能变。"""
        params = BoardParams()
        board = _render_board_image(params)
        page = _fit_to_page(_compose_sheet(params))

        assert _px_to_mm(board.width) == pytest.approx(5 * params.square_size_mm, abs=0.05)
        assert page.width == _mm_to_px(210.0)

    def test_page_grows_with_the_board(self):
        small = _compose_sheet(BoardParams(square_size_mm=16.0, marker_size_mm=12.0))
        large = _compose_sheet(BoardParams(square_size_mm=24.0, marker_size_mm=18.0))
        assert _px_to_mm(large.height) - _px_to_mm(small.height) == pytest.approx(7 * (24.0 - 16.0), abs=0.5)


class TestLayout:
    @pytest.mark.parametrize(
        "params",
        [
            BoardParams(),
            BoardParams(squares_x=3, squares_y=4, square_size_mm=15.0, marker_size_mm=11.0),  # 板比文字窄
            BoardParams(squares_x=7, squares_y=9, square_size_mm=25.0, marker_size_mm=19.0),
        ],
    )
    def test_header_text_never_overflows_the_page(self, params):
        """页宽必须容得下最长的页眉行 —— 小板配长参数行时最容易越界。"""
        sheet = _compose_sheet(params)
        widest = max(_text_size(_font(_mm_to_px(size_mm)), text)[0] for size_mm, text in _header_lines(params))
        assert sheet.width >= widest, "页眉文字宽于纸面,会被裁掉"

    def test_board_sits_below_the_header_and_above_the_ruler(self):
        """版面三块不重叠:页眉高度写死过就发生过板图盖住正文。"""
        params = BoardParams()
        sheet = _compose_sheet(params)
        board = _render_board_image(params)
        header_h = sum(_text_size(_font(_mm_to_px(size_mm)), text)[1] for size_mm, text in _header_lines(params))
        assert sheet.height > header_h + board.height, "页面高度不足以同时放下页眉、板图与标尺"


class TestPrintableSheet:
    def test_invalid_params_never_reach_the_printer(self, tmp_path):
        with pytest.raises(BoardParamsError):
            write_board_pdf(BoardParams(squares_x=5, squares_y=5), tmp_path / "bad.pdf")
        assert not (tmp_path / "bad.pdf").exists()

    def test_preview_is_a_png_data_uri(self):
        uri = render_preview_data_uri(BoardParams(), max_px=200)
        assert uri.startswith("data:image/png;base64,")
