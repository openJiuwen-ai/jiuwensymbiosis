# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""可打印标定板生成:参数校验 + 预览 + 带校验标尺的 PDF。

标定板的物理尺寸是整条标定链上唯一「错了却全程静默通过」的输入:方格实际边长
与配置值不符时,求解、质量门禁、reload 冒烟全部照常通过,只是解出来的外参带一个
系统性比例误差。因此本模块的产物不只是一张板图,而是一页**自证尺寸**的纸:

1. 页面按 ``dpi`` 存成 PDF(PDF 带物理尺寸;``cv2.imwrite`` 的裸 PNG 不写 pHYs,
   打印机只能猜),
2. 板下方印一条 ``_RULER_MM`` 毫米的校验标尺 —— 打印后量它就知道缩放对不对,
3. 页眉印死参数与「按 100% 实际大小打印」的告警。

板图本身来自 ``scripts/calibrate/handeye_board.generate_board_image``(标定 CLI
与 GUI 共用同一个生成器,避免两套板定义漂移)。
"""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["BoardParams", "BoardParamsError", "render_preview_data_uri", "write_board_pdf"]

# 300 dpi:激光/喷墨打印机的通用清晰档,再高对 ArUco 角点检测无增益。
_DPI = 300
_MM_PER_INCH = 25.4
# 校验标尺长度。50 mm 足够让 5% 的打印缩放变成肉眼可见的 2.5 mm 偏差。
_RULER_MM = 50.0
# A4 可打印区域(mm),用于提示板是否超出单页。
_A4_W_MM, _A4_H_MM = 210.0, 297.0
_PRINTABLE_W_MM, _PRINTABLE_H_MM = _A4_W_MM - 20.0, _A4_H_MM - 20.0
# cv2.aruco 预定义字典的 marker 容量(ChArUco 用掉约半数格子)。
_DICT_CAPACITY = {"DICT_4X4_50": 50, "DICT_4X4_100": 100, "DICT_4X4_250": 250, "DICT_5X5_50": 50}


class BoardParamsError(ValueError):
    """标定板参数不合法(面向操作者的中文原因)。"""


@dataclass(frozen=True)
class BoardParams:
    """一块 ChArUco / 棋盘格标定板的可打印规格。

    ``validate`` 只拦下**物理上做不出来或检测不了**的参数;版面偏大等尺寸建议
    通过 :meth:`warnings` 返回,不阻断 —— 用户可能用 A3 或分页拼板。
    """

    kind: str = "charuco"
    squares_x: int = 5
    squares_y: int = 7
    square_size_mm: float = 20.0
    marker_size_mm: float = 15.0
    aruco_dict: str = "DICT_4X4_50"

    def validate(self) -> None:
        """不合法时抛 :class:`BoardParamsError`。"""
        if self.kind not in ("charuco", "chessboard"):
            raise BoardParamsError(f"板型只能是 charuco 或 chessboard，当前为 {self.kind!r}。")
        for name, value in (("横向格数", self.squares_x), ("纵向格数", self.squares_y)):
            if value < 3:
                raise BoardParamsError(f"{name}至少为 3，当前为 {value}。")
        if self.square_size_mm <= 0:
            raise BoardParamsError(f"方格边长必须为正数，当前为 {self.square_size_mm}。")
        if self.squares_x == self.squares_y:
            raise BoardParamsError(
                f"横纵格数不能相同({self.squares_x}x{self.squares_y}):正方形板旋转 90° 后无法区分朝向，"
                "会让位姿解算出现 90° 歧义。请改成长方形，例如 5x7。"
            )
        if self.kind != "charuco":
            return
        if not 0 < self.marker_size_mm < self.square_size_mm:
            raise BoardParamsError(
                f"marker 边长必须大于 0 且小于方格边长({self.square_size_mm} mm)，当前为 {self.marker_size_mm} mm。"
            )
        needed = (self.squares_x * self.squares_y) // 2
        capacity = _DICT_CAPACITY.get(self.aruco_dict)
        if capacity is not None and needed > capacity:
            raise BoardParamsError(
                f"{self.squares_x}x{self.squares_y} 的 ChArUco 板需要 {needed} 个 marker，"
                f"超出字典 {self.aruco_dict} 的 {capacity} 个上限。请减少格数或换更大的字典。"
            )

    def warnings(self) -> list[str]:
        """返回不阻断打印、但操作者应当知道的提醒。"""
        notes: list[str] = []
        width_mm, height_mm = self.board_size_mm()
        if width_mm > _PRINTABLE_W_MM or height_mm > _PRINTABLE_H_MM:
            notes.append(
                f"板面 {width_mm:.0f}×{height_mm:.0f} mm 超出 A4 可打印区域"
                f"({_PRINTABLE_W_MM:.0f}×{_PRINTABLE_H_MM:.0f} mm)。请用更大的纸，"
                "或缩小方格边长/格数 —— 不要在打印对话框里缩放。"
            )
        if self.square_size_mm < 15.0:
            notes.append(f"方格边长 {self.square_size_mm:.1f} mm 偏小，相机稍远就检测不到角点。常用范围是 15–30 mm。")
        if self.kind == "charuco" and self.marker_size_mm / self.square_size_mm > 0.85:
            notes.append("marker 与方格边长过于接近，黑白边界变窄会降低识别率。建议 marker 取方格的 0.7–0.8 倍。")
        return notes

    def board_size_mm(self) -> tuple[float, float]:
        """板面物理尺寸(mm),不含页边距。"""
        return self.squares_x * self.square_size_mm, self.squares_y * self.square_size_mm

    def summary(self) -> str:
        """一行参数摘要,同时印在纸上和显示在界面里。

        印在纸上是为了重新打印时能对上参数,所以写成人读得懂的形式,而不是内部字段名。
        """
        width_mm, height_mm = self.board_size_mm()
        name = "ChArUco" if self.kind == "charuco" else "棋盘格"
        marker = f" · marker {self.marker_size_mm:g} mm" if self.kind == "charuco" else ""
        return (
            f"{name} {self.squares_x}×{self.squares_y} · 方格 {self.square_size_mm:g} mm{marker}"
            f" · 板面 {width_mm:.0f}×{height_mm:.0f} mm"
        )

    def to_board_spec(self):
        """转成标定 CLI 的 ``BoardSpec``(延迟导入:``scripts.calibrate`` 需要 OpenCV)。"""
        from scripts.calibrate.handeye_board import BoardSpec

        return BoardSpec(
            kind=self.kind,
            squares_x=self.squares_x,
            squares_y=self.squares_y,
            square_size_mm=self.square_size_mm,
            marker_size_mm=self.marker_size_mm if self.kind == "charuco" else None,
            aruco_dict=self.aruco_dict,
        )


def _mm_to_px(mm: float) -> int:
    return int(round(mm * _DPI / _MM_PER_INCH))


def _font(size_px: int) -> ImageFont.ImageFont:
    """取一个能渲染中文的字体;找不到时回落到 PIL 内置位图字体。

    回落时中文会渲染成方块,所以纸面文案里每条中文提示都配了等义英文 —— 板子可能
    在任何一台机器上生成,不能假设装了 CJK 字体。
    """
    resolved = _cjk_font()
    if resolved is not None:
        path, index = resolved
        try:
            return ImageFont.truetype(path, size_px, index=index)
        except OSError as exc:
            logger.debug("字体 %s[%d] 不可用: %s", path, index, exc)
    return ImageFont.load_default()


def _has_chinese_glyphs(font: ImageFont.FreeTypeFont) -> bool:
    """该字体是否真的画得出汉字。

    与一个私用区码位的位图逐字节对比:缺字形时两者都落到同一个 .notdef 方块,位图
    完全相同。这比查 fontconfig 元数据可靠 —— 元数据说支持,不等于这个 face 里有。
    """
    try:
        sample = font.getmask("标")
        undefined = font.getmask("\ue000")
    except Exception as exc:
        logger.debug("字形探测失败: %s", exc)
        return False
    return sample.size != undefined.size or bytes(sample) != bytes(undefined)


@lru_cache(maxsize=1)
def _cjk_font() -> tuple[str, int] | None:
    """定位一个含中文字形的字体,返回 ``(路径, face 索引)``;找不到返回 ``None``。

    **不信任 fontconfig 的查询结果**:``fc-match :lang=zh`` 匹配不到时会返回它认为
    最接近的字体而不是失败(在纯拉丁环境里就会返回 DejaVuSans),照单全收会让整页
    中文变成方块。因此逐个候选实际渲染一个汉字来验证。结果缓存,只探测一次。
    """
    for path in _candidate_font_files():
        # Noto CJK 的 .ttc 把简中 face 放在 index 2;单 face 字体只有 index 0。
        for index in (2, 0):
            try:
                font = ImageFont.truetype(path, 24, index=index)
            except OSError:
                continue
            if _has_chinese_glyphs(font):
                logger.debug("标定板使用中文字体 %s[%d]", path, index)
                return path, index
    logger.warning("未找到中文字体，标定板 PDF 的中文将显示为方块(英文提示不受影响)。")
    return None


def _candidate_font_files() -> list[str]:
    """候选字体文件:常见安装位置在前,再补 fontconfig 列出的中文字体。"""
    known = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    fc_list = shutil.which("fc-list")
    if fc_list is None:
        return known
    try:
        out = subprocess.run(
            [fc_list, ":lang=zh", "file"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("fc-list 查询中文字体失败: %s", exc)
        return known
    if out.returncode != 0:
        return known
    listed = sorted({line.split(":", 1)[0].strip() for line in out.stdout.splitlines() if line.strip()})
    return known + [path for path in listed if path not in known]


def _render_board_image(params: BoardParams) -> Image.Image:
    """调标定 CLI 的板生成器产出板图(灰度 PIL 图),尺寸严格按 ``_DPI`` 计算。"""
    from scripts.calibrate.handeye_board import generate_board_image

    # generate_board_image 只写文件,不返回数组,所以经一个临时 PNG 中转。
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "board.png"
        generate_board_image(params.to_board_spec(), png, dpi=_DPI, margin_px=0)
        with Image.open(png) as img:
            return img.convert("L").copy()


def _text_size(font: ImageFont.ImageFont, text: str) -> tuple[int, int]:
    """文本的像素宽高。版面按实测尺寸排布,不靠估算的行高——估小了正文会被板图盖住。"""
    left, top, right, bottom = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _draw_ruler(draw: ImageDraw.ImageDraw, x0: int, y0: int, font: ImageFont.ImageFont) -> int:
    """在 ``(x0, y0)`` 画一条 ``_RULER_MM`` 毫米的校验标尺,返回占用高度(px)。

    标尺是这页纸的自证机制:量它就知道打印有没有被缩放,不必去量方格(方格里有
    marker,边界不好下尺)。刻度朝上、文字在下,整块贴在页面左下角,与板图之间留白,
    以免被误当成板的一部分。
    """
    length_px = _mm_to_px(_RULER_MM)
    tick_h = _mm_to_px(4.0)
    baseline = y0 + tick_h
    hairline = max(1, _mm_to_px(0.25))
    draw.line([(x0, baseline), (x0 + length_px, baseline)], fill=0, width=max(1, _mm_to_px(0.4)))
    for millimetre in range(int(_RULER_MM) + 1):
        x = x0 + _mm_to_px(float(millimetre))
        if millimetre % 10 == 0:
            height = tick_h
        elif millimetre % 5 == 0:
            height = int(tick_h * 0.55)
        else:
            height = int(tick_h * 0.3)
        draw.line([(x, baseline - height), (x, baseline)], fill=0, width=hairline)
    for millimetre in (0, int(_RULER_MM)):
        label = f"{millimetre}"
        width, _height = _text_size(font, label)
        draw.text((x0 + _mm_to_px(float(millimetre)) - width // 2, baseline + _mm_to_px(1.0)), label, fill=0, font=font)

    caption = f"校验标尺 / ruler:{_RULER_MM:g} mm"
    _, caption_h = _text_size(font, caption)
    caption_y = baseline + _mm_to_px(1.0) + _text_size(font, "0")[1] + _mm_to_px(1.2)
    draw.text((x0, caption_y), caption, fill=0, font=font)
    return caption_y + caption_h - y0


def _header_lines(params: BoardParams) -> list[tuple[float, str]]:
    """页眉各行:(字号 mm, 文本)。参数与打印警告分行,避免挤成一条长行溢出页面。"""
    return [
        (4.6, "手眼标定板 / Hand-eye calibration board"),
        (3.4, params.summary()),
        (3.4, "务必按 100% 实际大小打印，不要选「适应页面」"),
        (3.0, "PRINT AT 100% / ACTUAL SIZE - DO NOT FIT TO PAGE"),
    ]


def _compose_sheet(params: BoardParams) -> Image.Image:
    """把页眉 + 板图 + 左下角校验标尺合成一页(白底灰度图)。

    页面尺寸由内容实测决定:高度按各块实际占用累加,宽度取板宽与最长文本的较大者,
    因此小板不会被长文本撑破版心,长参数行也不会越出纸面。
    """
    board = _render_board_image(params)
    margin = _mm_to_px(12.0)
    gap_after_header = _mm_to_px(6.0)
    gap_before_ruler = _mm_to_px(9.0)

    header = [(_font(_mm_to_px(size_mm)), text) for size_mm, text in _header_lines(params)]
    line_gap = _mm_to_px(1.6)
    header_h = sum(_text_size(font, text)[1] for font, text in header) + line_gap * (len(header) - 1)

    ruler_font = _font(_mm_to_px(3.0))
    ruler_h = _mm_to_px(4.0) + _mm_to_px(1.0) + _text_size(ruler_font, "0")[1] + _mm_to_px(1.2)
    ruler_h += _text_size(ruler_font, "校验标尺")[1]

    content_w = max(
        board.width,
        _mm_to_px(_RULER_MM),
        max(_text_size(font, text)[0] for font, text in header),
    )
    width = content_w + 2 * margin
    height = margin + header_h + gap_after_header + board.height + gap_before_ruler + ruler_h + margin

    sheet = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(sheet)

    y = margin
    for font, text in header:
        draw.text((margin, y), text, fill=0, font=font)
        y += _text_size(font, text)[1] + line_gap

    board_top = margin + header_h + gap_after_header
    sheet.paste(board, (margin, board_top))
    _draw_ruler(draw, margin, board_top + board.height + gap_before_ruler, ruler_font)
    return sheet


def _fit_to_page(sheet: Image.Image) -> Image.Image:
    """把内容居中放进一页 A4;放不下时保持原尺寸。

    输出标准纸张是为了少一个出错的机会:页面本来就是 A4 时,打印对话框里「实际大小」
    与「适应页面」得到的结果一致,选错也不会缩放。而一个 152×208 mm 的非标准页面
    很容易被驱动判成"需要缩放到纸张",那正是会静默毁掉标定的那种错误。

    内容超出 A4 时不强行缩放 —— 缩放就等于把方格改小了。此时 :meth:`BoardParams.warnings`
    已经提示改用更大的纸或减小板面。
    """
    page_w, page_h = _mm_to_px(_A4_W_MM), _mm_to_px(_A4_H_MM)
    if sheet.width > page_w or sheet.height > page_h:
        return sheet
    page = Image.new("L", (page_w, page_h), color=255)
    page.paste(sheet, ((page_w - sheet.width) // 2, (page_h - sheet.height) // 2))
    return page


def write_board_pdf(params: BoardParams, path: str | Path) -> Path:
    """把标定板渲染成一页 A4 PDF 并写盘,返回落盘路径。

    PDF 以 ``_DPI`` 作为分辨率写入,因此页面带正确的物理尺寸 —— 打印机按「实际
    大小」输出时,方格边长即为 ``params.square_size_mm``。
    """
    params.validate()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    page = _fit_to_page(_compose_sheet(params))
    page.convert("RGB").save(out, format="PDF", resolution=float(_DPI))
    logger.info("标定板 PDF 已生成: %s (%s)", out, params.summary())
    return out


def render_preview_data_uri(params: BoardParams, *, max_px: int = 520) -> str:
    """渲染供界面预览的 PNG data URI(等比缩小,只为看版面,不能打印)。

    与 PDF 用同一张 A4 页面,这样预览里板占纸面的比例就是打印出来的比例。
    """
    params.validate()
    sheet = _fit_to_page(_compose_sheet(params))
    scale = min(1.0, max_px / max(sheet.width, sheet.height))
    if scale < 1.0:
        sheet = sheet.resize((max(1, int(sheet.width * scale)), max(1, int(sheet.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
