import math
from typing import Generator, Callable

from core import CustomImage
from type import Color, ImageVertex

LineDrawer = Callable[[ImageVertex, ImageVertex, CustomImage, Color], None]


def float_range(start: float, end: float, step: float, accur: int) -> Generator[float, None, None]:
    cur = start
    while cur < end:
        yield cur
        cur = round(cur + step, accur)


def star(line: LineDrawer, color: Color) -> CustomImage:
    img = CustomImage(200, 200)
    for i in range(0, 12):
        alpha = (2 * math.pi * i) / 13
        v0 = 100, round(100 + 95 * math.cos(alpha))
        v1 = 100, round(100 + 95 * math.sin(alpha))
        line(v0, v1, img, color)
    return img
