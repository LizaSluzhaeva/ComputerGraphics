import math
from typing import Generator, Callable

from main import CustomImage, Color

LineDrawer = Callable[[int, int, int, int, CustomImage, Color], None]

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def float_range(start: float, end: float, step: float, accur: int) -> Generator[float, None, None]:
    cur = start
    while cur < end:
        yield cur
        cur = round(cur + step, accur)


def star(line: LineDrawer, color: Color) -> CustomImage:
    img = CustomImage(200, 200)
    for i in range(0, 12):
        alpha = (2 * math.pi * i) / 13
        x0 = 100
        y0 = 100
        x1 = 100 + 95 * math.cos(alpha)
        y1 = 100 + 95 * math.sin(alpha)
        line(x0, y0, round(x1), round(y1), img, color)

    return img
