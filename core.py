import math
import re
from sys import float_info
from typing import Generator, Callable

from PIL import Image

from type import ImageVertex, Color, Obj2DVertex, BarisentricVertex, ObjVertex, ObjFace, Obj2DFace


class CustomImage:
    def __init__(self, width: int, height: int):
        self.__img = Image.new('RGB', (width, height))

    def set(self, v: ImageVertex, color: Color):
        x, y = v
        if 0 <= x < self.width() and 0 <= y < self.height():
            self.__img.putpixel(v, color)

    def get(self, v: ImageVertex) -> Color:
        return self.__img.getpixel(v)

    def width(self) -> int:
        return self.__img.width

    def height(self) -> int:
        return self.__img.height

    def fill(self, color: Color):
        for i in range(self.width()):
            for j in range(self.height()):
                self.set((i, j), color)

    def line(self, v0: ImageVertex, v1: ImageVertex, color: Color):
        """
        Алгоритм Брезенхема
        """
        x0, y0 = v0
        x1, y1 = v1

        if x0 == x1 and y0 == y1:
            return

        steep = False
        if abs(x0 - x1) < abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = x1 - x0
        dy = y1 - y0
        d_error = abs(dy / float(dx))
        error = 0
        y = y0

        for x in range(x0, x1, 1):
            if steep:
                self.set((round(y), round(x)), color)
            else:
                self.set((round(x), round(y)), color)
            error += d_error
            if error >= 0.5:
                if y1 > y0:
                    step = 1
                else:
                    step = -1
                y += step
                error -= 1

    def save(self, filename: str):
        self.__img.save(filename)

    def show(self):
        self.__img.show()


class ObjModel:
    def __init__(self, path: str):
        self.__vertices: list[ObjVertex] = []
        self.__faces: list[tuple[int, int, int]] = []

        with open(path) as file:
            self.__parse(file.readlines())

    def __parse(self, lines: list[str]):
        for line in lines:
            words = re.split('\\s+', line)
            if words[0] == 'v':
                xyz = float(words[1]), float(words[2]), float(words[3])
                self.__vertices.append(xyz)
            elif words[0] == 'f':
                v1 = re.split('/', words[1])[0]
                v2 = re.split('/', words[2])[0]
                v3 = re.split('/', words[3])[0]
                vs = int(v1) - 1, int(v2) - 1, int(v3) - 1
                self.__faces.append(vs)

    def vertices(self) -> Generator[ObjVertex, None, None]:
        for v in self.__vertices:
            yield v

    def faces(self) -> Generator[ObjFace, None, None]:
        for v0, v1, v2 in self.__faces:
            yield self.__vertices[v0], self.__vertices[v1], self.__vertices[v2]

    def shift(self, val: tuple[float, float, float]):
        self.transform(lambda v: (v[0] + val[0], v[1] + val[1], v[2] + val[2]))

    def rotate(self, angle: tuple[float, float, float]):
        angle_x, angle_y, angle_z = angle

        sin_x = math.sin(angle_x)
        cos_x = math.cos(angle_x)
        sin_y = math.sin(angle_y)
        cos_y = math.cos(angle_y)
        sin_z = math.sin(angle_z)
        cos_z = math.cos(angle_z)

        def rotation(v: ObjVertex) -> ObjVertex:
            x, y, z = v

            new_x = cos_y * cos_z * x \
                    + cos_y * sin_z * y \
                    + sin_y * z
            new_y = -(sin_x * sin_y * cos_z + cos_y * sin_z) * x \
                    + (-sin_x * sin_y * sin_z + cos_x * cos_z) * y \
                    + sin_x * cos_y * z
            new_z = (-cos_x * sin_y * cos_z + sin_x * sin_z) * x \
                    - (cos_x * sin_y * sin_z + sin_x * cos_y) * y \
                    + cos_x * cos_z * z

            return new_x, new_y, new_z

        self.transform(rotation)

    def transform(self, transformation: Callable[[ObjVertex], ObjVertex]):
        new_vertices = []
        for v in self.vertices():
            new_vertices.append(transformation(v))

        self.__vertices = new_vertices


def barisentrik_coordinates(v: ImageVertex, v0: Obj2DVertex, v1: Obj2DVertex, v2: Obj2DVertex) -> BarisentricVertex:
    x, y = v
    x0, y0 = v0
    x1, y1 = v1
    x2, y2 = v2

    lambda0 = safe_division((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2),
                            (x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
    lambda1 = safe_division((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0),
                            (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
    lambda2 = safe_division((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1),
                            (x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))

    return lambda0, lambda1, lambda2


def safe_division(numerator: float, denominator: float) -> float:
    if numerator == 0:
        return 0
    if denominator == 0:
        return float_info.max
    return numerator / denominator


def draw_face(f: Obj2DFace, img: CustomImage, color: Color):
    v0, v1, v2 = f

    x0, y0 = v0
    x1, y1 = v1
    x2, y2 = v2

    x_min = round(max(min(x0, x1, x2), 0))
    y_min = round(max(min(y0, y1, y2), 0))
    x_max = round(min(max(x0, x1, x2), img.width() - 1))
    y_max = round(min(max(y0, y1, y2), img.height() - 1))

    for i in range(x_min, x_max + 1):
        for j in range(y_min, y_max + 1):
            v = (i, j)
            l0, l1, l2 = barisentrik_coordinates(v, v0, v1, v2)
            if l0 > 0 and l1 > 0 and l2 > 0:
                img.set(v, color)


def face_normal(f: ObjFace) -> ObjVertex:
    v0, v2, v1 = f

    x0, y0, z0 = v0
    x1, y1, z1 = v1
    x2, y2, z2 = v2

    x = (y2 - y0) * (z1 - z0) - (z2 - z0) * (y1 - y0)
    y = (z2 - z0) * (x1 - x0) - (x2 - x0) * (z1 - z0)
    z = (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)

    return x, y, z


def vec_norm(v: ObjVertex) -> float:
    x, y, z = v
    return math.sqrt(x * x + y * y + z * z)


def face_angle_cos(f: ObjFace) -> float:
    v = face_normal(f)
    norm = vec_norm(v)
    return v[2] / norm if norm != 0 else 0
