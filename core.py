from typing import Callable

from PIL import Image
import re

# Типы для последующей типизации параметров и возвращаемых значений
Color = tuple[int, int, int]
Transformation = Callable[[float, float, float], tuple[int, int]]


class CustomImage:
    def __init__(self, width: int, height: int):
        self.__img = Image.new('RGB', (width, height))

    def set(self, x: int, y: int, color: Color):
        if 0 <= x < self.width() and 0 <= y < self.height():
            self.__img.putpixel((x, y), color)

    def get(self, x: int, y: int) -> Color:
        return self.__img.getpixel((x, y))

    def width(self) -> int:
        return self.__img.width

    def height(self) -> int:
        return self.__img.height

    def fill(self, color: Color):
        for i in range(self.width()):
            for j in range(self.height()):
                self.set(i, j, color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Color):
        """
        Алгоритм Брезенхема
        """
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
                self.set(round(y), round(x), color)
            else:
                self.set(round(x), round(y), color)
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
        self.__vertices: list[tuple[float, float, float]] = []
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

    def draw_vertices(self, img: CustomImage, color: Color, transform: Transformation):
        for x, y, z in self.__vertices:
            x, y = transform(x, y, z)
            img.set(x, y, color)

    def draw_faces(self, img: CustomImage, color: Color, transform: Transformation):
        for v1, v2, v3 in self.__faces:
            x, y, z = self.__vertices[v1]
            x0, y0 = transform(x, y, z)
            x, y, z = self.__vertices[v2]
            x1, y1 = transform(x, y, z)
            x, y, z = self.__vertices[v3]
            x2, y2 = transform(x, y, z)
            img.line(x0, y0, x1, y1, color)
            img.line(x1, y1, x2, y2, color)
            img.line(x0, y0, x2, y2, color)
