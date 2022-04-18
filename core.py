import math
import re
from sys import float_info
from typing import Generator, Optional, Callable

from PIL import Image

from names import WHITE
from type import *


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

    def get_internal(self) -> Image.Image:
        return self.__img


class CustomAnimation:
    def __init__(self):
        self.__images: list[CustomImage] = []

    def add_image(self, img: CustomImage):
        self.__images.append(img)

    def save(self, filename: str, duration: int):
        if not filename.endswith('.gif'):
            raise Exception('File must be in .gif format')
        if len(self.__images) == 0:
            raise Exception('No images to animate')

        self.__images[0].get_internal().save(
            filename,
            save_all=True,
            append_images=list(map(lambda img: img.get_internal(), self.__images[1:])),
            duration=duration,
            loop=0
        )


_FaceVertex = tuple[int, Optional[int], Optional[int]]
_InternalFace = tuple[_FaceVertex, _FaceVertex, _FaceVertex]


class ObjModel:
    def __init__(self, model_path: str, texture_path: str = None):
        self.__vertices: list[ObjVertex] = []
        self.__normals: list[ObjNormal] = []
        self.__textures: list[ObjTexture] = []
        self.__faces: list[_InternalFace] = []
        self.__texture_img: Optional[Image.Image] = None

        with open(model_path) as file:
            self.__parse_model(file.readlines())

        if texture_path is not None:
            self.__texture_img = Image.open(texture_path).convert('RGB')

    def __parse_model(self, lines: list[str]):
        for line in lines:
            words = re.split('\\s+', line)
            words = list(filter(lambda s: len(s) > 0, words))
            if len(words) == 0:
                continue

            elem_name = words[0]
            if elem_name == 'v':
                xyz = float(words[1]), float(words[2]), float(words[3])
                self.__vertices.append(xyz)
            elif elem_name == 'vn':
                xyz = float(words[1]), float(words[2]), float(words[3])
                self.__normals.append(xyz)
            elif elem_name == 'vt':
                xy = float(words[1]), float(words[2])
                self.__textures.append(xy)
            elif elem_name == 'f':
                v1 = self.__parse_vertex(words[1])
                v2 = self.__parse_vertex(words[2])
                for word in words[3:]:
                    v3 = self.__parse_vertex(word)
                    self.__faces.append((v1, v2, v3))
                    v2 = v3

    @staticmethod
    def __parse_vertex(s: str) -> _FaceVertex:
        data = re.split('/', s)
        if len(data) == 1:
            return int(data[0]) - 1, None, None
        elif len(data) == 2:
            return int(data[0]) - 1, int(data[1]) - 1, None
        else:
            return int(data[0]) - 1, int(data[1]) - 1, int(data[2]) - 1

    def vertices(self) -> Generator[ObjVertex, None, None]:
        for v in self.__vertices:
            yield v

    def faces(self) -> Generator[ObjFace, None, None]:
        for f in self.__faces:
            yield self.__face_to_triangle(f)

    def __face_to_triangle(self, f: _InternalFace) -> ObjFace:
        v0, v1, v2 = f
        return self.__vertices[v0[0]], self.__vertices[v1[0]], self.__vertices[v2[0]]

    def draw(self, img: CustomImage, scale: Scale):
        z_buffer = init_z_buffer(img)

        for f in self.__faces:
            self.__draw_face(img, scale, f, z_buffer)

    def __draw_face(self, img: CustomImage, scale: Scale, f: _InternalFace, z_buffer: ZBuffer):
        t = self.__face_to_triangle(f)

        cos = face_angle_cos(t)
        if cos > 0:
            return

        z0, z1, z2 = t[0][2], t[1][2], t[2][2]
        t = adapt_triangle_to_image(t, img, scale)

        vn0, vn1, vn2 = self.__face_normals(f)
        vt0, vt1, vt2 = self.__face_textures(f)

        for v in suitable_pixels(t, img):
            lmd0, lmd1, lmd2 = barisentrik_coordinates(v, t)
            if lmd0 < 0 or lmd1 < 0 or lmd2 < 0:
                continue

            i, j = v
            z = lmd0 * z0 + lmd1 * z1 + lmd2 * z2
            if z >= z_buffer[i][j]:
                continue
            z_buffer[i][j] = z

            if vn0 is not None and vn1 is not None and vn2 is not None:
                l0 = vec_cos(vn0)
                l1 = vec_cos(vn1)
                l2 = vec_cos(vn2)
                intensity = lmd0 * l0 + lmd1 * l1 + lmd2 * l2
            else:
                intensity = cos

            if vt0 is not None and vt1 is not None and vt2 is not None and self.__texture_img is not None:
                vtx0, vty0 = vt0
                vtx1, vty1 = vt1
                vtx2, vty2 = vt2
                texture_x = round(self.__texture_img.width * (lmd0 * vtx0 + lmd1 * vtx1 + lmd2 * vtx2))
                texture_y = round(self.__texture_img.height * (lmd0 * vty0 + lmd1 * vty1 + lmd2 * vty2))
                color = self.__texture_img.getpixel((texture_x, texture_y))
            else:
                color = WHITE

            img.set(v, intensity_color(color, intensity))

    def __face_normals(self, f: _InternalFace) -> tuple[Optional[ObjNormal], Optional[ObjNormal], Optional[ObjNormal]]:
        return self.__normal(f[0][2]), self.__normal(f[1][2]), self.__normal(f[2][2])

    def __face_textures(self, f: _InternalFace) -> tuple[Optional[ObjTexture], Optional[ObjTexture], Optional[ObjTexture]]:
        return self.__texture(f[0][2]), self.__texture(f[1][2]), self.__texture(f[2][2])

    def __normal(self, idx: int) -> Optional[ObjNormal]:
        if idx is not None and 0 <= idx < len(self.__normals):
            return self.__normals[idx]
        else:
            return None

    def __texture(self, idx: int) -> Optional[ObjTexture]:
        if idx is not None and 0 <= idx < len(self.__textures):
            return self.__textures[idx]
        else:
            return None

    def shift(self, val: Shift):
        self.__transform_vertices(lambda v: (v[0] + val[0], v[1] + val[1], v[2] + val[2]))

    def rotate(self, angle: Angle):
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
            new_y = -(sin_x * sin_y * cos_z + cos_x * sin_z) * x \
                    + (-sin_x * sin_y * sin_z + cos_x * cos_z) * y \
                    + sin_x * cos_y * z
            new_z = (-cos_x * sin_y * cos_z + sin_x * sin_z) * x \
                    - (cos_x * sin_y * sin_z + sin_x * cos_z) * y \
                    + cos_x * cos_y * z

            return new_x, new_y, new_z

        self.__transform_vertices(rotation)
        self.__transform_normals(rotation)

    def __transform_vertices(self, transformation: Callable[[ObjVertex], ObjVertex]):
        new_vertices = []
        for v in self.__vertices:
            new_vertices.append(transformation(v))
        self.__vertices = new_vertices

    def __transform_normals(self, transformation: Callable[[ObjVertex], ObjVertex]):
        new_normals = []
        for vn in self.__normals:
            new_normals.append(transformation(vn))
        self.__normals = new_normals


def init_z_buffer(img: CustomImage) -> ZBuffer:
    return [[float_info.max for _ in range(img.height())] for _ in range(img.width())]


def suitable_pixels(f: Obj2DFace, img: CustomImage) -> Generator[ImageVertex, None, None]:
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
            yield i, j


def adapt_triangle_to_image(f: ObjFace, img: CustomImage, scale: Scale) -> Obj2DFace:
    v0, v1, v2 = f

    v0 = adapt_vertex_to_image(v0, img, scale)
    v1 = adapt_vertex_to_image(v1, img, scale)
    v2 = adapt_vertex_to_image(v2, img, scale)

    return v0, v1, v2


def adapt_vertex_to_image(v: ObjVertex, img: CustomImage, scale: Scale) -> Obj2DVertex:
    x, y, z = v
    ax, ay = scale
    img_x, img_y = img.width() / 2, img.height() / 2
    return safe_division(ax * x + img_x * z, z), safe_division(ay * y + img_y * z, z)


def intensity_color(color: Color, intensity: float) -> Color:
    r, g, b = color
    intensity = abs(intensity)
    return round(r * intensity), round(g * intensity), round(b * intensity)


def barisentrik_coordinates(v: ImageVertex, f: Obj2DFace) -> BarisentricVertex:
    x, y = v
    v0, v1, v2 = f
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
    return vec_cos(v)


def vec_cos(v: ObjVertex) -> float:
    norm = vec_norm(v)
    return safe_division(v[2], norm)
