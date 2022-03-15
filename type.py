from typing import Callable

Color = tuple[int, int, int]

ObjVertex = tuple[float, float, float]
Obj2DVertex = tuple[float, float]
BarisentricVertex = tuple[float, float, float]
ImageVertex = tuple[int, int]

ObjFace = tuple[ObjVertex, ObjVertex, ObjVertex]
Obj2DFace = tuple[Obj2DVertex, Obj2DVertex, Obj2DVertex]
