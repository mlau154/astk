import typing

import numpy as np

from astk.geom.point import Point3D, Origin3D, Point2D, Origin2D
from astk.units.length import Length
from astk.units.unit import Unit


__all__ = [
    "Vector2D",
    "Vector3D"
]


class Vector2D:
    def __init__(self, p0: Point2D, p1: Point2D):
        self.p0 = p0
        self.p1 = p1

    def value(self):
        return [self.p1.x - self.p0.x, self.p1.y - self.p0.y]

    def normalized_value(self) -> typing.List[float]:
        mag = self.mag()
        return [xy / mag for xy in self.value()]

    def get_normalized_vector(self) -> "Vector2D":
        return Vector2D.from_array(np.array(self.normalized_value()))

    def as_array(self):
        return np.array([xyz.m for xyz in self.value()])

    @classmethod
    def from_array(cls, arr: np.ndarray):
        return cls(p0=Origin2D(), p1=Point2D.from_array(arr))

    def dot(self, other: "Vector2D") -> float or Unit:
        A = self.value()
        B = other.value()
        return A[0] * B[0] + A[1] * B[1]

    def mag(self) -> Length:
        return Length(m=np.sqrt(np.sum(self.as_array()**2)))


class Vector3D:
    def __init__(self, p0: Point3D, p1: Point3D):
        self.p0 = p0
        self.p1 = p1

    def value(self):
        return [self.p1.x - self.p0.x, self.p1.y - self.p0.y, self.p1.z - self.p0.z]

    def normalized_value(self) -> typing.List[float]:
        mag = self.mag()
        return [xyz / mag for xyz in self.value()]

    def get_normalized_vector(self) -> "Vector3D":
        return Vector3D.from_array(np.array(self.normalized_value()))

    def as_array(self):
        return np.array([xyz.m for xyz in self.value()])

    @classmethod
    def from_array(cls, arr: np.ndarray):
        return cls(p0=Origin3D(), p1=Point3D.from_array(arr))

    def cross(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(p0=Origin3D(),
                        p1=Point3D.from_array(
                            np.cross(self.as_array(), other.as_array())
                        ))

    def dot(self, other: "Vector3D") -> float or Unit:
        A = self.value()
        B = other.value()
        return A[0] * B[0] + A[1] * B[1] + A[2] * B[2]

    def mag(self) -> Length:
        return Length(m=np.sqrt(np.sum(self.as_array()**2)))