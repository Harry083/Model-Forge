"""Reads a COLMAP sparse model exported as text (cameras/images/points3D.txt) and works out
where the subject is, which way is up, and how to frame it."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SparseModel:
    image_names: list[str] = field(default_factory=list)
    centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # camera positions (world)
    forwards: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # viewing directions (world)
    ups: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))       # camera "up" (world)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.uint8))
    errors: np.ndarray = field(default_factory=lambda: np.zeros(0))
    track_lengths: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    view_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # mean centre of cameras seeing each point


def _quat_to_matrix(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _data_lines(path: Path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                yield line


def read_text_model(folder: str | Path) -> SparseModel:
    folder = Path(folder)
    model = SparseModel()

    id_to_index: dict[int, int] = {}
    centers, forwards, ups = [], [], []
    lines = _data_lines(folder / "images.txt")
    for header in lines:
        next(lines, None)  # the 2D observations line that follows every image
        parts = header.split()
        image_id = int(parts[0])
        qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        rot = _quat_to_matrix(qw, qx, qy, qz)  # world -> camera
        id_to_index[image_id] = len(centers)
        centers.append(-rot.T @ np.array([tx, ty, tz]))
        forwards.append(rot.T @ np.array([0.0, 0.0, 1.0]))
        ups.append(rot.T @ np.array([0.0, -1.0, 0.0]))  # image rows grow downwards
        model.image_names.append(" ".join(parts[9:]))
    if centers:
        model.centers = np.array(centers)
        model.forwards = np.array(forwards)
        model.ups = np.array(ups)

    pts, cols, errs, tracks, views = [], [], [], [], []
    for line in _data_lines(folder / "points3D.txt"):
        parts = line.split()
        pts.append([float(v) for v in parts[1:4]])
        cols.append([int(v) for v in parts[4:7]])
        errs.append(float(parts[7]))
        image_ids = [int(v) for v in parts[8::2]]
        tracks.append(len(image_ids))
        seen = [id_to_index[i] for i in image_ids if i in id_to_index]
        views.append(model.centers[seen].mean(axis=0) if seen else np.zeros(3))
    if pts:
        model.points = np.array(pts)
        model.colors = np.array(cols, dtype=np.uint8)
        model.errors = np.array(errs)
        model.track_lengths = np.array(tracks, dtype=np.int64)
        model.view_points = np.array(views)
    return model


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:  # opposite: turn half-way round any perpendicular axis
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def focus_point(model: SparseModel) -> np.ndarray:
    """Where the cameras' viewing rays pass closest to each other — the thing being photographed.
    Falls back to the median sparse point when the cameras all look the same way (e.g. a wall)."""
    fallback = np.median(model.points, axis=0) if len(model.points) else np.zeros(3)
    if len(model.centers) < 2:
        return fallback
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(model.centers, model.forwards):
        m = np.eye(3) - np.outer(d, d)
        a += m
        b += m @ c
    if np.linalg.cond(a) > 1e4:
        return fallback
    point = np.linalg.solve(a, b)
    # the rays must meet in front of the cameras, not behind them
    if np.median(np.einsum("ij,ij->i", model.forwards, point - model.centers)) <= 0:
        return fallback
    return point


@dataclass
class Framing:
    focus: np.ndarray
    radius: float           # crop radius around the focus (world units), inf when not cropping
    rotation: np.ndarray    # world -> upright model space (+Y up, first camera on +Z)
    center: np.ndarray      # translated to the origin
    scale: float            # makes the subject's largest dimension 1 unit


def compute_framing(model: SparseModel, crop: bool, crop_factor: float = 0.8) -> Framing:
    focus = focus_point(model)
    radius = float("inf")
    if crop and len(model.centers):
        radius = crop_factor * float(np.median(np.linalg.norm(model.centers - focus, axis=1)))

    up = model.ups.mean(axis=0) if len(model.ups) else np.array([0.0, 1.0, 0.0])
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
    rotation = _rotation_between(up, np.array([0.0, 1.0, 0.0]))

    # spin about the vertical so the first photo looks at the front of the model
    if len(model.centers):
        v = rotation @ (model.centers[0] - focus)
        yaw = np.arctan2(v[0], v[2])
        cy, sy = np.cos(-yaw), np.sin(-yaw)
        rotation = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ rotation

    pts = model.points
    if len(pts):
        inside = np.linalg.norm(pts - focus, axis=1) <= radius
        if inside.sum() >= 10:
            pts = pts[inside]
    if len(pts) >= 10:
        local = (pts - focus) @ rotation.T
        lo, hi = np.percentile(local, 1, axis=0), np.percentile(local, 99, axis=0)
        center = focus + rotation.T @ ((lo + hi) / 2)
        extent = float((hi - lo).max())
    else:
        center, extent = focus, 1.0
    return Framing(focus, radius, rotation, center, 1.0 / extent if extent > 0 else 1.0)
