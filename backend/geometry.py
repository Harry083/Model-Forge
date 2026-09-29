"""Point-cloud and mesh processing: PLY in/out, normals, clean-up, and GLB/OBJ export.

Everything here works on plain numpy arrays:
  points  (N, 3) float  positions
  colors  (N, 3) uint8  sRGB vertex colours
  normals (N, 3) float  unit normals (optional)
  faces   (M, 3) int    triangle vertex indices (meshes only)
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


class GeometryError(RuntimeError):
    pass


@dataclass
class Geometry:
    points: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None

    @property
    def is_mesh(self) -> bool:
        return self.faces is not None and len(self.faces) > 0


# ---------------------------------------------------------------- PLY ----

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply(path: str | Path) -> Geometry:
    """Reads the binary/ASCII PLY files COLMAP writes (points, or triangle meshes)."""
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise GeometryError(f"Not a PLY file: {path}")
        fmt = None
        elements: list[dict] = []
        while True:
            line = fh.readline()
            if not line:
                raise GeometryError(f"Truncated PLY header: {path}")
            parts = line.decode("ascii", "replace").split()
            if not parts or parts[0] in ("comment", "obj_info"):
                continue
            if parts[0] == "end_header":
                break
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                elements.append({"name": parts[1], "count": int(parts[2]), "props": []})
            elif parts[0] == "property":
                if parts[1] == "list":
                    elements[-1]["props"].append(("list", parts[4], _PLY_TYPES[parts[2]], _PLY_TYPES[parts[3]]))
                else:
                    elements[-1]["props"].append(("scalar", parts[2], _PLY_TYPES[parts[1]]))
        body = fh.read()

    if fmt not in ("binary_little_endian", "ascii"):
        raise GeometryError(f"Unsupported PLY format '{fmt}' in {path}")

    data: dict[str, np.ndarray] = {}
    if fmt == "binary_little_endian":
        offset = 0
        for el in elements:
            dtype_fields = []
            for prop in el["props"]:
                if prop[0] == "scalar":
                    dtype_fields.append((prop[1], "<" + prop[2]))
                else:  # COLMAP only writes triangle lists, which makes the record fixed-size
                    dtype_fields.append((prop[1] + "_n", "<" + prop[2]))
                    dtype_fields.append((prop[1], "<" + prop[3], (3,)))
            dtype = np.dtype(dtype_fields)
            arr = np.frombuffer(body, dtype=dtype, count=el["count"], offset=offset)
            offset += dtype.itemsize * el["count"]
            for prop in el["props"]:
                if prop[0] == "list" and el["count"] and not np.all(arr[prop[1] + "_n"] == 3):
                    raise GeometryError("Only triangle meshes are supported")
            data[el["name"]] = arr
    else:
        lines = body.decode("ascii", "replace").splitlines()
        pos = 0
        for el in elements:
            rows = [ln.split() for ln in lines[pos:pos + el["count"]]]
            pos += el["count"]
            if el["name"] == "face":
                data["face"] = {"vertex_indices": np.array([[int(v) for v in r[1:4]] for r in rows], dtype=np.int64)}
            else:
                names = [p[1] for p in el["props"] if p[0] == "scalar"]
                table = np.array(rows, dtype=np.float64).reshape(-1, len(names))
                data[el["name"]] = {n: table[:, i] for i, n in enumerate(names)}

    vertex = data.get("vertex")
    if vertex is None:
        raise GeometryError(f"PLY has no vertices: {path}")

    def cols(*names):
        if all(n in _field_names(vertex) for n in names):
            return np.stack([np.asarray(vertex[n]) for n in names], axis=1)
        return None

    points = cols("x", "y", "z").astype(np.float64)
    colors = cols("red", "green", "blue")
    if colors is not None:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    normals = cols("nx", "ny", "nz")
    if normals is not None:
        normals = normals.astype(np.float64)

    faces = None
    if "face" in data:
        face = data["face"]
        key = next((k for k in ("vertex_indices", "vertex_index") if k in _field_names(face)), None)
        if key is not None:
            faces = np.asarray(face[key]).astype(np.int64).reshape(-1, 3)
    return Geometry(points, colors, normals, faces)


def _field_names(arr) -> tuple:
    return arr.dtype.names if hasattr(arr, "dtype") else tuple(arr.keys())


def write_ply(path: str | Path, geo: Geometry) -> None:
    n = len(geo.points)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if geo.normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if geo.colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vert = np.empty(n, dtype=np.dtype(fields))
    vert["x"], vert["y"], vert["z"] = geo.points.T
    if geo.normals is not None:
        vert["nx"], vert["ny"], vert["nz"] = geo.normals.T
    if geo.colors is not None:
        vert["red"], vert["green"], vert["blue"] = geo.colors.T

    header = ["ply", "format binary_little_endian 1.0", "comment Made with Model Forge", f"element vertex {n}"]
    header += [f"property {'float' if t == '<f4' else 'uchar'} {name}" for name, t in fields]
    if geo.is_mesh:
        header += [f"element face {len(geo.faces)}", "property list uchar int vertex_indices"]
    header.append("end_header")

    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(vert.tobytes())
        if geo.is_mesh:
            face = np.empty(len(geo.faces), dtype=np.dtype([("n", "u1"), ("i", "<i4", (3,))]))
            face["n"] = 3
            face["i"] = geo.faces
            fh.write(face.tobytes())


# ------------------------------------------------------------ normals ----

def estimate_normals(points: np.ndarray, view_points: np.ndarray, k: int = 16) -> np.ndarray:
    """PCA normals from each point's k nearest neighbours, flipped to face `view_points`
    (for each point, a camera position that saw it) so the Poisson surface closes the right way."""
    k = max(3, min(k, len(points)))
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    neigh = points[idx]  # (N, k, 3)
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered)
    _, vecs = np.linalg.eigh(cov)
    normals = vecs[:, :, 0]  # eigenvector of the smallest eigenvalue
    flip = np.einsum("ij,ij->i", normals, view_points - points) < 0
    normals[flip] *= -1
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def statistical_outliers(points: np.ndarray, k: int = 12, std_ratio: float = 2.0) -> np.ndarray:
    """Boolean mask of points to keep: those whose mean neighbour distance isn't unusually large."""
    if len(points) <= k:
        return np.ones(len(points), dtype=bool)
    dist, _ = cKDTree(points).query(points, k=k + 1)
    mean_d = dist[:, 1:].mean(axis=1)
    return mean_d <= mean_d.mean() + std_ratio * mean_d.std()


# ------------------------------------------------------------ clean-up ----

def filter_vertices(geo: Geometry, keep: np.ndarray) -> Geometry:
    """Drops vertices where keep is False (and any triangle that used one)."""
    remap = np.full(len(geo.points), -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    faces = None
    if geo.faces is not None:
        faces = remap[geo.faces]
        faces = faces[(faces >= 0).all(axis=1)]
    return Geometry(
        geo.points[keep],
        None if geo.colors is None else geo.colors[keep],
        None if geo.normals is None else geo.normals[keep],
        faces,
    )


def remove_unreferenced(geo: Geometry) -> Geometry:
    if not geo.is_mesh:
        return geo
    used = np.zeros(len(geo.points), dtype=bool)
    used[geo.faces.ravel()] = True
    return filter_vertices(geo, used)


def keep_largest_component(geo: Geometry) -> Geometry:
    """Keeps only the connected piece of the mesh with the most triangles (drops floating debris)."""
    if not geo.is_mesh:
        return geo
    n = len(geo.points)
    f = geo.faces
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    count, labels = connected_components(graph, directed=False)
    if count <= 1:
        return geo
    face_labels = labels[f[:, 0]]
    biggest = np.bincount(face_labels).argmax()
    return remove_unreferenced(Geometry(geo.points, geo.colors, geo.normals, f[face_labels == biggest]))


def vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = points[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # area-weighted
    vn = np.zeros_like(points)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def transform(geo: Geometry, rotation: np.ndarray, center: np.ndarray, scale: float) -> Geometry:
    """p' = scale * R (p - center); normals are only rotated."""
    points = (geo.points - center) @ rotation.T * scale
    normals = None if geo.normals is None else geo.normals @ rotation.T
    return Geometry(points, geo.colors, normals, geo.faces)


# -------------------------------------------------------------- export ----

def _srgb_to_linear(colors: np.ndarray) -> np.ndarray:
    c = colors.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def write_glb(path: str | Path, geo: Geometry, name: str = "model") -> None:
    """A single-mesh glTF 2.0 binary: triangles (with normals) or a point cloud, with vertex colours."""
    positions = geo.points.astype(np.float32)
    chunks: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []

    def add(array: np.ndarray, comp: int, typ: str, target: int, **extra) -> int:
        offset = sum(len(c) for c in chunks)
        raw = np.ascontiguousarray(array).tobytes()
        chunks.append(raw + b"\0" * (-len(raw) % 4))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        accessors.append({"bufferView": len(views) - 1, "componentType": comp,
                          "count": int(len(array)), "type": typ, **extra})
        return len(accessors) - 1

    attributes = {"POSITION": add(positions, 5126, "VEC3", 34962,
                                  min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist())}
    if geo.is_mesh:
        normals = geo.normals if geo.normals is not None else vertex_normals(geo.points, geo.faces)
        attributes["NORMAL"] = add(normals.astype(np.float32), 5126, "VEC3", 34962)
    if geo.colors is not None:
        attributes["COLOR_0"] = add(_srgb_to_linear(geo.colors), 5126, "VEC3", 34962)

    primitive: dict = {"attributes": attributes, "material": 0, "mode": 4 if geo.is_mesh else 0}
    if geo.is_mesh:
        primitive["indices"] = add(geo.faces.astype(np.uint32).ravel(), 5125, "SCALAR", 34963)

    binary = b"".join(chunks)
    doc = {
        "asset": {"version": "2.0", "generator": "Model Forge"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [primitive]}],
        "materials": [{"name": "vertex-colour", "doubleSided": True,
                       "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                                                "metallicFactor": 0.0, "roughnessFactor": 0.9}}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A))
        fh.write(js)
        fh.write(struct.pack("<II", len(binary), 0x004E4942))
        fh.write(binary)


def write_obj(path: str | Path, geo: Geometry) -> None:
    """Wavefront OBJ with per-vertex colours (`v x y z r g b`), read by Blender, MeshLab and most tools."""
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("# Made with Model Forge\n")
        if geo.colors is not None:
            table = np.hstack([geo.points, geo.colors.astype(np.float64) / 255.0])
            np.savetxt(fh, table, fmt="v %.6f %.6f %.6f %.4f %.4f %.4f")
        else:
            np.savetxt(fh, geo.points, fmt="v %.6f %.6f %.6f")
        if geo.is_mesh:
            np.savetxt(fh, geo.faces + 1, fmt="f %d %d %d")
