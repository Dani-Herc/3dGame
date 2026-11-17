#!/usr/bin/env python3
"""
Procedural generator for a stylised (Pixar-like) cartoon fox.

Builds a smooth, soft character from signed-distance-field primitives,
polygonises it, decimates it to a game-friendly triangle budget, skins it
to a quadruped skeleton and bakes a set of game animations.

Output: assets/fox.glb  (glTF 2.0 binary, skinned mesh + animations)

Run:  python3 tools/build_fox.py
Deps: numpy, scikit-image, pyfqmr
"""
import json
import math
import os
import struct
import sys

import numpy as np
import pyfqmr
from skimage.measure import marching_cubes

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "fox.glb")

VOXEL = 0.0045          # marching cubes resolution (metres)
TARGET_TRIS = 26000     # body triangle budget after decimation
FPS = 30


# ----------------------------------------------------------------------------
# small math helpers
# ----------------------------------------------------------------------------
def V(*a):
    return np.array(a, dtype=np.float64)


def length(v):
    return np.sqrt(np.einsum("...i,...i->...", v, v))


def normalize(v):
    return v / np.linalg.norm(v)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def srgb(hexstr):
    c = np.array([int(hexstr[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def mix(a, b, t):
    t = np.asarray(t)[..., None]
    return a * (1 - t) + b * t


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_mat(r):  # degrees, applied X then Y then Z
    rx, ry, rz = np.radians(r)
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def mat_to_quat(m):
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


# ----------------------------------------------------------------------------
# SDF primitives
# ----------------------------------------------------------------------------
def sd_ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = length(q)
    k1 = length(q / r)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_roundcone(p, a, b, r1, r2):
    """Inigo Quilez' round cone between two arbitrary points."""
    ba = b - a
    l2 = ba @ ba
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    w = pa * l2 - np.outer(y, ba)
    x2 = np.einsum("ij,ij->i", w, w)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d1 = np.sqrt(x2 + z2) * il2 - r2
    d2 = np.sqrt(x2 + y2) * il2 - r1
    d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d1,
                    np.where(np.sign(y) * a2 * y2 < k, d2, d3))


def warp_flat(p, c, n, f):
    """Squash space along direction n around c by factor f (bijective)."""
    return p + (f - 1.0) * ((p - c) @ n)[:, None] * n


# ----------------------------------------------------------------------------
# Palette (linear)
# ----------------------------------------------------------------------------
ORANGE = srgb("#F2862E")
ORANGE_TOP = srgb("#D65F1C")
ORANGE_DEEP = srgb("#C24E16")
CREAM = srgb("#FFF3E3")
CREAM_WARM = srgb("#FBE3CC")
SOCK = srgb("#3D2720")
EARTIP = srgb("#34231E")


def fur(p):
    """Orange fur with a darker back - the base for most parts."""
    t = smoothstep(0.40, 0.55, p[:, 1])
    return mix(ORANGE, ORANGE_TOP, t)


def solid(c):
    return lambda p, x=None: np.tile(c, (len(p), 1))


# ----------------------------------------------------------------------------
# Skeleton  (rest pose: all bone rotations identity, fox faces +Z, Y up,
#            fox's left side is +X)
# ----------------------------------------------------------------------------
def mirror(name, parent, pos):
    out = []
    for side, sx in (("L", 1), ("R", -1)):
        par = parent.replace("#", side) if parent else None
        out.append((name.replace("#", side), par, (pos[0] * sx, pos[1], pos[2])))
    return out


BONES = [
    ("root", None, (0, 0, 0)),
    ("hips", "root", (0, 0.40, -0.13)),
    ("spine", "hips", (0, 0.41, 0.0)),
    ("chest", "spine", (0, 0.43, 0.12)),
    ("neck", "chest", (0, 0.50, 0.20)),
    ("head", "neck", (0, 0.63, 0.28)),
]
BONES += mirror("ear_#", "head", (0.075, 0.785, 0.30))
BONES += mirror("eye_#", "head", (0.063, 0.705, 0.418))
BONES += mirror("upperarm_#", "chest", (0.075, 0.38, 0.17))
BONES += mirror("forearm_#", "upperarm_#", (0.072, 0.23, 0.15))
BONES += mirror("hand_#", "forearm_#", (0.07, 0.065, 0.18))
BONES += mirror("thigh_#", "hips", (0.085, 0.38, -0.15))
BONES += mirror("shin_#", "thigh_#", (0.09, 0.25, -0.09))
BONES += mirror("foot_#", "shin_#", (0.09, 0.13, -0.19))
BONES += [
    ("tail1", "hips", (0, 0.43, -0.23)),
    ("tail2", "tail1", (0, 0.47, -0.36)),
    ("tail3", "tail2", (0, 0.52, -0.49)),
    ("tail4", "tail3", (0, 0.56, -0.61)),
]
BONE_INDEX = {b[0]: i for i, b in enumerate(BONES)}
BONE_POS = {b[0]: V(*b[2]) for b in BONES}
BONE_PARENT = {b[0]: b[1] for b in BONES}


# ----------------------------------------------------------------------------
# Body parts
# ----------------------------------------------------------------------------
class Part:
    def __init__(self, name, bone, sdf, color, k=0.03, sub=None, wbias=0.0):
        self.name, self.bone, self._sdf, self.color, self.k = name, bone, sdf, color, k
        self.sub = sub          # optional carving SDF
        self.wbias = wbias      # skin-weight bias (positive = weaker)

    def eval(self, p):
        d = self._sdf(p)
        ds = None
        if self.sub is not None:
            ds = self.sub(p)
            d = np.maximum(d, -ds)
        return d, ds


def build_parts():
    P = []
    E, RC = sd_ellipsoid, sd_roundcone

    # --- torso -------------------------------------------------------------
    def belly_col(c0):
        def f(p, x=None):
            c = c0(p)
            return mix(c, CREAM, smoothstep(0.345, 0.305, p[:, 1]))
        return f

    P.append(Part("hips", "hips", lambda p: E(p, V(0, 0.405, -0.13), V(0.118, 0.118, 0.135)),
                  belly_col(fur), k=0.05))
    P.append(Part("belly", "spine", lambda p: E(p, V(0, 0.40, 0.0), V(0.108, 0.108, 0.13)),
                  belly_col(fur), k=0.05))
    P.append(Part("chest", "chest", lambda p: E(p, V(0, 0.425, 0.12), V(0.122, 0.132, 0.12)),
                  belly_col(fur), k=0.05))
    P.append(Part("chest_fluff", "chest",
                  lambda p: E(p, V(0, 0.452, 0.20), V(0.088, 0.112, 0.078)),
                  solid(CREAM), k=0.035))
    P.append(Part("chest_tuft", "chest",
                  lambda p: RC(p, V(0, 0.40, 0.235), V(0, 0.335, 0.255), 0.045, 0.012),
                  solid(CREAM), k=0.03))

    def neck_col(p, x=None):
        zaxis = 0.17 + (p[:, 1] - 0.47) * (0.11 / 0.15)
        return mix(fur(p), CREAM, smoothstep(0.015, 0.045, p[:, 2] - zaxis))

    P.append(Part("neck", "neck", lambda p: RC(p, V(0, 0.47, 0.17), V(0, 0.62, 0.28), 0.088, 0.082),
                  neck_col, k=0.05))

    # --- head ----------------------------------------------------------------
    def cranium_col(p, x=None):
        c = fur(p)
        c = mix(c, CREAM, smoothstep(0.645, 0.62, p[:, 1]) * smoothstep(0.30, 0.36, p[:, 2]))
        return c

    P.append(Part("cranium", "head", lambda p: E(p, V(0, 0.70, 0.31), V(0.148, 0.128, 0.132)),
                  cranium_col, k=0.04))
    P.append(Part("forehead", "head", lambda p: E(p, V(0, 0.755, 0.335), V(0.10, 0.07, 0.09)),
                  lambda p, x=None: fur(p), k=0.04))
    for sx in (1, -1):
        P.append(Part("cheek", "head",
                      lambda p, sx=sx: E(p, V(0.088 * sx, 0.645, 0.35), V(0.078, 0.062, 0.078)),
                      solid(CREAM), k=0.035))
        P.append(Part("tuft_a", "head",
                      lambda p, sx=sx: RC(p, V(0.115 * sx, 0.648, 0.31), V(0.222 * sx, 0.602, 0.285), 0.048, 0.008),
                      solid(CREAM), k=0.03))
        P.append(Part("tuft_b", "head",
                      lambda p, sx=sx: RC(p, V(0.11 * sx, 0.615, 0.30), V(0.19 * sx, 0.56, 0.262), 0.036, 0.006),
                      solid(CREAM), k=0.025))

    def muzzle_col(p, x=None):
        yaxis = 0.668 - (p[:, 2] - 0.40) * (0.024 / 0.13)
        c = mix(fur(p), CREAM, smoothstep(0.004, -0.012, p[:, 1] - yaxis))
        return c

    P.append(Part("muzzle", "head",
                  lambda p: RC(p, V(0, 0.668, 0.40), V(0, 0.644, 0.535), 0.064, 0.03),
                  muzzle_col, k=0.035))
    P.append(Part("chin", "head",
                  lambda p: E(p, V(0, 0.625, 0.44), V(0.05, 0.03, 0.06)),
                  solid(CREAM), k=0.03))

    # --- ears ------------------------------------------------------------------
    for sx in (1, -1):
        base = V(0.078 * sx, 0.775, 0.302)
        tip = V(0.158 * sx, 0.975, 0.275)
        axis = normalize(tip - base)
        n = normalize(np.cross(axis, V(0, 0, 1)) * 0 + V(0.18 * sx, 0.12, 1.0))
        n = normalize(n - (n @ axis) * axis)
        fl = 2.3

        def ear_sdf(p, base=base, tip=tip, n=n, fl=fl):
            q = warp_flat(p, base, n, fl)
            return RC(q, base, tip, 0.064, 0.009)

        def ear_sub(p, base=base, tip=tip, n=n, fl=fl, axis=axis):
            off = n * 0.026 + axis * 0.028
            q = warp_flat(p, base + off, n, fl * 0.75)
            return RC(q, base + off, tip + n * 0.012 - axis * 0.012, 0.046, 0.004)

        def ear_col(p, x, base=base, axis=axis, n=n):
            t = ((p - base) @ axis) / np.linalg.norm(tip - base)
            back = fur(p)
            back = mix(back, EARTIP, smoothstep(0.62, 0.78, t))
            inner = mix(CREAM_WARM, CREAM, smoothstep(0.2, 0.7, t))
            return mix(back, inner, smoothstep(0.012, -0.002, x))

        P.append(Part("ear", "ear_" + ("L" if sx > 0 else "R"), ear_sdf, ear_col, k=0.022, sub=ear_sub))

    # --- front legs -------------------------------------------------------------
    for sx, s in ((1, "L"), (-1, "R")):
        def ua_col(p, x=None):
            return mix(fur(p), SOCK, smoothstep(0.27, 0.2, p[:, 1]))
        P.append(Part("upperarm", "upperarm_" + s,
                      lambda p, sx=sx: RC(p, V(0.075 * sx, 0.40, 0.16), V(0.072 * sx, 0.23, 0.15), 0.056, 0.034),
                      ua_col, k=0.045))
        P.append(Part("forearm", "forearm_" + s,
                      lambda p, sx=sx: RC(p, V(0.072 * sx, 0.23, 0.15), V(0.07 * sx, 0.065, 0.18), 0.034, 0.027),
                      lambda p, x=None: mix(fur(p), SOCK, smoothstep(0.25, 0.19, p[:, 1])), k=0.015))
        P.append(Part("hand", "hand_" + s,
                      lambda p, sx=sx: E(p, V(0.07 * sx, 0.036, 0.205), V(0.043, 0.036, 0.058)),
                      solid(SOCK), k=0.02))

    # --- hind legs ----------------------------------------------------------------
    for sx, s in ((1, "L"), (-1, "R")):
        P.append(Part("thigh_mass", "thigh_" + s,
                      lambda p, sx=sx: E(p, V(0.088 * sx, 0.345, -0.13), V(0.062, 0.102, 0.097)),
                      lambda p, x=None: fur(p), k=0.05))
        P.append(Part("thigh", "thigh_" + s,
                      lambda p, sx=sx: RC(p, V(0.085 * sx, 0.38, -0.15), V(0.09 * sx, 0.25, -0.09), 0.05, 0.034),
                      lambda p, x=None: fur(p), k=0.03))
        P.append(Part("shin", "shin_" + s,
                      lambda p, sx=sx: RC(p, V(0.09 * sx, 0.25, -0.09), V(0.09 * sx, 0.13, -0.19), 0.033, 0.025),
                      lambda p, x=None: mix(fur(p), SOCK, smoothstep(0.215, 0.16, p[:, 1])), k=0.015))
        P.append(Part("foot", "foot_" + s,
                      lambda p, sx=sx: RC(p, V(0.09 * sx, 0.13, -0.19), V(0.09 * sx, 0.045, -0.16), 0.025, 0.026),
                      solid(SOCK), k=0.012))
        P.append(Part("hpaw", "foot_" + s,
                      lambda p, sx=sx: E(p, V(0.09 * sx, 0.036, -0.14), V(0.043, 0.036, 0.058)),
                      solid(SOCK), k=0.02))

    # --- tail -----------------------------------------------------------------------
    tail_pts = [V(0, 0.425, -0.21), V(0, 0.47, -0.36), V(0, 0.52, -0.49), V(0, 0.56, -0.61), V(0, 0.59, -0.72)]
    tail_r = [0.048, 0.092, 0.112, 0.092, 0.014]

    def tail_col(p, x=None):
        c = mix(fur(p), ORANGE_DEEP, 0.15)
        return mix(c, CREAM, smoothstep(-0.575, -0.625, p[:, 2]))

    for i in range(4):
        a, b, r1, r2 = tail_pts[i], tail_pts[i + 1], tail_r[i], tail_r[i + 1]
        P.append(Part("tail%d" % (i + 1), "tail%d" % (i + 1),
                      lambda p, a=a, b=b, r1=r1, r2=r2: RC(p, a, b, r1, r2),
                      tail_col, k=0.04 if i == 0 else 0.02))
    return P


PARTS = build_parts()


def field(p, with_parts=False):
    ds, subs = [], []
    total = None
    for part in PARTS:
        d, s = part.eval(p)
        ds.append(d)
        subs.append(s)
        total = d if total is None else smin(total, d, part.k)
    if with_parts:
        return total, ds, subs
    return total


# ----------------------------------------------------------------------------
# Mesh generation
# ----------------------------------------------------------------------------
def build_body():
    lo = V(-0.27, -0.01, -0.80)
    hi = V(0.27, 1.01, 0.62)
    n = np.ceil((hi - lo) / VOXEL).astype(int) + 1
    print("grid", n, "=", n.prod() / 1e6, "M samples")
    xs = [lo[i] + np.arange(n[i]) * VOXEL for i in range(3)]
    vol = np.empty(tuple(n), dtype=np.float32)
    # slice along x to bound memory
    Y, Z = np.meshgrid(xs[1], xs[2], indexing="ij")
    yz = np.stack([Y.ravel(), Z.ravel()], 1)
    for i, x in enumerate(xs[0]):
        p = np.column_stack([np.full(len(yz), x), yz])
        vol[i] = field(p).reshape(n[1], n[2])
    verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(VOXEL,) * 3)
    verts = verts + lo
    print("marching cubes:", len(verts), "verts", len(faces), "tris")

    s = pyfqmr.Simplify()
    s.setMesh(verts.astype(np.float64), faces.astype(np.int32))
    s.simplify_mesh(target_count=TARGET_TRIS, aggressiveness=5, preserve_border=True, verbose=False)
    verts, faces, _ = s.getMesh()
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    print("decimated:", len(verts), "verts", len(faces), "tris")

    # project vertices back onto the iso-surface (two Newton steps)
    for _ in range(2):
        d, g = sdf_grad(verts)
        verts = verts - (d / np.maximum((g * g).sum(1), 1e-12))[:, None] * g

    d, g = sdf_grad(verts)
    normals = g / np.linalg.norm(g, axis=1, keepdims=True)
    # marching cubes winding: make sure faces point along the gradient
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    if (np.einsum("ij,ij->i", fn, normals[faces[:, 0]]) < 0).mean() > 0.5:
        faces = faces[:, [0, 2, 1]]

    # colours + skin weights from the parts
    _, ds, subs = field(verts, with_parts=True)
    D = np.stack(ds, 1)                                  # (N, parts)
    dmin = D.min(1, keepdims=True)
    wc = np.exp(-(D - dmin) / 0.006)
    wc /= wc.sum(1, keepdims=True)
    col = np.zeros((len(verts), 3))
    for j, part in enumerate(PARTS):
        m = wc[:, j] > 1e-4
        if m.any():
            col[m] += wc[m, j, None] * part.color(verts[m], None if subs[j] is None else subs[j][m])

    # baked soft ambient occlusion (SDF based)
    ao = np.zeros(len(verts))
    for i in range(1, 6):
        h = 0.014 * i
        ao += (h - field(verts + normals * h)) / (2 ** i)
    ao = np.clip(1.0 - 9.0 * ao, 0.0, 1.0)
    col *= (0.55 + 0.45 * ao)[:, None]

    # skin weights
    nb = len(BONES)
    wb = np.zeros((len(verts), nb))
    W = np.exp(-(D - dmin + np.array([p.wbias for p in PARTS])) / 0.014)
    for j, part in enumerate(PARTS):
        wb[:, BONE_INDEX[part.bone]] += W[:, j]
    joints, weights = top4(wb)
    return verts, normals, faces, col, joints, weights


def sdf_grad(p, e=4e-4):
    d = field(p)
    g = np.zeros_like(p)
    for i in range(3):
        o = np.zeros(3)
        o[i] = e
        g[:, i] = (field(p + o) - field(p - o)) / (2 * e)
    return d, g


def top4(wb):
    idx = np.argsort(-wb, axis=1)[:, :4]
    w = np.take_along_axis(wb, idx, 1)
    w[w < 0.02 * w[:, :1]] = 0
    w /= w.sum(1, keepdims=True)
    idx[w == 0] = 0
    return idx.astype(np.uint8), w.astype(np.float32)


def uv_sphere(seg=32, rings=20):
    vs, ns = [], []
    for i in range(rings + 1):
        th = math.pi * i / rings
        for j in range(seg + 1):
            ph = 2 * math.pi * j / seg
            v = (math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph))
            vs.append(v)
    vs = np.array(vs)
    f = []
    for i in range(rings):
        for j in range(seg):
            a = i * (seg + 1) + j
            b = a + seg + 1
            f += [(a, a + 1, b), (a + 1, b + 1, b)]
    f = np.array(f)
    # orientation check (outward)
    fn = np.cross(vs[f[:, 1]] - vs[f[:, 0]], vs[f[:, 2]] - vs[f[:, 0]])
    c = vs[f].mean(1)
    if (np.einsum("ij,ij->i", fn, c) < 0).mean() > 0.5:
        f = f[:, [0, 2, 1]]
    return vs, f


def ellipsoid_mesh(center, radii, R=np.eye(3), seg=24, rings=14):
    v, f = uv_sphere(seg, rings)
    n = v / radii             # normal transform for scale: inverse scale
    v = (v * radii) @ R.T + center
    n = n @ R.T
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return v, n, f


def build_eyes():
    """Layered cartoon eyes: sclera, iris ring, iris, pupil, highlights."""
    prims = {k: [] for k in ("sclera", "iris_dark", "iris", "pupil", "shine")}
    for s, sx in (("L", 1), ("R", -1)):
        c = BONE_POS["eye_" + s]
        R = rot_y(math.radians(17 * sx)) @ rot_x(math.radians(-4))
        rs = 0.043
        sc = V(rs, rs * 1.16, rs * 0.78)
        layers = [
            ("sclera", V(0, 0, 0), sc),
            ("iris_dark", V(0, 0.003, sc[2] - 0.0105), V(0.0355, 0.0395, 0.0135)),
            ("iris", V(0, 0.003, sc[2] - 0.0088), V(0.0305, 0.0340, 0.0128)),
            ("pupil", V(0, 0.003, sc[2] - 0.0052), V(0.0170, 0.0200, 0.0105)),
            ("shine", V(0.010 * sx, 0.016, sc[2] + 0.0040), V(0.0078, 0.0088, 0.0035)),
            ("shine", V(-0.008 * sx, -0.008, sc[2] + 0.0028), V(0.0034, 0.0034, 0.002)),
        ]
        for name, off, rad in layers:
            v, n, f = ellipsoid_mesh(c + R @ off, rad, R)
            prims[name].append((v, n, f, BONE_INDEX["eye_" + s]))
    # glossy nose
    v, n, f = ellipsoid_mesh(V(0, 0.657, 0.553), V(0.029, 0.021, 0.021), rot_x(math.radians(-12)))
    prims["nose"] = [(v, n, f, BONE_INDEX["head"])]
    return prims


# ----------------------------------------------------------------------------
# Posing / FK / IK
# ----------------------------------------------------------------------------
NB = len(BONES)
REST_LOCAL = np.array([BONE_POS[b] - (BONE_POS[BONE_PARENT[b]] if BONE_PARENT[b] else 0)
                       for b, _, _ in BONES])

LEGS = {
    "FL": ("upperarm_L", "forearm_L", "hand_L", "chest", V(0.07, 0, 0.205)),
    "FR": ("upperarm_R", "forearm_R", "hand_R", "chest", V(-0.07, 0, 0.205)),
    "HL": ("thigh_L", "shin_L", "foot_L", "hips", V(0.09, 0, -0.14)),
    "HR": ("thigh_R", "shin_R", "foot_R", "hips", V(-0.09, 0, -0.14)),
}
HIND_TOE = {"HL": V(0.09, 0.045, -0.16), "HR": V(-0.09, 0.045, -0.16)}


class Pose:
    def __init__(self):
        self.R = np.zeros((NB, 3))     # euler degrees
        self.T = np.zeros((NB, 3))     # translation offsets
        self.S = np.ones((NB, 3))

    def copy(self):
        p = Pose()
        p.R, p.T, p.S = self.R.copy(), self.T.copy(), self.S.copy()
        return p

    def rot(self, b, x=0, y=0, z=0):
        self.R[BONE_INDEX[b]] += (x, y, z)
        return self

    def move(self, b, x=0, y=0, z=0):
        self.T[BONE_INDEX[b]] += (x, y, z)
        return self

    def scale(self, b, x=1, y=1, z=1):
        self.S[BONE_INDEX[b]] *= (x, y, z)
        return self

    def fk(self):
        Rw = [None] * NB
        Pw = [None] * NB
        for i, (b, par, _) in enumerate(BONES):
            Rl = euler_mat(self.R[i])
            tl = REST_LOCAL[i] + self.T[i]
            if par is None:
                Rw[i], Pw[i] = Rl, tl
            else:
                j = BONE_INDEX[par]
                Rw[i] = Rw[j] @ Rl
                Pw[i] = Pw[j] + Rw[j] @ tl
        return Rw, Pw


def ang(v):  # angle of a (y, z) vector, matching a rotation about +X
    return math.atan2(v[1], v[0])


def rot2(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def solve_leg(pose, leg, target, paw_deg=0.0):
    """Two-bone IK in the sagittal plane + paw orientation (world, relative to rest)."""
    b1, b2, b3, parent, ground = LEGS[leg]
    i1, i2, i3 = BONE_INDEX[b1], BONE_INDEX[b2], BONE_INDEX[b3]
    hind = leg[0] == "H"
    paw = math.radians(paw_deg)
    if hind:
        toe_rest = HIND_TOE[leg]
        f0 = (toe_rest - BONE_POS[b3])[1:]
        toe_t = target + (toe_rest - ground)
        f = rot2(f0, paw)
        end_t = toe_t - np.array([0, f[0], f[1]])
    else:
        end_t = target + (BONE_POS[b3] - ground)
    Rw, Pw = pose.fk()
    ip = BONE_INDEX[parent]
    Rp, Pp = Rw[ip], Pw[ip]
    pitch = math.atan2(Rp[2, 1], Rp[1, 1])
    o = REST_LOCAL[i1] + pose.T[i1]
    loc = Rp.T @ (end_t - Pp) - o
    t2 = loc[1:]
    u0 = (BONE_POS[b2] - BONE_POS[b1])[1:]
    l0 = (BONE_POS[b3] - BONE_POS[b2])[1:]
    L1, L2 = np.linalg.norm(u0), np.linalg.norm(l0)
    d = np.linalg.norm(t2)
    d = np.clip(d, abs(L1 - L2) + 1e-4, (L1 + L2) * 0.999)
    tdir = t2 / max(np.linalg.norm(t2), 1e-9)
    t2 = tdir * d
    ca = np.clip((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1)
    alpha = math.acos(ca)
    rest_t = u0 + l0
    rest_sign = np.sign(u0[0] * rest_t[1] - u0[1] * rest_t[0])
    best = None
    for sgn in (1, -1):
        u = rot2(tdir, sgn * alpha) * L1
        s = np.sign(u[0] * t2[1] - u[1] * t2[0])
        if best is None or s == rest_sign:
            best = u
            if s == rest_sign:
                break
    u = best
    th1 = ang(u) - ang(u0)
    lower = ang(t2 - u) - ang(l0)
    th2 = lower - th1
    th3 = paw - pitch - lower
    base = pose.R[[i1, i2, i3], 0].copy()
    pose.R[i1, 0] = math.degrees(th1)
    pose.R[i2, 0] = math.degrees(th2)
    pose.R[i3, 0] = math.degrees(th3)
    return pose


def plant(pose, legs=None, offsets=None, paws=None):
    offsets = offsets or {}
    paws = paws or {}
    for leg in legs or LEGS:
        solve_leg(pose, leg, LEGS[leg][4] + offsets.get(leg, 0), paws.get(leg, 0))
    return pose


def fk_leg(pose, leg, a1, a2, a3):
    b1, b2, b3 = LEGS[leg][:3]
    pose.R[BONE_INDEX[b1], 0] = a1
    pose.R[BONE_INDEX[b2], 0] = a2
    pose.R[BONE_INDEX[b3], 0] = a3


def tail_chain(pose, x=0, y=0, z=0, wave_x=None, wave_y=None):
    for i in range(4):
        b = "tail%d" % (i + 1)
        pose.rot(b, x * (1 if i == 0 else 0.6), y * (1 if i == 0 else 0.6), z)
        if wave_x is not None:
            pose.rot(b, x=wave_x(i))
        if wave_y is not None:
            pose.rot(b, y=wave_y(i))


def blink_scale(t, times, dur=0.16):
    s = 1.0
    for bt in times:
        u = (t - bt) / dur
        if 0 <= u <= 1:
            s = min(s, 1 - 0.92 * math.sin(math.pi * u) ** 0.6)
    return s


def eyes(pose, sy):
    pose.scale("eye_L", 1, sy, 1)
    pose.scale("eye_R", 1, sy, 1)


TAU = 2 * math.pi
S = lambda x: math.sin(TAU * x)
C = lambda x: math.cos(TAU * x)


# ----------------------------------------------------------------------------
# Animation clips
# ----------------------------------------------------------------------------
def clip_idle(t, T=4.0):
    u = t / T
    p = Pose()
    br = S(u * 2)  # two breaths per loop
    p.move("hips", y=-0.004 + 0.003 * br)
    p.rot("spine", x=-1.0 * br)
    p.rot("chest", x=1.5 * br)
    p.rot("neck", x=-2 + 1.5 * S(u * 2 + 0.1))
    p.rot("head", x=2 + 2 * S(u * 2 + 0.2), y=10 * S(u) , z=5 * S(u + 0.1))
    # ear twitches
    tw = lambda c, w=0.18: max(0.0, math.sin(math.pi * (t - c) / w)) if 0 <= t - c <= w else 0.0
    p.rot("ear_L", x=-3 + 2 * S(u * 2), z=-14 * tw(1.6))
    p.rot("ear_R", x=-3 + 2 * S(u * 2 + 0.3), z=14 * tw(3.0) + 10 * tw(3.25))
    tail_chain(p, x=12, wave_y=lambda i: 9 * S(u * 2 - i * 0.09), wave_x=lambda i: 3 * S(u * 2 - i * 0.1))
    eyes(p, blink_scale(t, (1.15, 2.85, 3.05)))
    plant(p)
    return p


def gait(p, phase, stride, lift, duty, curl_f, curl_h, legs_phase):
    offs, paws = {}, {}
    for leg, ph in legs_phase.items():
        f = (phase + ph) % 1.0
        if f < duty:
            v = f / duty
            dz = stride * (0.5 - v)
            dy = 0.0
            a = 0.0
            if v > 0.75:    # heel-off
                a = -(v - 0.75) / 0.25 * (8 if leg[0] == "F" else -6)
        else:
            v = (f - duty) / (1 - duty)
            e = 0.5 - 0.5 * math.cos(math.pi * v)
            dz = stride * (-0.5 + e)
            dy = lift * math.sin(math.pi * v) ** 1.2
            a = (curl_f if leg[0] == "F" else curl_h) * math.sin(math.pi * min(1, v * 1.15))
        offs[leg] = V(0, dy, dz)
        paws[leg] = a
    plant(p, offsets=offs, paws=paws)


def clip_walk(t, T=1.0):
    u = t / T
    p = Pose()
    p.move("hips", y=-0.012 + 0.007 * C(2 * u + 0.1))
    p.rot("hips", x=1.5 * S(2 * u), y=4 * S(u), z=2.5 * S(u + 0.25))
    p.rot("spine", y=-2 * S(u))
    p.rot("chest", y=-3 * S(u), z=-3 * S(u + 0.25))
    p.rot("neck", x=-4 + 2 * S(2 * u + 0.25))
    p.rot("head", x=4 - 3 * S(2 * u + 0.35), y=3 * S(u), z=2 * S(u + 0.3))
    p.rot("ear_L", x=-4 + 4 * S(2 * u - 0.1), z=-3)
    p.rot("ear_R", x=-4 + 4 * S(2 * u - 0.1), z=3)
    tail_chain(p, x=8, wave_y=lambda i: 10 * S(u - 0.12 * i), wave_x=lambda i: 3 * S(2 * u - 0.15 * i))
    eyes(p, blink_scale(t, (0.55,)))
    gait(p, u, stride=0.16, lift=0.055, duty=0.62, curl_f=55, curl_h=35,
         legs_phase={"HL": 0.0, "FL": 0.25, "HR": 0.5, "FR": 0.75})
    return p


def clip_run(t, T=0.5):
    u = t / T
    p = Pose()
    p.move("hips", y=-0.03 + 0.035 * S(u - 0.05), z=0.01 * S(u))
    p.rot("hips", x=-9 * S(u + 0.1))
    p.rot("spine", x=6 * S(u + 0.1))
    p.rot("chest", x=4 + 6 * S(u + 0.1))
    p.rot("neck", x=-14 - 5 * S(u + 0.2))
    p.rot("head", x=12 + 6 * S(u + 0.3))
    p.rot("ear_L", x=-38 + 6 * S(u - 0.1), z=-8)
    p.rot("ear_R", x=-38 + 6 * S(u - 0.1), z=8)
    tail_chain(p, x=-6, wave_x=lambda i: 10 * S(u - 0.12 * i - 0.2), wave_y=lambda i: 3 * S(u - 0.1 * i))
    gait(p, u, stride=0.30, lift=0.09, duty=0.36, curl_f=80, curl_h=55,
         legs_phase={"HL": 0.0, "HR": 0.08, "FL": 0.52, "FR": 0.60})
    return p


def clip_sit(t, T=3.0):
    u = t / T
    p = Pose()
    p.move("hips", y=-0.175, z=0.03)
    p.rot("hips", x=-36)
    p.rot("spine", x=-12)
    p.rot("chest", x=-8 + 1.5 * S(2 * u))
    p.rot("neck", x=18)
    p.rot("head", x=32 + 2 * S(2 * u + 0.2), z=8 * S(u), y=6 * S(u + 0.3))
    p.rot("ear_L", x=2, z=-6 * max(0, S(u * 1 + 0.3)) ** 6)
    p.rot("ear_R", x=2)
    tail_chain(p, x=48, wave_y=lambda i: [30, 18, 14, 10][i] * S(2 * u - 0.1 * i),
               wave_x=lambda i: [-8, -16, -14, -10][i])
    eyes(p, blink_scale(t, (0.9, 2.3)))
    plant(p, offsets={"FL": V(0, 0, -0.02), "FR": V(0, 0, -0.02),
                      "HL": V(0.01, 0, 0.10), "HR": V(-0.01, 0, 0.10)},
          paws={"HL": -62, "HR": -62})
    return p


def keyed(keys, t, loop=False):
    """Hermite interpolation of a list of (time, Pose)."""
    times = [k[0] for k in keys]
    if t <= times[0]:
        return keys[0][1]
    if t >= times[-1]:
        return keys[-1][1]
    i = max(j for j in range(len(times) - 1) if times[j] <= t)
    t0, t1 = times[i], times[i + 1]
    s = (t - t0) / (t1 - t0)

    def arr(k):
        p = keys[k][1]
        return np.concatenate([p.R.ravel(), p.T.ravel(), p.S.ravel()])

    P0, P1 = arr(i), arr(i + 1)

    def tangent(k):
        if k == 0 or k == len(keys) - 1:
            return np.zeros_like(P0)
        return (arr(k + 1) - arr(k - 1)) / (times[k + 1] - times[k - 1]) * 0.8

    m0 = tangent(i) * (t1 - t0)
    m1 = tangent(i + 1) * (t1 - t0)
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    v = h00 * P0 + h10 * m0 + h01 * P1 + h11 * m1
    out = Pose()
    n = NB * 3
    out.R = v[:n].reshape(NB, 3)
    out.T = v[n:2 * n].reshape(NB, 3)
    out.S = v[2 * n:].reshape(NB, 3)
    return out


def rest_pose():
    p = Pose()
    tail_chain(p, x=12)
    p.rot("ear_L", x=-3)
    p.rot("ear_R", x=-3)
    return plant(p)


def make_jump():
    k = []
    k.append((0.0, rest_pose()))
    p = Pose()   # crouch
    p.move("hips", y=-0.075, z=-0.01).rot("hips", x=4).rot("chest", x=6).rot("neck", x=6).rot("head", x=-6)
    p.rot("ear_L", x=-22, z=-6).rot("ear_R", x=-22, z=6)
    tail_chain(p, x=-4, wave_x=lambda i: -6)
    eyes(p, 0.85)
    k.append((0.22, plant(p)))
    p = Pose()   # take-off
    p.move("hips", y=0.08, z=0.05).rot("hips", x=-26).rot("chest", x=-4).rot("neck", x=-4).rot("head", x=18)
    p.rot("ear_L", x=-30).rot("ear_R", x=-30)
    tail_chain(p, x=-10, wave_x=lambda i: -8)
    plant(p, legs=["HL", "HR"], offsets={"HL": V(0, 0, -0.05), "HR": V(0, 0, -0.05)}, paws={"HL": 25, "HR": 25})
    fk_leg(p, "FL", -55, 85, 40)
    fk_leg(p, "FR", -50, 80, 40)
    k.append((0.36, p))
    p = Pose()   # apex
    p.move("hips", y=0.34, z=0.04).rot("hips", x=-6).rot("chest", x=-2).rot("neck", x=-6).rot("head", x=8)
    p.rot("ear_L", x=-16, z=-10).rot("ear_R", x=-16, z=10)
    tail_chain(p, x=14, wave_x=lambda i: 10)
    fk_leg(p, "FL", -38, 60, 30)
    fk_leg(p, "FR", -42, 62, 30)
    fk_leg(p, "HL", -40, 45, -15)
    fk_leg(p, "HR", -36, 42, -15)
    k.append((0.58, p))
    p = Pose()   # reach for landing
    p.move("hips", y=0.10, z=0.02).rot("hips", x=12).rot("chest", x=2).rot("neck", x=-10).rot("head", x=-6)
    p.rot("ear_L", x=8).rot("ear_R", x=8)
    tail_chain(p, x=24, wave_x=lambda i: 6)
    fk_leg(p, "FL", -22, 8, 10)
    fk_leg(p, "FR", -18, 6, 10)
    fk_leg(p, "HL", 10, 30, -10)
    fk_leg(p, "HR", 14, 28, -10)
    k.append((0.78, p))
    p = Pose()   # land squash
    p.move("hips", y=-0.07).rot("hips", x=5).rot("chest", x=6).rot("neck", x=8).rot("head", x=-8)
    p.rot("ear_L", x=14, z=-10).rot("ear_R", x=14, z=10)
    tail_chain(p, x=26, wave_x=lambda i: 8)
    eyes(p, 0.7)
    k.append((0.88, plant(p)))
    p = rest_pose()
    p.move("hips", y=0.006)
    plant(p)
    k.append((1.02, p))
    k.append((1.2, rest_pose()))
    return lambda t: keyed(k, t)


def make_attack():
    k = [(0.0, rest_pose())]
    p = Pose()  # wind-up: crouch back, wiggle
    p.move("hips", y=-0.055, z=-0.05).rot("hips", x=8).rot("chest", x=8).rot("neck", x=10).rot("head", x=-14)
    p.rot("ear_L", x=12).rot("ear_R", x=12)
    tail_chain(p, x=10, wave_y=lambda i: 14 * (1 if i % 2 else -1))
    eyes(p, 0.75)
    k.append((0.28, plant(p)))
    p = Pose()  # pounce
    p.move("hips", y=0.05, z=0.10).rot("hips", x=-22).rot("chest", x=-6).rot("neck", x=-12).rot("head", x=14)
    p.rot("ear_L", x=-10).rot("ear_R", x=-10)
    tail_chain(p, x=-4, wave_x=lambda i: -6)
    plant(p, legs=["HL", "HR"], offsets={"HL": V(0, 0, 0.02), "HR": V(0, 0, 0.02)}, paws={"HL": 30, "HR": 30})
    fk_leg(p, "FL", -78, 40, 20)
    fk_leg(p, "FR", -70, 45, 20)
    eyes(p, 1.1)
    k.append((0.44, p))
    p = Pose()  # strike down
    p.move("hips", y=-0.03, z=0.12).rot("hips", x=10).rot("chest", x=10).rot("neck", x=8).rot("head", x=14)
    p.rot("ear_L", x=6).rot("ear_R", x=6)
    tail_chain(p, x=22, wave_x=lambda i: 6)
    plant(p, offsets={"FL": V(0, 0, 0.17), "FR": V(0, 0, 0.15), "HL": V(0, 0, 0.06), "HR": V(0, 0, 0.06)})
    eyes(p, 0.8)
    k.append((0.58, p))
    p = Pose()
    p.move("hips", y=-0.01, z=0.06).rot("hips", x=4).rot("head", x=4)
    tail_chain(p, x=14)
    plant(p, offsets={"FL": V(0, 0, 0.08), "FR": V(0, 0, 0.07), "HL": V(0, 0, 0.03), "HR": V(0, 0, 0.03)})
    k.append((0.78, p))
    k.append((1.0, rest_pose()))
    return lambda t: keyed(k, t)


def make_hit():
    k = [(0.0, rest_pose())]
    p = Pose()
    p.move("hips", y=0.01, z=-0.05).rot("hips", x=-10, z=4).rot("chest", x=-6, y=-6).rot("neck", x=-8)
    p.rot("head", x=-12, y=14, z=-10)
    p.rot("ear_L", x=-55, z=-15).rot("ear_R", x=-55, z=15)
    tail_chain(p, x=-18, wave_x=lambda i: -8)
    eyes(p, 0.25)
    k.append((0.09, plant(p, offsets={"FL": V(0, 0.03, -0.02)})))
    p = Pose()
    p.move("hips", y=-0.02, z=-0.03).rot("hips", x=-2).rot("head", x=4, y=6)
    p.rot("ear_L", x=-30).rot("ear_R", x=-30)
    tail_chain(p, x=-6)
    eyes(p, 0.6)
    k.append((0.24, plant(p)))
    k.append((0.55, rest_pose()))
    return lambda t: keyed(k, t)


def make_death():
    k = [(0.0, rest_pose())]
    p = Pose()
    p.move("hips", y=0.01, z=-0.04).rot("hips", x=-12).rot("head", x=-16, y=10)
    p.rot("ear_L", x=-50).rot("ear_R", x=-50)
    tail_chain(p, x=-10)
    eyes(p, 0.2)
    k.append((0.12, plant(p)))
    p = Pose()
    p.rot("root", z=22).move("root", x=0.06).move("hips", y=-0.05).rot("head", x=10, z=-10)
    p.rot("ear_L", x=-40).rot("ear_R", x=-40)
    tail_chain(p, x=-8)
    eyes(p, 0.5)
    fk_leg(p, "FL", -10, 10, 10)
    fk_leg(p, "FR", 10, 10, 5)
    fk_leg(p, "HL", 8, 6, 0)
    fk_leg(p, "HR", -6, 10, 0)
    k.append((0.45, p))
    p = Pose()
    p.rot("root", z=88).move("root", x=0.275, y=0.122).rot("hips", x=4)
    p.rot("neck", x=-6, z=-8).rot("head", x=-10, z=-18)
    p.rot("ear_L", x=-30, z=-12).rot("ear_R", x=-30, z=-12)
    tail_chain(p, x=-2, z=-6)
    eyes(p, 0.08)
    fk_leg(p, "FL", -28, 30, 20)
    fk_leg(p, "FR", -12, 18, 15)
    fk_leg(p, "HL", -20, 18, 10)
    fk_leg(p, "HR", 8, 20, 5)
    k.append((0.85, p))
    p2 = p.copy()
    p2.move("root", y=0.012)
    p2.rot("head", z=6)
    k.append((0.97, p2))
    p3 = p.copy()
    p3.rot("head", z=-4).rot("neck", z=-4)
    tail_chain(p3, z=-4)
    k.append((1.6, p3))
    return lambda t: keyed(k, t)


CLIPS = [
    ("Idle", clip_idle, 4.0),
    ("Walk", clip_walk, 1.0),
    ("Run", clip_run, 0.5),
    ("Jump", make_jump(), 1.2),
    ("Attack", make_attack(), 1.0),
    ("Hit", make_hit(), 0.55),
    ("Death", make_death(), 1.6),
    ("Sit", clip_sit, 3.0),
]


def bake(fn, T, loop):
    n = int(round(T * FPS)) + 1
    times = np.linspace(0, T, n)
    rots = np.zeros((n, NB, 4))
    trans = np.zeros((n, NB, 3))
    scales = np.zeros((n, NB, 3))
    for i, t in enumerate(times):
        tt = t % T if (loop and i == n - 1) else t
        p = fn(tt if not loop or i < n - 1 else 0.0)
        for b in range(NB):
            q = mat_to_quat(euler_mat(p.R[b]))
            if i > 0 and np.dot(q, rots[i - 1, b]) < 0:
                q = -q
            rots[i, b] = q
        trans[i] = REST_LOCAL + p.T
        scales[i] = p.S
    return times.astype(np.float32), rots.astype(np.float32), trans.astype(np.float32), scales.astype(np.float32)


# ----------------------------------------------------------------------------
# glTF writer
# ----------------------------------------------------------------------------
class GLB:
    def __init__(self):
        self.bin = bytearray()
        self.views, self.accessors = [], []

    def add(self, arr, ctype, typ, target=None, minmax=False, normalized=False):
        arr = np.ascontiguousarray(arr)
        while len(self.bin) % 4:
            self.bin += b"\0"
        off = len(self.bin)
        self.bin += arr.tobytes()
        view = {"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes}
        if target:
            view["target"] = target
        self.views.append(view)
        acc = {"bufferView": len(self.views) - 1, "componentType": ctype,
               "count": int(arr.shape[0]), "type": typ}
        if normalized:
            acc["normalized"] = True
        if minmax:
            a2 = arr.reshape(arr.shape[0], -1)
            acc["min"] = [float(x) for x in a2.min(0)]
            acc["max"] = [float(x) for x in a2.max(0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def write_glb(path, body, eyes_prims):
    verts, normals, faces, col, joints, weights = body
    g = GLB()
    F, U8, U16, U32 = 5126, 5121, 5123, 5125
    ARR, ELEM = 34962, 34963
    materials = [
        {"name": "Fur", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0,
                                                  "roughnessFactor": 0.78}},
        {"name": "EyeWhite", "pbrMetallicRoughness": {"baseColorFactor": list(srgb("#FFFDF8")) + [1],
                                                       "metallicFactor": 0.0, "roughnessFactor": 0.2}},
        {"name": "IrisDark", "pbrMetallicRoughness": {"baseColorFactor": list(srgb("#2E6B2A")) + [1],
                                                       "metallicFactor": 0.0, "roughnessFactor": 0.15}},
        {"name": "Iris", "pbrMetallicRoughness": {"baseColorFactor": list(srgb("#7CC242")) + [1],
                                                   "metallicFactor": 0.0, "roughnessFactor": 0.12},
         "emissiveFactor": list(srgb("#1E3A0C"))},
        {"name": "Pupil", "pbrMetallicRoughness": {"baseColorFactor": list(srgb("#0E0B0A")) + [1],
                                                    "metallicFactor": 0.0, "roughnessFactor": 0.08}},
        {"name": "EyeShine", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                                                       "metallicFactor": 0.0, "roughnessFactor": 0.1},
         "emissiveFactor": [1, 1, 1]},
        {"name": "Nose", "pbrMetallicRoughness": {"baseColorFactor": list(srgb("#1B1414")) + [1],
                                                   "metallicFactor": 0.0, "roughnessFactor": 0.22}},
    ]
    mat_index = {"sclera": 1, "iris_dark": 2, "iris": 3, "pupil": 4, "shine": 5, "nose": 6}

    prims = []
    col4 = np.ones((len(col), 4), dtype=np.float32)
    col4[:, :3] = col
    prims.append({
        "attributes": {
            "POSITION": g.add(verts.astype(np.float32), F, "VEC3", ARR, minmax=True),
            "NORMAL": g.add(normals.astype(np.float32), F, "VEC3", ARR),
            "COLOR_0": g.add(col4, F, "VEC4", ARR),
            "JOINTS_0": g.add(joints, U8, "VEC4", ARR),
            "WEIGHTS_0": g.add(weights, F, "VEC4", ARR),
        },
        "indices": g.add(faces.astype(np.uint32).ravel(), U32, "SCALAR", ELEM),
        "material": 0,
    })
    for name, parts in eyes_prims.items():
        vs, ns, fs, js = [], [], [], []
        off = 0
        for v, n, f, bone in parts:
            vs.append(v)
            ns.append(n)
            fs.append(f + off)
            js.append(np.full(len(v), bone))
            off += len(v)
        v = np.concatenate(vs).astype(np.float32)
        n = np.concatenate(ns).astype(np.float32)
        f = np.concatenate(fs).astype(np.uint16).ravel()
        j = np.zeros((len(v), 4), np.uint8)
        j[:, 0] = np.concatenate(js)
        w = np.zeros((len(v), 4), np.float32)
        w[:, 0] = 1
        prims.append({
            "attributes": {
                "POSITION": g.add(v, F, "VEC3", ARR, minmax=True),
                "NORMAL": g.add(n, F, "VEC3", ARR),
                "JOINTS_0": g.add(j, U8, "VEC4", ARR),
                "WEIGHTS_0": g.add(w, F, "VEC4", ARR),
            },
            "indices": g.add(f, U16, "SCALAR", ELEM),
            "material": mat_index[name],
        })

    # nodes: 0 = skinned mesh, 1.. = bones (both scene roots, as glTF recommends)
    nodes = [{"name": "FoxMesh", "mesh": 0, "skin": 0}]
    bone_node = {}
    for i, (b, par, _) in enumerate(BONES):
        bone_node[b] = 1 + i
        nodes.append({"name": b, "translation": [float(x) for x in REST_LOCAL[i]]})
    for b, par, _ in BONES:
        if par:
            nodes[bone_node[par]].setdefault("children", []).append(bone_node[b])

    ibm = np.zeros((NB, 16), np.float32)
    for i, (b, _, _) in enumerate(BONES):
        m = np.eye(4)
        m[:3, 3] = -BONE_POS[b]
        ibm[i] = m.T.ravel()   # column-major
    skin = {"name": "FoxRig", "joints": [bone_node[b] for b, _, _ in BONES],
            "skeleton": bone_node["root"], "inverseBindMatrices": g.add(ibm, F, "MAT4")}

    anims = []
    loops = {"Idle", "Walk", "Run", "Sit"}
    for name, fn, T in CLIPS:
        print("baking", name)
        times, rots, trans, scales = bake(fn, T, name in loops)
        tin = g.add(times, F, "SCALAR", minmax=True)
        samplers, channels = [], []
        for i, (b, _, _) in enumerate(BONES):
            samplers.append({"input": tin, "output": g.add(rots[:, i], F, "VEC4"), "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": bone_node[b], "path": "rotation"}})
            if b in ("root", "hips"):
                samplers.append({"input": tin, "output": g.add(trans[:, i], F, "VEC3"), "interpolation": "LINEAR"})
                channels.append({"sampler": len(samplers) - 1, "target": {"node": bone_node[b], "path": "translation"}})
            if b.startswith("eye_"):
                samplers.append({"input": tin, "output": g.add(scales[:, i], F, "VEC3"), "interpolation": "LINEAR"})
                channels.append({"sampler": len(samplers) - 1, "target": {"node": bone_node[b], "path": "scale"}})
        anims.append({"name": name, "samplers": samplers, "channels": channels})

    gltf = {
        "asset": {"version": "2.0", "generator": "3dGame tools/build_fox.py"},
        "scene": 0,
        "scenes": [{"name": "Fox", "nodes": [0, 1]}],
        "nodes": nodes,
        "meshes": [{"name": "Fox", "primitives": prims}],
        "skins": [skin],
        "materials": materials,
        "animations": anims,
        "accessors": g.accessors,
        "bufferViews": g.views,
        "buffers": [{"byteLength": len(g.bin)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    while len(js) % 4:
        js += b" "
    binb = bytes(g.bin)
    while len(binb) % 4:
        binb += b"\0"
    total = 12 + 8 + len(js) + 8 + len(binb)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<I4s", len(js), b"JSON") + js)
        fh.write(struct.pack("<I4s", len(binb), b"BIN\0") + binb)
    print("wrote", path, "%.1f KB" % (total / 1024))


def main():
    body = build_body()
    eyes_prims = build_eyes()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    write_glb(OUT, body, eyes_prims)


if __name__ == "__main__":
    main()
