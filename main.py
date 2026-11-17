"""Loewner-order lattice equivalence.

Usage: python main.py  (no arguments; settings are the constants above main(); output in OUT_DIR)."""

import csv
import hashlib
import json
import os
import time
from itertools import product

import numpy as np

E_S, NU_S = 1.0, 0.3
G_S = E_S / (2 * (1 + NU_S))
FE_MODEL_VERSION = 7
FIG_FONT = "Arial"


def sym(A):
    """Symmetric part of A (works on stacks of matrices: the last two axes are transposed)."""
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def tol_of(T):
    """Absolute tolerance used for all 'passes' tests (numerical zero entries)."""
    return 1e-9 * np.abs(T).max()


def vweights(n):
    """D = diag(1,1,1,2,2,2) (n=6, 3D) or diag(1,1,2) (n=3, 2D); identity for any other size (abstract matrices)."""
    return np.diag([1, 1, 2.0]) if n == 3 else (np.diag([1, 1, 1, 2, 2, 2.0]) if n == 6 else np.eye(n))


def sym_w(C, T):
    """sym(D (C - T)): the symmetric, energy-consistent form of the stiffness gap in tensorial Voigt."""
    return sym(vweights(C.shape[-1]) @ (C - T))


def margin(C, T):
    """m(C) = lambda_min(sym(D (C - T))) -- THE one-line diagnostic (works on stacks)."""
    return np.linalg.eigvalsh(sym_w(C, T))[..., 0]


def _canonical_modes(d):
    """The 'named' strain modes (tensorial Voigt) used to pick a representative in a degenerate eigenspace: 2D = deviatoric normal, equibiaxial, pure shear;"""
    if d == 2:
        return [np.array(v, float) for v in ([1, -1, 0], [1, 1, 0], [0, 0, 1])]
    c = [[2, -1, -1, 0, 0, 0], [-1, 2, -1, 0, 0, 0], [-1, -1, 2, 0, 0, 0], [1, 1, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
    return [np.array(v, float) for v in c]


def canonical_mode(C, T, rtol=1e-7):
    """Weakest strain mode eps*. If the minimum eigenvalue is degenerate (cubic symmetry) the eigenvector is arbitrary inside the eigenspace, so report the"""
    A = sym_w(C, T)
    w, V = np.linalg.eigh(A)
    n = A.shape[-1]
    Vs = V[:, w <= w[0] + rtol * max(abs(w).max(), 1e-300)]
    if Vs.shape[1] == 1:
        return V[:, 0]
    best, score = V[:, 0], 0.0
    for c in _canonical_modes(3 if n == 6 else 2):
        pc = Vs @ (Vs.T @ c)
        sc = np.linalg.norm(pc) / np.linalg.norm(c)
        if sc > score + 1e-12:
            best, score = pc / np.linalg.norm(pc), sc
    return best


def pass_loewner(C, T):
    """Loewner test: True iff C - T is PSD in the energy-weighted sense, i.e."""
    return margin(C, T) >= -tol_of(T)


def pass_componentwise(C, T):
    """Competing component-wise test: True iff EVERY entry C_ij >= T_ij (up to the same tolerance); works on stacks."""
    return np.all((C - T) >= -tol_of(T), axis=(-1, -2))


def classify(lo, cw):
    """0 BOTH, 1 LO_ONLY (necessity failure of CW), 2 CW_ONLY (sufficiency failure), 3 NEITHER."""
    return np.where(lo & cw, 0, np.where(lo & ~cw, 1, np.where(~lo & cw, 2, 3)))


def voigt_to_tensor(v, d):
    """Tensorial Voigt strain vector (eps_ij, shear not doubled) -> symmetric strain tensor."""
    if d == 2:
        return np.array([[v[0], v[2]], [v[2], v[1]]])
    return np.array([[v[0], v[5], v[4]],
                     [v[5], v[1], v[3]],
                     [v[4], v[3], v[2]]])


def beam2d_global(dvec, A, I, kappa=5 / 6):
    """6x6 stiffness of a 2D Timoshenko beam in GLOBAL axes."""
    L = np.linalg.norm(dvec)
    c, s = dvec / L
    phi = 12 * E_S * I / (kappa * G_S * A * L ** 2)
    k = np.zeros((6, 6))
    a = E_S * A / L
    k[np.ix_([0, 3], [0, 3])] = a * np.array([[1, -1], [-1, 1]])
    cb =E_S * I / ((1 + phi) * L ** 3)
    kb = cb * np.array([[12, 6 * L, -12, 6 * L],
                        [6 * L, (4 + phi) * L ** 2, -6 * L, (2 - phi) * L ** 2],
                        [-12, -6 * L, 12, -6 * L],
                        [6 * L, (2 - phi) * L ** 2, -6 * L, (4 + phi) * L ** 2]])
    k[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = kb
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    T = np.kron(np.eye(2), R)
    return T.T @ k @ T


def beam3d_global(dvec, A, Iy, Iz, J, kappa=0.9):
    """12x12 stiffness of a 3D Timoshenko beam in GLOBAL axes."""
    L = np.linalg.norm(dvec)
    k = np.zeros((12, 12))
    a = E_S * A / L
    k[0, 0] = k[6, 6] = a
    k[0, 6] = k[6, 0] = -a
    t = G_S * J / L
    k[3, 3] = k[9, 9] = t
    k[3, 9] = k[9, 3] = -t
    phi = 12 * E_S * Iz / (kappa * G_S * A * L ** 2)
    c = E_S * Iz / ((1 + phi) * L ** 3)
    kb = c * np.array([[12, 6 * L, -12, 6 * L],
                       [6 * L, (4 + phi) * L ** 2, -6 * L, (2 - phi) * L ** 2],
                       [-12, -6 * L, 12, -6 * L],
                       [6 * L, (2 - phi) * L ** 2, -6 * L, (4 + phi) * L ** 2]])
    k[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] = kb
    phi = 12 * E_S * Iy / (kappa * G_S * A * L ** 2)
    c = E_S * Iy / ((1 + phi) * L ** 3)
    kb = c * np.array([[12, -6 * L, -12, -6 * L],
                       [-6 * L, (4 + phi) * L ** 2, 6 * L, (2 - phi) * L ** 2],
                       [-12, 6 * L, 12, 6 * L],
                       [-6 * L, (2 - phi) * L ** 2, 6 * L, (4 + phi) * L ** 2]])
    k[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] = kb
    ex = dvec / L
    ref = np.array([0.0, 0.0, 1.0]) if abs(ex[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    ey = np.cross(ref, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    R = np.vstack([ex, ey, ez])
    T = np.kron(np.eye(4), R)
    return T.T @ k @ T


class Lattice:
    """Periodic lattice unit cell (orthogonal box)."""

    def __init__(self, name, dim, nodes, struts, cell):
        self.name, self.dim = name, dim
        self.nodes = np.asarray(nodes, float)
        self.struts = struts
        self.cell = np.asarray(cell, float)

    def strut_vec(self, i, j, nv):
        """End-to-end vector of the strut from node i to node j in the cell shifted by nv (periodic image)."""
        return self.nodes[j] + np.asarray(nv) * self.cell - self.nodes[i]


def _macro_coupling(dvec, d):
    """Maps tensorial-Voigt macro strain eps_V (eps_ij) -> relative end displacement E.d of a strut (d x n_strain matrix M with M @ eps_V = E @ dvec)."""
    if d == 2:
        M = np.array([[dvec[0], 0, dvec[1]], [0, dvec[1], dvec[0]]])
    else:
        d1, d2, d3 = dvec
        M = np.array([[d1, 0, 0, 0, d3, d2],
                      [0, d2, 0, d3, 0, d1],
                      [0, 0, d3, d2, d1, 0]])
    return M


CORR_PARAMS = {}


def corr_of(lat):
    """Calibration constants (ka, kb, pois, pa, pb) of the lattice's own topology (`lat.name`, a key of CORR_PARAMS)."""
    return CORR_PARAMS[lat.name]


def _cross(a):
    """Skew-symmetric matrix [a]x with [a]x @ b = a x b (used for the rigid-arm kinematics of the joint zones)."""
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def end_overlap_factors(lat, size):
    """g for every strut end = max over the other struts j meeting at the node of e_ij / r_i, e_ij = (r_j + r_i cos th)/sin th (length of strut i inside"""
    d = lat.dim
    inc = {}
    for s_idx, (i, j, nv, kind) in enumerate(lat.struts):
        dv = lat.strut_vec(i, j, nv)
        u = dv / np.linalg.norm(dv)
        rr = size[kind] if d == 3 else 0.5 * size[kind]
        inc.setdefault(i, []).append((s_idx, 0, u, rr))
        inc.setdefault(j, []).append((s_idx, 1, -u, rr))
    g = {}
    for lst in inc.values():
        for (s_idx, end, u, ri) in lst:
            best = 0.3
            for (s2, end2, u2, rj) in lst:
                if s2 == s_idx and end2 == end:
                    continue
                c = float(np.clip(u @ u2, -1, 1))
                sn = max(np.sqrt(1 - c * c), 1e-6)
                best = max(best, (rj + ri * c) / sn / ri)
            g[(s_idx, end)] = best
    return g


def _corrected_element(lat, size, s_idx, kind, dv, corr, g):
    """Global stiffness (6x6 in 2D, 12x12 in 3D) of ONE strut with rigid joint zones ('modified beam')."""
    ka, kb, _, pa, pb = corr
    d = lat.dim
    s = size[kind]
    rr = s if d == 3 else 0.5 * s
    gi, gj = g[(s_idx, 0)], g[(s_idx, 1)]
    eb_i, eb_j = kb * rr * gi ** pb, kb * rr * gj ** pb
    Ln = np.linalg.norm(dv)
    ex = dv / Ln
    eb_m = min(0.5 * (eb_i + eb_j), 0.45 * Ln)
    ea = min(ka * rr * 0.5 * (gi ** pa + gj ** pa), 0.45 * Ln)
    Lb, La = Ln - 2 * eb_m, Ln - 2 * ea
    P = np.outer(ex, ex)
    if d == 3:
        A = np.pi * s ** 2
        I = np.pi * s ** 4 / 4
        J = 2 * I
        k = beam3d_global(dv * Lb / Ln, A, I, I, J).copy()
        da, dt = E_S * A * (1 / La - 1 / Lb), G_S * J * (1 / La - 1 / Lb)
        for (r0, c0) in ((0, 0), (0, 6), (6, 0), (6, 6)):
            sg = 1 if r0 == c0 else -1
            k[r0:r0 + 3, c0:c0 + 3] += da * sg * P
            k[r0 + 3:r0 + 6, c0 + 3:c0 + 6] += dt * sg * P
        T = np.eye(12)
        for end, e_end in ((0, eb_i), (1, eb_j)):
            a = (1 if end == 0 else -1) * min(e_end, 0.45 * Ln) * ex
            T[6 * end:6 * end + 3, 6 * end + 3:6 * end + 6] = -_cross(a)
    else:
        k = beam2d_global(dv * Lb / Ln, s, s ** 3 / 12).copy()
        da = E_S * s * (1 / La - 1 / Lb)
        for (r0, c0) in ((0, 0), (0, 3), (3, 0), (3, 3)):
            k[r0:r0 + 2, c0:c0 + 2] += da * (1 if r0 == c0 else -1) * P
        T = np.eye(6)
        for end, e_end in ((0, eb_i), (1, eb_j)):
            a = (1 if end == 0 else -1) * min(e_end, 0.45 * Ln) * ex
            T[3 * end, 3 * end + 2] = -a[1]
            T[3 * end + 1, 3 * end + 2] = a[0]
    return T.T @ k @ T


def homogenize(lat, size, fluct=None, corr=None):
    """Effective stiffness C (tensorial Voigt) of the periodic frame + relative density sum(A L)/V (beam-sum, overlaps counted twice)."""
    d = lat.dim
    ndn = 3 if d == 2 else 6
    ns = 3 if d == 2 else 6
    nn = len(lat.nodes)
    N = ndn * nn
    V = float(np.prod(lat.cell))
    K = np.zeros((N, N))
    B = np.zeros((N, ns))
    A0 = np.zeros((ns, ns))
    vol = 0.0
    g = end_overlap_factors(lat, size) if corr is not None else None
    for s_idx, (i, j, nv, kind) in enumerate(lat.struts):
        dv = lat.strut_vec(i, j, nv)
        L = np.linalg.norm(dv)
        s = size[kind]
        Ae = s if d == 2 else np.pi * s ** 2
        if corr is None:
            if d == 2:
                ke = beam2d_global(dv, s, s ** 3 / 12)
            else:
                I = np.pi * s ** 4 / 4
                ke = beam3d_global(dv, Ae, I, I, 2 * I)
        else:
            ke = _corrected_element(lat, size, s_idx, kind, dv, corr, g)
        vol += Ae * L
        idx = np.r_[i * ndn + np.arange(ndn), j * ndn + np.arange(ndn)]
        G = np.zeros((2 * ndn, ns))
        G[ndn:ndn + d] = _macro_coupling(dv, d)
        np.add.at(K, (idx[:, None], idx[None, :]), ke)
        np.add.at(B, idx, ke @ G)
        A0 += G.T @ ke @ G
    free = np.arange(d, N)
    Kf = K[np.ix_(free, free)]
    Kf = Kf + 1e-13 * np.trace(Kf) / len(free) * np.eye(len(free))
    X = np.linalg.solve(Kf, B[free])
    if fluct is not None:
        u = np.zeros(N)
        u[free] = -(X @ np.asarray(fluct, float))
        Cen = sym((A0 - B[free].T @ X) / V)
        return np.linalg.inv(vweights(ns)) @ Cen, vol / V, u.reshape(nn, ndn)[:, :d]
    Cen = sym((A0 - B[free].T @ X) / V)
    C = np.linalg.inv(vweights(ns)) @ Cen
    if corr is not None:
        pois = corr[2]
        sizes_L = [(size[k], np.linalg.norm(lat.strut_vec(i, j, nv)), k) for (i, j, nv, k) in lat.struts]
        w = [(sz if d == 2 else np.pi * sz ** 2) * ln for sz, ln, _ in sizes_L]
        dl = [(sz if d == 2 else 2 * sz) / ln for sz, ln, _ in sizes_L]
        xbar = float(np.dot(w, dl) / np.sum(w))
        C = C.copy()
        C[:d, :d] += pois * xbar * float(np.mean(np.diag(C)[:d]))
    return C, vol / V


def _honeycomb2d(name, theta_deg, h, l=1.0):
    """Hexagonal (theta > 0) or re-entrant (theta < 0) honeycomb cell with 4 nodes A,B,C,D: vertical walls of length h, inclined walls of length l at angle"""
    th = np.deg2rad(theta_deg)
    w, s = l * np.cos(th), l * np.sin(th)
    Lx, Ly = 2 * w, 2 * h + 2 * s
    nodes = [(0, 0), (0, h), (w, h + s), (w, 2 * h + s)]
    st = [(0, 1, (0, 0), 'v'), (2, 3, (0, 0), 'v'),
          (1, 2, (0, 0), 'h'), (2, 1, (1, 0), 'h'),
          (0, 3, (0, -1), 'h'), (3, 0, (1, 1), 'h')]
    return Lattice(name, 2, nodes, st, (Lx, Ly))


TOPO_2D = ["2D_grid", "2D_xbrace", "2D_triangular", "2D_honeycomb", "2D_reentrant"]


def _make_lattice_2d(topo):
    """Unstretched 2D cell of the named topology."""
    grid = [(0, 0, (1, 0), 'h'), (0, 0, (0, 1), 'v')]
    diag = [(1, 0, (0, 0), 'd'), (1, 0, (1, 0), 'd'), (1, 0, (0, 1), 'd'), (1, 0, (1, 1), 'd')]
    if topo == "2D_grid":
        return Lattice(topo, 2, [(0, 0)], grid, (1, 1))
    if topo == "2D_xbrace":
        return Lattice(topo, 2, [(0, 0), (.5, .5)], grid + diag, (1, 1))
    if topo == "2D_triangular":
        s3 = np.sqrt(3)
        st = [(0, 0, (1, 0), 'h'), (1, 1, (1, 0), 'h'), (0, 1, (0, 0), 'v'),
              (0, 1, (-1, 0), 'v'), (1, 0, (0, 1), 'v'), (1, 0, (1, 1), 'v')]
        return Lattice(topo, 2, [(0, 0), (.5, s3 / 2)], st, (1, s3))
    if topo == "2D_honeycomb":
        return _honeycomb2d(topo, +30.0, 1.0)
    if topo == "2D_reentrant":
        return _honeycomb2d(topo, -30.0, 2.0)
    raise KeyError(topo)


def make_lattice_2d(topo, alpha=1.0):
    """2D cell; alpha = y/x aspect ratio (cell and node y-coordinates stretched by alpha, as alpha stretches z in 3D)."""
    lat = _make_lattice_2d(topo)
    if alpha != 1.0:
        lat.nodes = lat.nodes * np.array([1.0, alpha])
        lat.cell = lat.cell * np.array([1.0, alpha])
    return lat


def size_2d(th, tv):
    """Strut thicknesses of the 2D kinds: horizontal th, vertical tv, diagonal = their mean."""
    return {'h': th, 'v': tv, 'd': 0.5 * (th + tv)}


TOPO_3D = ["3D_SC", "3D_BCC", "3D_FCC", "3D_octet", "3D_diamond", "3D_reentrant"]


def _auto_struts(fr, rules, nmax=1):
    """Struts of a cubic lattice from bond-length rules in fractional coordinates: every node pair (i, j) in periodic image nv whose distance equals a"""
    out = []
    n = len(fr)
    for i in range(n):
        for j in range(i, n):
            for nv in product(range(-nmax, nmax + 1), repeat=3):
                if i == j and nv <= (0, 0, 0):
                    continue
                dv = np.asarray(fr[j]) + nv - np.asarray(fr[i])
                dl = np.linalg.norm(dv)
                for ell, pred in rules:
                    if abs(dl - ell) < 1e-9 and pred(i, j, dv):
                        out.append((i, j, nv, 's'))
                        break
    return out


def _make_lattice_3d(topo, alpha=1.0):
    """3D cell of the named topology with the cell stretched by alpha along z."""
    ax =np.array([1.0, 1.0, alpha])
    anyb = lambda i, j, dv: True
    fcc = [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]
    if topo == "3D_SC":
        fr, rules = [(0, 0, 0)], [(1.0, anyb)]
    elif topo == "3D_BCC":
        fr, rules = [(0, 0, 0), (.5, .5, .5)], [(np.sqrt(3) / 2, anyb)]
    elif topo in ("3D_FCC", "3D_octet"):
        fr = fcc
        pred = anyb if topo == "3D_octet" else (lambda i, j, dv: not (i > 0 and j > 0))
        rules = [(np.sqrt(2) / 2, pred)]
    elif topo == "3D_diamond":
        t = [(.25, .25, .25), (.75, .75, .25), (.75, .25, .75), (.25, .75, .75)]
        fr, rules = fcc + t, [(np.sqrt(3) / 4, anyb)]
    elif topo == "3D_reentrant":
        base = make_lattice_2d("2D_reentrant")
        Lx, Ly = base.cell
        cell = (Lx, Ly, alpha * Lx)
        nodes = np.c_[base.nodes, np.zeros(len(base.nodes))]
        st = [(i, j, tuple(nv) + (0,), k) for (i, j, nv, k) in base.struts]
        st += [(i, i, (0, 0, 1), 'z') for i in range(len(nodes))]
        return Lattice(topo, 3, nodes, st, cell)
    else:
        raise KeyError(topo)
    st = _auto_struts(fr, rules)
    return Lattice(topo, 3, np.asarray(fr) * ax, st, ax)


def make_lattice_3d(topo, alpha=1.0, axis=2):
    """alpha stretches the cell along z (axis=2, the default: aspect z/x with y = x) or along y (axis=1: aspect y/x with z = x)."""
    if axis == 2:
        return _make_lattice_3d(topo, alpha)
    lat = _make_lattice_3d(topo, 1.0)
    f = np.array([1.0, alpha, 1.0])
    lat.nodes = lat.nodes * f
    lat.cell = lat.cell * f
    return lat


def radius_for_density(lat, rho):
    """Strut radius giving BEAM-SUM relative density rho = pi r^2 sum(L) / V (all struts share one radius; joint overlaps counted twice, hence the later"""
    wL = sum(np.linalg.norm(lat.strut_vec(i, j, nv)) for (i, j, nv, k) in lat.struts)
    return np.sqrt(rho * np.prod(lat.cell) / (np.pi * wL))


def size_3d(r):
    """Strut radius per strut kind in 3D (one radius for every kind)."""
    return {'s': r, 'h': r, 'v': r, 'z': r}


def thickness_for_density_2d(lat, rho):
    """Uniform strut thickness t (size_2d(t, t)) giving relative density rho."""
    wL = sum(np.linalg.norm(lat.strut_vec(i, j, nv)) for (i, j, nv, k) in lat.struts)
    return rho * float(np.prod(lat.cell)) / wL


S_CORR_MIN = 5.0


def min_slenderness(lat, size):
    """Minimum over struts of L/d."""
    vals = []
    for (i, j, nv, k) in lat.struts:
        L = np.linalg.norm(lat.strut_vec(i, j, nv))
        sz = size[k]
        d = sz if lat.dim == 2 else np.sqrt(4.0 * (np.pi * sz ** 2) / np.pi)
        vals.append(L / d)
    return float(min(vals))


class Pool:
    """All FE-evaluated designs of one topology on a (fine) 2-parameter grid."""

    def __init__(self, dim, topo, axes, C, rho, par, slender=None, conv=None, vox_actual=None):
        self.dim, self.topo, self.axes = dim, topo, axes
        self.C, self.rho, self.par = C, rho, par
        self.slender = slender
        self.conv = np.ones_like(rho, bool) if conv is None else conv
        self.vox_actual = np.zeros_like(rho) if vox_actual is None else vox_actual

    def sub(self, k=2):
        return Pool(self.dim, self.topo, [a[::k] for a in self.axes],
                    self.C[::k, ::k], self.rho[::k, ::k], self.par[::k, ::k],
                    None if self.slender is None else self.slender[::k, ::k],
                    self.conv[::k, ::k], self.vox_actual[::k, ::k])

    def flat(self):
        n = self.C.shape[-1]
        return self.C.reshape(-1, n, n), self.rho.reshape(-1), self.par.reshape(-1, 2)


def iso_stiffness(E, nu, d):
    """Isotropic stiffness in TENSORIAL Voigt (sigma_V = C eps_V, eps_ij not doubled): shear entries = 2*mu."""
    if d == 2:
        c = E / (1 - nu ** 2)
        return np.array([[c, nu * c, 0], [nu * c, c, 0], [0, 0, c * (1 - nu)]])
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[range(3), range(3)] += 2 * mu
    C[range(3, 6), range(3, 6)] = 2 * mu
    return C


TARGET_SPEC = dict(name="target", ratio=0.02, nu=NU_S)


def set_target(ratio=None, nu=None):
    TARGET_SPEC.update(ratio=TARGET_SPEC["ratio"] if ratio is None else float(ratio), nu=TARGET_SPEC["nu"] if nu is None else float(nu))


def target_ratio():
    return TARGET_SPEC["ratio"]


def target_stiffness(dim):
    """Isotropic target in tensorial Voigt (2D: plane stress), in units of the reference solid's E_s."""
    return iso_stiffness(target_ratio(), TARGET_SPEC["nu"], dim)


def refined_optimum(pool, T, kind):
    """Sub-grid minimum-density design that passes the 'lo' (Loewner) or 'cw' (component-wise) test: for every alpha column the pass/fail margin g (>= 0 <=>"""
    tol = tol_of(T)
    if kind == 'lo':
        g = gi = margin(pool.C, T) + tol
    else:
        g = (pool.C - T).min(axis=(-1, -2)) + tol
        gi = componentwise_margin_field(pool.C, T)
    nr = g.shape[0]
    best = None
    for b in range(g.shape[1]):
        ok = np.flatnonzero(g[:, b] >= 0)
        if len(ok) == 0:
            continue
        a = int(ok[0])
        if a == 0:
            a0, w = 0, 0.0
        elif gi[a - 1, b] < 0 < gi[a, b] - gi[a - 1, b]:
            a0, w = a - 1, float(np.clip(-gi[a - 1, b] / (gi[a, b] - gi[a - 1, b]), 0.0, 1.0))
        else:
            a0, w = a - 1, 1.0
        r = pool.rho[a0, b] + w * (pool.rho[min(a0 + 1, nr - 1), b] - pool.rho[a0, b])
        if best is None or r < best['rho'] - 1e-12:
            best = dict(rho=float(r), alpha=float(pool.par[a0, b, 1]), a0=a0, b=b, w=w)
    if best is not None:
        a0, b, w = best['a0'], best['b'], best['w']
        best['C'] = (1 - w) * pool.C[a0, b] + w * pool.C[min(a0 + 1, nr - 1), b]
    return best


_FE_CACHE = {}
_TOPO_CACHE = {}


def _fe_ke(d, nu, hs):
    """Stiffness of a rectangular quad4 (plane stress, thickness 1) / hex8 element with sides hs (E = 1) on its compatible dofs."""
    key = (d, round(nu, 12), tuple(np.round(hs, 12)))
    if key in _FE_CACHE:
        return _FE_CACHE[key]
    corners = np.array(list(product([0, 1], repeat=d)), float)
    g = 0.5 / np.sqrt(3.0)
    gp = [0.5 - g, 0.5 + g]
    Dm = iso_stiffness(1.0, nu, d) @ np.linalg.inv(vweights(3 if d == 2 else 6))
    nn, ns = len(corners), (3 if d == 2 else 6)
    nu_dof, na = nn * d, d * d
    Kuu, Kua, Kaa = np.zeros((nu_dof, nu_dof)), np.zeros((nu_dof, na)), np.zeros((na, na))
    shear_row = {(1, 2): 3, (0, 2): 4, (0, 1): 5}
    for xi in product(gp, repeat=d):
        dN = np.zeros((nn, d))
        for a_, c in enumerate(corners):
            for i in range(d):
                f = 1.0
                for j in range(d):
                    f *= (1.0 if c[j] == 1 else -1.0) if j == i else (xi[j] if c[j] == 1 else 1 - xi[j])
                dN[a_, i] = f / hs[i]
        B = np.zeros((ns, nu_dof))
        for a_ in range(nn):
            if d == 2:
                B[0, 2 * a_], B[1, 2 * a_ + 1] = dN[a_, 0], dN[a_, 1]
                B[2, 2 * a_], B[2, 2 * a_ + 1] = dN[a_, 1], dN[a_, 0]
            else:
                x, y, z = dN[a_]
                B[0, 3 * a_], B[1, 3 * a_ + 1], B[2, 3 * a_ + 2] = x, y, z
                B[3, 3 * a_ + 1], B[3, 3 * a_ + 2] = z, y
                B[4, 3 * a_], B[4, 3 * a_ + 2] = z, x
                B[5, 3 * a_], B[5, 3 * a_ + 1] = y, x
        Bi = np.zeros((ns, na))
        for i in range(d):
            gp_ = 4.0 * (1.0 - 2.0 * xi[i]) / hs[i]
            for j in range(d):
                col = i * d + j
                if i == j:
                    Bi[j, col] += gp_
                else:
                    Bi[2 if d == 2 else shear_row[tuple(sorted((i, j)))], col] += gp_
        w = (0.5 ** d) * float(np.prod(hs))
        Kuu += w * B.T @ Dm @ B
        Kua += w * B.T @ Dm @ Bi
        Kaa += w * Bi.T @ Dm @ Bi
    _FE_CACHE[key] = (corners, Kuu - Kua @ np.linalg.solve(Kaa, Kua.T))
    return _FE_CACHE[key]


def _fe_topology(ns):
    """Element -> global dof connectivity of a periodic structured grid with ns elements per axis (cached per grid shape)."""
    ns = tuple(ns)
    if ns not in _TOPO_CACHE:
        d = len(ns)
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in ns], indexing='ij'), -1).reshape(-1, d)
        corners = np.array(list(product([0, 1], repeat=d)))
        nodes = (idx[:, None, :] + corners[None, :, :]) % np.array(ns)
        nid = np.ravel_multi_index(tuple(nodes.transpose(2, 0, 1)), ns)
        _TOPO_CACHE[ns] = (nid[:, :, None] * d + np.arange(d)).reshape(len(idx), -1)
    return _TOPO_CACHE[ns]


def _solve_spd(K, F, rtol=1e-8):
    """Solve K X = F for SPD K (F has one column per load case)."""
    if K.shape[0] <= 2500:
        from scipy.sparse.linalg import splu
        return splu(K.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(F), True

    dg = K.diagonal()[:, None]
    X = np.zeros_like(F)
    R = F.copy()
    Z = R / dg
    Pd = Z.copy()
    rz = (R * Z).sum(0)
    b0 = np.maximum(np.linalg.norm(F, axis=0), 1e-300)
    Kc = K.tocsr()
    converged = False
    for _ in range(40000):
        KP = Kc @ Pd
        den = (Pd * KP).sum(0)
        alpha = np.where(den > 0, rz / np.where(den > 0, den, 1.0), 0.0)
        X += alpha * Pd
        R -= alpha * KP
        if np.all(np.linalg.norm(R, axis=0) <= rtol * b0):
            converged = True
            break
        Z = R / dg
        rz_new = (R * Z).sum(0)
        beta = np.where(rz > 0, rz_new / np.where(rz > 0, rz, 1.0), 0.0)
        Pd = Z + beta * Pd
        rz = rz_new
    if not converged:
        from scipy.sparse.linalg import splu
        X = splu(K.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(F)
    return X, converged


def periodic_voxel_homogenize(dens, nu=NU_S, tol=1e-9, cell=None):
    """Periodic homogenisation of a pixel (2D, quad4 plane stress) / voxel (3D, hex8) density field (E_s = 1) on a cell of edge lengths `cell` (default:"""
    from scipy.sparse import coo_matrix
    d = dens.ndim
    ns_ = 3 if d == 2 else 6
    cell = np.ones(d) if cell is None else np.asarray(cell, float)
    hs = cell / np.array(dens.shape)
    corners, Ke = _fe_ke(d, nu, hs)
    edof_all = _fe_topology(dens.shape)
    s_all = dens.reshape(-1)
    act = np.flatnonzero(s_all > tol)
    s = s_all[act]
    udof, inv = np.unique(edof_all[act], return_inverse=True)
    edof = inv.reshape(len(act), -1)
    N, nd = len(udof), edof.shape[1]
    rows = np.repeat(edof, nd, axis=1).ravel()
    cols = np.tile(edof, (1, nd)).ravel()
    K = coo_matrix(((s[:, None, None] * Ke[None]).ravel(), (rows, cols)), shape=(N, N)).tocsc()
    K = K + coo_matrix((np.full(N, 1e-12 * K.diagonal().mean()), (np.arange(N), np.arange(N))), shape=(N, N)).tocsc()
    U0 = np.stack([(corners * hs) @ voigt_to_tensor(np.eye(ns_)[k], d) for k in range(ns_)], -1).reshape(-1, ns_)
    F = np.zeros((N, ns_))
    fe = Ke @ U0
    for k in range(ns_):
        F[:, k] = -np.bincount(edof.ravel(), weights=(s[:, None] * fe[None, :, k]).ravel(), minlength=N)
    free = np.arange(d, N)
    u = np.zeros((N, ns_))
    u[free], converged = _solve_spd(K[free][:, free], F[free])
    Ue = U0[None] + u[edof]
    KU = np.einsum('ab,ebk->eak', Ke, Ue)
    C = np.einsum('e,eak,eal->kl', s, Ue, KU) / float(np.prod(cell))
    return np.linalg.inv(vweights(ns_)) @ sym(C), float(dens.mean()), converged


def rasterize_lattice(lat, size, ns):
    """Partial-volume density of the union of the lattice's struts (3D: circular radius size[kind]; 2D: strip of width size[kind]) on an ns grid over the"""
    d = lat.dim
    cell = lat.cell
    hs = cell / np.array(ns)
    h = float(hs.mean())
    dens = np.zeros(ns)
    axes = [(np.arange(n) + 0.5) * hh for n, hh in zip(ns, hs)]
    for (i, j, nv, k) in lat.struts:
        a0 = lat.nodes[i]
        b0 = a0 + lat.strut_vec(i, j, nv)
        r = size[k] if d == 3 else 0.5 * size[k]
        for sh in product(range(-2, 3), repeat=d):
            off = np.array(sh) * cell
            a, b = a0 + off, b0 + off
            lo = np.minimum(a, b) - r - h
            hi = np.maximum(a, b) + r + h
            if np.any(hi < 0) or np.any(lo > cell):
                continue
            sl, ax_l, ok = [], [], True
            for q in range(d):
                i0 = max(int(np.floor(lo[q] / hs[q])), 0)
                i1 = min(int(np.ceil(hi[q] / hs[q])), ns[q])
                if i1 <= i0:
                    ok = False
                    break
                sl.append(slice(i0, i1))
                ax_l.append(axes[q][i0:i1])
            if not ok:
                continue
            P = np.stack(np.meshgrid(*ax_l, indexing='ij'), -1)
            ab = b - a
            t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
            dist = np.linalg.norm(P - (a + t[..., None] * ab), axis=-1)
            tgt = tuple(sl)
            dens[tgt] = np.maximum(dens[tgt], np.clip((r + 0.5 * h - dist) / h, 0.0, 1.0))
    return dens


def voxel_grid(lat, size, vox_per_diameter, nmin=12, nmax=56):
    """Grid (n per axis) with cubic voxels of size ~ (thinnest strut diameter)/vox_per_diameter, clipped to [nmin, nmax] per axis (nmax caps the cost; where"""
    d_min = min(size[k] if lat.dim == 2 else 2.0 * size[k] for (_, _, _, k) in lat.struts)
    h = d_min / vox_per_diameter
    return tuple(int(np.clip(round(c / h), nmin, nmax)) for c in lat.cell)


def write_csv(path, rows, fields=None):
    """Write a list of dict rows to CSV (columns = `fields`, default the keys of the first row; extra keys are ignored)."""
    fields = fields or list(rows[0])
    base, ext = os.path.splitext(path)
    target, k = path, 0
    while True:
        try:
            f = open(target, "w", newline="")
            break
        except PermissionError:
            k += 1
            target = f"{base}_new{k}{ext}"
    if target != path:
        print(f"WARNING: {path} is locked; wrote {target} instead")
    with f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def design_volume_curve(lat, n_pts=11):
    """True solid volume fraction (rasterised union of struts, overlaps counted once) vs strut size for this cell geometry: returns (sizes, rho_true) with"""
    d = lat.dim
    rho_b = np.geomspace(0.01, 4.0, n_pts)
    rads, vols = [], []
    for rb in rho_b:
        if d == 2:
            t = thickness_for_density_2d(lat, rb)
            sz, rad = size_2d(t, t), t
        else:
            rad = radius_for_density(lat, rb)
            sz = size_3d(rad)
        ns = voxel_grid(lat, sz, 8, nmin=16, nmax=(96 if d == 3 else 256))
        rads.append(rad)
        vols.append(float(rasterize_lattice(lat, sz, ns).mean()))
    vols = np.maximum.accumulate(np.array(vols)) + np.arange(n_pts) * 1e-9
    return np.array(rads), vols


FE_SETTINGS = dict(vox=5, nmax3=40, nmax2=200, vox_max=8)


def hybrid_C(lat, size):
    """Modified beam if L/d >= S_CORR_MIN, else periodic continuum FE. FE voxels-per-diameter is ramped from FE_SETTINGS['vox'] (default 5) up to"""
    d = lat.dim
    sl = min_slenderness(lat, size)
    if sl >= S_CORR_MIN:
        return dict(C=homogenize(lat, size, corr=corr_of(lat))[0],
                    model=0, converged=True, ns=None, vox_actual=0.0)

    v0 = FE_SETTINGS['vox']
    vmax = FE_SETTINGS.get('vox_max', 8)
    vox_target = int(np.clip(round(v0 * np.sqrt(S_CORR_MIN / max(sl, 1.5))), v0, vmax))
    ns = voxel_grid(lat, size, vox_target,
                    nmin=(12 if d == 3 else 24),
                    nmax=(FE_SETTINGS['nmax3'] if d == 3 else FE_SETTINGS['nmax2']))
    d_min = min(size[k] if d == 2 else 2.0 * size[k] for (_, _, _, k) in lat.struts)
    vox_actual = float(d_min / (np.asarray(lat.cell) / np.asarray(ns)).max())

    C, _, conv = periodic_voxel_homogenize(rasterize_lattice(lat, size, ns), cell=lat.cell)
    return dict(C=C, model=1, converged=conv, ns=ns, vox_actual=vox_actual)


def figure_pool(topo, dim, rhos, alphas, axis=2):
    """Display pool for Figures 5-21 on (relative density = true solid fraction, alpha): hybrid corrected-beam / continuum-FE stiffness."""
    nr, na = len(rhos), len(alphas)
    n = 3 if dim == 2 else 6
    rhos = np.asarray(rhos, float)
    alphas = np.asarray(alphas, float)
    C = np.zeros((nr, na, n, n))
    rho = np.tile(rhos[:, None], (1, na))
    par = np.zeros((nr, na, 2))
    sl, rad, model = np.zeros((nr, na)), np.zeros((nr, na)), np.zeros((nr, na), int)
    conv = np.ones((nr, na), bool)
    vox_actual_map = np.zeros((nr, na))
    Cb = np.zeros_like(C)
    lats, sizes = [], {}
    for b, al in enumerate(alphas):
        lat = make_lattice_2d(topo, al) if dim == 2 else make_lattice_3d(topo, al, axis)
        lats.append(lat)
        rads, vols = design_volume_curve(lat)
        for a, rt in enumerate(rhos):
            r = float(np.interp(rt, vols, rads))
            sizes[(a, b)] = size_2d(r, r) if dim == 2 else size_3d(r)
            par[a, b] = (rt, al)
            sl[a, b] = min_slenderness(lat, sizes[(a, b)])
            rad[a, b] = r
    rows_c = sorted(set(range(0, nr, 2)) | {nr - 1})
    b_unit = int(np.argmin(np.abs(alphas - 1.0)))
    b_unit = b_unit if np.isclose(alphas[b_unit], 1.0) else -1
    cols_c = sorted(set(range(0, na, 2)) | {na - 1} | ({b_unit} if b_unit >= 0 else set()))
    need = []
    for b in range(na):
        for a in range(nr):
            if sl[a, b] >= S_CORR_MIN or (a in rows_c and b in cols_c) or b == b_unit:
                out = hybrid_C(lats[b], sizes[(a, b)])
                C[a, b]          = out['C']
                Cb[a, b]         = out['C'] if out['model'] == 0 else homogenize(lats[b], sizes[(a, b)], corr=corr_of(lats[b]))[0]
                model[a, b]      = out['model']
                conv[a, b]       = out['converged']
                vox_actual_map[a, b] = out['vox_actual']
            else:
                need.append((a, b))
    for (a, b) in need:
        a0 = max(i for i in rows_c if i <= a)
        a1 = min(i for i in rows_c if i >= a)
        b0 = max(i for i in cols_c if i <= b)
        b1 = min(i for i in cols_c if i >= b)
        wa = 0.0 if a1 == a0 else (rhos[a] - rhos[a0]) / (rhos[a1] - rhos[a0])
        wb = 0.0 if b1 == b0 else (alphas[b] - alphas[b0]) / (alphas[b1] - alphas[b0])
        R = C - Cb
        res = ((1 - wa) * (1 - wb) * R[a0, b0] + wa * (1 - wb) * R[a1, b0] + (1 - wa) * wb * R[a0, b1] + wa * wb * R[a1, b1])
        C[a, b] = sym(homogenize(lats[b], sizes[(a, b)], corr=corr_of(lats[b]))[0] + res)
        model[a, b] = 2
        conv[a, b] = bool(conv[a0, b0] and conv[a1, b0] and conv[a0, b1] and conv[a1, b1])
        corners = [vox_actual_map[a0, b0], vox_actual_map[a1, b0],
                   vox_actual_map[a0, b1], vox_actual_map[a1, b1]]
        corners = [c for c in corners if c > 0]
        vox_actual_map[a, b] = min(corners) if corners else 0.0
    pool = Pool(dim, topo, [rhos, alphas], C, rho, par, sl, conv, vox_actual_map)
    pool.rad, pool.model = rad, model
    return pool


def figure_pool_cached(cache_dir, topo, dim, rhos, alphas, axis=2):
    """figure_pool() with an on-disk cache (npz) keyed by a SHA-1 of the actual grid bytes, the beam calibration, FE settings, S_CORR_MIN and"""
    key = (np.asarray(rhos,   float).tobytes()
         + np.asarray(alphas, float).tobytes()
         + repr(CORR_PARAMS).encode()
         + str((S_CORR_MIN, FE_MODEL_VERSION, FE_SETTINGS)).encode()
         + str((topo, dim, axis)).encode())
    tag = f"{topo}_{dim}D_ax{axis}_{hashlib.sha1(key).hexdigest()[:16]}.npz"
    path = os.path.join(cache_dir, tag)
    if os.path.exists(path):
        z = np.load(path)
        pool = Pool(dim, topo, [z['rhos'], z['alphas']],
                    z['C'], z['rho'], z['par'], z['sl'], z['conv'], z['vox_actual'])
        pool.rad, pool.model = z['rad'], z['model']
        return pool
    pool = figure_pool(topo, dim, rhos, alphas, axis)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, rhos=pool.axes[0], alphas=pool.axes[1], C=pool.C, rho=pool.rho, par=pool.par,
             sl=pool.slender, rad=pool.rad, model=pool.model, conv=pool.conv, vox_actual=pool.vox_actual)
    return pool


def _pool_job(a):
    """Worker: build one display pool."""
    cache_dir, topo, dim, rhos, alphas, axis, corr, fe = a
    CORR_PARAMS.update(corr)
    FE_SETTINGS.update(fe)
    pool = figure_pool(topo, dim, rhos, alphas, axis) if cache_dir is None else figure_pool_cached(cache_dir, topo, dim, rhos, alphas, axis)
    return (topo, dim, axis), pool


def prefetch_pools(cache_dir, jobs, rhos, alphas, workers):
    """Build the pools of `jobs` = [(topo, dim, axis), ...] in parallel (one process per pool; cache_dir=None disables the disk cache)."""
    from concurrent.futures import ProcessPoolExecutor
    args = [(cache_dir, t, dm, rhos, alphas, ax, dict(CORR_PARAMS), dict(FE_SETTINGS)) for (t, dm, ax) in jobs]
    args.sort(key=lambda a: -a[2])
    ready = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(args))) as ex:
        for k, pool in ex.map(_pool_job, args):
            ready[k] = pool
            print(f"  pool {k[1]}D {k[0]:16s} axis {k[2]} ready", flush=True)
    return ready


def fig_rho_rows(quick):
    """Relative-density rows of the display pools: dense where the (cheap) modified beam applies, coarser above."""
    if quick:
        return np.r_[np.arange(0.02, 0.1801, 0.04), np.arange(0.3, 1.0001, 0.1)]
    return np.r_[np.arange(0.02, 0.2001, 0.02), np.arange(0.25, 1.0001, 0.05)]


def _entry_errors(C, Cr):
    """Largest relative error of a stiffness against a reference: every diagonal entry, normal off-diagonals (relative to sqrt(C_ii C_jj)) and"""
    n = C.shape[0]
    d = 2 if n == 3 else 3
    e = [abs(C[i, i] / Cr[i, i] - 1) for i in range(n)]
    e += [abs(C[i, j] - Cr[i, j]) / np.sqrt(Cr[i, i] * Cr[j, j]) for i in range(d) for j in range(i + 1, d)]
    Dw = vweights(n)
    lam_r = np.linalg.eigvalsh(sym(Dw @ Cr))[0]
    lam_c = np.linalg.eigvalsh(sym(Dw @ C))[0]
    e.append(abs(lam_c / lam_r - 1) if abs(lam_r) > 1e-12 * np.abs(Cr).max() else 0.0)
    return float(max(e))


def _cubic_entries(C):
    """(C11, C12/C11, shear, lambda_min) -- averaged over equivalent entries."""
    n = C.shape[0]
    d = 2 if n == 3 else 3
    c11 = float(np.mean(np.diag(C)[:d]))
    off = float(np.mean([C[i, j] for i in range(d) for j in range(d) if i != j]))
    sh = float(np.mean(np.diag(C)[d:]))
    lam = float(np.linalg.eigvalsh(sym(vweights(n) @ C))[0])
    return c11, off, sh, lam


CALIB_RATIOS = (5, 6, 7, 8, 9, 10, 12)
CALIB_RATIOS_QUICK = (5, 8, 12)
_CALIB_LO = np.array([0.0, 0.0, 0.0, 0.3, 0.3])
_CALIB_HI = np.array([2.0, 3.0, 2.0, 6.0, 6.0])
_CALIB_START = {2: (0.103, 0.594, 0.265, 2.558, 1.075), 3: (0.168, 0.767, 0.433, 3.032, 1.078)}


def _calib_lattice_size(topo, ratio):
    """Lattice of `topo` and strut size for which the MINIMUM strut slenderness (L/d, d = t in 2D, 2r in 3D) equals `ratio`."""
    lat = make_lattice_3d(topo) if topo in TOPO_3D else make_lattice_2d(topo)
    L0 = float(np.linalg.norm(lat.strut_vec(*lat.struts[0][:3])))
    mk = (lambda x: size_3d(x / 2)) if lat.dim == 3 else (lambda x: size_2d(x, x))
    k = min_slenderness(lat, mk(L0 / ratio)) / ratio
    return lat, mk(L0 * k / ratio)


def _calib_reference(job):
    """Continuum-FE reference stiffness of one (topology, L/d) sample."""
    topo, ratio = job
    lat, size = _calib_lattice_size(topo, ratio)
    ns = voxel_grid(lat, size, 8, nmin=24, nmax=(96 if lat.dim == 3 else 256))
    return topo, ratio, periodic_voxel_homogenize(rasterize_lattice(lat, size, ns), cell=lat.cell)[0]


def _signed_entry_errors(C, Cr):
    """Signed counterpart of _entry_errors (least-squares residuals): diagonal entries, normal off-diagonals and lambda_min(sym(D C))."""
    n = C.shape[0]
    d = 2 if n == 3 else 3
    e = [C[i, i] / Cr[i, i] - 1 for i in range(n)]
    e += [(C[i, j] - Cr[i, j]) / np.sqrt(Cr[i, i] * Cr[j, j]) for i in range(d) for j in range(i + 1, d)]
    Dw = vweights(n)
    lam_r = np.linalg.eigvalsh(sym(Dw @ Cr))[0]
    lam_c = np.linalg.eigvalsh(sym(Dw @ C))[0]
    e.append(lam_c / lam_r - 1 if abs(lam_r) > 1e-12 * np.abs(Cr).max() else 0.0)
    return np.array(e)


def _calib_fit_cell(topo, ratios, refs):
    """Constants (ka, kb, pois, pa, pb) of ONE topology: multi-start bounded least squares on the signed entry errors over all `ratios`, then a Nelder-Mead"""
    from scipy.optimize import least_squares, minimize
    lat = make_lattice_3d(topo) if topo in TOPO_3D else make_lattice_2d(topo)
    d = lat.dim
    data = {r: (_calib_lattice_size(topo, r)[1], refs[(topo, r)]) for r in ratios}
    p_ref = np.array(_CALIB_START[d])

    def errs(p):
        return {r: 100 * _entry_errors(homogenize(lat, sz, corr=tuple(p))[0], Cr) for r, (sz, Cr) in data.items()}

    def resid(p):
        return np.concatenate([_signed_entry_errors(homogenize(lat, sz, corr=tuple(p))[0], Cr) for sz, Cr in data.values()])

    best = None
    for p0 in (p_ref, np.array([0.3, 0.8, 0.2, 1.5, 1.0]), p_ref * 0.5, p_ref * 1.5):
        p0 = np.clip(p0, _CALIB_LO + 1e-3, _CALIB_HI - 1e-3)
        sol = least_squares(resid, p0, bounds=(_CALIB_LO, _CALIB_HI), x_scale=np.maximum(p0, 0.1), max_nfev=300)
        if best is None or sol.cost < best.cost:
            best = sol
    p = best.x
    worst = lambda q: max(errs(np.clip(q, _CALIB_LO, _CALIB_HI)).values())
    pm = np.clip(minimize(worst, p, method="Nelder-Mead", options=dict(maxiter=400, xatol=1e-4, fatol=1e-4)).x, _CALIB_LO, _CALIB_HI)
    if worst(pm) < worst(p):
        p = pm
    return p, errs(p_ref), errs(p)


def _calib_fit_job(a):
    """Picklable wrapper of _calib_fit_cell for ProcessPoolExecutor."""
    return _calib_fit_cell(*a)


def default_workers():
    """Default number of worker processes: all logical cores but one (at most 8)."""
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def calibrate_corr_params(out, quick=False, workers=None):
    """Calibrate CORR_PARAMS separately for every cell of TOPO_2D and TOPO_3D against the periodic continuum FE (hex8/quad4 + Wilson modes) at the strut"""
    from concurrent.futures import ProcessPoolExecutor
    ratios = CALIB_RATIOS_QUICK if quick else CALIB_RATIOS
    cells = TOPO_2D + TOPO_3D
    cache = os.path.join(out, "calibration_fe_refs.npz")
    Z = dict(np.load(cache)) if os.path.exists(cache) else {}
    key = lambda t, r: f"{t}|{r}"
    todo = [(t, r) for t in cells for r in ratios if key(t, r) not in Z]
    todo.sort(key=lambda j: -(j[0] in TOPO_3D) * j[1])
    workers = workers or default_workers()
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for topo, ratio, Cr in ex.map(_calib_reference, todo):
                Z[key(topo, ratio)] = Cr
                print(f"  calibration FE reference {topo:14s} L/d {ratio:4.1f}", flush=True)
        np.savez(cache, **Z)
    refs = {(t, r): Z[key(t, r)] for t in cells for r in ratios}
    result = {}
    fit_args = [(t, ratios, {k: v for k, v in refs.items() if k[0] == t}) for t in cells]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(cells))) as ex:
            fits = list(ex.map(_calib_fit_job, fit_args))
    else:
        fits = [_calib_fit_job(a) for a in fit_args]
    for topo, (p, e_old, e_new) in zip(cells, fits):
        result[topo] = tuple(round(float(x), 4) for x in p)
        print(f"  {topo:14s} {result[topo]}  max err {max(e_old.values()):5.2f}% (old shared set) -> {max(e_new.values()):5.2f}% (fitted)", flush=True)
    return result


FIG_CONES = "Fig01_Loewner_cones"
FIG_SCATTER = {2: "Fig22_target_modulus_scatter_2D", 3: "Fig23_target_modulus_scatter_3D"}
FIG_DENSITY = "Fig24_optimal_density"
FIG_LWCW = "Fig25_LW_CW_optima_vs_target"
FIG_MESH = "Fig04_mesh_convergence"
CELL_FIGNO = {"2D_grid": 5, "2D_xbrace": 6, "2D_triangular": 7, "2D_honeycomb": 8, "2D_reentrant": 9,
              "3D_SC": (10, 11), "3D_BCC": (12, 13), "3D_FCC": (14, 15), "3D_octet": (16, 17), "3D_diamond": (18, 19), "3D_reentrant": (20, 21)}
assert set(CELL_FIGNO) == set(TOPO_2D) | set(TOPO_3D), "CELL_FIGNO must list exactly the cells of TOPO_2D + TOPO_3D (add the paper figure numbers of a new cell here)"


def cell_figure(topo, panel):
    """(file stem, paper label) of one panel of a cell figure."""
    n = CELL_FIGNO[topo]
    if isinstance(n, int):
        return f"Fig{n:02d}_{topo}_{panel}", f"Figure {n}({panel})"
    geo, res = n
    if panel == "a":
        return f"Fig{geo:02d}_{topo}_geometry", f"Figure {geo}"
    if panel == "b":
        return f"Fig{res:02d}_{topo}_a", f"Figure {res}(a)"
    if panel == "c":
        return f"Fig{res:02d}_{topo}_b", f"Figure {res}(b)"
    raise KeyError(panel)


def safe_regime_scan(pools_by_dim, n=14):
    """For isotropic targets of increasing stiffness ratio E_T/E_s (the target's nu, TARGET_SPEC['nu']): density of the component-wise optimum of every cell"""
    out = {}
    ratios = np.unique(np.r_[np.geomspace(0.003, 0.12, n), target_ratio()])
    for dim, pools in pools_by_dim.items():
        recs = []
        for E in ratios:
            T = iso_stiffness(E, TARGET_SPEC["nu"], dim)
            for topo, p in pools.items():
                cw = refined_optimum(p, T, 'cw')
                if cw is not None:
                    recs.append(dict(topo=topo, ratio=float(E), rho=cw['rho'], safe=bool(pass_loewner(cw['C'], T)),
                                     actual=bool(np.isclose(E, target_ratio()))))
        out[dim] = recs
    return out


def _ax_style(ax):
    """Common axes style: hide the top and right spines."""
    ax.spines[['top', 'right']].set_visible(False)


def _apply_figure_fonts():
    """Use FIG_FONT for all text of every figure, including mathtext (call after importing pyplot, before drawing)."""
    import matplotlib
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FIG_FONT, "DejaVu Sans"],
                                "mathtext.fontset": "custom", "mathtext.rm": FIG_FONT, "mathtext.it": f"{FIG_FONT}:italic",
                                "mathtext.bf": f"{FIG_FONT}:bold", "mathtext.sf": FIG_FONT, "mathtext.cal": FIG_FONT})


def _savefig(fig, path, dpi=130):
    """Save the figure tightly cropped and close it (frees memory in the long figure loops)."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def _cap_title(name):
    """Capitalise the first letter of a topology name used as a title ('2D_grid' -> '2D_Grid')."""
    import re
    return re.sub(r"^(\d+D_)?([a-z])", lambda m: (m.group(1) or "") + m.group(2).upper(), name)


def fig_01_loewner_cones(path):
    """Figure 1 (Fig01_Loewner_cones.svg) -- geometry of the Loewner-feasible set for an isotropic target with lambda_T = mu_T = 1 (units of mu_T;"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    _apply_figure_fonts()
    lam, mu = 1.0, 1.0
    T11, T12, T44 = lam + 2 * mu, lam, 2 * mu
    col = {"hyd": "#4c78a8", "dev": "#59a14f", "shr": "#f28e2b", "cw": "#d1495b"}
    fig = plt.figure(figsize=(12.4, 5.4))

    ax = fig.add_subplot(1, 2, 1)
    x0, x1, y0, y1 = T11 - 1, T11 + 3, T12 - 3, T12 + 3
    xs = np.linspace(T11, x1, 100)
    s_n = T11 + T12
    ax.fill_between(xs, s_n - xs, xs - 2 * mu, color="#4c78a8", alpha=0.25)
    ax.fill_between(xs, T12, y1, color=col["cw"], alpha=0.12, hatch="///")
    ax.plot(xs, s_n - xs, color=col["hyd"], lw=1.6)
    ax.plot(xs, xs - 2 * mu, color=col["dev"], lw=1.6, ls="--")
    ax.scatter([T11], [T12], c="k", s=40, zorder=5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks(range(int(x0), int(x1) + 1))
    ax.set_yticks(range(int(y0), int(y1) + 1))
    ax.set_xlabel(r"$C_{11}$ (= $C_{22}$)")
    ax.set_ylabel(r"$C_{12}$")
    ax.set_title("2D plane stress", fontsize=10)
    ax.legend(handles=[Line2D([], [], color=col["hyd"], lw=1.6, label=r"$C_{11}+C_{12}\geq 2\lambda_T+2\mu_T$"),
                       Line2D([], [], color=col["dev"], lw=1.6, ls="--", label=r"$C_{11}-C_{12}\geq 2\mu_T$"),
                       Patch(facecolor=col["hyd"], alpha=0.3, label=r"LW-feasible cone"),
                       Patch(facecolor=col["cw"], alpha=0.2, hatch="///", label=r"CW-feasible region ($C_{ij}\geq T_{ij}$)"),
                       Line2D([], [], marker="o", ls="", color="k", markersize=5, label=r"target $T$")],
              fontsize=7.5, loc="lower left")
    _ax_style(ax)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    from scipy.spatial import HalfspaceIntersection
    win = [(2.0, 6.0), (-2.0, 5.0), (1.0, 5.0)]
    box_hs = [(-1, 0, 0, -win[0][0]), (1, 0, 0, win[0][1]), (0, -1, 0, -win[1][0]), (0, 1, 0, win[1][1]), (0, 0, -1, -win[2][0]), (0, 0, 1, win[2][1])]
    cone_hs = {"hyd": (-1, -2, 0, -(T11 + 2 * T12)),
               "dev": (-1, 1, 0, -2 * mu),
               "shr": (0, 0, -1, -T44)}
    cw_hs = {"cw1": (-1, 0, 0, -T11), "cw2": (0, -1, 0, -T12), "cw3": (0, 0, -1, -T44)}

    def polytope_faces(constraints, interior):
        """Faces of {x: a.x <= b for all constraints} intersected with the window; returns {name: ordered vertex array} for the named (non-window) constraints."""
        allh = list(constraints.values()) + box_hs
        hs = HalfspaceIntersection(np.array([[a1, a2, a3, -bb] for (a1, a2, a3, bb) in allh]), np.asarray(interior, float))
        V = hs.intersections
        out = {}
        for name, (a1, a2, a3, bb) in constraints.items():
            n = np.array([a1, a2, a3], float)
            Vf = V[np.abs(V @ n - bb) < 1e-8]
            Vf = np.unique(np.round(Vf, 9), axis=0)
            if len(Vf) < 3:
                continue
            c = Vf.mean(0)
            e1 = Vf[0] - c
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            order = np.argsort(np.arctan2((Vf - c) @ e2, (Vf - c) @ e1))
            out[name] = Vf[order]
        return out

    for k, f in polytope_faces(cone_hs, (4.5, 1.2, 3.0)).items():
        ax.add_collection3d(Poly3DCollection([f], facecolor=col[k], alpha=0.35, edgecolor="k", linewidths=0.7))
    for k, f in polytope_faces(cw_hs, (4.0, 2.0, 3.0)).items():
        ax.add_collection3d(Poly3DCollection([f], facecolor=col["cw"], alpha=0.12, edgecolor=col["cw"], linewidths=0.7))
    ax.scatter([T11], [T12], [T44], c="k", marker="o", s=40, depthshade=False)
    ax.set_xlabel(r"$C_{11}$")
    ax.set_ylabel(r"$C_{12}$")
    ax.set_zlabel(r"$C_{44}$", rotation=270)
    ax.set_xlim(*win[0])
    ax.set_ylim(*win[1])
    ax.set_zlim(*win[2])
    ax.set_xticks(range(2, 7))
    ax.set_yticks(range(-2, 6))
    ax.set_zticks(range(1, 6))
    ax.view_init(28, -55)
    ax.zaxis.set_label_position('lower')
    ax.zaxis.set_ticks_position('lower')
    ax.set_title("3D isotropic continuum", fontsize=10)
    ax.legend(handles=[Patch(facecolor=col["hyd"], alpha=0.5, edgecolor="k", label=r"$C_{11}+2C_{12}\geq 3\lambda_T+2\mu_T$"),
                       Patch(facecolor=col["dev"], alpha=0.5, edgecolor="k", label=r"$C_{11}-C_{12}\geq 2\mu_T$"),
                       Patch(facecolor=col["shr"], alpha=0.5, edgecolor="k", label=r"$2C_{44}\geq 4\mu_T$"),
                       Patch(facecolor=col["cw"], alpha=0.25, edgecolor=col["cw"], label=r"CW-feasible region ($C_{ij}\geq T_{ij}$)"),
                       Line2D([], [], marker="o", ls="", color="k", markersize=5, label=r"target $T$")],
              fontsize=7.5, loc="upper left")
    _savefig(fig, path)


def _clip_segment(p, q, lo, hi):
    """Liang-Barsky clipping of the segment p-q to the box [lo, hi]; None if it lies outside."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    t0, t1 = 0.0, 1.0
    for k in range(len(p)):
        if abs(d[k]) < 1e-14:
            if p[k] < lo[k] - 1e-12 or p[k] > hi[k] + 1e-12:
                return None
        else:
            ta, tb = (lo[k] - p[k]) / d[k], (hi[k] - p[k]) / d[k]
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return None
    return np.array([p + t0 * d, p + t1 * d])


def _display_offset(lat):
    """Shift that centres the bounding box of the cell's struts on the cell box [0, cell] (purely a drawing choice: struts are node-to-node vectors, so a"""
    pts = np.array([q for (i, j, nv, k) in lat.struts for q in (lat.nodes[i], lat.nodes[i] + lat.strut_vec(i, j, nv))])
    return (pts.min(0) + pts.max(0)) / 2 - lat.cell / 2


def _tile_segments(lat, n=3):
    """Struts of an n x n (x n) block of cells, clipped to the block; those of the central cell are returned separately."""
    off = _display_offset(lat)
    base = [(lat.nodes[i] - off, lat.nodes[i] - off + lat.strut_vec(i, j, nv)) for (i, j, nv, k) in lat.struts]
    lo, hi = np.zeros(lat.dim), n * lat.cell
    black, red = [], []
    centre = (n // 2,) * lat.dim
    for sh_i in product(range(n), repeat=lat.dim):
        sh = np.array(sh_i) * lat.cell
        for a, b in base:
            seg = _clip_segment(a + sh, b + sh, lo, hi)
            if seg is not None:
                (red if sh_i == centre else black).append(seg)
    return black, red, lo, hi


def _box_edges(lo, hi):
    V = np.array(list(product(*zip(lo, hi))))
    return [np.array([V[a], V[b]]) for a in range(len(V)) for b in range(a + 1, len(V)) if np.sum(V[a] != V[b]) == 1]


def _deformed_cell(lat, size, v, amp_frac=0.12):
    """Undeformed and deformed struts of ONE cell for the macro strain eps_V = v (weakest mode): displacement = macro E (x - centre) + periodic fluctuation"""
    d = lat.dim
    v = np.asarray(v, float) / np.linalg.norm(v)
    u = homogenize(lat, size, fluct=v, corr=corr_of(lat))[2]
    E = voigt_to_tensor(v, d)
    cen = lat.cell / 2
    nodes = lat.nodes - _display_offset(lat)
    mean = np.mean([E @ (x - cen) + uu for x, uu in zip(nodes, u)], axis=0)
    seg0, disp = [], []
    for (i, j, nv, k) in lat.struts:
        xi, xj = nodes[i], nodes[j] + np.asarray(nv) * lat.cell
        seg0.append(np.array([xi, xj]))
        disp.append(np.array([E @ (xi - cen) + u[i] - mean, E @ (xj - cen) + u[j] - mean]))
    scale = amp_frac * lat.cell.min() / max(np.abs(np.array(disp)).max(), 1e-300)
    seg1 = [a + scale * dd for a, dd in zip(seg0, disp)]
    box0 = _box_edges(np.zeros(d), lat.cell)
    box1 = [b + scale * np.array([E @ (x - cen) - mean for x in b]) for b in box0]
    return seg0, seg1, box0, box1, scale


_SCATTER_COLORS = {"2D_grid": "#E69F00", "2D_xbrace": "#0072B2", "2D_triangular": "#D55E00", "2D_honeycomb": "#009E73", "2D_reentrant": "#7B3294",
              "3D_SC": "#0072B2", "3D_BCC": "#D55E00", "3D_FCC": "#009E73", "3D_octet": "#CC79A7",
              "3D_diamond": "#7B3294", "3D_reentrant": "#444444"}


def fig_22_23_target_scatter(safe, out_dir):
    """Figure 22 (Fig22_target_modulus_scatter_2D.svg) and Figure 23 (Fig23_target_modulus_scatter_3D.svg) -- safe-regime maps (one figure per dimension):"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _apply_figure_fonts()
    for dim in (2, 3):
        recs = safe[dim]
        topos = list(dict.fromkeys(r['topo'] for r in recs))
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        for topo in topos:
            col = _SCATTER_COLORS[topo]
            for flag in (True, False):
                pts = [r for r in recs if r['topo'] == topo and r['safe'] == flag]
                if pts:
                    ax.scatter([r['rho'] for r in pts], [r['ratio'] for r in pts], s=38, marker='o',
                               facecolors=col if flag else 'none', edgecolors=col, linewidths=1.3, alpha=0.95, zorder=3)
        ax.axhline(target_ratio(), color='k', lw=0.8, ls='--')
        ax.set_yscale('log')
        ax.set_xlabel(VVS_LABEL)
        ax.set_ylabel(r"$E_T/E_s$")
        ax.set_title(f"{dim}D cells", fontsize=10)
        handles = [Line2D([], [], marker='o', ls='', mfc=_SCATTER_COLORS[t], mec=_SCATTER_COLORS[t], markersize=6, label=t) for t in topos]
        handles += [Line2D([], [], marker='o', ls='', mfc='0.35', mec='0.35', markersize=6, label="Loewner feasible"),
                    Line2D([], [], marker='o', ls='', mfc='none', mec='0.35', markersize=6, label="Not Loewner feasible")]
        handles.append(Line2D([], [], color='k', lw=0.8, ls='--', label=rf"Target, $E_T/E_s$ = {target_ratio():.2f}"))
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
        _ax_style(ax)
        fig.tight_layout()
        name = FIG_SCATTER[dim] + ".svg"
        _savefig(fig, os.path.join(out_dir, name), dpi=200)


def fig_25_lw_cw_optima(pools_by_dim, safe, out_dir):
    """Figure 25 (Fig25_LW_CW_optima_vs_target.svg/.csv): V_s/V_T of the Loewner optimum (black line) and of the component-wise optimum (markers) of"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _apply_figure_fonts()
    red = "#C0392B"
    panels = {}
    for dim in (2, 3):
        recs = safe[dim]
        for topo in dict.fromkeys(r['topo'] for r in recs):
            cw = sorted([r for r in recs if r['topo'] == topo], key=lambda r: r['ratio'])
            lw = []
            for r in cw:
                T = iso_stiffness(r['ratio'], TARGET_SPEC["nu"], dim)
                best = refined_optimum(pools_by_dim[dim][topo], T, 'lo')
                lw.append(None if best is None else best)
            panels[(dim, topo)] = (cw, lw)
    fig, axs = plt.subplots(2, 6, figsize=(10.4, 4.9), sharey=True)
    xmax = max(0.68, 1.05 * max(r['rho'] for dim in (2, 3) for r in safe[dim]))
    for (dim, topo), (cw, lw) in panels.items():
        k = [t for t in panels if t[0] == dim].index((dim, topo))
        ax = axs[dim - 2, k]
        y = [r['ratio'] for r in cw]
        ok = [i for i, b in enumerate(lw) if b is not None]
        ax.plot([lw[i]['rho'] for i in ok], [y[i] for i in ok], color='k', lw=1.8, zorder=2)
        for flag in (True, False):
            pts = [r for r in cw if r['safe'] == flag]
            ax.scatter([r['rho'] for r in pts], [r['ratio'] for r in pts], s=22, marker='o', facecolors=red if flag else 'white', edgecolors=red,
                       linewidths=1.2, zorder=3)
        ax.axhline(target_ratio(), color='0.5', lw=0.9, ls='--', zorder=1)
        ax.set_yscale('log')
        ax.set_xlim(0, xmax)
        ax.set_xticks([0.0, 0.3, 0.6])
        ax.set_title(topo.replace("_", " "), fontsize=9, loc='left')
        ax.tick_params(labelsize=8)
        if dim == 3:
            ax.set_xlabel(VVS_LABEL, fontsize=9)
        _ax_style(ax)
    ylo, yhi = min(r['ratio'] for r in safe[2]) / 1.15, max(r['ratio'] for r in safe[2]) * 1.2
    for ax in axs[:, 0]:
        ax.set_ylim(ylo, yhi)
        ax.set_yticks([0.003, 0.01, 0.03, 0.1])
        ax.set_yticklabels(["0.003", "0.01", "0.03", "0.1"])
        ax.set_ylabel(r"$E_T/E_s$")
    axs[0, 5].axis('off')
    axs[0, 5].legend(handles=[Line2D([], [], color='k', lw=1.8, label="Loewner optimum"),
                              Line2D([], [], marker='o', ls='', mfc=red, mec=red, markersize=5, label="CW optimum, Loewner-feasible"),
                              Line2D([], [], marker='o', ls='', mfc='white', mec=red, markersize=5, label="CW optimum, not Loewner-feasible"),
                              Line2D([], [], color='0.5', lw=0.9, ls='--', label=rf"Reference target $E_T/E_s = {target_ratio():.2f}$")],
                     loc='center left', fontsize=8, frameon=False)
    fig.tight_layout()
    _save_pair(fig, out_dir, FIG_LWCW)


def componentwise_margin_field(C, T):
    """Continuous scalar whose zero level is the component-wise boundary: min over entries of (C_ij - T_ij)/|T|max, ignoring structurally zero entries (zero"""
    Tm = np.abs(T).max()
    dlt = (C - T) / Tm
    struct0 = (np.abs(T) < 1e-12 * Tm) & (np.abs(C).max(axis=(0, 1)) < 1e-8 * Tm)
    return np.where(struct0, np.inf, dlt).min(axis=(-1, -2))


def _cell_geometry(lat, centred=True):
    """Struts and nodes of ONE unit cell, drawn periodic-complete: every strut (and every periodic image of it, shifts -2..2) clipped to the cell box [0,"""
    d = lat.dim
    off = _display_offset(lat) if centred else np.zeros(d)
    base = [(lat.nodes[i] - off, lat.nodes[i] - off + lat.strut_vec(i, j, nv)) for (i, j, nv, k) in lat.struts]
    lo, hi = np.zeros(d), np.asarray(lat.cell, float)
    segs, seen = [], set()
    for sh_i in product(range(-2, 3), repeat=d):
        sh = np.array(sh_i) * lat.cell
        for a, b in base:
            seg = _clip_segment(a + sh, b + sh, lo, hi)
            if seg is None or np.linalg.norm(seg[1] - seg[0]) < 1e-9:
                continue
            key = tuple(sorted((tuple(np.round(seg[0], 6)), tuple(np.round(seg[1], 6)))))
            if key not in seen:
                seen.add(key)
                segs.append(seg)
    nodes, nseen = [], set()
    for sh_i in product(range(-1, 2), repeat=d):
        for x in lat.nodes - off:
            p = x + np.array(sh_i) * lat.cell
            if np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9):
                key = tuple(np.round(p, 6))
                if key not in nseen:
                    nseen.add(key)
                    nodes.append(p)
    return segs, np.array(nodes), lo, hi


def _finish_axes(ax, pts, dim, pad_frac=0.06):
    """Limits / aspect / no axes for a drawing whose extent is given by the point sets `pts`."""
    pts = np.vstack(pts)
    lo, hi = pts.min(0), pts.max(0)
    pad = pad_frac * (hi - lo)
    for a, (l, h) in zip("xyz"[:dim], zip(lo - pad, hi + pad)):
        getattr(ax, f"set_{a}lim")(l, h)
    if dim == 2:
        ax.set_aspect('equal')
    else:
        ax.set_box_aspect(hi - lo + 2 * pad)
    ax.set_axis_off()


def _annotation_2d(lat, topo, par, path):
    """Lattice annotation, 2D: (left) ONE unit cell with nodes, struts and dimensions; (right) 3 x 3 tiling with the central cell in black."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    segs, nodes, lo, hi = _cell_geometry(lat)
    neigh, centre, tlo, thi = _tile_segments(lat)
    Lx, Ly = lat.cell
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 5.0))
    a1.add_collection(LineCollection(_box_edges(lo, hi), colors='0.45', linewidths=1.0, linestyles='--'))
    a1.add_collection(LineCollection(segs, colors='k', linewidths=2.6))
    a1.scatter(nodes[:, 0], nodes[:, 1], s=28, c='k', zorder=4)
    pad = 0.10 * max(Lx, Ly)
    a1.annotate('', xy=(0, -pad), xytext=(Lx, -pad), arrowprops=dict(arrowstyle='<->', lw=0.9))
    a1.text(Lx / 2, -1.35 * pad, f"$L_x$ = {Lx:.3g}", ha='center', va='top', fontsize=9)
    a1.annotate('', xy=(-pad, 0), xytext=(-pad, Ly), arrowprops=dict(arrowstyle='<->', lw=0.9))
    a1.text(-1.35 * pad, Ly / 2, f"$L_y$ = {Ly:.3g}", ha='right', va='center', rotation=90, fontsize=9)
    a1.set_xlim(-3.2 * pad, Lx + pad)
    a1.set_ylim(-3.2 * pad, Ly + pad)
    a1.set_aspect('equal')
    a1.set_axis_off()
    a1.legend(handles=[Line2D([], [], color='k', lw=2.6, label="Strut"),
                       Line2D([], [], color='k', marker='o', ls='', markersize=5, label="Node (with periodic images on the boundary)"),
                       Line2D([], [], color='0.45', lw=1, ls='--', label="Unit-cell boundary")],
              loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=8, frameon=False)
    a2.add_collection(LineCollection(neigh, colors='0.65', linewidths=1.0))
    a2.add_collection(LineCollection(centre, colors='k', linewidths=2.4))
    a2.add_collection(LineCollection(_box_edges(lat.cell, 2 * lat.cell), colors='k', linewidths=1.0, linestyles='--'))
    a2.add_collection(LineCollection(_box_edges(tlo, thi), colors='0.65', linewidths=0.8))
    _finish_axes(a2, [tlo, thi], 2)
    a2.set_title("3 x 3 tiling", fontsize=11)
    a2.legend(handles=[Line2D([], [], color='k', lw=2.4, label="Central unit cell"),
                       Line2D([], [], color='0.65', lw=1, label="Neighbouring cells")],
              loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=8, frameon=False)
    fig.tight_layout()
    _savefig(fig, path, dpi=200)


def _annotation_3d(lat, topo, par, path):
    """Lattice annotation, 3D: isometric view of ONE unit cell + its XY, YZ and ZX orthographic projections."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    segs, nodes, lo, hi = _cell_geometry(lat, centred=(lat.name == '3D_reentrant'))
    cell = np.asarray(lat.cell, float)
    fig = plt.figure(figsize=(9.4, 9.2))
    axi = fig.add_subplot(2, 2, 1, projection='3d')
    axi.add_collection3d(Line3DCollection(_box_edges(lo, hi), colors='0.55', linewidths=0.9, linestyles='--'))
    axi.add_collection3d(Line3DCollection(segs, colors='k', linewidths=1.8))
    axi.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=16, c='k', depthshade=False)
    m = float(cell.min())
    org, ln = np.full(3, -0.40 * m), np.full(3, 0.30 * m)
    for e, (lab, col) in enumerate((("x", "#c0392b"), ("y", "#27ae60"), ("z", "#2471a3"))):
        tip = org.copy()
        tip[e] += ln[e]
        axi.plot([org[0], tip[0]], [org[1], tip[1]], [org[2], tip[2]], color=col, lw=1.8)
        axi.text(*(tip + 0.05 * m * (np.arange(3) == e)), lab, color=col, fontsize=10, ha='center', va='center')
    axi.set_xlim(org[0], cell[0])
    axi.set_ylim(org[1], cell[1])
    axi.set_zlim(org[2], cell[2])
    axi.set_box_aspect(cell - org)
    axi.view_init(elev=22, azim=-58)
    axi.set_axis_off()
    views = [((0, 1), 2, "x", "y", "XY view (from +z)"), ((1, 2), 0, "y", "z", "YZ view (from +x)"),
             ((2, 0), 1, "z", "x", "ZX view (from +y)")]
    for pos, ((ia, ib), idep, la, lb, ttl) in zip((2, 3, 4), views):
        ax = fig.add_subplot(2, 2, pos)
        for sg in segs:
            ax.plot(sg[:, ia], sg[:, ib], color='k', lw=2.0, solid_capstyle='round')
        ax.scatter(nodes[:, ia], nodes[:, ib], c='k', s=18, zorder=4)
        ax.plot([0, cell[ia], cell[ia], 0, 0], [0, 0, cell[ib], cell[ib], 0], color='0.55', lw=0.9, ls='--')
        ax.set_aspect('equal')
        ax.set_xlim(-0.08 * cell[ia], 1.08 * cell[ia])
        ax.set_ylim(-0.08 * cell[ib], 1.08 * cell[ib])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(la)
        ax.set_ylabel(lb, rotation=0, labelpad=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(ttl, fontsize=11)
    fig.tight_layout()
    _savefig(fig, path, dpi=200)


def select_design(pool, T):
    """Design marked in Figures 5-21: at alpha = 1 (the display grids contain alpha = 1 exactly) the MINIMUM V_s/V_T at which the Loewner stiffness margin m"""
    b = int(np.argmin(np.abs(np.asarray(pool.axes[1]) - 1.0)))
    rhos = np.asarray(pool.axes[0], float)
    m = margin(pool.C[:, b], T)
    ok = np.flatnonzero(m >= -tol_of(T))
    if len(ok) == 0:
        a0 = a1 = int(np.argmax(m))
        w, rule = 0.0, "FALLBACK (margin < 0 on the whole alpha = 1 column): row of largest margin, nearest to zero from below"
    elif ok[0] == 0:
        a0 = a1 = 0
        w, rule = 0.0, "FALLBACK (margin >= 0 already at the lowest V_s/V_T row): lowest row, margin nearest to zero from above"
    else:
        a1 = int(ok[0])
        a0 = a1 - 1
        w = float(np.clip(-m[a0] / (m[a1] - m[a0]), 0.0, 1.0)) if m[a1] > m[a0] else 1.0
        rule = "alpha = 1; minimum V_s/V_T at which the Loewner margin is zero (linear interpolation between the last failing and first passing row)"
    mix = lambda arr: (1 - w) * arr[a0, b] + w * arr[a1, b]
    C = sym(mix(pool.C))
    near = a1 if w > 0.5 else a0
    vox = [x for x in (pool.vox_actual[a0, b], pool.vox_actual[a1, b]) if x > 0]
    m_sel = float((1 - w) * m[a0] + w * m[a1])
    lo, cw = bool(m_sel >= -tol_of(T)), bool(pass_componentwise(C, T))
    return dict(rho=float(rhos[a0] + w * (rhos[a1] - rhos[a0])), alpha=float(pool.axes[1][b]), C=C, m=m_sel, m_of_interpolated_C=float(margin(C, T)),
                vec=canonical_mode(C, T), rad=float(mix(pool.rad)), slender=float(mix(pool.slender)), model=int(pool.model[near, b]),
                conv=bool(pool.conv[a0, b] and pool.conv[a1, b]), vox=float(min(vox)) if vox else 0.0,
                cls=int(classify(np.array(lo), np.array(cw))), a0=a0, a1=a1, w=w, b=b, rule=rule)


_SEL_SQUARE_SIDE_PT = 8.0
_SEL_CIRCLE_DIAM_PT = 2 * 0.70 * _SEL_SQUARE_SIDE_PT
_SEL_CIRCLE_S = (2 * 0.70 * np.sqrt(90.0)) ** 2
VVS_LABEL = r"$V_s/V_T$"


def fig_cell(topo, dim, pool, T, key, out_dir):
    """Figures 5-21 (file names from cell_figure(); ONE svg per subplot)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    _apply_figure_fonts()
    sel = select_design(pool, T)
    par, v = (sel['rho'], sel['alpha']), sel['vec']
    v = v * np.sign(v[np.argmax(np.abs(v))])
    rad_k = sel['rad']
    if dim == 2:
        lat = make_lattice_2d(topo, sel['alpha'])
        size = size_2d(rad_k, rad_k)
    else:
        lat = make_lattice_3d(topo, sel['alpha'])
        size = size_3d(rad_k)
    Coll = LineCollection if dim == 2 else Line3DCollection
    path = {p: os.path.join(out_dir, cell_figure(topo, p)[0] + ".svg") for p in "abc"}

    (_annotation_2d if dim == 2 else _annotation_3d)(lat, topo, par, path['a'])

    fig = plt.figure(figsize=(5.6, 6.0))
    axb = fig.add_subplot(1, 1, 1, projection=None if dim == 2 else '3d')
    s0, s1, b0, b1, sc = _deformed_cell(lat, size, v)
    axb.add_collection(Coll(s0, colors='k', linewidths=1.5))
    axb.add_collection(Coll(s1, colors='r', linewidths=1.5))
    axb.add_collection(Coll(b0, colors='k', linewidths=1.0, linestyles='--'))
    axb.add_collection(Coll(b1, colors='r', linewidths=1.0, linestyles='--'))
    _finish_axes(axb, [x for sg in s0 + s1 + b0 + b1 for x in sg], dim)
    axb.legend(handles=[Line2D([], [], color='k', lw=1.5, label="Undeformed lattice"), Line2D([], [], color='r', lw=1.5, label="Deformed lattice"),
                        Line2D([], [], color='k', lw=1, ls='--', label="Undeformed cell boundary"),
                        Line2D([], [], color='r', lw=1, ls='--', label="Deformed cell boundary")],
               loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    _savefig(fig, path['b'], dpi=200)

    def stiffness_figure(pl, sl, xlabel, title, fpath):
        """Draw the zero contours of the Loewner margin, the component-wise margin field and the slenderness field (= S_CORR_MIN) over (alpha, V_s/V_T) of pool"""
        fig, ax = plt.subplots(figsize=(5.2, 5.6))
        X, Y = np.meshgrid(pl.axes[1], pl.axes[0])
        m = np.maximum.accumulate(margin(pl.C, T), axis=0)
        cwf = np.maximum.accumulate(componentwise_margin_field(pl.C, T), axis=0)

        def despike(f, level):
            """One-column spikes: an interior alpha column (not alpha = 1) whose lowest crossing lies >= 2 density rows above that of both neighbouring columns is"""
            f = f.copy()
            for bb in range(1, f.shape[1] - 1):
                if np.isclose(pl.axes[1][bb], 1.0):
                    continue
                cr = [int(np.argmax(f[:, c] >= level)) if (f[:, c] >= level).any() else f.shape[0] for c in (bb - 1, bb, bb + 1)]
                if cr[1] >= max(cr[0], cr[2]) + 2:
                    f[:, bb] = 0.5 * (f[:, bb - 1] + f[:, bb + 1])
            return f
        m, cwf = despike(m, 0.0), despike(cwf, -1e-9)
        fields = ((m, 0.0, 'k', "Loewner boundary", 'loewner', (m >= 0).any() and (m < 0).any()),
                  (cwf, -1e-9, 'r', "Component-wise boundary", 'cw', None),
                  (pl.slender, S_CORR_MIN, 'b', f"Beam validity limit (L/d = {S_CORR_MIN:g})", 'beam', None))
        handles, flags = [], {}
        for f, level, col, lab, name, ok in fields:
            ok = (f.min() < level < f.max()) if ok is None else ok
            flags[name] = bool(ok)
            if ok:
                ax.contour(X, Y, f, [level], colors=col, linewidths=2.0, linestyles='solid')
                handles.append(Line2D([], [], color=col, lw=2, label=lab))
        ax.scatter([sl['alpha']], [sl['rho']], marker='o', s=_SEL_CIRCLE_S, c='k', zorder=6, clip_on=False)
        handles.append(Line2D([], [], marker='o', ls='', color='k', markersize=_SEL_CIRCLE_DIAM_PT, label="Selected design"))
        ax.set_xlim(pl.axes[1][0], pl.axes[1][-1])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(VVS_LABEL)
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
        fig.tight_layout()
        _savefig(fig, fpath, dpi=200)
        return flags

    stiffness_figure(pool, sel, f"a ({'y/x' if dim == 2 else 'z/x'})", _cap_title(topo), path['c'])


_VERIF_STYLE = {"3D_SC": ("#0072B2", "o"), "3D_BCC": ("#D55E00", "s"), "3D_octet": ("#009E73", "^"),
                "2D_grid": ("#E69F00", "D"), "2D_triangular": ("#56B4E9", "P"), "2D_honeycomb": ("#444444", "X"),
                "3D_FCC": ("#CC79A7", "v"), "3D_diamond": ("#5E3C99", "<"), "3D_reentrant": ("#8C510A", ">"),
                "2D_xbrace": ("#B8A800", "*"), "2D_reentrant": ("#882255", "h")}
_VERIF_FALLBACK = [("#000000", "p"), ("#777777", "d"), ("#AA4499", "H"), ("#44AA99", "8")]
VERIF_RATIOS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)


def _vstyle(cell, _cache={}):
    if cell in _VERIF_STYLE:
        return _VERIF_STYLE[cell]
    if cell not in _cache:
        _cache[cell] = _VERIF_FALLBACK[len(_cache) % len(_VERIF_FALLBACK)]
    return _cache[cell]


def verification_sweep(quick=False, ratios=None, cells=None, cache_dir=None, recompute=False):
    """Plain / modified beam vs periodic continuum FE for a slenderness sweep, for EVERY cell (default TOPO_2D + TOPO_3D) and every L/d in `ratios` (default"""
    cells = tuple(cells) if cells else tuple(TOPO_2D) + tuple(TOPO_3D)
    ratios = tuple(ratios) if ratios else ((3, 5, 8) if quick else VERIF_RATIOS)
    Zc, Zv, pv = {}, {}, None
    if cache_dir and not recompute:
        pc = os.path.join(cache_dir, "calibration_fe_refs.npz")
        if os.path.exists(pc):
            Zc = dict(np.load(pc))
    if cache_dir:
        pv = os.path.join(cache_dir, "verification_fe_refs.npz")
        if os.path.exists(pv) and not recompute:
            Zv = dict(np.load(pv))
    rows = []
    for topo in cells:
        for ratio in ratios:
            lat, sz = _calib_lattice_size(topo, ratio)
            ns = voxel_grid(lat, sz, 8, nmin=24, nmax=(96 if lat.dim == 3 else 256))
            key = f"{topo}|{ratio}"
            Cr, src, conv = None, "computed", True
            if key in Zc:
                Cr, src = Zc[key], "calibration cache"
            elif key in Zv:
                Cr, src = Zv[key], "verification cache"
            if Cr is None:
                Cr, _, conv = periodic_voxel_homogenize(rasterize_lattice(lat, sz, ns), cell=lat.cell)
                if pv:
                    Zv[key] = Cr
                    np.savez(pv, **Zv)
            rho = float(np.mean(rasterize_lattice(lat, sz, ns)))
            Cp = homogenize(lat, sz)[0]
            Cc = homogenize(lat, sz, corr=corr_of(lat))[0]
            row = dict(cell=topo, dim=lat.dim, L_over_d=ratio, min_slenderness=float(min_slenderness(lat, sz)), rho_fe=rho,
                       err_plain=_entry_errors(Cp, Cr), err_corrected=_entry_errors(Cc, Cr), fe_source=src, fe_converged=bool(conv))
            for tag, C in (("fe", Cr), ("plain", Cp), ("corr", Cc)):
                for name, v in zip(("C11", "C12", "Css", "lmin"), _cubic_entries(C)):
                    row[f"{name}_{tag}"] = v
            rows.append(row)
            print(f"  sweep {topo:13s} L/d {ratio:4.1f}  plain {100 * row['err_plain']:6.1f}%  corrected {100 * row['err_corrected']:6.1f}%   [{src}]", flush=True)
    return rows


def _save_pair(fig, out, stem):
    """Save a figure as <stem>.svg (vector)."""
    _savefig(fig, os.path.join(out, stem + ".svg"), dpi=300)


def _nice_limits(lo, hi):
    """Round a positive data range outwards to 1-2-5 values (for a log axis)."""
    nice = [m * 10.0 ** e for e in range(-4, 6) for m in (1, 2, 5)]
    return max(v for v in nice if v <= lo / 1.15), min(v for v in nice if v >= hi * 1.15)


def fig_verification(out, quick=False, rows=None, ratios=None, fig2_min_ld=None, cache_dir=None, recompute=False):
    """Two stand-alone verification figures (each its own svg), made from one slenderness sweep against continuum FE, for ALL cells:"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    _apply_figure_fonts()
    fig2_min_ld = S_CORR_MIN if fig2_min_ld is None else float(fig2_min_ld)
    rows = rows or verification_sweep(quick, ratios=ratios, cache_dir=cache_dir, recompute=recompute)
    os.makedirs(out, exist_ok=True)
    cells = list(dict.fromkeys(r["cell"] for r in rows))
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "axes.linewidth": 0.8, "font.family": "sans-serif"}):
        fig, ax = plt.subplots(figsize=(5.9, 3.9))
        errs, dropped = {"plain": [], "corr": []}, []
        vals = [r["lmin_fe"] for r in rows if r["L_over_d"] >= fig2_min_ld and r["lmin_fe"] > 0]
        vals += [r[f"lmin_{t}"] for r in rows if r["L_over_d"] >= fig2_min_ld and r["lmin_fe"] > 0 for t in ("plain", "corr") if r[f"lmin_{t}"] > 0]
        lo, hi = min(vals), max(vals)
        g = np.array([lo / 1.4, hi * 1.4])
        ylow = g[0] * 1.25
        n_nonpos = 0
        for r in rows:
            ref = r["lmin_fe"]
            if r["L_over_d"] < fig2_min_ld:
                continue
            if ref <= 0:
                dropped.append((r["cell"], r["L_over_d"]))
                continue
            c, mk = _vstyle(r["cell"])
            for tag, filled in (("plain", False), ("corr", True)):
                v = r[f"lmin_{tag}"]
                errs[tag].append(abs(v / ref - 1))
                y = v if v > 0 else ylow
                if v <= 0:
                    n_nonpos += 1
                    ax.annotate("", xy=(ref, g[0] * 1.02), xytext=(ref, ylow), arrowprops=dict(arrowstyle="->", color=c, lw=0.7), zorder=2)
                ax.plot(ref, y, mk, ms=4.6, mfc=c if filled else "none", mec=c, mew=0.9, alpha=0.95 if filled else 0.7, zorder=3)
        ax.fill_between(g, g * 0.95, g * 1.05, color='0.55', alpha=0.55, lw=0, zorder=1)
        ax.plot(g, g * 0.95, color='0.35', lw=0.5, zorder=2)
        ax.plot(g, g * 1.05, color='0.35', lw=0.5, zorder=2)
        ax.plot(g, g, 'k-', lw=0.9, zorder=2)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(g); ax.set_ylim(g)
        ax.set_xlabel(r"$\lambda_{\min,\,\mathrm{continuum\ FE}}/E_s$"); ax.set_ylabel(r"$\lambda_{\min,\,\mathrm{beam\ model}}/E_s$")
        ax.set_title(r"Beam simplification error ($L/d \geq %g$)" % fig2_min_ld)
        h = [Line2D([], [], marker=_vstyle(c)[1], ls='', mfc=_vstyle(c)[0], mec=_vstyle(c)[0], ms=4.5, label=c) for c in cells]
        h += [Line2D([], [], marker='o', ls='', mfc='k', mec='k', ms=4.5, label="modified beam"),
              Line2D([], [], marker='o', ls='', mfc='none', mec='k', ms=4.5, label="plain beam"),
              Patch(facecolor='0.55', alpha=0.55, label="$\\pm$5 % error")]
        if n_nonpos:
            h += [Line2D([], [], marker=r'$\downarrow$', ls='', color='0.2', ms=6, label=r"beam $\lambda_{\min}\leq 0$ (drawn at lower edge)")]
        ax.legend(handles=h, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=False, handletextpad=0.3, borderaxespad=0.0)
        _ax_style(ax)
        fig.tight_layout()
        _save_pair(fig, out, "Fig02_beam_simplification_error")
        if dropped or n_nonpos:
            print(f"  Figure 2: {n_nonpos} model points with beam lambda_min <= 0 drawn at the lower edge; samples without a positive FE lambda_min (not drawable): {dropped or 'none'}")

        fig, ax = plt.subplots(figsize=(5.9, 3.9))
        allv = []
        for c in cells:
            rr = sorted((r for r in rows if r["cell"] == c), key=lambda r: r["L_over_d"])
            x = [r["L_over_d"] for r in rr]
            em, ep = [100 * r["err_corrected"] for r in rr], [100 * r["err_plain"] for r in rr]
            allv += [v for v in em + ep if v > 0]
            col, mk = _vstyle(c)
            ax.plot(x, em, '-', color=col, lw=1.3, marker=mk, ms=3.6, mfc=col, mec=col)
            ax.plot(x, ep, '--', color=col, lw=1.0)
        ax.axvline(S_CORR_MIN, color='0.5', lw=0.9)
        ax.axhline(5, color='0.5', lw=0.7, ls=':')
        ylo, yhi = _nice_limits(min(allv), max(allv))
        ax.set_yscale('log'); ax.set_ylim(ylo, yhi)
        xs = sorted(set(r["L_over_d"] for r in rows))
        ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4); ax.set_xticks(xs)
        ax.text(S_CORR_MIN - 0.05, yhi / 1.12, "$L/d=%g$" % S_CORR_MIN, color='0.35', ha='right', va='top', fontsize=7.5)
        ax.set_xlabel("Min. strut slenderness $L/d$"); ax.set_ylabel("Largest entry error vs FE  [%]")
        ax.set_title("Beam model error vs slenderness")
        h = [Line2D([], [], color=_vstyle(c)[0], lw=1.3, marker=_vstyle(c)[1], ms=3.6, label=c) for c in cells]
        h += [Line2D([], [], color='k', lw=1.3, label="modified beam"), Line2D([], [], color='k', lw=1.0, ls='--', label="plain beam"),
              Line2D([], [], color='0.5', lw=0.9, label="beam validity limit"), Line2D([], [], color='0.5', lw=0.7, ls=':', label="5 % error")]
        ax.legend(handles=h, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=False, handlelength=2.2, borderaxespad=0.0)
        _ax_style(ax)
        fig.tight_layout()
        _save_pair(fig, out, "Fig03_error_vs_slenderness")


def calibration_residuals(topo, params, refs, ratios=CALIB_RATIOS):
    """Residual vector of the calibration: signed errors e1 (all diagonal entries, shear diagonals included), e2 (normal off-diagonals, relative to"""
    lat = make_lattice_3d(topo) if topo in TOPO_3D else make_lattice_2d(topo)
    return np.concatenate([_signed_entry_errors(homogenize(lat, _calib_lattice_size(topo, r)[1], corr=tuple(params))[0], refs[(topo, r)])
                           for r in ratios])


def write_calibration_objective(path, refs_file, ratios=CALIB_RATIOS):
    """Table 1 data: J(theta) and the largest error at the current CORR_PARAMS, constants on a bound, number of residuals."""
    Z = np.load(refs_file)
    rows = []
    for topo in TOPO_2D + TOPO_3D:
        p = np.array(CORR_PARAMS[topo])
        e = calibration_residuals(topo, p, {(topo, r): Z[f"{topo}|{r}"] for r in ratios}, ratios)
        on_bound = [n for n, x, lo, hi in zip(("ka", "kb", "kappa_p", "pa", "pb"), p, _CALIB_LO, _CALIB_HI) if np.isclose(x, lo) or np.isclose(x, hi)]
        rows.append(dict(cell=topo, ka=p[0], kb=p[1], kappa_p=p[2], pa=p[3], pb=p[4], on_bound=" ".join(on_bound) or "-", J=float(e @ e),
                         largest_error_pct=100 * float(np.abs(e).max()), n_residuals=e.size))
    write_csv(path, rows)
    return rows


MESH_VOX = (4, 6, 8, 12, 16)
MESH_MAX_ELEMENTS = {2: 1_500_000, 3: 850_000}
MESH_SLOPE_STEP = 1.03


def mesh_case(topo, ld, vox, T, with_boundary=False):
    """FE homogenisation of cell `topo` at alpha = 1 with minimum strut slenderness `ld` and `vox` voxels across the thinnest strut (no grid cap, no minimum"""
    lat, size = _calib_lattice_size(topo, ld)
    ns = voxel_grid(lat, size, vox, nmin=4, nmax=10 ** 4)
    d_min = min(size[k] if lat.dim == 2 else 2.0 * size[k] for (_, _, _, k) in lat.struts)

    def solve(sz):
        C, rho_r, conv = periodic_voxel_homogenize(rasterize_lattice(lat, sz, ns), cell=lat.cell)
        return C, rho_r, conv

    C, rho_r, conv = solve(size)
    out = dict(cell=topo, L_over_d=float(ld), vox_target=vox, grid="x".join(map(str, ns)), n_elements=int(np.prod(ns)),
               vox_actual=float(d_min / (np.asarray(lat.cell) / np.asarray(ns)).max()), converged=bool(conv),
               lam_min=float(np.linalg.eigvalsh(sym(vweights(C.shape[0]) @ C))[0]), rho_r=float(rho_r), margin=float(margin(C, T)))
    if with_boundary:
        C2, rho2, conv2 = solve({k: v * MESH_SLOPE_STEP for k, v in size.items()})
        slope = (float(margin(C2, T)) - out['margin']) / (rho2 - rho_r)
        out.update(rho_B=float(rho_r - out['margin'] / slope), slope=float(slope), converged=bool(conv and conv2))
    return out


def grid_convergence(vox, vals):
    """Richardson extrapolation and grid convergence index (Celik et al."""
    vox, vals = np.asarray(vox, float), np.asarray(vals, float)
    if len(vals) < 3:
        return dict(p=np.nan, ext=np.nan, gci_pct=np.nan, note="< 3 levels")
    v3, v2, v1 = vox[-3:]
    f3, f2, f1 = vals[-3:]
    r21, r32 = v1 / v2, v2 / v3
    e21, e32 = f2 - f1, f3 - f2
    note, p = "", 1.0
    if e21 != 0 and e32 / e21 > 0:
        p = abs(np.log(abs(e32 / e21))) / np.log(r21)
        for _ in range(100):
            s = np.sign(e32 / e21)
            p_new = abs(np.log(abs(e32 / e21)) + np.log((r21 ** p - s) / (r32 ** p - s))) / np.log(r21)
            if abs(p_new - p) < 1e-6:
                break
            p = p_new
        p = float(np.clip(p, 0.5, 4.0))
    else:
        note = "assumed p=1"
    ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    return dict(p=float(p), ext=float(ext), gci_pct=float(100 * 1.25 * abs((f2 - f1) / f1) / (r21 ** p - 1)), note=note)


def _ladder(topo, ld, ladder=MESH_VOX):
    """Ladder levels of cell `topo` at slenderness `ld` whose grid stays within MESH_MAX_ELEMENTS."""
    lat, size = _calib_lattice_size(topo, ld)
    cap = MESH_MAX_ELEMENTS[lat.dim]
    return [v for v in ladder if int(np.prod(voxel_grid(lat, size, v, nmin=4, nmax=10 ** 4))) <= cap]


def affordable_ld(topo, ld, levels=(4, 6, 8, 12)):
    """Largest slenderness <= ld (step 0.1) at which all `levels` fit into MESH_MAX_ELEMENTS, or None."""
    x = ld
    while x > 1.0:
        if len(_ladder(topo, x, levels)) == len(levels):
            return round(x, 1)
        x -= 0.1
    return None


def mesh_convergence_cell(topo, sel_ld, T, ladder=MESH_VOX, log=print, known=()):
    """Mesh-convergence rows of one cell at alpha = 1: (i) the selected design (minimum strut slenderness `sel_ld`; lambda_min, rasterised rho and the"""
    cases = [("selected", sel_ld), ("L/d=5", 5.0)]
    if any(len(_ladder(topo, ld, ladder)) < 3 for _, ld in cases):
        aff = affordable_ld(topo, min(sel_ld, 5.0))
        if aff:
            cases.append(("affordable", aff))
    have = {(r['case'], round(r['L_over_d'], 3), r['vox_target']): r for r in known}
    rows = []
    for case, ld in cases:
        for v in _ladder(topo, ld, ladder):
            r = have.get((case, round(ld, 3), v))
            if r is None:
                r = mesh_case(topo, ld, v, T, with_boundary=(case == "selected"))
                r['case'] = case
            rows.append(r)
            log(f"  mesh {topo:13s} {case:8s} L/d {ld:6.2f}  v={v:2d} grid {r['grid']:>11s}  lam_min {r['lam_min']:.6e}  rho_r {r['rho_r']:.5f}"
                + (f"  rho_B {r['rho_B']:.5f}" if 'rho_B' in r else ""))
    return rows


def mesh_summary_rows(rows, rho_map=None):
    """One row per (cell, case): deviations of the 8-voxel and of the finest level from the extrapolated value, GCI, observed order."""
    out = []
    for cell in dict.fromkeys(r['cell'] for r in rows):
        for case in ("selected", "L/d=5", "affordable"):
            R = sorted([r for r in rows if r['cell'] == cell and r['case'] == case], key=lambda r: r['vox_target'])
            if not R:
                continue
            row = dict(cell=cell, case=case, L_over_d=R[0]['L_over_d'], levels=",".join(str(r['vox_target']) for r in R), finest_grid=R[-1]['grid'])
            for key, lab in (("lam_min", "lam"), ("rho_r", "rho_r"), ("rho_B", "rho_B")):
                if key not in R[0]:
                    continue
                vx, vals = [r['vox_target'] for r in R], [r[key] for r in R]
                g = grid_convergence(vx, vals)
                ref = g['ext'] if np.isfinite(g['ext']) else vals[-1]
                dev = lambda x: 100 * (x / ref - 1)
                row.update({f"{lab}_ext": ref, f"{lab}_dev4_pct": dev(vals[0]) if vx[0] == 4 else np.nan,
                            f"{lab}_dev8_pct": dev(vals[vx.index(8)]) if 8 in vx else np.nan, f"{lab}_devfine_pct": dev(vals[-1]),
                            f"{lab}_gci_pct": g['gci_pct'], f"{lab}_p": g['p'], f"{lab}_note": g['note']})
            if rho_map and cell in rho_map and case == "selected":
                row['rho_map'] = rho_map[cell]
                row['rho_B_ext_vs_map_pct'] = 100 * (row['rho_B_ext'] / rho_map[cell] - 1)
            out.append(row)
    return out


def fig_04_mesh_convergence(rows, out_dir):
    """Figure 4: deviation of lambda_min (a: L/d = 5, b: selected design) and of the Loewner-boundary density rho_B (c: selected design) from the"""
    import matplotlib.pyplot as plt
    _apply_figure_fonts()
    fig, axg = plt.subplots(2, 2, figsize=(7.4, 5.6))
    axs = list(axg.ravel()[:3])
    panels = (("L/d=5", "lam_min", r"(a) $\lambda_{\min}$, $L/d$ = 5"), ("selected", "lam_min", r"(b) $\lambda_{\min}$, selected design"),
              ("selected", "rho_B", r"(c) $\rho_B$, selected design"))
    for ax, (case, key, title) in zip(axs, panels):
        for cell in TOPO_2D + TOPO_3D:
            R = sorted([r for r in rows if r['cell'] == cell and r['case'] == case], key=lambda r: r['vox_target'])
            if len(R) < 3 and case == "L/d=5":
                R = sorted([r for r in rows if r['cell'] == cell and r['case'] == "affordable"], key=lambda r: r['vox_target'])
            if len(R) < 3:
                continue
            vx, vals = [r['vox_target'] for r in R], [r[key] for r in R]
            ext = grid_convergence(vx, vals)['ext']
            dev = [100 * (x / ext - 1) for x in vals]
            col, mk = _vstyle(cell)
            ax.plot(vx, dev, marker=mk, color=col, ms=4, lw=1.1, label=cell)
        ax.axhline(0, color='k', lw=0.6)
        ax.axhspan(-5, 5, color='0.9', zorder=0)
        ax.axvline(8, color='0.5', lw=0.8, ls=':')
        ax.set_xlabel("Voxels across struts")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([4, 6, 8, 12, 16])
        _ax_style(ax)
    axs[0].set_ylabel("Deviation (%)")
    axs[2].set_ylabel("Deviation (%)")
    handles, labels = axs[2].get_legend_handles_labels()
    axg[1, 1].axis('off')
    axg[1, 1].legend(handles, labels, fontsize=7.5, ncol=2, loc='upper left', frameon=False)
    fig.tight_layout()
    _save_pair(fig, out_dir, FIG_MESH)


def _map_fields(pool, T, kind):
    """(g, gi) of refined_optimum for the whole map: g >= 0 <=> the design passes the test `kind` ('lo' / 'cw'), gi = smooth field used to interpolate the"""
    tol = tol_of(T)
    if kind == 'lo':
        g = margin(pool.C, T) + tol
        return g, g
    return (pool.C - T).min(axis=(-1, -2)) + tol, componentwise_margin_field(pool.C, T)


def _map_crossing(pool, fields, b):
    """Density at which column b of the design map crosses zero: the linear interpolation between the last failing and the first passing row used by"""
    g, gi = fields[0][:, b], fields[1][:, b]
    ok = np.flatnonzero(g >= 0)
    if len(ok) == 0:
        return np.nan, None
    a = int(ok[0])
    if a == 0:
        return float(pool.rho[0, b]), 0
    w = float(np.clip(-gi[a - 1] / (gi[a] - gi[a - 1]), 0.0, 1.0)) if gi[a - 1] < 0 < gi[a] - gi[a - 1] else 1.0
    return float(pool.rho[a - 1, b] + w * (pool.rho[a, b] - pool.rho[a - 1, b])), a - 1


def _strut_size(lat, r):
    """Strut-size dict of a design of strut radius (3D) / thickness (2D) r, as in figure_pool."""
    return size_2d(r, r) if lat.dim == 2 else size_3d(r)


def direct_boundary(lat, rads, vols, rhos, T, kind, a_fail, tol_rho=1e-4):
    """Boundary density of the hybrid model evaluated DIRECTLY (no interpolation between density rows, no FE-minus-beam reconstruction): bisection of the"""
    n_eval = [0]

    def passes(r):
        n_eval[0] += 1
        out = hybrid_C(lat, _strut_size(lat, float(np.interp(r, vols, rads))))
        return bool((pass_loewner if kind == 'lo' else pass_componentwise)(out['C'], T)), out['model']

    lo_i, hi_i = max(a_fail, 0), min(a_fail + 1, len(rhos) - 1)
    while lo_i > 0 and passes(rhos[lo_i])[0]:
        hi_i, lo_i = lo_i, lo_i - 1
    while hi_i < len(rhos) - 1 and not passes(rhos[hi_i])[0]:
        lo_i, hi_i = hi_i, hi_i + 1
    lo, hi = float(rhos[lo_i]), float(rhos[hi_i])
    model = 0
    while hi - lo > tol_rho:
        mid = 0.5 * (lo + hi)
        ok, model = passes(mid)
        lo, hi = (lo, mid) if ok else (mid, hi)
    return 0.5 * (lo + hi), model, n_eval[0]


def interpolation_error_cell(topo, pool, T, n_pixels=20, seed=0, tol_rho=1e-4, log=print, known=None, rho_design=0.5):
    """Interpolation error of the design map of one cell."""
    dim = pool.dim
    alphas = np.asarray(pool.axes[1], float)
    rhos = np.asarray(pool.axes[0], float)
    known_b, known_p = known if known else ([], [])
    mk = lambda al: make_lattice_2d(topo, al) if dim == 2 else make_lattice_3d(topo, al, 2)
    brows = list(known_b)
    lattices = {}
    if not brows:
        best = refined_optimum(pool, T, 'lo')
        cols = {}
        for label, a in (("0.7", 0.7), ("1", 1.0), ("1.05", 1.05), ("1.4", 1.4), ("opt", best['alpha'] if best else None)):
            if a is not None:
                cols.setdefault(int(np.argmin(np.abs(alphas - a))), []).append(label)
        fields = {kind: _map_fields(pool, T, kind) for kind in ('lo', 'cw')}
        for b, labels in cols.items():
            lattices[b] = mk(alphas[b])
            curve = design_volume_curve(lattices[b])
            for kind in ('lo', 'cw'):
                r_map, a_fail = _map_crossing(pool, fields[kind], b)
                if a_fail is None:
                    continue
                r_dir, model, n_eval = direct_boundary(lattices[b], curve[0], curve[1], rhos, T, kind, a_fail, tol_rho)
                brows.append(dict(cell=topo, alpha=float(alphas[b]), alpha_label="/".join(labels), criterion={'lo': 'Loewner', 'cw': 'CW'}[kind],
                                  rho_map=r_map, rho_direct=r_dir, diff_pct=100 * (r_map / r_dir - 1), boundary_model=['beam', 'FE'][model], n_eval=n_eval))
                log(f"  interp {topo:13s} alpha {alphas[b]:.4f} {kind}: map {r_map:.5f} direct {r_dir:.5f}  ({100 * (r_map / r_dir - 1):+.2f} %)")
    done = {(r.get('sample', 'all'), round(r['rho'], 6), round(r['alpha'], 6)): r for r in known_p}
    prows = []
    rec = np.argwhere(pool.model == 2)
    for sample, sel_seed, rho_cap in (("all", seed, None), ("design", seed + 1, rho_design)):
        pix = rec if rho_cap is None else rec[rhos[rec[:, 0]] <= rho_cap]
        if not len(pix):
            continue
        rng = np.random.default_rng(sel_seed)
        for a, b in pix[rng.choice(len(pix), size=min(n_pixels, len(pix)), replace=False)]:
            key = (sample, round(float(rhos[a]), 6), round(float(alphas[b]), 6))
            if key in done:
                prows.append(dict(done[key], sample=sample))
                continue
            lat = lattices[b] if b in lattices else mk(alphas[b])
            direct = hybrid_C(lat, _strut_size(lat, float(pool.rad[a, b])))
            Cd, Cm = direct['C'], pool.C[a, b]
            prows.append(dict(cell=topo, sample=sample, rho=float(rhos[a]), alpha=float(alphas[b]), model_direct=['beam', 'FE'][direct['model']],
                              entry_error_pct=100 * _entry_errors(Cm, Cd), margin_map=float(margin(Cm, T)), margin_direct=float(margin(Cd, T)),
                              margin_diff_over_Tmax_pct=100 * (float(margin(Cm, T)) - float(margin(Cd, T))) / float(np.abs(T).max())))
            log(f"  interp {topo:13s} pixel ({sample}) rho {rhos[a]:.3f} alpha {alphas[b]:.4f}: entry error {prows[-1]['entry_error_pct']:.2f} %")
    return brows, prows


def interpolation_summary_rows(brows, prows):
    """Table 5 rows: per cell the largest and mean |difference| of the boundary densities (%), the mean signed difference of the Loewner boundary (negative"""
    out = []
    for cell in dict.fromkeys(r['cell'] for r in brows):
        B = [r for r in brows if r['cell'] == cell]
        d = np.array([r['diff_pct'] for r in B])
        lo = np.array([r['diff_pct'] for r in B if r['criterion'] == 'Loewner'])
        row = dict(cell=cell, n_boundaries=len(B), max_abs_diff_pct=float(np.abs(d).max()), mean_abs_diff_pct=float(np.abs(d).mean()),
                   mean_signed_diff_lw_pct=float(lo.mean()) if len(lo) else np.nan)
        for sample in ("all", "design"):
            e = np.array([r['entry_error_pct'] for r in prows if r['cell'] == cell and r.get('sample', 'all') == sample] or [np.nan])
            row.update({f"n_pixels_{sample}": int(np.isfinite(e).sum()), f"median_entry_error_{sample}_pct": float(np.nanmedian(e)),
                        f"max_entry_error_{sample}_pct": float(np.nanmax(e)), f"mean_entry_error_{sample}_pct": float(np.nanmean(e))})
        out.append(row)
    return out


def _study_cell(job):
    """Worker: the requested studies for ONE cell."""
    cell, what, out, corr, fe, target, quick, recompute = job
    CORR_PARAMS.update(corr)
    FE_SETTINGS.update(fe)
    set_target(*target)
    dim = 3 if cell in TOPO_3D else 2
    T = target_stiffness(dim)
    folder = os.path.join(out, "studies")
    os.makedirs(folder, exist_ok=True)
    path = lambda study: os.path.join(folder, f"{study}_{cell}.json")
    todo = [s for s in ("mesh", "interp") if s in what]
    pool = figure_pool_cached(os.path.join(out, "pool_cache"), cell, dim, fig_rho_rows(quick),
                              np.unique(np.r_[np.linspace(0.7, 1.4, 5 if quick else 9), 1.0]), 2)
    if "mesh" in todo:
        sel = select_design(pool, T)
        known = json.load(open(path("mesh")))["rows"] if (os.path.exists(path("mesh")) and not recompute) else []
        rows = mesh_convergence_cell(cell, sel['slender'], T, known=known)
        with open(path("mesh"), "w") as f:
            json.dump(dict(rho_map=float(sel['rho']), L_over_d=float(sel['slender']), rows=rows), f)
    if "interp" in todo:
        old = json.load(open(path("interp"))) if (os.path.exists(path("interp")) and not recompute) else None
        brows, prows = interpolation_error_cell(cell, pool, T, known=(old["boundaries"], old["pixels"]) if old else None)
        with open(path("interp"), "w") as f:
            json.dump(dict(boundaries=brows, pixels=prows), f)
    return cell


def report_studies(out):
    """Table 4 + Figure 4 (mesh study) and Table 5 (interpolation study) from the per-cell results in <out>/studies."""
    import matplotlib
    matplotlib.use("Agg")
    folder = os.path.join(out, "studies")
    load = lambda study, cell: json.load(open(os.path.join(folder, f"{study}_{cell}.json"))) if os.path.exists(os.path.join(folder, f"{study}_{cell}.json")) else None
    cells = TOPO_2D + TOPO_3D
    mesh = {c: v for c, v in ((c, load("mesh", c)) for c in cells) if v}
    if mesh:
        rows = [r for c in mesh for r in mesh[c]['rows']]
        write_csv(os.path.join(out, "Table4.csv"), mesh_summary_rows(rows, {c: v['rho_map'] for c, v in mesh.items()}))
        fig_04_mesh_convergence(rows, out)
    inter = {c: v for c, v in ((c, load("interp", c)) for c in cells) if v}
    if inter:
        write_csv(os.path.join(out, "Table5.csv"), interpolation_summary_rows([r for c in inter for r in inter[c]['boundaries']],
                                                                             [r for c in inter for r in inter[c]['pixels']]))


def fig_24_optimal_density(rows, out_dir):
    """Figure 24 (Fig24_optimal_density.svg) -- graphical form of table T5: for every cell the minimum V_s/V_T found with the Loewner test (black circle)"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _apply_figure_fonts()
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.4), gridspec_kw=dict(width_ratios=[5, 8]))
    for ax, dim in zip(axs, ("2D", "3D")):
        R = sorted([r for r in rows if r['dim'] == dim], key=lambda r: float(r['rho_loewner']))
        xmax = 1.2 * max(max(float(r['rho_loewner']), float(r['rho_componentwise'])) for r in R)
        for k, r in enumerate(R):
            lo, cw = float(r['rho_loewner']), float(r['rho_componentwise'])
            safe = str(r['cw_optimum_is_loewner_feasible']) == 'True'
            ax.plot([lo, cw], [k, k], color='0.65', lw=1.6, zorder=1)
            ax.scatter([lo], [k], s=46, c='k', zorder=3)
            ax.scatter([cw], [k], s=46, marker='s', facecolors='#d1495b' if safe else 'white', edgecolors='#d1495b', linewidths=1.4, zorder=3)
            pct = f"{100 * (cw - lo) / lo:+.0f} %".replace('-', '−')
            ax.text(max(lo, cw) + 0.035 * xmax, k, pct, va='center', fontsize=7.5, color='0.25')
        ax.set_yticks(range(len(R)))
        ax.set_yticklabels([r['cell'].split('_', 1)[-1] for r in R])
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xlabel(VVS_LABEL)
        ax.set_title(f"({'a' if dim == '2D' else 'b'}) {dim}", fontsize=10)
        _ax_style(ax)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color='k', markersize=6, label="Loewner optimum"),
                        Line2D([], [], marker='s', ls='', mfc='#d1495b', mec='#d1495b', markersize=6, label="Component-wise optimum, Loewner-feasible (safe)"),
                        Line2D([], [], marker='s', ls='', mfc='white', mec='#d1495b', markersize=6, label="Component-wise optimum, not Loewner-feasible (unsafe)")],
               loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    name = FIG_DENSITY + ".svg"
    _savefig(fig, os.path.join(out_dir, name))


def optima_rows(pools, Ts):
    """Loewner and component-wise optima of every cell at the reference target (input of Figure 24)."""
    rows = []
    for dim in (2, 3):
        for topo, pool in pools[dim].items():
            lw, cw = refined_optimum(pool, Ts[dim], 'lo'), refined_optimum(pool, Ts[dim], 'cw')
            rows.append(dict(dim=f"{dim}D", cell=topo, rho_loewner=lw['rho'], rho_componentwise=cw['rho'],
                             cw_optimum_is_loewner_feasible=bool(pass_loewner(cw['C'], Ts[dim]))))
    return rows


def _lw_optimum(pool, T):
    """Loewner optimum over alpha, with the min."""
    best = refined_optimum(pool, T, 'lo')
    if best is not None:
        a0, b, w = best['a0'], best['b'], best['w']
        best['slender'] = float((1 - w) * pool.slender[a0, b] + w * pool.slender[min(a0 + 1, pool.slender.shape[0] - 1), b])
    return best


def table2_rows(pools, safe):
    """Table 2: CW vs Loewner optimum along the E_T/E_s sweep, slenderness of the Loewner optimum at the reference target, largest target evaluated with the"""
    rows = []
    for dim in (2, 3):
        for topo, pool in pools[dim].items():
            recs = sorted((r for r in safe[dim] if r['topo'] == topo), key=lambda r: r['ratio'])
            lw = [_lw_optimum(pool, iso_stiffness(r['ratio'], TARGET_SPEC["nu"], dim)) for r in recs]
            pct = lambda i: 100 * (recs[i]['rho'] / lw[i]['rho'] - 1) if lw[i] else np.nan
            ref = min(range(len(recs)), key=lambda i: abs(np.log(recs[i]['ratio'] / target_ratio())))
            beam = [r['ratio'] for r, l in zip(recs, lw) if l and l['slender'] >= S_CORR_MIN]
            row = dict(cell=topo)
            row.update({f"cw_vs_lw_at_{recs[i]['ratio']:.3g}_pct": pct(i) for i in (0, ref, len(recs) - 1)})
            row.update(n_targets=len(recs), n_cw_optimum_loewner_feasible=sum(r['safe'] for r in recs),
                       Ld_loewner_optimum_at_reference=lw[ref]['slender'] if lw[ref] else np.nan,
                       largest_ET_Es_beam_evaluated=("none" if not beam else f"all ({beam[-1]:.3g})" if len(beam) == len(recs) else f"{max(beam):.3g}"))
            rows.append(row)
    return rows


def table3_rows(pools, Ts):
    """Table 3: entry-wise ratios C_ij/T_ij of the Loewner optimum, number of entries with C_ij < T_ij, binding entry of the CW optimum."""
    rows = []
    for dim in (2, 3):
        T = Ts[dim]
        n = T.shape[0]
        nz = T != 0
        for topo, pool in pools[dim].items():
            C, Cc = refined_optimum(pool, T, 'lo')['C'], refined_optimum(pool, T, 'cw')['C']
            d = {(i, j): Cc[i, j] - T[i, j] for i in range(n) for j in range(i, n) if nz[i, j]}
            dmin = min(d.values())
            bind = " = ".join(f"C{i + 1}{j + 1}" for (i, j), v in d.items() if v - dmin <= 1e-6 * np.abs(T).max())
            rows.append(dict(cell=topo, min_Cii_over_Tii_normal=min(C[i, i] / T[i, i] for i in range(dim)),
                             min_Cij_over_Tij_normal_offdiag=min(C[i, j] / T[i, j] for i in range(dim) for j in range(i + 1, dim)),
                             min_Css_over_Tss_shear=min(C[i, i] / T[i, i] for i in range(dim, n)),
                             n_entries_Cij_below_Tij=int(((C < T - tol_of(T)) & nz).sum()), binding_entry_at_cw_optimum=bind))
    return rows


OUT_DIR = "paper_output"
QUICK = False
NO_CACHE = False
RECOMPUTE_FE = False
WORKERS = default_workers()
MESH_WORKERS = 1


def main():
    """Run driver: calibration, design pools, Figures 1-25 and Tables 1-5 of the manuscript, written to OUT_DIR."""
    if QUICK:
        FE_SETTINGS.update(vox=4, vox_max=6, nmax3=28, nmax2=120)
    FE_SETTINGS["vox_max"] = max(FE_SETTINGS["vox_max"], FE_SETTINGS["vox"])
    import matplotlib
    matplotlib.use("Agg")
    out = OUT_DIR
    os.makedirs(out, exist_ok=True)
    o = lambda n: os.path.join(out, n)
    T0 = time.time()
    log = lambda msg: print(f"{msg}  [{time.time() - T0:.0f}s]", flush=True)
    CORR_PARAMS.update(calibrate_corr_params(out, QUICK, WORKERS))
    write_calibration_objective(o("Table1.csv"), o("calibration_fe_refs.npz"), CALIB_RATIOS_QUICK if QUICK else CALIB_RATIOS)
    log("Table 1 written")
    fig_01_loewner_cones(o(FIG_CONES + ".svg"))
    fig_verification(out, QUICK, cache_dir=out, recompute=RECOMPUTE_FE)
    log("Figures 1-3 written")

    rhos, alphas = fig_rho_rows(QUICK), np.unique(np.r_[np.linspace(0.7, 1.4, 5 if QUICK else 9), 1.0])
    cdir = None if NO_CACHE else o("pool_cache")
    jobs = [(t, 2, 2) for t in TOPO_2D] + [(t, 3, 2) for t in TOPO_3D]
    ready = prefetch_pools(cdir, jobs, rhos, alphas, WORKERS) if WORKERS > 1 else {}
    pools = {2: {}, 3: {}}
    for t, dm, ax_ in jobs:
        pools[dm][t] = ready[(t, dm, ax_)] if (t, dm, ax_) in ready else (
            figure_pool(t, dm, rhos, alphas, ax_) if cdir is None else figure_pool_cached(cdir, t, dm, rhos, alphas, ax_))
    log("design pools ready")
    Ts = {2: target_stiffness(2), 3: target_stiffness(3)}
    safe = safe_regime_scan(pools)
    write_csv(o("Table2.csv"), table2_rows(pools, safe))
    write_csv(o("Table3.csv"), table3_rows(pools, Ts))
    log("Tables 2-3 written")

    fig_22_23_target_scatter(safe, out)
    fig_24_optimal_density(optima_rows(pools, Ts), out)
    fig_25_lw_cw_optima(pools, safe, out)
    log("Figures 22-25 written")
    for dim in (2, 3):
        for topo in (TOPO_2D if dim == 2 else TOPO_3D):
            fig_cell(topo, dim, pools[dim][topo], Ts[dim], TARGET_SPEC["name"], out)
    log("Figures 5-21 written")

    jobs = [(c, {"mesh", "interp"}, out, dict(CORR_PARAMS), dict(FE_SETTINGS), (target_ratio(), TARGET_SPEC['nu']), QUICK, RECOMPUTE_FE)
            for c in TOPO_3D + TOPO_2D]
    if MESH_WORKERS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(MESH_WORKERS, len(jobs))) as ex:
            for c in ex.map(_study_cell, jobs):
                log(f"  studies {c} done")
    else:
        for j in jobs:
            log(f"  studies {_study_cell(j)} done")
    report_studies(out)
    log("Table 4, Figure 4 and Table 5 written")


if __name__ == "__main__":
    main()
