"""Drivable-area signed distance field (SDF) on the E grid -- stage T, IMPL_SPEC section 3.5.

What the field is
-----------------
``sdf[i, j]`` = signed Euclidean distance (metres) from the centre of E-grid cell ``(i, j)`` to the
boundary of the official DAC drivable region, **positive inside, negative outside**, clipped to
``+-SDF_CLIP`` (10 m).  Stored per token as float16 ``[320, 256]`` (``save_sdf`` / ``load_sdf``).

Drivable region = union of the metric-cache ``drivable_area_map`` polygons whose type is one of the four
layers the official DAC uses (``pdm_scorer._calculate_ego_area``: ROADBLOCK, INTERSECTION, DRIVABLE_AREA,
CARPARK_AREA).  LANE / LANE_CONNECTOR polygons are in the same map but are NOT DAC layers and are not used.
Official DAC fails a trajectory iff at some of the 41 LQR-tracked states (0..4 s, 0.1 s) at least one ego
corner lies in none of those polygons  <=>  the corner is outside their union  <=>  ``sdf(corner) < 0``
(a corner exactly on the boundary is "outside" for shapely; measure zero).

Frames, grid, units
-------------------
N frame: NAVSIM ego frame at t0 -- rear-axle origin, x forward, y left, heading CCW from +x (radians).
E grid (row = x forward, col = y left, col 0 = LEFT):
    row i -> x      = -8 + 0.125 + 0.25 i      (i = 0..319, cells cover x in [-8, 72] m)
    col j -> y_left = 32 - 0.125 - 0.25 j      (j = 0..255, cells cover y in [32, -32] m)
The metric cache queries the map with radius 100 m around the t0 ego *centre*
(``MetricCacheProcessor._map_radius``); every E cell is < 80 m away, so the grid is fully covered.

Construction (exact at the cell centres; not a distance transform)
------------------------------------------------------------------
sign      : ``shapely.contains_xy(union, centre)`` -- the point-in-polygon predicate the scorer itself uses;
magnitude : distance from the centre to the union boundary (exact GEOS point-segment distance, nearest
            boundary segment from an STRtree, searched up to ``SDF_CLIP``).
So the only approximations left are the bilinear interpolation between centres and float16 rounding
(<= 0.004 m below 10 m).  Pilot on 12 navtest tokens (points within 2 m of the boundary): |error| median
0.000 m, p99 0.015-0.046 m; the cell-centre distance-transform prototype (review/t_sdf.py) had median
0.07-0.11 m, p99 0.21-0.25 m, a +0.01..+0.03 m bias, and -- because a grid distance transform cannot
see road edges just outside the grid -- errors up to 9.8 m near the grid edge (1 of 12 tokens).  So this
module does NOT follow the prototype's distance transform.
Build cost ~0.5 s/token + ~0.2 s metric-cache load; ~82 KB/token (compressed npz).

Validation (tools/refiner/validate_sdf.py; 550 navtest tokens = 300 random + 250 DAC-fail enriched; numbers in
report/refiner_T/sdf_validation.json): the exact union reproduces the official DAC on all 1,100 scored trajectories
(original + human); ``min corner sdf < 0`` over the 41 LQR-tracked footprints agrees with the official DAC on
548/550 original trajectories (0 false alarms, AUC 0.9999; random subset 299/300, AUC 0.9993).  The 2 misses are
sub-cell slivers (< ~10 cm gaps between adjacent map polygons, penetration 0.001 / 0.012 m) that a 0.25 m grid
cannot represent.  Stored-field error at corners within 2 m of the boundary: median 0.0001 m, p99 0.008 m,
max 0.12 m (at such slivers / vertices).

Lookup (torch, differentiable)
------------------------------
``sample_sdf`` interpolates bilinearly between cell centres (identical to ``grid_sample(...,
align_corners=False)`` inside the grid) and is differentiable w.r.t. the query points (and the field).
A point is **in grid** iff it lies in the hull of the cell centres, x in [-7.875, 71.875] and
y in [-31.875, 31.875] m; outside it the returned value is 0 and ``valid`` is False -- no border copy and no
zero-padding blend (spec 3.7: out-of-grid corners are counted and excluded by the caller).
Non-finite query points are also invalid.

Deviation from the spec text (documented, interface unchanged)
--------------------------------------------------------------
* The valid lookup domain is the hull of the cell centres, i.e. the grid extent minus half a cell
  (0.125 m) on each side, because bilinear interpolation is undefined in the outer half cell without
  padding.  Points there are reported as out of grid.
* Values are clipped to +-10 m (spec gives no clip; the DAC surrogate only uses values near its 0.2 m margin).
* Storage layout: ``sdf/<subset>/<token>.npz`` with subset = navtrain (stage-T train + dev pool) or navtest
  (``sdf_path``); a field depends only on the token, so it is not duplicated per train/dev split.

Ego footprint (``ego_corners``): nuPlan Pacifica, half length 2.588 m, half width 1.1485 m, box centre
1.461 m ahead of the rear axle along the heading (= sf_common / metric-cache vehicle parameters); corner
order FL, RL, RR, FR (= nuPlan OrientedBox / pdm ``BBCoordsIndex``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# ----------------------------------------------------------------------------------------- constants
E_X0 = -8.0            # x of the grid's rear edge [m]  (row 0 side)
E_X1 = 72.0            # x of the grid's front edge [m]
E_Y0 = 32.0            # y_left of the grid's left edge [m] (col 0 side)
E_Y1 = -32.0           # y_left of the grid's right edge [m]
E_RES = 0.25           # cell size [m]
E_H = 320              # rows (x)
E_W = 256              # cols (y_left, col 0 = left)
SDF_CLIP = 10.0        # |sdf| clip [m]
SDF_VERSION = "e_grid_dac_exact_v1"   # bump when the bytes of a stored field would change

# valid lookup domain = hull of the cell centres
X_MIN_VALID = E_X0 + E_RES / 2        # -7.875
X_MAX_VALID = E_X1 - E_RES / 2        # 71.875
Y_MIN_VALID = E_Y1 + E_RES / 2        # -31.875
Y_MAX_VALID = E_Y0 - E_RES / 2        # 31.875

# official DAC layers (pdm_scorer._calculate_ego_area)
DAC_LAYER_NAMES = ("ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA")

# ego vehicle (nuPlan Pacifica; identical to sf_common and the metric-cache vehicle parameters)
EGO_HALF_LEN = 2.588
EGO_HALF_WID = 1.1485
EGO_RA2C = 1.461       # rear axle -> box centre along the heading [m]

SDF_ROOT = Path("/home/external-user/ssd/yongjae_refiner/sdf")
METRIC_CACHE_MAP_RADIUS = 100.0       # metric_cache_processor._map_radius (around the t0 ego centre)


# ----------------------------------------------------------------------------------------- grid
def grid_centres() -> Tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates (float64): xs [320] (x forward, row order), ys [256] (y left, col order)."""
    xs = E_X0 + E_RES * (np.arange(E_H) + 0.5)
    ys = E_Y0 - E_RES * (np.arange(E_W) + 0.5)
    return xs, ys


def grid_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """X, Y [320, 256] float64 cell centres in N (indexing 'ij': X[i, j] = xs[i], Y[i, j] = ys[j])."""
    xs, ys = grid_centres()
    return np.meshgrid(xs, ys, indexing="ij")


# ----------------------------------------------------------------------------------------- torch lookup
def sample_sdf(sdf: torch.Tensor, xy: torch.Tensor,
               sdf_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinear SDF lookup at N-frame points, differentiable w.r.t. ``xy`` (and ``sdf``).

    sdf       : [S, 320, 256] (or [320, 256] with ``xy`` [..., 2]); any float dtype (float16 storage is
                upcast to xy's dtype if float64, else float32).
    xy        : [B, ..., 2] query points (x forward, y left, metres, N frame).
    sdf_index : optional long [B]: field row used for batch element b (e.g. 13 drafts of a token share one
                field).  Default: b if S == B, 0 if S == 1.
    returns   : (value [B, ...] float, valid [B, ...] bool).  value = 0 where not valid (outside the hull of
                the cell centres, or non-finite point); its gradient there is 0.
    """
    if sdf.dim() == 2:
        v, m = sample_sdf(sdf[None], xy[None], None)
        return v[0], m[0]
    if sdf.dim() != 3 or tuple(sdf.shape[1:]) != (E_H, E_W):
        raise ValueError(f"sdf must be [S, {E_H}, {E_W}], got {tuple(sdf.shape)}")
    if xy.shape[-1] != 2:
        raise ValueError(f"xy must end with 2, got {tuple(xy.shape)}")
    S, H, W = sdf.shape
    B = xy.shape[0]
    dtype = torch.float64 if xy.dtype == torch.float64 else torch.float32
    if sdf_index is None:
        if S == B:
            sdf_index = torch.arange(B, device=xy.device)
        elif S == 1:
            sdf_index = torch.zeros(B, dtype=torch.long, device=xy.device)
        else:
            raise ValueError(f"sdf has {S} fields but xy has batch {B}; pass sdf_index")
    sdf_index = sdf_index.to(device=xy.device, dtype=torch.long)
    if sdf_index.shape != (B,):
        raise ValueError(f"sdf_index must be [{B}], got {tuple(sdf_index.shape)}")

    field = sdf.to(device=xy.device, dtype=dtype).reshape(-1)       # [S*H*W]
    pts = xy.reshape(B, -1, 2).to(dtype)
    u = (pts[..., 0] - E_X0) / E_RES - 0.5                           # fractional row (cell centre i at u = i)
    v = (E_Y0 - pts[..., 1]) / E_RES - 0.5                           # fractional col (cell centre j at v = j)
    valid = torch.isfinite(u) & torch.isfinite(v) & (u >= 0) & (u <= H - 1) & (v >= 0) & (v <= W - 1)
    zero = torch.zeros((), dtype=dtype, device=xy.device)
    u = torch.where(valid, u, zero)
    v = torch.where(valid, v, zero)
    i0 = torch.clamp(torch.floor(u.detach()), 0, H - 2).long()
    j0 = torch.clamp(torch.floor(v.detach()), 0, W - 2).long()
    a = u - i0.to(dtype)                                             # in [0, 1]
    b = v - j0.to(dtype)
    base = sdf_index.view(B, 1) * (H * W) + i0 * W + j0              # [B, P]
    f00, f10 = field[base], field[base + W]
    f01, f11 = field[base + 1], field[base + W + 1]
    val = (f00 * (1 - a) * (1 - b) + f10 * a * (1 - b) + f01 * (1 - a) * b + f11 * a * b)
    val = torch.where(valid, val, zero)
    out_shape = xy.shape[:-1]
    return val.reshape(out_shape), valid.reshape(out_shape)


def ego_corners(poses: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Rear-axle poses [..., 3] (x, y, heading; N frame) -> ego footprint corners [..., 4, 2], order
    FL, RL, RR, FR.  ``margin`` inflates half length and half width (metres)."""
    x, y, h = poses[..., 0], poses[..., 1], poses[..., 2]
    c, s = torch.cos(h), torch.sin(h)
    cx, cy = x + EGO_RA2C * c, y + EGO_RA2C * s
    hl, hw = EGO_HALF_LEN + margin, EGO_HALF_WID + margin
    out = []
    for sl, sw in ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)):   # FL, RL, RR, FR
        out.append(torch.stack([cx + sl * hl * c - sw * hw * s, cy + sl * hl * s + sw * hw * c], -1))
    return torch.stack(out, -2)


def corner_sdf(sdf: torch.Tensor, poses: torch.Tensor, sdf_index: Optional[torch.Tensor] = None,
               margin: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """SDF at the 4 ego corners of rear-axle poses [B, T, 3] -> (value [B, T, 4], valid [B, T, 4])."""
    return sample_sdf(sdf, ego_corners(poses, margin), sdf_index)


# ----------------------------------------------------------------------------------------- numpy reference
def sample_sdf_np(sdf: np.ndarray, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """float64 numpy twin of ``sample_sdf`` for one field [320, 256]; xy [..., 2] -> (value, valid)."""
    f = np.asarray(sdf, np.float64)
    xy = np.asarray(xy, np.float64)
    u = (xy[..., 0] - E_X0) / E_RES - 0.5
    v = (E_Y0 - xy[..., 1]) / E_RES - 0.5
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= E_H - 1) & (v >= 0) & (v <= E_W - 1)
    u, v = np.where(valid, u, 0.0), np.where(valid, v, 0.0)
    i0 = np.clip(np.floor(u), 0, E_H - 2).astype(np.int64)
    j0 = np.clip(np.floor(v), 0, E_W - 2).astype(np.int64)
    a, b = u - i0, v - j0
    val = (f[i0, j0] * (1 - a) * (1 - b) + f[i0 + 1, j0] * a * (1 - b)
           + f[i0, j0 + 1] * (1 - a) * b + f[i0 + 1, j0 + 1] * a * b)
    return np.where(valid, val, 0.0), valid


def global_to_n(xy: np.ndarray, rear_axle_xyh: Sequence[float]) -> np.ndarray:
    """Global (map) points [..., 2] -> N frame of the t0 rear axle (x0, y0, heading0)."""
    x0, y0, h0 = (float(a) for a in rear_axle_xyh)
    xy = np.asarray(xy, np.float64)
    c, s = np.cos(h0), np.sin(h0)
    dx, dy = xy[..., 0] - x0, xy[..., 1] - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy], -1)


# ----------------------------------------------------------------------------------------- construction
def drivable_union(drivable_area_map, rear_axle_xyh: Sequence[float], crop: bool = True,
                   clip: float = SDF_CLIP):
    """Union of the official DAC-layer polygons of a metric-cache ``drivable_area_map`` (PDMDrivableMap),
    transformed to N.  ``crop``: intersect with the grid box grown by clip + 2 m (the SDF inside the grid
    is unchanged by that; the cut edges are > clip away from every cell).  Returns (geometry, info)."""
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    idx = drivable_area_map.get_indices_of_map_type([SemanticMapLayer[n] for n in DAC_LAYER_NAMES])
    # same list the scorer iterates in PDMOccupancyMap.points_in_polygons
    geoms = [drivable_area_map._geometries[i] for i in idx]
    n_invalid = 0
    fixed = []
    for g in geoms:
        if not g.is_valid:
            n_invalid += 1
            g = shapely.make_valid(g)
        fixed.append(g)
    x0, y0, h0 = (float(a) for a in rear_axle_xyh)
    c, s = np.cos(h0), np.sin(h0)
    union = shapely.unary_union(fixed) if fixed else shapely.Polygon()
    union = shapely.transform(union, lambda p: np.stack([c * (p[:, 0] - x0) + s * (p[:, 1] - y0),
                                                          -s * (p[:, 0] - x0) + c * (p[:, 1] - y0)], 1))
    if crop:
        g = clip + 2.0
        union = shapely.intersection(union, shapely.box(E_X0 - g, E_Y1 - g, E_X1 + g, E_Y0 + g))
    return union, dict(n_polys=len(geoms), n_invalid=n_invalid)


def _boundary_segments(geom) -> np.ndarray:
    """All boundary segments [n, 2, 2] of a (multi)polygon (exteriors and holes)."""
    import shapely
    segs = []
    for part in shapely.get_parts(shapely.boundary(geom)):
        co = shapely.get_coordinates(part)
        if len(co) >= 2:
            segs.append(np.stack([co[:-1], co[1:]], 1))
    return np.concatenate(segs) if segs else np.zeros((0, 2, 2))


def signed_distance_exact(geom_n, xy: np.ndarray, clip: Optional[float] = None) -> np.ndarray:
    """Exact signed distance (+ inside) of N-frame points [..., 2] to a shapely region (float64).
    Used for construction (at cell centres) and for validation at arbitrary points."""
    import shapely
    xy = np.asarray(xy, np.float64)
    shp = xy.shape[:-1]
    px, py = xy.reshape(-1, 2).T
    shapely.prepare(geom_n)
    inside = shapely.contains_xy(geom_n, px, py)
    segs = _boundary_segments(geom_n)
    far = np.inf if clip is None else float(clip)
    dist = np.full(px.shape, far)
    if len(segs):
        tree = shapely.STRtree(shapely.linestrings(segs))
        pts = shapely.points(px, py)
        kw = {} if clip is None else dict(max_distance=float(clip))
        (qi, _), d = tree.query_nearest(pts, return_distance=True, all_matches=False, **kw)
        dist[qi] = np.minimum(d, far)
    return np.where(inside, dist, -dist).reshape(shp)


def rasterize_sdf(geom_n, clip: float = SDF_CLIP) -> np.ndarray:
    """Shapely region in N -> float32 SDF [320, 256] on the E grid (+ inside, clipped to +-clip)."""
    X, Y = grid_mesh()
    sd = signed_distance_exact(geom_n, np.stack([X, Y], -1), clip=clip)
    return np.clip(sd, -clip, clip).astype(np.float32)


def build_sdf(drivable_area_map, rear_axle_xyh: Sequence[float],
              clip: float = SDF_CLIP) -> Tuple[np.ndarray, Dict]:
    """Metric-cache drivable map + t0 rear axle (global x, y, heading) -> (float32 SDF [320, 256], info)."""
    union, info = drivable_union(drivable_area_map, rear_axle_xyh, crop=True, clip=clip)
    sdf = rasterize_sdf(union, clip)
    info.update(frac_inside=float((sdf > 0).mean()), n_segments=int(len(_boundary_segments(union))))
    return sdf, info


def build_sdf_from_metric_cache(mc) -> Tuple[np.ndarray, Dict]:
    """``MetricCache`` object -> (float32 SDF [320, 256], info incl. the global rear-axle pose)."""
    ra = mc.ego_state.rear_axle
    xyh = (float(ra.x), float(ra.y), float(ra.heading))
    sdf, info = build_sdf(mc.drivable_area_map, xyh)
    info["ego_xyh"] = xyh
    return sdf, info


# ----------------------------------------------------------------------------------------- storage
def sdf_path(token: str, subset: str = "navtrain", root: Union[str, Path] = SDF_ROOT) -> Path:
    """Per-token file: <root>/<subset>/<token>.npz, subset in {navtrain (stage-T train+dev pool), navtest}."""
    return Path(root) / subset / f"{token}.npz"


def save_sdf(path: Union[str, Path], sdf: np.ndarray, token: str = "", ego_xyh=(np.nan,) * 3,
             info: Optional[Dict] = None) -> int:
    """Atomic compressed save (float16 field + provenance); returns the file size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = info or {}
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        tmp, sdf=np.asarray(sdf, np.float32).astype(np.float16), version=np.array(SDF_VERSION),
        token=np.array(token), grid=np.array([E_X0, E_Y0, E_RES, E_H, E_W, SDF_CLIP], np.float64),
        ego_xyh=np.asarray(ego_xyh, np.float64), layers=np.array(",".join(DAC_LAYER_NAMES)),
        n_polys=np.int32(info.get("n_polys", -1)), n_invalid=np.int32(info.get("n_invalid", -1)))
    os.replace(tmp, path)
    return path.stat().st_size


def load_sdf(path: Union[str, Path], check: bool = True) -> np.ndarray:
    """float16 [320, 256] field from a ``save_sdf`` file (checks version and grid unless check=False)."""
    with np.load(path) as z:
        sdf = z["sdf"]
        if check:
            if str(z["version"]) != SDF_VERSION:
                raise ValueError(f"{path}: SDF version {z['version']} != {SDF_VERSION}")
            if not np.array_equal(z["grid"], np.array([E_X0, E_Y0, E_RES, E_H, E_W, SDF_CLIP])):
                raise ValueError(f"{path}: grid {z['grid']} does not match the E grid")
    if sdf.shape != (E_H, E_W) or sdf.dtype != np.float16:
        raise ValueError(f"{path}: bad field {sdf.shape} {sdf.dtype}")
    return sdf
