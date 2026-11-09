#!/usr/bin/env python3
"""
pylidar_tls_canopy

Pulse geometry for drone lidar (DLS) LAZ flightlines

John Armston
University of Maryland
November 2026
"""

import os
import types
import numpy as np
import laspy
import rasterio as rio
from numba import njit


QA_INVALID = 0
QA_ANCHOR = 1
QA_INTERP = 2
QA_EXTRAP = 3
QA_BORROW = 4
QA_NAMES = {QA_INVALID: 'invalid', QA_ANCHOR: 'anchor', QA_INTERP: 'interpolated',
            QA_EXTRAP: 'extrapolated', QA_BORROW: 'borrowed'}

RECORD_DTYPE = np.dtype([('X', '<i4'), ('Y', '<i4'), ('Z', '<i4'), ('t', '<f8'),
                         ('rn', 'u1'), ('cls', 'u1'), ('ch', 'u1'), ('sa', '<i2')])
SCAN_ANGLE_SCALE = 0.006


class FlightlineProcessor:
    """
    Pulse geometry for one LAZ flightline, cropped to a voxel grid.

    Returns are read within zmin - zbelow and ztop + zabove, and within the grid
    bounds expanded horizontally by (ztop + zabove - zmin + zbelow) *
    tan(max_zenith) + margin, which contains every return of a pulse with a
    return in, or above, the grid at a view zenith <= max_zenith. ztop is the
    canopy top elevation (default: grid zmax); returns above ztop + zabove are
    treated as noise, and zmin - zbelow must lie below the terrain within the
    expanded area.

    Ground returns are z < DTM + ground_threshold and returns below
    DTM - noise_threshold are removed when dtm_filename is given; otherwise
    ground is classification == ground_class and noise classes are removed.

    Pulses are grouped by (scanner_channel, gps_time). Beam directions are the
    unit vector from first to last return for pulses with >= 2 returns, and are
    interpolated in time within contiguous scan-line runs for single-return
    pulses. Runs without multi-return pulses take the direction at the nearest
    scan angle in neighbouring runs. Pulses with a view zenith > max_zenith are
    flagged invalid. With assume_nadir, valid pulses are given a vertical
    direction (0, 0, -1), as in workflows that treat every pulse as nadir.

    process() holds the cropped flightline in memory (about 50 bytes per
    return). iter_buckets() spills cropped returns to gps_time buckets and
    yields one bucket at a time, so memory scales with the bucket size.

    Pulse set attributes
    --------------------
    x, y, z : float64 (n_returns,) sorted by pulse then return number
    gnd : bool (n_returns,)
    start : int64 (n_pulses,) index of the first return of each pulse
    count : uint8 (n_pulses,) number of returns
    ux, uy, uz : float32 (n_pulses,) unit beam direction (away from the sensor)
    qa : uint8 (n_pulses,) direction source (QA_* constants)
    time : float64 (n_pulses,) gps_time
    core : bool (n_pulses,) pulses inside the bucket (iter_buckets only)
    summary : dict of counts
    """

    NOISE_CLASSES = (7, 18)

    def __init__(self, path, voxel_bounds, max_zenith=70.0, margin=5.0, ztop=None,
                 zabove=1.0, zbelow=5.0, dtm_filename=None, ground_threshold=0.25,
                 noise_threshold=1.0, ground_class=2, min_separation=0.1, gap_factor=50.0,
                 max_borrow_runs=4, assume_nadir=False, chunk_size=2_000_000):
        self.path = path
        self.voxel_bounds = np.asarray(voxel_bounds, dtype=np.float64)
        self.max_zenith = float(max_zenith)
        self.margin = float(margin)
        self.ztop = self.voxel_bounds[5] if ztop is None else float(ztop)
        self.zabove = float(zabove)
        self.zbelow = float(zbelow)
        self.dtm_filename = dtm_filename
        self.ground_threshold = float(ground_threshold)
        self.noise_threshold = float(noise_threshold)
        self.ground_class = int(ground_class)
        self.min_separation = float(min_separation)
        self.gap_factor = float(gap_factor)
        self.max_borrow_runs = int(max_borrow_runs)
        self.assume_nadir = bool(assume_nadir)
        self.chunk_size = int(chunk_size)
        self.summary = {}
        self._dtm = None

    @property
    def crop_bounds(self):
        """(xmin, ymin, zmin, xmax, ymax, zmax) of the returns that are read."""
        xmin, ymin, zmin, xmax, ymax, zmax = self.voxel_bounds
        z0 = zmin - self.zbelow
        z1 = self.ztop + self.zabove
        b = (z1 - z0) * np.tan(np.radians(self.max_zenith)) + self.margin
        return np.array([xmin - b, ymin - b, z0, xmax + b, ymax + b, z1])

    def process(self):
        """Read the cropped flightline and derive pulse geometry in memory."""
        parts = list(self._iter_records())
        if not parts:
            raise ValueError(f'No returns within the crop bounds of {self.path}')
        ps = self._build(np.concatenate(parts))
        self.__dict__.update({k: v for k, v in vars(ps).items() if k != 'summary'})
        self.summary.update(ps.summary)
        return self

    def iter_buckets(self, spill_dir, bucket_seconds=2.0):
        """
        Yield pulse sets for consecutive gps_time buckets of bucket_seconds.
        Each set includes the neighbouring buckets as context; core flags the
        pulses that belong to the bucket. Spill files are removed afterwards.
        """
        os.makedirs(spill_dir, exist_ok=True)
        tag = f'{os.path.splitext(os.path.basename(self.path))[0]}_{os.getpid()}_{id(self)}'
        files = {}
        try:
            for rec in self._iter_records():
                b = np.floor(rec['t'] / bucket_seconds).astype(np.int64)
                order = np.argsort(b, kind='stable')
                b = b[order]
                rec = rec[order]
                ub, i0 = np.unique(b, return_index=True)
                i1 = np.append(i0[1:], b.shape[0])
                for k in range(ub.shape[0]):
                    fn = files.setdefault(int(ub[k]), os.path.join(spill_dir, f'{tag}_{int(ub[k])}.bin'))
                    with open(fn, 'ab') as f:
                        rec[i0[k]:i1[k]].tofile(f)
            for bi in sorted(files):
                parts = [np.fromfile(files[j], dtype=RECORD_DTYPE)
                         for j in (bi - 1, bi, bi + 1) if j in files]
                t0 = bi * bucket_seconds
                ps = self._build(np.concatenate(parts), core_window=(t0, t0 + bucket_seconds))
                for k, v in ps.summary.items():
                    self.summary[k] = self.summary.get(k, 0) + v
                yield ps
        finally:
            for fn in files.values():
                if os.path.exists(fn):
                    os.remove(fn)

    def _iter_records(self):
        """Yield the cropped returns of each chunk as RECORD_DTYPE arrays."""
        cb = self.crop_bounds
        n_read = 0
        n_keep = 0
        with laspy.open(self.path, laz_backend=_laz_backend()) as f:
            hdr = f.header
            self._scale = np.asarray(hdr.scale, dtype=np.float64)
            self._offset = np.asarray(hdr.offset, dtype=np.float64)
            dims = set(hdr.point_format.dimension_names)
            has_channel = 'scanner_channel' in dims
            sa_name = 'scan_angle' if 'scan_angle' in dims else 'scan_angle_rank'
            sa_mult = 1.0 if sa_name == 'scan_angle' else 1.0 / SCAN_ANGLE_SCALE
            lim = [(cb[i] - self._offset[i]) / self._scale[i] for i in range(3)]
            lim += [(cb[i + 3] - self._offset[i]) / self._scale[i] for i in range(3)]
            for chunk in f.chunk_iterator(self.chunk_size):
                n_read += len(chunk)
                X = np.asarray(chunk.X)
                Y = np.asarray(chunk.Y)
                Z = np.asarray(chunk.Z)
                m = _crop_mask(X, Y, Z, *lim)
                n = int(np.count_nonzero(m))
                if n == 0:
                    continue
                rec = np.empty(n, dtype=RECORD_DTYPE)
                rec['X'] = X[m]
                rec['Y'] = Y[m]
                rec['Z'] = Z[m]
                rec['t'] = np.asarray(chunk.gps_time)[m]
                rec['rn'] = np.asarray(chunk.return_number)[m]
                rec['cls'] = np.asarray(chunk.classification)[m]
                rec['ch'] = np.asarray(chunk.scanner_channel)[m] if has_channel else 0
                rec['sa'] = np.round(np.asarray(chunk[sa_name])[m] * sa_mult)
                n_keep += n
                yield rec
        self.summary.update(n_points_read=int(n_read), n_points_kept=int(n_keep))

    def _load_dtm(self):
        if self._dtm is None:
            with rio.open(self.dtm_filename) as src:
                dtm = src.read(1).astype(np.float64)
                if src.nodata is not None:
                    dtm[dtm == src.nodata] = np.nan
                self._dtm = (dtm, src.bounds.left, src.bounds.top, src.res[0])
        return self._dtm

    def _build(self, rec, core_window=None):
        """Sorted, classified pulses with beam directions from cropped returns."""
        rec = rec[np.lexsort((rec['rn'], rec['t'], rec['ch']))]
        ps = types.SimpleNamespace()
        x = rec['X'] * self._scale[0] + self._offset[0]
        y = rec['Y'] * self._scale[1] + self._offset[1]
        z = rec['Z'] * self._scale[2] + self._offset[2]
        if self.dtm_filename is not None:
            zg = sample_raster(x, y, *self._load_dtm())
            noise = z < zg - self.noise_threshold
            gnd = z < zg + self.ground_threshold
        else:
            noise = np.isin(rec['cls'], self.NOISE_CLASSES)
            gnd = rec['cls'] == self.ground_class
        keep = ~noise
        ps.x, ps.y, ps.z, ps.gnd = x[keep], y[keep], z[keep], gnd[keep]
        t = rec['t'][keep]
        ch = rec['ch'][keep]
        sa = rec['sa'][keep].astype(np.float32) * np.float32(SCAN_ANGLE_SCALE)
        t_noise = rec['t'][noise]
        del rec, x, y, z, gnd

        start = _pulse_starts(t, ch)
        count = np.diff(np.append(start, t.shape[0]))
        ps.start = start
        ps.count = np.minimum(count, 255).astype(np.uint8)
        ps.time = t[start]
        p_ch = ch[start]
        p_sa = sa[start]

        ux, uy, uz, qa = _anchor_directions(ps.x, ps.y, ps.z, ps.start, ps.count,
                                            self.min_separation)
        dt = np.diff(ps.time)
        dt = dt[dt > 0]
        gap = self.gap_factor * (np.median(dt) if dt.size else 1.0)
        run_start = _run_starts(ps.time, p_ch, gap)
        _interpolate_runs(ps.time, run_start, ux, uy, uz, qa)
        _borrow_directions(p_sa, p_ch, run_start, ux, uy, uz, qa, self.max_borrow_runs)
        _limit_zenith(uz, qa, np.cos(np.radians(self.max_zenith)))
        if self.assume_nadir:
            valid = qa != QA_INVALID
            ux[valid] = 0
            uy[valid] = 0
            uz[valid] = -1
        ps.ux, ps.uy, ps.uz, ps.qa = ux, uy, uz, qa

        if core_window is None:
            ps.core = np.ones(ps.start.shape[0], dtype=bool)
            core_noise = t_noise.shape[0]
        else:
            ps.core = (ps.time >= core_window[0]) & (ps.time < core_window[1])
            core_noise = int(np.count_nonzero((t_noise >= core_window[0]) & (t_noise < core_window[1])))
        c = ps.core
        n_core_ret = int(ps.count[c].sum(dtype=np.int64))
        ret_core = np.repeat(c, ps.count)
        ps.summary = dict(n_pulses=int(np.count_nonzero(c)), n_returns=n_core_ret,
                          n_noise=int(core_noise), n_ground=int(np.count_nonzero(ps.gnd[ret_core])),
                          n_runs=int(np.count_nonzero(c[run_start])))
        ps.summary.update({f'n_{QA_NAMES[k]}': int(np.count_nonzero(qa[c] == k)) for k in QA_NAMES})
        return ps


def _laz_backend():
    available = laspy.LazBackend.detect_available()
    if laspy.LazBackend.LazrsParallel in available:
        return laspy.LazBackend.LazrsParallel
    return None


@njit(cache=True)
def _crop_mask(X, Y, Z, x0, y0, z0, x1, y1, z1):
    n = X.shape[0]
    m = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if X[i] >= x0 and X[i] <= x1 and Y[i] >= y0 and Y[i] <= y1 and Z[i] >= z0 and Z[i] <= z1:
            m[i] = True
    return m


@njit(cache=True)
def _pulse_starts(t, ch):
    n = t.shape[0]
    flag = np.zeros(n, dtype=np.bool_)
    flag[0] = True
    for i in range(1, n):
        if t[i] != t[i - 1] or ch[i] != ch[i - 1]:
            flag[i] = True
    return np.flatnonzero(flag)


@njit(cache=True)
def _anchor_directions(x, y, z, start, count, min_separation):
    n = start.shape[0]
    ux = np.full(n, np.nan, dtype=np.float32)
    uy = np.full(n, np.nan, dtype=np.float32)
    uz = np.full(n, np.nan, dtype=np.float32)
    qa = np.zeros(n, dtype=np.uint8)
    for p in range(n):
        if count[p] < 2:
            continue
        s = start[p]
        e = s + count[p] - 1
        vx = x[e] - x[s]
        vy = y[e] - y[s]
        vz = z[e] - z[s]
        r = np.sqrt(vx * vx + vy * vy + vz * vz)
        if r >= min_separation:
            ux[p] = vx / r
            uy[p] = vy / r
            uz[p] = vz / r
            qa[p] = QA_ANCHOR
    return ux, uy, uz, qa


@njit(cache=True)
def _run_starts(t, ch, gap):
    n = t.shape[0]
    flag = np.zeros(n, dtype=np.bool_)
    flag[0] = True
    for i in range(1, n):
        if ch[i] != ch[i - 1] or t[i] - t[i - 1] > gap:
            flag[i] = True
    return np.flatnonzero(flag)


@njit(cache=True)
def _normalize_inplace(ux, uy, uz, p):
    r = np.sqrt(ux[p] * ux[p] + uy[p] * uy[p] + uz[p] * uz[p])
    if r > 0:
        ux[p] /= r
        uy[p] /= r
        uz[p] /= r


@njit(cache=True)
def _interpolate_runs(t, run_start, ux, uy, uz, qa):
    """
    Linear interpolation in time between anchors within each run; linear
    extrapolation from the two nearest anchors outside the anchored span.
    """
    n = t.shape[0]
    nrun = run_start.shape[0]
    maxlen = 0
    for r in range(nrun):
        e = run_start[r + 1] if r + 1 < nrun else n
        maxlen = max(maxlen, e - run_start[r])
    anchors = np.empty(maxlen, dtype=np.int64)
    for r in range(nrun):
        s = run_start[r]
        e = run_start[r + 1] if r + 1 < nrun else n
        m = 0
        for p in range(s, e):
            if qa[p] == QA_ANCHOR:
                anchors[m] = p
                m += 1
        if m == 0:
            continue
        j = 0
        for p in range(s, e):
            if j < m and anchors[j] == p:
                j += 1
                continue
            if j == 0:
                a, b, flag = anchors[0], anchors[min(1, m - 1)], QA_EXTRAP
            elif j == m:
                a, b, flag = anchors[max(m - 2, 0)], anchors[m - 1], QA_EXTRAP
            else:
                a, b, flag = anchors[j - 1], anchors[j], QA_INTERP
            dt = t[b] - t[a]
            f = (t[p] - t[a]) / dt if dt != 0 else 0.0
            ux[p] = ux[a] + f * (ux[b] - ux[a])
            uy[p] = uy[a] + f * (uy[b] - uy[a])
            uz[p] = uz[a] + f * (uz[b] - uz[a])
            _normalize_inplace(ux, uy, uz, p)
            qa[p] = flag


@njit(cache=True)
def _borrow_directions(sa, ch, run_start, ux, uy, uz, qa, max_runs):
    """
    Directions for runs without anchors: the pulse with the nearest scan angle in
    each of the closest anchored runs before and after (same channel), averaged.
    """
    n = sa.shape[0]
    nrun = run_start.shape[0]
    has_anchor = np.zeros(nrun, dtype=np.bool_)
    for r in range(nrun):
        s = run_start[r]
        e = run_start[r + 1] if r + 1 < nrun else n
        for p in range(s, e):
            if qa[p] == QA_ANCHOR:
                has_anchor[r] = True
                break
    for r in range(nrun):
        if has_anchor[r]:
            continue
        s = run_start[r]
        e = run_start[r + 1] if r + 1 < nrun else n
        cands = np.full(2, -1, dtype=np.int64)
        k = 0
        for step in (-1, 1):
            rr = r + step
            nchk = 0
            while rr >= 0 and rr < nrun and nchk < max_runs:
                if ch[run_start[rr]] != ch[s]:
                    break
                if has_anchor[rr]:
                    cands[k] = rr
                    break
                rr += step
                nchk += 1
            k += 1
        for p in range(s, e):
            sx = 0.0
            sy = 0.0
            sz = 0.0
            m = 0
            for c in range(2):
                rr = cands[c]
                if rr < 0:
                    continue
                cs = run_start[rr]
                ce = run_start[rr + 1] if rr + 1 < nrun else n
                best = -1
                bestd = np.inf
                for q in range(cs, ce):
                    if qa[q] == QA_ANCHOR or qa[q] == QA_INTERP or qa[q] == QA_EXTRAP:
                        d = abs(sa[q] - sa[p])
                        if d < bestd:
                            bestd = d
                            best = q
                if best >= 0:
                    sx += ux[best]
                    sy += uy[best]
                    sz += uz[best]
                    m += 1
            if m > 0:
                ux[p] = sx
                uy[p] = sy
                uz[p] = sz
                _normalize_inplace(ux, uy, uz, p)
                qa[p] = QA_BORROW


@njit(cache=True)
def _limit_zenith(uz, qa, min_abs_cos):
    for p in range(uz.shape[0]):
        if qa[p] != QA_INVALID and abs(uz[p]) < min_abs_cos:
            qa[p] = QA_INVALID


@njit(cache=True)
def sample_raster(x, y, data, xmin, ymax, res):
    """Nearest-cell raster values at (x, y); NaN outside the raster."""
    out = np.full(x.shape[0], np.nan)
    nrow, ncol = data.shape
    for i in range(x.shape[0]):
        col = int(np.floor((x[i] - xmin) / res))
        row = int(np.floor((ymax - y[i]) / res))
        if row >= 0 and col >= 0 and row < nrow and col < ncol:
            out[i] = data[row, col]
    return out
