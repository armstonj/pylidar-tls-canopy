#!/usr/bin/env python3
"""
pylidar_tls_canopy

Code for voxelization

John Armston
University of Maryland
December 2022
"""

import os
import json
import numpy as np
import laspy

import multiprocessing as mp
from tqdm.auto import tqdm

from numba import njit

import rasterio as rio
from rasterio.enums import Resampling
from affine import Affine

from . import riegl_io
from . import RIO_DEFAULT_PROFILE
from .flightline import FlightlineProcessor, sample_raster


# Accumulator rows of the traversal kernel
ACC_HITS = 0
ACC_MISS = 1
ACC_OCCL = 2
ACC_GND = 3
ACC_NBEAM = 4
ACC_PATH = 5
ACC_PCOS = 6
ACC_PSIN = 7
ACC_PXT = 8
ACC_PTHETA = 9
ACC_PUX = 10
ACC_PUY = 11
ACC_PHIT = 12
ACC_NENTER = 13
ACC_HITS2 = 14
ACC_PEFF = 15
ACC_PHITEFF = 16
NACC = 17

MAX_XTHETA = 100.0


class VoxelModel:
    """
    Voxel Pgap model from multiple scan positions or flightlines.

    The configuration file holds bounds, resolution, nx, ny, nz, nodata, dtm and
    positions, where positions maps a name to the grid filenames written by
    TLSVoxelGrid or DLSVoxelGrid. Grids are read north-up (row 0 = ymax).
    """
    def __init__(self, config_file, dtm_filename=None):
        self.config_file = config_file
        self.load_config()
        if dtm_filename is not None:
            self.dtm = dtm_filename
        if getattr(self, 'dtm', None) is not None:
            self.load_dtm()

    def load_config(self):
        """Load the VoxelModel configuration file."""
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        for k in config:
            setattr(self, k, config[k])
        self.npos = len(self.positions)

    def load_dtm(self):
        """Voxel grid flagging voxels whose centre lies below the DTM (1)."""
        with rio.open(self.dtm, 'r') as src:
            dtm_data = src.read(1)
            dtm_xmin = src.bounds.left
            dtm_ymax = src.bounds.top
            dtm_res = src.res[0]
        self.dtm_voxelgrid = create_ground_voxel_grid(self.nx, self.ny, self.nz,
            self.bounds[0], self.bounds[4], self.bounds[2], self.resolution,
            dtm_data, dtm_xmin, dtm_ymax, dtm_res)

    def read_voxelgrids(self, z=0, names=('pgap', 'zeni', 'vwts')):
        """Read layer z of the named grids for all positions."""
        sh = (self.npos, self.ny, self.nx)
        self.voxelgrids = {}
        for k in names:
            self.voxelgrids[k] = np.empty(sh, dtype=np.float32)
            for i, p in enumerate(self.positions):
                with rio.open(self.positions[p][k], 'r') as src:
                    self.voxelgrids[k][i] = src.read(z + 1)

    def _has_grid(self, name):
        return all(name in self.positions[p] for p in self.positions)

    def run_linear_model(self, min_n=3, weights=True):
        """
        Jupp et al. (2009) linear model per voxel:
        -ln Pgap(theta) = paiv + (2/pi) tan(theta) paih

        paiv is the vertically projected plant area (intercept, theta = 0) and
        paih the horizontally projected plant area (slope).
        """
        sh = (self.nz, self.ny, self.nx)
        paiv = np.empty(sh, dtype=np.float32)
        paih = np.empty(sh, dtype=np.float32)
        nscans = np.empty(sh, dtype=np.uint8)
        xname = 'xtheta' if self._has_grid('xtheta') else 'zeni'
        for i in range(self.nz):
            self.read_voxelgrids(z=i, names=('pgap', xname, 'vwts'))
            x = self.voxelgrids[xname]
            if xname == 'zeni':
                x = np.where(x != self.nodata, np.abs(2 * np.tan(x) / np.pi), self.nodata).astype(np.float32)
            w = self.voxelgrids['vwts'] if weights else np.ones_like(self.voxelgrids['vwts'])
            nscans[i] = np.sum(self.voxelgrids['vwts'] > 0, axis=0, dtype=np.uint8)
            paiv[i], paih[i] = run_linear_model_numba(x, self.voxelgrids['pgap'], w,
                null=self.nodata, min_n=min_n)
        return paiv, paih, nscans

    def run_attenuation_model(self, min_n=2, min_xtheta_range=0.2, max_iter=500, tol=1e-8,
                              estimator='mle', effective=False, ci=None):
        """
        Per-voxel attenuation model lambda(theta) = a |cos(theta)| + b (2/pi) sin(theta)
        fitted by Poisson maximum likelihood to the hits and free path length
        sums of each position (a, b >= 0).

        a is the attenuation of a vertical beam (m-1), a + b the plant area
        density and degrees(arctan2(b, a)) the mean leaf angle. a and b are
        estimated where >= min_n positions observe the voxel with a spread of
        (2/pi) tan(theta) >= min_xtheta_range.

        lambda is the attenuation pooled over all positions, estimated wherever
        the voxel is observed (see pooled_attenuation for estimator, effective
        and ci). Requires the path_hit, nenter and hits2 grids for
        estimator='bias_corrected' or ci, and path_eff, path_hit_eff for effective.

        Returns
        -------
        dict of (nz, ny, nx) float32 grids: a, b, pad, mla, lambda, npos
        (and lambda_lo, lambda_hi when ci is given)
        """
        sh = (self.nz, self.ny, self.nx)
        keys = ['a', 'b', 'pad', 'mla', 'lambda'] + (['lambda_lo', 'lambda_hi'] if ci else [])
        out = {k: np.full(sh, self.nodata, dtype=np.float32) for k in keys}
        out['npos'] = np.zeros(sh, dtype=np.uint8)
        names = ['hits', 'path', 'path_cos', 'path_sin']
        if estimator != 'mle' or ci:
            names += ['path_hit', 'nenter', 'hits2']
        if effective:
            names += ['path_eff', 'path_hit_eff']
        for i in range(self.nz):
            self.read_voxelgrids(z=i, names=names)
            v = self.voxelgrids
            res = fit_attenuation_numba(v['hits'], v['path'], v['path_cos'], v['path_sin'],
                self.nodata, min_n, min_xtheta_range, max_iter, tol)
            for k, r in zip(('a', 'b', 'pad', 'mla', 'lambda', 'npos'), res):
                out[k][i] = r
            if estimator != 'mle' or ci or effective:
                total = lambda k: np.where(v[k] != self.nodata, v[k], 0).sum(axis=0, dtype=np.float64)
                lam, lo, hi = pooled_attenuation(total('hits'),
                    total('path_eff' if effective else 'path'),
                    total('path_hit_eff' if effective else 'path_hit') if estimator != 'mle' else None,
                    total('nenter') if estimator != 'mle' else None,
                    total('hits2') if ci else None, estimator=estimator, ci=ci)
                observed = out['npos'][i] > 0
                out['lambda'][i] = np.where(observed, lam, self.nodata)
                if ci:
                    out['lambda_lo'][i] = np.where(observed, lo, self.nodata)
                    out['lambda_hi'][i] = np.where(observed, hi, self.nodata)
        return out

    def run_hinge_model(self, zenith_range=(52.5, 62.5)):
        """
        Plant area density from positions viewing each voxel within zenith_range
        (degrees), where G(theta) ~ 0.5 for any leaf angle distribution:
        pad = (sum hits / sum free path) / 0.5
        """
        sh = (self.nz, self.ny, self.nx)
        pad = np.full(sh, self.nodata, dtype=np.float32)
        lo, hi = np.radians(zenith_range[0]), np.radians(zenith_range[1])
        for i in range(self.nz):
            self.read_voxelgrids(z=i, names=('hits', 'path', 'zeni'))
            h = self.voxelgrids['hits']
            p = self.voxelgrids['path']
            zen = self.voxelgrids['zeni']
            sel = (p > 0) & (p != self.nodata) & (zen >= lo) & (zen <= hi)
            hs = np.where(sel, h, 0).sum(axis=0)
            ps = np.where(sel, p, 0).sum(axis=0)
            np.divide(hs, 0.5 * ps, out=pad[i], where=ps > 0)
        return pad

    def run_occlusion_voxelgrid(self, null=-9999):
        """Fraction of beams occluded before reaching each voxel, and the beam count."""
        sh = (self.nz, self.ny, self.nx)
        voxelgrids = {}
        for k in ('occl', 'miss', 'hits'):
            voxelgrids[k] = np.zeros(sh, dtype=np.float32)
            for p in self.positions:
                with rio.open(self.positions[p][k], 'r') as src:
                    voxelgrids[k] += src.read()
        nbeam = voxelgrids['occl'] + voxelgrids['miss'] + voxelgrids['hits']
        poccl = np.full(sh, null, dtype=np.float32)
        np.divide(voxelgrids['occl'], nbeam, out=poccl, where=nbeam > 0)
        return poccl, nbeam

    def _fill(self, grid, fill):
        valid = grid != self.nodata
        out = np.where(valid, grid, 0).astype(np.float32)
        if fill == 'layer_mean':
            for i in range(grid.shape[0]):
                if valid[i].any():
                    out[i][~valid[i]] = grid[i][valid[i]].mean()
        elif fill != 'zero':
            raise ValueError(f'Unknown fill policy: {fill}')
        return out, valid

    def get_cover_profile(self, paiv, fill='zero'):
        """
        Vertical canopy cover profile from the vertically projected PAI per voxel,
        accumulated from the top layer down. Unobserved voxels take zero PAI
        (fill='zero') or the layer mean of observed voxels (fill='layer_mean').
        """
        paiv_f, _ = self._fill(paiv, fill)
        cover = 1 - np.exp(-paiv_f)
        cover_z = np.zeros_like(cover)
        cover_z[-1] = cover[-1]
        for i in range(cover.shape[0] - 2, -1, -1):
            cover_z[i] = cover_z[i + 1] + (1 - cover_z[i + 1]) * cover[i]
        return cover_z

    def get_vertical_pgap(self, lambda0, fill='zero'):
        """
        Vertical gap probability below the top of the grid to the bottom of each
        layer, exp(-sum lambda0 dz), from the vertical-beam attenuation (m-1).

        Returns
        -------
        pgap_z : (nz, ny, nx) float32
        observed : (nz, ny, nx) bool, voxels with an estimate
        """
        lam, observed = self._fill(lambda0, fill)
        tau = np.cumsum((lam * self.resolution)[::-1], axis=0)[::-1]
        return np.exp(-tau).astype(np.float32), observed


class _VoxelGridBase:
    """
    Shared voxel grid definition, traversal and output for one scan position or
    flightline. Points are stored per pulse in compressed form: return i of pulse
    p is element start[p] + i of x, y, z and gnd.
    """
    def __init__(self, profile=None, k1=0.0):
        self.profile = dict(RIO_DEFAULT_PROFILE if profile is None else profile)
        self.nodata = self.profile['nodata']
        self.k1 = float(k1)

    def _init_voxel_grid(self, bounds, voxelsize):
        bounds = np.asarray(bounds, dtype=np.float64)
        self.voxelsize = float(voxelsize)
        n = np.floor((bounds[3:6] - bounds[0:3]) / self.voxelsize + 1e-9).astype(int)
        self.nx, self.ny, self.nz = int(n[0]), int(n[1]), int(n[2])
        self.bounds = np.concatenate([bounds[0:3], bounds[0:3] + n * self.voxelsize])
        self.nvox = self.nx * self.ny * self.nz
        self.profile.update(height=self.ny, width=self.nx, count=self.nz,
            transform=Affine(self.voxelsize, 0.0, self.bounds[0], 0.0,
            -self.voxelsize, self.bounds[4]), nodata=self.nodata, interleave='band')

    def _traverse(self, origin, use_first_return):
        acc = np.zeros((NACC, self.nvox), dtype=np.float64)
        traverse_pulses(self.x, self.y, self.z, self.gnd, self.start, self.count,
            self.ux, self.uy, self.uz, np.asarray(origin, dtype=np.float64),
            use_first_return, self.bounds[0:3].copy(), self.nx, self.ny, self.nz,
            self.voxelsize, acc, self.k1)
        return acc

    def _make_grids(self, acc, save_counts):
        nd = self.nodata
        path = acc[ACC_PATH]
        nshots = acc[ACC_HITS] + acc[ACC_MISS]
        observed = path > 0

        g = {}
        g['pgap'] = np.full(self.nvox, nd)
        np.divide(acc[ACC_MISS], nshots, out=g['pgap'], where=nshots > 0)
        g['zeni'] = np.full(self.nvox, nd)
        np.divide(acc[ACC_PTHETA], path, out=g['zeni'], where=observed)
        g['xtheta'] = np.full(self.nvox, nd)
        np.divide(acc[ACC_PXT], path, out=g['xtheta'], where=observed)
        g['vwts'] = np.where(observed, path, nd)
        g['vcls'] = self.classify_voxels(acc[ACC_HITS], acc[ACC_MISS], acc[ACC_OCCL], acc[ACC_GND])
        g['hits'] = acc[ACC_HITS]
        g['path'] = path
        g['path_cos'] = acc[ACC_PCOS]
        g['path_sin'] = acc[ACC_PSIN]
        g['path_hit'] = acc[ACC_PHIT]
        g['nenter'] = acc[ACC_NENTER]
        g['hits2'] = acc[ACC_HITS2]
        if self.k1 > 0:
            g['path_eff'] = acc[ACC_PEFF]
            g['path_hit_eff'] = acc[ACC_PHITEFF]
        if save_counts:
            g['miss'] = acc[ACC_MISS]
            g['occl'] = acc[ACC_OCCL]
            g['gnd'] = acc[ACC_GND]
            g['nbeam'] = acc[ACC_NBEAM]
            g['azim'] = np.full(self.nvox, nd)
            az = np.mod(np.arctan2(acc[ACC_PUX], acc[ACC_PUY]), 2 * np.pi)
            g['azim'][observed] = az[observed]
        self.voxelgrids = g

    def classify_voxels(self, hits, miss, occl, gnd):
        """
        Class       Value   Hits    Misses   Occluded   Ground
        Observed    5       >0      >=0      >=0        >=0
        Empty       4       =0      >0       >=0        =0
        Hidden      3       =0      =0       >0         =0
        Unobserved  2       =0      =0       =0         =0
        Ground      1       =0      >=0      >=0        >0
        """
        c = np.full_like(hits, self.nodata)
        c[hits > 0] = 5
        c[(hits == 0) & (miss > 0)] = 4
        c[(hits == 0) & (miss == 0) & (occl > 0)] = 3
        c[(hits == 0) & (miss == 0) & (occl == 0)] = 2
        c[(hits == 0) & (gnd > 0)] = 1
        return c

    def write_grids(self, prefix):
        """Write each grid to {prefix}_{name}.tif (north-up, one band per layer)."""
        self.filenames = {}
        shape = (self.nz, self.ny, self.nx)
        for k, grid in self.voxelgrids.items():
            self.filenames[k] = f'{prefix}_{k}.tif'
            with rio.Env():
                with rio.open(self.filenames[k], 'w', **self.profile) as dst:
                    dst.write(grid.reshape(shape)[:, ::-1, :].astype(self.profile['dtype']))
                    levels = overview_levels(self.nx, self.ny)
                    if levels:
                        dst.build_overviews(levels, Resampling.average)
                    for i in range(self.nz):
                        dst.set_band_description(i + 1, f'{self.bounds[2] + i * self.voxelsize:.2f}m')


class TLSVoxelGrid(_VoxelGridBase):
    """
    Voxel grid of a RIEGL TLS scan position.
    """
    def __init__(self, dtm_filename=None, profile=None, k1=0.0):
        super().__init__(profile=profile, k1=k1)
        self.dtm_filename = dtm_filename

    def classify_ground(self, thres=0.25):
        """Flag returns below the DTM + thres as ground."""
        if self.dtm_filename is None:
            self.gnd = np.zeros(self.x.shape[0], dtype=bool)
            return
        with rio.open(self.dtm_filename, 'r') as src:
            dtm = src.read(1).astype(np.float64)
            if src.nodata is not None:
                dtm[dtm == src.nodata] = np.nan
            zg = sample_raster(self.x, self.y, dtm, src.bounds.left, src.bounds.top, src.res[0])
        self.gnd = self.z < zg + thres

    def add_riegl_scan_position(self, rxp_file, transform_file, rdbx_file=None, query_str=None):
        """Add a scan position from RXP (and optionally MTA-corrected RDBX) files."""
        with riegl_io.RXPFile(rxp_file, transform_file=transform_file, query_str=query_str) as rxp:
            self.ux = rxp.get_data('beam_direction_x').astype(np.float32)
            self.uy = rxp.get_data('beam_direction_y').astype(np.float32)
            self.uz = rxp.get_data('beam_direction_z').astype(np.float32)
            t = rxp.transform
            self.origin = np.zeros(3) if t is None else np.array(t[3, 0:3], dtype=np.float64)
            npulse = self.ux.shape[0]
            if rdbx_file:
                scanline = rxp.get_data('scanline')
                scanline_idx = rxp.get_data('scanline_idx')
                with riegl_io.RDBFile(rdbx_file, transform_file=transform_file, query_str=query_str) as rdb:
                    m = np.int64(np.max(scanline_idx)) + 1
                    pid = scanline.astype(np.int64) * m + scanline_idx
                    sort_idx = np.argsort(pid)
                    rid = rdb.get_data('scanline').astype(np.int64) * m + rdb.get_data('scanline_idx')
                    pos = np.searchsorted(pid, rid, sorter=sort_idx)
                    pos = np.clip(pos, 0, npulse - 1)
                    pulse_idx = sort_idx[pos]
                    valid = pid[pulse_idx] == rid
                    self._set_points(pulse_idx[valid], rdb.get_data('target_index')[valid],
                        rdb.get_data('x')[valid], rdb.get_data('y')[valid], rdb.get_data('z')[valid], npulse)
            else:
                count = rxp.get_data('target_count')
                pulse_idx = np.repeat(np.arange(npulse, dtype=np.int64), count)
                self._set_points(pulse_idx, rxp.get_data('target_index'), rxp.get_data('x'),
                    rxp.get_data('y'), rxp.get_data('z'), npulse)
        self.classify_ground()

    def _set_points(self, pulse_idx, target_index, x, y, z, npulse):
        order = np.lexsort((target_index, pulse_idx))
        self.x = np.asarray(x, dtype=np.float64)[order]
        self.y = np.asarray(y, dtype=np.float64)[order]
        self.z = np.asarray(z, dtype=np.float64)[order]
        count = np.bincount(pulse_idx, minlength=npulse)
        self.count = np.minimum(count, 255).astype(np.uint8)
        self.start = (np.cumsum(count) - count).astype(np.int64)

    def voxelize_scan(self, bounds, voxelsize, save_counts=True):
        """Voxelize the scan on the grid defined by bounds and voxelsize."""
        self._init_voxel_grid(bounds, voxelsize)
        acc = self._traverse(self.origin, False)
        self._make_grids(acc, save_counts)


class DLSVoxelGrid(_VoxelGridBase):
    """
    Voxel grid of a drone lidar (DLS) LAZ flightline.

    Usage
    -----
        grid = DLSVoxelGrid()
        grid.voxelize_flightline('flight.laz', bounds, 0.5, spill_dir='/scratch/dir',
                                 dtm_filename='dtm.tif')
        grid.write_grids('output/flight')

    or, holding the cropped flightline in memory:

        grid.add_laz_flightline('flight.laz', bounds, dtm_filename='dtm.tif')
        grid.voxelize_scan(bounds, 0.5)
    """
    ZENITH_EDGES = np.arange(0, 91, 1.0)
    AZIMUTH_EDGES = np.arange(0, 361, 10.0)

    def add_laz_flightline(self, laz_file, voxel_bounds, **kwargs):
        """Load the pulses of a flightline that can cross voxel_bounds (see FlightlineProcessor)."""
        proc = FlightlineProcessor(laz_file, voxel_bounds, **kwargs).process()
        self.x, self.y, self.z, self.gnd = proc.x, proc.y, proc.z, proc.gnd
        self.start, self.count = proc.start, proc.count
        self.ux, self.uy, self.uz = proc.ux, proc.uy, proc.uz
        self.qa = proc.qa
        self.time = proc.time
        self.summary = proc.summary

    def voxelize_scan(self, bounds, voxelsize, save_counts=True):
        """Voxelize the loaded flightline on the grid defined by bounds and voxelsize."""
        self._init_voxel_grid(bounds, voxelsize)
        acc = np.zeros((NACC, self.nvox), dtype=np.float64)
        self.angle_hist = np.zeros((self.ZENITH_EDGES.size - 1, self.AZIMUTH_EDGES.size - 1), dtype=np.int64)
        self._accumulate(self, self.qa > 0, acc)
        self._make_grids(acc, save_counts)

    def voxelize_flightline(self, laz_file, bounds, voxelsize, spill_dir, bucket_seconds=2.0,
                            save_counts=True, **kwargs):
        """
        Voxelize a flightline one gps_time bucket at a time (see
        FlightlineProcessor.iter_buckets); spill_dir holds the temporary bucket files.
        """
        self._init_voxel_grid(bounds, voxelsize)
        acc = np.zeros((NACC, self.nvox), dtype=np.float64)
        self.angle_hist = np.zeros((self.ZENITH_EDGES.size - 1, self.AZIMUTH_EDGES.size - 1), dtype=np.int64)
        proc = FlightlineProcessor(laz_file, bounds, **kwargs)
        for ps in proc.iter_buckets(spill_dir, bucket_seconds=bucket_seconds):
            self._accumulate(ps, (ps.qa > 0) & ps.core, acc)
        self.summary = proc.summary
        self._make_grids(acc, save_counts)

    def _accumulate(self, ps, use, acc):
        ux = np.where(use, ps.ux, np.float32(np.nan))
        traverse_pulses(ps.x, ps.y, ps.z, ps.gnd, ps.start, ps.count, ux, ps.uy, ps.uz,
            np.zeros(3), True, self.bounds[0:3].copy(), self.nx, self.ny, self.nz,
            self.voxelsize, acc, self.k1)
        cross = rays_cross_grid(ps.x, ps.y, ps.z, ps.start, ps.count, ux, ps.uy, ps.uz,
            self.bounds)
        zen = np.degrees(np.arccos(np.minimum(np.abs(ps.uz[cross]), 1)))
        azi = np.mod(np.degrees(np.arctan2(ps.ux[cross], ps.uy[cross])), 360)
        self.angle_hist += np.histogram2d(zen, azi, bins=(self.ZENITH_EDGES, self.AZIMUTH_EDGES))[0].astype(np.int64)

    def write_summary(self, prefix):
        """Write the flightline summary and zenith/azimuth histogram of grid-crossing pulses."""
        rep = dict(summary=self.summary, bounds=self.bounds.tolist(), voxelsize=self.voxelsize,
                   zenith_edges=self.ZENITH_EDGES.tolist(), azimuth_edges=self.AZIMUTH_EDGES.tolist(),
                   angle_hist=self.angle_hist.tolist())
        fn = f'{prefix}_summary.json'
        with open(fn, 'w') as f:
            json.dump(rep, f)
        return fn


@njit(cache=True)
def create_ground_voxel_grid(nx, ny, nz, xmin, ymax, zmin, binsize,
    dem, dem_xmin, dem_ymax, dem_binsize):
    """Flag voxels (north-up) whose centre lies below the DEM (1)."""
    outdata = np.zeros((nz, ny, nx), dtype=np.float32)
    for z in range(nz):
        zc = zmin + z * binsize + binsize / 2
        for y in range(ny):
            yc = ymax - (y * binsize + binsize / 2)
            dem_row = int((dem_ymax - yc) // dem_binsize)
            for x in range(nx):
                xc = xmin + (x * binsize + binsize / 2)
                dem_col = int((xc - dem_xmin) // dem_binsize)
                if (dem_row >= 0) & (dem_col >= 0) & (dem_row < dem.shape[0]) & (dem_col < dem.shape[1]):
                    if zc < dem[dem_row, dem_col]:
                        outdata[z, y, x] = 1
    return outdata


@njit(cache=True)
def _slab(o, d, hi, tin, tout):
    if d != 0.0:
        t1 = -o / d
        t2 = (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tin:
            tin = t1
        if t2 < tout:
            tout = t2
    elif o < 0.0 or o > hi:
        tout = -np.inf
    return tin, tout


@njit(cache=True)
def _axis_init(p, d, i, vs, tin):
    if d > 0:
        return 1, tin + ((i + 1) * vs - p) / d, vs / d
    elif d < 0:
        return -1, tin + (i * vs - p) / d, -vs / d
    return 0, np.inf, np.inf


@njit(cache=True)
def _effective_path(z, k1):
    if k1 <= 0.0:
        return z
    return -np.log(1.0 - min(k1 * z, 0.999999)) / k1


@njit(cache=True)
def traverse_pulses(x, y, z, gnd, start, count, ux, uy, uz, origin, use_first_return,
                    bmin, nx, ny, nz, vs, acc, k1=0.0):
    """
    Accumulate per-voxel beam statistics along each pulse (Amanatides and Woo, 1987).

    Each pulse is a ray with unit direction u from origin, or from its first
    return when use_first_return is True (the ray then extends upstream of the
    first return). Returns are placed by their distance along the ray, each
    intercepting 1 / count of the beam; returns before grid entry reduce the
    beam entering the grid. Pulses with a NaN direction are skipped.

    acc rows (per voxel):
      hits    canopy-intercepted beam weight
      miss    beam weight leaving the voxel
      occl    beam weight intercepted before the voxel
      gnd     ground-intercepted beam weight
      nbeam   number of pulses entering the voxel
      path    free path length (beam weight x distance travelled in the voxel)
      path_cos, path_sin, path_xtheta, path_theta, path_ux, path_uy
              free path weighted |cos(theta)|, (2/pi) sin(theta), (2/pi) tan(theta),
              theta, ux and uy, with theta = arccos(|uz|)
      path_hit     free path of the canopy-intercepted beam weight
      nenter       number of pulses entering the voxel with beam weight > 0
      hits2        sum of squared canopy hit weights
      path_eff, path_hit_eff
                   path and path_hit with each free path z replaced by the
                   effective free path -ln(1 - k1 z) / k1 (Pimont et al., 2018);
                   equal to path and path_hit when k1 = 0
    """
    gx = nx * vs
    gy = ny * vs
    gz = nz * vs
    two_pi = 2.0 / np.pi
    for p in range(start.shape[0]):
        dx = np.float64(ux[p])
        dy = np.float64(uy[p])
        dz = np.float64(uz[p])
        if dx != dx:
            continue
        n = count[p]
        s = start[p]
        if use_first_return:
            if n == 0:
                continue
            ox = x[s] - bmin[0]
            oy = y[s] - bmin[1]
            oz = z[s] - bmin[2]
            tin = -np.inf
        else:
            ox = origin[0] - bmin[0]
            oy = origin[1] - bmin[1]
            oz = origin[2] - bmin[2]
            tin = 0.0
        tout = np.inf
        tin, tout = _slab(ox, dx, gx, tin, tout)
        tin, tout = _slab(oy, dy, gy, tin, tout)
        tin, tout = _slab(oz, dz, gz, tin, tout)
        if not (tin < tout):
            continue

        cos_t = abs(dz)
        sin_t = np.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        theta = np.arccos(min(cos_t, 1.0))
        xt = two_pi * sin_t / cos_t if cos_t > 0 else MAX_XTHETA
        if xt > MAX_XTHETA:
            xt = MAX_XTHETA
        w = 1.0 / n if n > 0 else 0.0

        wm = 1.0
        k = 0
        while k < n:
            tk = (x[s + k] - bmin[0] - ox) * dx + (y[s + k] - bmin[1] - oy) * dy + (z[s + k] - bmin[2] - oz) * dz
            if tk >= tin:
                break
            wm -= w
            k += 1
        if wm < 0:
            wm = 0.0

        px = ox + tin * dx
        py = oy + tin * dy
        pz = oz + tin * dz
        ix = min(max(int(np.floor(px / vs)), 0), nx - 1)
        iy = min(max(int(np.floor(py / vs)), 0), ny - 1)
        iz = min(max(int(np.floor(pz / vs)), 0), nz - 1)
        sx, tmx, tdx = _axis_init(px, dx, ix, vs, tin)
        sy, tmy, tdy = _axis_init(py, dy, iy, vs, tin)
        sz, tmz, tdz = _axis_init(pz, dz, iz, vs, tin)

        t = tin
        while True:
            vidx = ix + nx * (iy + ny * iz)
            if tmx <= tmy and tmx <= tmz:
                tn = tmx
                axis = 0
            elif tmy <= tmz:
                tn = tmy
                axis = 1
            else:
                tn = tmz
                axis = 2
            last = tn >= tout
            if last:
                tn = tout

            acc[ACC_OCCL, vidx] += 1.0 - wm
            acc[ACC_NBEAM, vidx] += 1.0
            if wm > 0:
                acc[ACC_NENTER, vidx] += 1.0
            fp = 0.0
            tp = t
            while k < n:
                tk = (x[s + k] - bmin[0] - ox) * dx + (y[s + k] - bmin[1] - oy) * dy + (z[s + k] - bmin[2] - oz) * dz
                if tk >= tn:
                    break
                if tk > tp:
                    fp += wm * (tk - tp)
                    tp = tk
                wk = min(w, wm)
                wm -= w
                if wm < 0:
                    wm = 0.0
                zk = max(tk - t, 0.0)
                ze = _effective_path(zk, k1)
                acc[ACC_PEFF, vidx] += wk * ze
                if gnd[s + k]:
                    acc[ACC_GND, vidx] += w
                else:
                    acc[ACC_HITS, vidx] += w
                    acc[ACC_PHIT, vidx] += wk * zk
                    acc[ACC_HITS2, vidx] += w * w
                    acc[ACC_PHITEFF, vidx] += wk * ze
                k += 1
            if tn > tp:
                fp += wm * (tn - tp)
            acc[ACC_PEFF, vidx] += wm * _effective_path(tn - t, k1)
            acc[ACC_MISS, vidx] += wm
            acc[ACC_PATH, vidx] += fp
            acc[ACC_PCOS, vidx] += fp * cos_t
            acc[ACC_PSIN, vidx] += fp * two_pi * sin_t
            acc[ACC_PXT, vidx] += fp * xt
            acc[ACC_PTHETA, vidx] += fp * theta
            acc[ACC_PUX, vidx] += fp * dx
            acc[ACC_PUY, vidx] += fp * dy

            if last:
                break
            t = tn
            if axis == 0:
                ix += sx
                tmx += tdx
                if ix < 0 or ix >= nx:
                    break
            elif axis == 1:
                iy += sy
                tmy += tdy
                if iy < 0 or iy >= ny:
                    break
            else:
                iz += sz
                tmz += tdz
                if iz < 0 or iz >= nz:
                    break


@njit(cache=True)
def rays_cross_grid(x, y, z, start, count, ux, uy, uz, bounds):
    """Pulses (with a valid direction) whose ray through the first return crosses the grid."""
    n = start.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    gx = bounds[3] - bounds[0]
    gy = bounds[4] - bounds[1]
    gz = bounds[5] - bounds[2]
    for p in range(n):
        if count[p] == 0 or ux[p] != ux[p]:
            continue
        s = start[p]
        tin = -np.inf
        tout = np.inf
        tin, tout = _slab(x[s] - bounds[0], np.float64(ux[p]), gx, tin, tout)
        tin, tout = _slab(y[s] - bounds[1], np.float64(uy[p]), gy, tin, tout)
        tin, tout = _slab(z[s] - bounds[2], np.float64(uz[p]), gz, tin, tout)
        out[p] = tin < tout
    return out


@njit(cache=True)
def run_linear_model_numba(xtheta, pgap, weights, null=-9999, min_n=3):
    """
    Weighted least squares fit of -ln Pgap = paiv + xtheta paih per voxel
    (Jupp et al., 2009), with xtheta = (2/pi) tan(theta) and both terms >= 0.
    """
    sh = (pgap.shape[1], pgap.shape[2])
    paiv = np.full(sh, null, dtype=np.float32)
    paih = np.full(sh, null, dtype=np.float32)
    for x in range(pgap.shape[2]):
        for y in range(pgap.shape[1]):
            valid = ((xtheta[:, y, x] != null) &
                     (pgap[:, y, x] != null) &
                     (weights[:, y, x] != null))
            n = np.count_nonzero(valid)
            if n < min_n:
                continue
            xt = xtheta[valid, y, x].astype(np.float64)
            pg = pgap[valid, y, x].astype(np.float64)
            wi = weights[valid, y, x].astype(np.float64)
            k = np.full(n, -np.log(1e-5))
            for j in range(n):
                if pg[j] > 0:
                    k[j] = -np.log(pg[j])
            sw = np.sqrt(wi)
            A = np.ones((n, 2))
            A[:, 1] = xt
            for j in range(n):
                A[j, 0] *= sw[j]
                A[j, 1] *= sw[j]
            B = k * sw
            result = np.linalg.lstsq(A, B)[0]
            v, h = result[0], result[1]
            if v < 0:
                v = 0.0
                pos = xt > 0
                h = np.mean(k[pos] / xt[pos]) if np.any(pos) else 0.0
            elif h < 0:
                h = 0.0
                v = np.mean(k)
            paiv[y, x] = v
            paih[y, x] = h
    return paiv, paih


@njit(cache=True)
def fit_attenuation_numba(hits, path, path_cos, path_sin, null, min_n, min_xtheta_range,
                          max_iter, tol):
    """
    Poisson ML fit of hits_j ~ a path_cos_j + b path_sin_j (a, b >= 0) per voxel
    by multiplicative (EM) updates over the positions j observing the voxel.
    """
    npos, ny, nx = hits.shape
    a_out = np.full((ny, nx), null, dtype=np.float32)
    b_out = np.full((ny, nx), null, dtype=np.float32)
    pad_out = np.full((ny, nx), null, dtype=np.float32)
    mla_out = np.full((ny, nx), null, dtype=np.float32)
    lam_out = np.full((ny, nx), null, dtype=np.float32)
    n_out = np.zeros((ny, nx), dtype=np.uint8)
    for y in range(ny):
        for x in range(nx):
            hs = 0.0
            ps = 0.0
            n = 0
            rmin = np.inf
            rmax = -np.inf
            for j in range(npos):
                pj = path[j, y, x]
                if pj > 0 and pj != null:
                    n += 1
                    hs += hits[j, y, x]
                    ps += pj
                    r = path_sin[j, y, x] / path_cos[j, y, x] if path_cos[j, y, x] > 0 else MAX_XTHETA
                    rmin = min(rmin, r)
                    rmax = max(rmax, r)
            n_out[y, x] = min(n, 255)
            if n == 0:
                continue
            lam_out[y, x] = hs / ps
            if n < min_n or rmax - rmin < min_xtheta_range:
                continue
            sc = 0.0
            ss = 0.0
            for j in range(npos):
                pj = path[j, y, x]
                if pj > 0 and pj != null:
                    sc += path_cos[j, y, x]
                    ss += path_sin[j, y, x]
            a = hs / (sc + ss)
            b = a
            if hs > 0:
                for it in range(max_iter):
                    na = 0.0
                    nb_ = 0.0
                    for j in range(npos):
                        pj = path[j, y, x]
                        if pj > 0 and pj != null:
                            mu = a * path_cos[j, y, x] + b * path_sin[j, y, x]
                            if mu > 0:
                                na += hits[j, y, x] * path_cos[j, y, x] / mu
                                nb_ += hits[j, y, x] * path_sin[j, y, x] / mu
                    a_new = a * na / sc if sc > 0 else 0.0
                    b_new = b * nb_ / ss if ss > 0 else 0.0
                    d = abs(a_new - a) + abs(b_new - b)
                    a = a_new
                    b = b_new
                    if d < tol * (a + b + 1e-12):
                        break
            else:
                a = 0.0
                b = 0.0
            a_out[y, x] = a
            b_out[y, x] = b
            pad_out[y, x] = a + b
            mla_out[y, x] = np.degrees(np.arctan2(b, a)) if a + b > 0 else null
    return a_out, b_out, pad_out, mla_out, lam_out, n_out


def pooled_attenuation(hits, path, path_hit=None, nenter=None, hits2=None, estimator='mle', ci=None):
    """
    Attenuation coefficient (m-1) from summed canopy hit weights and free paths.

    estimator
      'mle'             hits / path (free path length MLE, Pimont et al., 2018)
      'bias_corrected'  (hits - (hits / nenter) * (path_hit / path)) / path, the
                        low-sampling bias correction of Pimont et al. (2018) as
                        formulated in Pimont, Soma and Dupuy (2019); >= 0
    ci
      two-sided level (e.g. 0.95) of an exact Poisson-rate (Garwood) interval on
      hits per unit free path, using the effective hit count hits**2 / hits2
      for fractional hit weights

    Returns
    -------
    lam, lo, hi (lo and hi are None without ci)
    """
    from scipy.stats import chi2
    hits = np.asarray(hits, dtype=np.float64)
    path = np.asarray(path, dtype=np.float64)
    ok = path > 0
    lam = np.full(hits.shape, np.nan)
    np.divide(hits, path, out=lam, where=ok)
    if estimator == 'bias_corrected':
        corr = np.zeros(hits.shape)
        good = ok & (nenter > 0)
        corr[good] = hits[good] / nenter[good] * path_hit[good] / path[good]
        lam = np.where(ok, np.maximum(hits - corr, 0) / np.where(ok, path, 1), np.nan)
    elif estimator != 'mle':
        raise ValueError(f'Unknown estimator: {estimator}')
    lo = hi = None
    if ci:
        a = 1 - ci
        pos = hits > 0
        neff = np.zeros(hits.shape)
        np.divide(hits ** 2, hits2, out=neff, where=pos & (hits2 > 0))
        scale = np.ones(hits.shape)
        np.divide(hits, neff, out=scale, where=neff > 0)
        den = np.where(ok, 2 * path, np.nan)
        lo = np.where(neff > 0, scale * chi2.ppf(a / 2, 2 * np.maximum(neff, 1e-12)), 0.0) / den
        hi = scale * chi2.ppf(1 - a / 2, 2 * (neff + 1)) / den
    return lam, lo, hi


def overview_levels(nx, ny, factors=(2, 4, 8)):
    """Overview factors that leave at least 2 x 2 cells."""
    return [f for f in factors if min(nx, ny) // f >= 2]


def write_voxelgrid(vmodel, data, filename):
    """Write a (nz, ny, nx) north-up voxel grid defined by a VoxelModel."""
    profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': vmodel.nodata,
               'width': vmodel.nx, 'height': vmodel.ny, 'count': vmodel.nz,
               'transform': Affine(vmodel.resolution, 0.0, vmodel.bounds[0], 0.0,
                                   -vmodel.resolution, vmodel.bounds[4]),
               'blockxsize': 256, 'blockysize': 256, 'tiled': True,
               'compress': 'deflate', 'interleave': 'band'}
    elev = vmodel.bounds[2] + np.arange(vmodel.nz) * vmodel.resolution
    with rio.Env():
        with rio.open(filename, 'w', **profile) as dst:
            dst.write(data.astype(np.float32))
            levels = overview_levels(vmodel.nx, vmodel.ny)
            if levels:
                dst.build_overviews(levels, Resampling.average)
            for i in range(vmodel.nz):
                dst.set_band_description(i + 1, f'{elev[i]:.02f}m')


def multivoxel_worker(inputs):
    """Voxelize a single scan and return (name, filenames)."""
    config, name, rxp_fn, transform_fn, rdbx_fn, prefix = inputs
    vgrid = TLSVoxelGrid(dtm_filename=config['dtm'])
    vgrid.add_riegl_scan_position(rxp_fn, transform_fn, rdbx_file=rdbx_fn)
    vgrid.voxelize_scan(config['bounds'], config['resolution'], save_counts=True)
    vgrid.write_grids(prefix)
    return name, vgrid.filenames


def run_multivoxel_grids(rxp_fn, transform_fn, config, rdbx_file=None, outdir='.', ncpu=1):
    """Voxelize multiple scans in parallel, adding their filenames to config['positions']."""
    tasks = []
    for i, fn in enumerate(rxp_fn):
        name = os.path.splitext(os.path.basename(fn))[0]
        rdbx_fn = None if rdbx_file is None else rdbx_file[i]
        tasks.append((config, name, fn, transform_fn[i], rdbx_fn, os.path.join(outdir, name)))
    with mp.Pool(ncpu) as pool:
        for name, filenames in tqdm(pool.imap_unordered(multivoxel_worker, tasks), total=len(tasks)):
            config['positions'][name] = filenames
    return config


def voxel_bounds_from_file(path):
    """(xmin, ymin, zmin, xmax, ymax, zmax) from a LAS/LAZ header."""
    with laspy.open(path) as f:
        hdr = f.header
    mins = np.asarray(hdr.min)
    maxs = np.asarray(hdr.max)
    return (mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2])
