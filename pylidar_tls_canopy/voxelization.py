#!/usr/bin/env python3
"""
pylidar_tls_canopy

Code for voxelization

John Armston
University of Maryland
December 2022
"""

import os
import sys
import json
import numpy as np

import multiprocessing as mp
from tqdm.notebook import tqdm

import numba as nb
from numba import njit

import rasterio as rio
from rasterio.enums import Resampling
from affine import Affine

from . import riegl_io
from . import RIO_DEFAULT_PROFILE

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import laspy


class VoxelModel:
    """
    Class for a voxel Pgap model using multiple TLS scans
    """
    def __init__(self, config_file, dtm_filename=None):
        self.config_file = config_file
        self.load_config()
        self.dtm_filename = dtm_filename
        if self.dtm is not None:
            self.load_dtm()

    def load_config(self):
        """
        Load the VoxelModel configuration file
        nz,ny,nz,resolution,bounds,nodata,positions
        """
        with open(self.config_file, 'r') as f:
            config = json.load(f)
            for k in config:
                setattr(self, k, config[k])
        self.npos = len(self.positions)

    def load_dtm(self):
        """
        Load the DTM voxel grid
        """
        with rio.open(self.dtm, 'r') as src:
            dtm_data = src.read(1)
            dtm_xmin = src.bounds.left
            dtm_ymax = src.bounds.top
            dtm_res = src.res[0]

        self.dtm_voxelgrid = create_ground_voxel_grid(self.nx, self.ny, self.nz,
            self.bounds[3], self.bounds[4], self.bounds[4], self.resolution,
            dtm_data, dtm_xmin, dtm_ymax, dtm_res)

    def read_voxelgrids(self, z=0, names=['pgap','zeni','vwts']):
        """
        Read all the data required (e.g., to invert the Pgap model)
        """
        sh = (self.npos, self.ny, self.nx)
        self.voxelgrids = {}
        for k in names:
            self.voxelgrids[k] = np.empty(sh, dtype=np.float32)
            for i,p in enumerate(self.positions):
                with rio.open(self.positions[p][k],'r') as src:
                    self.voxelgrids[k][i] = src.read(z+1)

    def run_linear_model(self, min_n=3, weights=True):
        """
        Run the linear model from Jupp et al. (2009) to get the
        vertical and horizontal projected area
        """
        sh = (self.nz, self.ny, self.nx)
        paiv = np.empty(sh, dtype=np.float32)
        paih = np.empty(sh, dtype=np.float32)
        nscans = np.empty(sh, dtype=np.uint8)
        for i in range(self.nz):
            self.read_voxelgrids(z=i, names=['pgap','zeni','vwts'])
            if weights:
                w = self.voxelgrids['vwts']
            else:
                w = np.ones(self.voxelgrids['vwts'].shape, dtype=np.float32)
            nscans[i] = np.sum(self.voxelgrids['vwts'] > 0, axis=0, dtype=np.uint8)
            paiv[i],paih[i] = run_linear_model_numba(self.voxelgrids['zeni'], 
                self.voxelgrids['pgap'], w, null=self.nodata, min_n=min_n)

        return paiv,paih,nscans        

    def run_occlusion_voxelgrid(self, null=-9999):
        """
        Compute a weighted average of scan occlusion for each voxel
        """
        sh = (self.nz, self.ny, self.nx)
        voxelgrids = {}
        for k in ('occl','miss','hits'):
            voxelgrids[k] = np.zeros(sh, dtype=np.float32)
            for p in self.positions:
                with rio.open(self.positions[p][k],'r') as src:
                    voxelgrids[k] += src.read()
        nbeam = voxelgrids['occl'] + voxelgrids['miss'] + voxelgrids['hits']
        poccl = np.full(sh, null, dtype=np.float32)
        np.divide(voxelgrids['occl'], nbeam, out=poccl, where=nbeam>0)
        return poccl,nbeam

    def get_cover_profile(self, paiv):
        """
        Get the vertical canopy cover profile using conditional probability
        """
        pgap = np.zeros_like(paiv)
        np.exp(-paiv, out=pgap, where=paiv != self.nodata)        

        cover = np.zeros_like(pgap)
        np.subtract(1, pgap, out=cover, where=paiv != self.nodata)

        cover_z = np.zeros_like(cover)
        for i in range(cover.shape[0]-2,-1,-1):
            p_o = cover_z[i+1]
            cover_z[i] = p_o + (1 - p_o) * cover[i]
        
        return cover_z


class TLSVoxelGrid:
    """
    Class for a voxel grid of a TLS scan
    """
    def __init__(self, dtm_filename=None, profile=RIO_DEFAULT_PROFILE):
        self.dtm_filename = dtm_filename
        self.nodata = profile['nodata']
        self.profile = profile
        if self.dtm_filename is not None:
            self.load_dtm()

    def load_dtm(self):
        with rio.open(self.dtm_filename, 'r') as src:
            self.dtm_data = src.read(1)
            self.dtm_xmin = src.bounds.left
            self.dtm_ymax = src.bounds.top
            self.dtm_res = src.res[0]

    def classify_ground(self, x, y, z, target_count, thres=0.25):
        """
        Classify points as ground (1) or canopy (0)
        """
        if self.dtm_filename is not None:
            dtm_vals = extract_ground_by_pulse(x, y, target_count, self.dtm_data,
                self.dtm_xmin, self.dtm_ymax, self.dtm_res, nodata=self.nodata)
            self.ground = np.zeros(z.shape, dtype=bool)
            np.less(z, dtm_vals + thres, out=self.ground, 
                where=dtm_vals != self.nodata)
        else:
            self.ground = np.zeros(z.shape, dtype=bool)    

    def add_riegl_scan_position(self, rxp_file, transform_file, rdbx_file=None, query_str=None):
        """
        Add a scan position using rxp
        """
        with riegl_io.RXPFile(rxp_file, transform_file=transform_file, query_str=query_str) as rxp:
            self.count = rxp.get_data('target_count')

            self.dx = rxp.get_data('beam_direction_x')
            self.dy = rxp.get_data('beam_direction_y')
            self.dz = rxp.get_data('beam_direction_z')

            self.x0 = rxp.transform[3,0]
            self.y0 = rxp.transform[3,1]
            self.z0 = rxp.transform[3,2]

            self.pulse_scanline = rxp.get_data('scanline')
            self.pulse_scanline_idx = rxp.get_data('scanline_idx')

            if rdbx_file:
                with riegl_io.RDBFile(rdbx_file, transform_file=transform_file, query_str=query_str) as rdb:
                    self.points = rdb.get_points_by_pulse(['x','y','z'], self.pulse_scanline, self.pulse_scanline_idx)
            else:
                self.points = rxp.get_points_by_pulse(['x','y','z'])

        self.classify_ground(self.points['x'], self.points['y'], self.points['z'], self.count)

    def _init_voxel_grid(self, bounds, voxelsize):
        """
        Initialize the voxel grid
        """
        self.bounds = nb.typed.List(bounds)
        self.voxelsize = float(voxelsize)

        self.nx = int( (bounds[3] - bounds[0]) // voxelsize)
        self.ny = int( (bounds[4] - bounds[1]) // voxelsize)
        self.nz = int( (bounds[5] - bounds[2]) // voxelsize)

        self.voxdimx = bounds[3] - bounds[0]
        self.voxdimy = bounds[4] - bounds[1]
        self.voxdimz = bounds[5] - bounds[2]

        self.nvox = int(self.nx * self.ny * self.nz)

        self.profile.update(height=self.ny, width=self.nx, count=self.nz,
            transform=Affine(voxelsize, 0.0, bounds[0], 0.0,
            -voxelsize, bounds[4]), nodata=self.nodata)

    def voxelize_scan(self, bounds, voxelsize, save_counts=True):
        """
        Voxelize the scan data
        """
        self._init_voxel_grid(bounds, voxelsize)
        hits = np.zeros(self.nvox, dtype=float)
        miss = np.zeros(self.nvox, dtype=float)
        occl = np.zeros(self.nvox, dtype=float)
        zeni = np.zeros(self.nvox, dtype=float)
        phit = np.zeros(self.nvox, dtype=float)
        pmiss = np.zeros(self.nvox, dtype=float)

        run_traverse_voxels_tls(self.x0, self.y0, self.z0, self.ground, self.points['x'].data, 
            self.points['y'].data, self.points['z'].data, self.dx, self.dy, self.dz, 
            self.count, self.voxdimx, self.voxdimy, self.voxdimz, self.nx, self.ny, 
            self.nz, self.bounds, self.voxelsize, hits, miss, occl, zeni, phit, pmiss)

        self.voxelgrids = {}
        nshots = hits + miss
        nbeams = nshots + occl

        self.voxelgrids['vwts'] = np.full(self.nvox, self.nodata, dtype=float)
        np.add(phit, pmiss, out=self.voxelgrids['vwts'], where=nshots > 0)

        self.voxelgrids['pgap'] = np.full(self.nvox, self.nodata, dtype=float)
        np.divide(miss, nshots, out=self.voxelgrids['pgap'], where=nshots > 0)

        self.voxelgrids['zeni'] = np.full(self.nvox, self.nodata, dtype=float)
        np.divide(zeni, nbeams, out=self.voxelgrids['zeni'], where=nbeams > 0)

        self.voxelgrids['vcls'] = self.classify_voxels(hits, miss, occl)

        if save_counts:
            self.voxelgrids['hits'] = hits
            self.voxelgrids['miss'] = miss
            self.voxelgrids['occl'] = occl
            self.voxelgrids['phit'] = phit
            self.voxelgrids['pmiss'] = pmiss

    def classify_voxels(self, hits, miss, occl):
        """
        Classification of voxels
        
        Class       Value   Hits    Misses   Occluded
        Observed    5       >0      >=0      >=0
        Empty       4       =0      >0       >=0
        Hidden      3       =0      =0       >0
        Unobserved  2       =0      =0       =0
        Ground      1
        """
        classification = np.full_like(hits, self.nodata)

        idx = (hits > 0) & (miss >= 0) & (occl >= 0)
        classification[idx] = 5

        idx = (hits == 0) & (miss > 0) & (occl >= 0)
        classification[idx] = 4

        idx = (hits == 0) & (miss == 0) & (occl > 0)
        classification[idx] = 3

        idx = (hits == 0) & (miss == 0) & (occl == 0)
        classification[idx] = 2

        return classification

    def write_grids(self, prefix):
        """
        Write the results to file
        """
        self.filenames = {}
        new_shape = (self.nz, self.ny, self.nx)
        for k in self.voxelgrids:
            with rio.Env():
                self.filenames[k] = f'{prefix}_{k}.tif'
                with rio.open(self.filenames[k], 'w', **self.profile) as dst:
                    dst.write(self.voxelgrids[k].reshape(new_shape))
                    dst.build_overviews([2,4,8], Resampling.average)
                    for i in range(self.nz):
                        height = self.bounds[2] + i * self.voxelsize
                        description = f'{height:.02f}m'
                        dst.set_band_description(i+1, description)

    
class DLSVoxelGrid:
    """
    Voxel grid for a drone lidar (DLS) flightline.
 
    Parameters
    ----------
    profile : dict
        Rasterio write profile.
 
    Usage
    -----
        grid = DLSVoxelGrid()
        grid.add_laz_flightline('flight.laz')
        grid.voxelize_scan(bounds, voxelsize=1.0)
        grid.write_grids('output/flight')
    """
 
    def __init__(self, profile=None):
        self.profile = dict(RIO_DEFAULT_PROFILE if profile is None else profile)
        self.nodata  = self.profile['nodata']
 
    # ── Data loading ──────────────────────────────────────────────────────────
 
    def add_laz_flightline(self, laz_file, voxel_bounds=None, t_scale=2.0):
        """
        Load a LAZ flightline and derive all voxel traversal inputs.
 
        Parameters
        ----------
        laz_file : str
        voxel_bounds : (xmin, ymin, zmin, xmax, ymax, zmax) or None
            Derived from file header if not provided.
        t_scale : float
            Passed to FlightlineProcessor; controls beam origin offset.
        """
        if voxel_bounds is None:
            voxel_bounds = voxel_bounds_from_file(laz_file)
 
        self.proc = FlightlineProcessor(laz_file, voxel_bounds, t_scale=t_scale)
        self.proc.process()
 
        # Pulse-level arrays (equivalent to TLS per-pulse fields)
        pi = self.proc.pulse_inputs
        self.x0    = pi['x0']   # array — varies with drone position
        self.y0    = pi['y0']
        self.z0    = pi['z0']
        self.dx    = pi['dx']
        self.dy    = pi['dy']
        self.dz    = pi['dz']
        self.count = pi['count']
 
        # Return-level coordinates shaped as (n_pulses, max_returns) for njit
        pbp = self.proc.points_by_pulse
        self.points = {'x': pbp['x'], 'y': pbp['y'], 'z': pbp['z']}
        self.ground = pbp['gnd']
 
    # ── Voxel grid ────────────────────────────────────────────────────────────
 
    def _init_voxel_grid(self, bounds, voxelsize):
        self.bounds    = nb.typed.List(bounds)
        self.voxelsize = float(voxelsize)
        self.nx = int((bounds[3] - bounds[0]) // voxelsize)
        self.ny = int((bounds[4] - bounds[1]) // voxelsize)
        self.nz = int((bounds[5] - bounds[2]) // voxelsize)
        self.voxdimx = bounds[3] - bounds[0]
        self.voxdimy = bounds[4] - bounds[1]
        self.voxdimz = bounds[5] - bounds[2]
        self.nvox = int(self.nx * self.ny * self.nz)
        self.profile.update(
            height=self.ny, width=self.nx, count=self.nz,
            transform=Affine(voxelsize, 0.0, bounds[0], 0.0, -voxelsize, bounds[4]),
            nodata=self.nodata,
        )
 
    def voxelize_scan(self, bounds, voxelsize, save_counts=True):
        """
        Voxelize the flightline data.
 
        Parameters
        ----------
        bounds : (xmin, ymin, zmin, xmax, ymax, zmax)
        voxelsize : float
        save_counts : bool
            If True, raw hit/miss/occl/phit/pmiss grids are retained.
 
        Note: run_traverse_voxels_dls differs from the TLS version in that
        x0, y0, z0 are per-pulse arrays rather than scalars.
        """
        self._init_voxel_grid(bounds, voxelsize)
 
        hits  = np.zeros(self.nvox, dtype=float)
        miss  = np.zeros(self.nvox, dtype=float)
        occl  = np.zeros(self.nvox, dtype=float)
        zeni  = np.zeros(self.nvox, dtype=float)
        phit  = np.zeros(self.nvox, dtype=float)
        pmiss = np.zeros(self.nvox, dtype=float)
 
        run_traverse_voxels_dls(
            self.x0, self.y0, self.z0,
            self.ground,
            self.points['x'], self.points['y'], self.points['z'],
            self.dx, self.dy, self.dz,
            self.count,
            self.voxdimx, self.voxdimy, self.voxdimz,
            self.nx, self.ny, self.nz,
            self.bounds, self.voxelsize,
            hits, miss, occl, zeni, phit, pmiss,
        )
 
        self.voxelgrids = {}
        nshots = hits + miss
        nbeams = nshots + occl
 
        self.voxelgrids['vwts'] = np.full(self.nvox, self.nodata, dtype=float)
        np.add(phit, pmiss, out=self.voxelgrids['vwts'], where=nshots > 0)
 
        self.voxelgrids['pgap'] = np.full(self.nvox, self.nodata, dtype=float)
        np.divide(miss, nshots, out=self.voxelgrids['pgap'], where=nshots > 0)
 
        self.voxelgrids['zeni'] = np.full(self.nvox, self.nodata, dtype=float)
        np.divide(zeni, nbeams, out=self.voxelgrids['zeni'], where=nbeams > 0)
 
        self.voxelgrids['vcls'] = self.classify_voxels(hits, miss, occl)
 
        if save_counts:
            self.voxelgrids['hits']  = hits
            self.voxelgrids['miss']  = miss
            self.voxelgrids['occl']  = occl
            self.voxelgrids['phit']  = phit
            self.voxelgrids['pmiss'] = pmiss
 
    def classify_voxels(self, hits, miss, occl):
        """
        Classification of voxels.
 
        Class       Value   Hits    Misses   Occluded
        Observed    5       >0      >=0      >=0
        Empty       4       =0      >0       >=0
        Hidden      3       =0      =0       >0
        Unobserved  2       =0      =0       =0
        Ground      1
        """
        classification = np.full_like(hits, self.nodata)
        classification[(hits > 0)]                               = 5
        classification[(hits == 0) & (miss > 0)]                = 4
        classification[(hits == 0) & (miss == 0) & (occl > 0)]  = 3
        classification[(hits == 0) & (miss == 0) & (occl == 0)] = 2
        return classification
 
    def write_grids(self, prefix):
        """Write voxel grids to GeoTIFF files."""
        self.filenames = {}
        new_shape = (self.nz, self.ny, self.nx)
        for k, grid in self.voxelgrids.items():
            self.filenames[k] = f'{prefix}_{k}.tif'
            with rio.Env():
                with rio.open(self.filenames[k], 'w', **self.profile) as dst:
                    dst.write(grid.reshape(new_shape))
                    dst.build_overviews([2, 4, 8], Resampling.average)
                    for i in range(self.nz):
                        height = self.bounds[2] + i * self.voxelsize
                        dst.set_band_description(i + 1, f'{height:.2f}m')


class FlightlineProcessor:
    """
    Derives voxel traversal inputs from a single drone lidar LAZ flightline.
 
    Parameters
    ----------
    path : str
    voxel_bounds : (xmin, ymin, zmin, xmax, ymax, zmax)
    t_scale : float
        Beam origin is placed t_scale * grid_diagonal behind each return.
 
    Usage
    -----
        proc = FlightlineProcessor(path, voxel_bounds)
        proc.process()
        proc.voxelize(my_njit_fn)
    """
 
    FIELDS = ['X', 'Y', 'Z', 'gps_time', 'return_number',
              'number_of_returns', 'classification', 'edge_of_flight_line']
 
    def __init__(self, path, voxel_bounds, t_scale=2.0):
        self.path         = path
        self.voxel_bounds = voxel_bounds
        self.t_scale      = t_scale
        self.inputs       = None   # populated by process()
 
    # ── Public interface ──────────────────────────────────────────────────────
 
    def process(self):
        """Run the full pipeline. Results stored in self.inputs."""
        self._read_points()
        self._sort_points()
        self._build_pulse_table()
        self._assign_segments()
        self._compute_directions()
        self._interpolate_directions()
        self._compute_beam_origins()
        self._expand_to_returns()
        print(f"  {self.n_pulses:,} pulses  |  {len(self.x1):,} returns  |  "
              f"{self._multi.sum():,} multi-return  |  {(~self._multi).sum():,} single-return")
        return self
 
    def voxelize(self, njit_fn, *args, **kwargs):
        """
        Call a numba njit voxelization function with the derived inputs.
 
        The njit function is expected to accept:
            x0, y0, z0, x1, y1, z1, dx, dy, dz, gnd, point_count
        followed by any additional positional or keyword arguments.
 
        Parameters
        ----------
        njit_fn : callable
            Compiled numba function.
        *args, **kwargs
            Passed through to njit_fn after the standard inputs.
 
        Returns
        -------
        Whatever njit_fn returns.
        """
        if self.inputs is None:
            raise RuntimeError("Call process() before voxelize().")
        inp = self.inputs
        return njit_fn(
            inp['x0'], inp['y0'], inp['z0'],
            inp['x1'], inp['y1'], inp['z1'],
            inp['dx'], inp['dy'], inp['dz'],
            inp['gnd'], inp['point_count'],
            *args, **kwargs,
        )
 
    @property
    def pulse_inputs(self):
        """
        Pulse-level arrays, available after process().
        Used by DLSVoxelGrid to match the TLS per-pulse interface.
        """
        if self.inputs is None:
            raise RuntimeError("Call process() first.")
        return dict(
            x0=self._p_x0, y0=self._p_y0, z0=self._p_z0,
            dx=self._p_ux, dy=self._p_uy, dz=self._p_uz,
            count=self._actual_counts,  # actual observed, not declared number_of_returns
            x=self._p_x_first, y=self._p_y_first, z=self._p_z_first,
        )
 
    @property
    def points_by_pulse(self):
        """
        Return-level arrays reshaped to (n_pulses, max_returns) as expected
        by run_traverse_voxels_dls.  Ragged pulses are padded with 0 / False.
 
        Returns
        -------
        dict with keys: x, y, z (float64) and gnd (bool)
        """
        if self.inputs is None:
            raise RuntimeError("Call process() first.")
        max_ret = int(self._actual_counts.max())
        n = self.n_pulses
 
        x   = np.zeros((n, max_ret), dtype=np.float64)
        y   = np.zeros((n, max_ret), dtype=np.float64)
        z   = np.zeros((n, max_ret), dtype=np.float64)
        gnd = np.zeros((n, max_ret), dtype=bool)
 
        for i in range(n):
            s = self._first_idx[i]
            e = self._last_idx[i] + 1
            k = e - s
            x[i, :k]   = self.x1[s:e]
            y[i, :k]   = self.y1[s:e]
            z[i, :k]   = self.z1[s:e]
            gnd[i, :k] = self.inputs['gnd'][s:e].astype(bool)
 
        return dict(x=x, y=y, z=z, gnd=gnd)
 
    # ── Pipeline steps ────────────────────────────────────────────────────────
 
    def _read_points(self):
        buf = {f: [] for f in self.FIELDS}
        with laspy.open(self.path) as f:
            self._scale  = f.header.scale
            self._offset = f.header.offset
            for chunk in f.chunk_iterator(1_000_000):
                for field in self.FIELDS:
                    buf[field].append(np.asarray(chunk[field]))
        self._d = {f: np.concatenate(buf[f]) for f in self.FIELDS}
 
    def _sort_points(self):
        d = self._d
        x1 = d['X'] * self._scale[0] + self._offset[0]
        y1 = d['Y'] * self._scale[1] + self._offset[1]
        z1 = d['Z'] * self._scale[2] + self._offset[2]
        del d['X'], d['Y'], d['Z']
 
        order = np.lexsort((d['return_number'], d['gps_time']))
        self.x1 = x1[order]
        self.y1 = y1[order]
        self.z1 = z1[order]
        self._d = {f: d[f][order] for f in d}
        del order
 
    def _build_pulse_table(self):
        d = self._d
        _, first_idx = np.unique(d['gps_time'], return_index=True)
        last_idx          = np.empty_like(first_idx)
        last_idx[:-1]     = first_idx[1:] - 1
        last_idx[-1]      = len(self.x1) - 1
 
        actual_counts = last_idx - first_idx + 1
        if actual_counts.sum() != len(self.x1):
            raise ValueError(
                f"Return count mismatch: {actual_counts.sum():,} != {len(self.x1):,}")
 
        self.n_pulses      = len(first_idx)
        self._first_idx    = first_idx
        self._last_idx     = last_idx
        self._actual_counts = actual_counts
        self._p_time       = d['gps_time'][first_idx]
        self._p_nret       = d['number_of_returns'][first_idx]
        self._p_edge       = np.maximum.reduceat(d['edge_of_flight_line'], first_idx)
        self._p_x_first    = self.x1[first_idx]
        self._p_y_first    = self.y1[first_idx]
        self._p_z_first    = self.z1[first_idx]
        self._p_x_last     = self.x1[last_idx]
        self._p_y_last     = self.y1[last_idx]
        self._p_z_last     = self.z1[last_idx]
 
    def _assign_segments(self):
        p_segment       = np.cumsum(self._p_edge)
        seg_starts      = np.concatenate([[0], np.where(np.diff(p_segment))[0] + 1])
        seg_ends        = np.concatenate([seg_starts[1:], [self.n_pulses]])
        self._p_segment = p_segment
        self._seg_starts = seg_starts
        self._seg_ends   = seg_ends
 
    def _compute_directions(self):
        """Direction vectors for multi-return pulses."""
        n = self.n_pulses
        self._p_ux = np.full(n, np.nan)
        self._p_uy = np.full(n, np.nan)
        self._p_uz = np.full(n, np.nan)
 
        multi = self._p_nret > 1
        self._multi = multi
 
        vx = self._p_x_last[multi] - self._p_x_first[multi]
        vy = self._p_y_last[multi] - self._p_y_first[multi]
        vz = self._p_z_last[multi] - self._p_z_first[multi]
        self._p_ux[multi], self._p_uy[multi], self._p_uz[multi] = _normalize(vx, vy, vz)
 
    def _interpolate_directions(self):
        """
        Temporal interpolation within flightline segments for single-return
        pulses. Falls back to 2D spatial interpolation for segments that
        contain no multi-return pulses at all.
        """
        multi  = self._multi
        single = ~multi
        n_spatial_fallback = 0
 
        for s, e in zip(self._seg_starts, self._seg_ends):
            seg_multi  = np.where(multi[s:e])[0] + s
            seg_single = np.where(single[s:e])[0] + s
            if len(seg_single) == 0:
                continue
            if len(seg_multi) == 0:
                n_spatial_fallback += len(seg_single)
                continue
 
            anchor_t = self._p_time[seg_multi]
            query_t  = self._p_time[seg_single]
            for arr in [self._p_ux, self._p_uy, self._p_uz]:
                arr[seg_single] = np.interp(query_t, anchor_t, arr[seg_multi])
 
            dx, dy, dz = _normalize(
                self._p_ux[seg_single],
                self._p_uy[seg_single],
                self._p_uz[seg_single],
            )
            self._p_ux[seg_single] = dx
            self._p_uy[seg_single] = dy
            self._p_uz[seg_single] = dz
 
        self._spatial_fallback(n_spatial_fallback)
 
    def _spatial_fallback(self, n_flagged):
        """2D spatial interpolation for segments with no multi-return pulses."""
        still_nan = np.isnan(self._p_ux)
        if not still_nan.any():
            return
 
        print(f"  Spatial post-processing {still_nan.sum():,} pulses "
              f"({n_flagged:,} flagged during segment pass)")
 
        anchor_mask = self._multi & ~np.isnan(self._p_ux)
        anchor_xy   = np.column_stack([
            (self._p_x_first[anchor_mask] + self._p_x_last[anchor_mask]) / 2,
            (self._p_y_first[anchor_mask] + self._p_y_last[anchor_mask]) / 2,
        ])
        query_xy = np.column_stack([
            self._p_x_first[still_nan], self._p_y_first[still_nan]
        ])
 
        for arr in [self._p_ux, self._p_uy, self._p_uz]:
            lin  = LinearNDInterpolator(anchor_xy, arr[anchor_mask], fill_value=np.nan)
            vals = lin(query_xy)
            outside = np.isnan(vals)
            if outside.any():
                nn = NearestNDInterpolator(anchor_xy, arr[anchor_mask])
                vals[outside] = nn(query_xy[outside])
            arr[still_nan] = vals
 
        dx, dy, dz = _normalize(
            self._p_ux[still_nan], self._p_uy[still_nan], self._p_uz[still_nan]
        )
        self._p_ux[still_nan] = dx
        self._p_uy[still_nan] = dy
        self._p_uz[still_nan] = dz
 
    def _compute_beam_origins(self):
        xmin, ymin, zmin, xmax, ymax, zmax = self.voxel_bounds
        diag = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
        t = diag * self.t_scale
 
        # Beam origin: project back along unit direction to outside the grid
        self._p_x0 = self._p_x_first - t * self._p_ux
        self._p_y0 = self._p_y_first - t * self._p_uy
        self._p_z0 = self._p_z_first - t * self._p_uz
 
        # Displacement vectors from x0 to x_last so that t=1 reaches the
        # last return — required by the voxel traversal parameterisation
        self._p_dx = self._p_x_last - self._p_x0
        self._p_dy = self._p_y_last - self._p_y0
        self._p_dz = self._p_z_last - self._p_z0
 
    def _expand_to_returns(self):
        """Expand all pulse-level arrays to per-return using actual counts."""
        ri = np.repeat(np.arange(self.n_pulses), self._actual_counts)
        self.inputs = dict(
            x0=self._p_x0[ri], y0=self._p_y0[ri], z0=self._p_z0[ri],
            x1=self.x1,        y1=self.y1,         z1=self.z1,
            dx=self._p_ux[ri], dy=self._p_uy[ri],  dz=self._p_uz[ri],
            gnd=(self._d['classification'] == 2).astype(np.uint8),
            point_count=self._p_nret[ri],
        )


@njit
def extract_ground_by_pulse(x, y, target_count, data, xmin, ymax, binsize, nodata=-9999):
    outdata = np.full(x.shape, nodata, dtype=data.dtype)
    for i in range(target_count.shape[0]):
        for j in range(target_count[i]):
            col = int( (x[i,j] - xmin) // binsize )
            row = int( (ymax - y[i,j]) // binsize )
            if (row >= 0) & (col >= 0) & (row < data.shape[0]) & (col < data.shape[1]):
                outdata[i,j] = data[row, col]
    return outdata


@njit
def create_ground_voxel_grid(nx, ny, nz, xmin, ymax, zmin, binsize,
    dem, dem_xmin, dem_ymax, dem_binsize):
    outdata = np.zeros((nz,ny,nx), dtype=np.float32)
    zval = zmin + np.arange(nz, dtype=np.float32) * binsize + binsize / 2
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                xc = xmin + (x * binsize + binsize / 2)
                yc = ymax - (y * binsize + binsize / 2)
                dem_col = int( (xc - dem_xmin) // dem_binsize )
                dem_row = int( (dem_ymax - yc) // dem_binsize )
                if (dem_row >= 0) & (dem_col >= 0) & (dem_row < dem.shape[0]) & (dem_col < dem.shape[1]):
                    dem_val = dem[dem_row, dem_col]
                    if zval[z] < dem_val:
                        outdata[z,y,x] = 1
    return outdata


@njit
def run_traverse_voxels_tls(x0, y0, z0, gnd, x, y, z, dx, dy, dz, target_count, voxdimx, voxdimy, voxdimz,
                        nx, ny, nz, bounds, voxelsize, hits, miss, occl, zeni, phit, pmiss):
    """
    Loop through each pulse and run voxel traversal
    """
    max_nreturns = np.max(target_count)
    vox_idx = np.empty(max_nreturns, dtype=np.uint32)
    for i in range(target_count.shape[0]):        
        traverse_voxels(x0, y0, z0, gnd[i,:], x[i,:], y[i,:], z[i,:], dx[i], dy[i], dz[i],
            nx, ny, nz, voxdimx, voxdimy, voxdimz, bounds, voxelsize, target_count[i],
            hits, miss, occl, zeni, phit, pmiss, vox_idx)


@njit
def run_traverse_voxels_dls(x0, y0, z0, gnd, x, y, z, dx, dy, dz, target_count, voxdimx, voxdimy, voxdimz,
                        nx, ny, nz, bounds, voxelsize, hits, miss, occl, zeni, phit, pmiss):
    """
    Loop through each pulse and run voxel traversal
    """
    max_nreturns = np.max(target_count)
    vox_idx = np.empty(max_nreturns, dtype=np.uint32)
    for i in range(target_count.shape[0]):        
        traverse_voxels(x0[i], y0[i], z0[i], gnd[i,:], x[i,:], y[i,:], z[i,:], dx[i], dy[i], dz[i],
            nx, ny, nz, voxdimx, voxdimy, voxdimz, bounds, voxelsize, target_count[i],
            hits, miss, occl, zeni, phit, pmiss, vox_idx)    


@njit
def traverse_voxels(x0, y0, z0, gnd, x1, y1, z1, dx, dy, dz, nx, ny, nz, voxdimx, voxdimy, voxdimz,
                    bounds, voxelsize, target_count, hits, miss, occl, zeni, phit, pmiss, vox_idx):
    """
    A fast and simple voxel traversal algorithm through a 3D voxel space (J. Amanatides and A. Woo, 1987)
    Inputs:
       x0, y0, z0
       gnd
       x1, y1, z1
       dx, dy, dz
       nX, nY, nZ
       bounds
       voxelsize
       target_count
       vox_idx
    Outputs:
       hits
       miss
       occl
       zeni
       phit
       pmiss
    """
    intersect, tmin, tmax = grid_intersection(x0, y0, z0, dx, dy, dz, bounds)    
    if intersect:
        
        tmin = max(0, tmin)
        tmax = min(1, tmax)

        startX = x0 + tmin * dx
        startY = y0 + tmin * dy
        startZ = z0 + tmin * dz

        r = np.sqrt(dx**2 + dy**2 + dz**2)
        theta = np.arccos(dz / r)
        
        x = np.floor( ((startX - bounds[0]) / voxdimx) * nx )
        y = np.floor( ((startY - bounds[1]) / voxdimy) * ny )
        z = np.floor( ((startZ - bounds[2]) / voxdimz) * nz )              
        
        for i in range(target_count):
            px = np.floor( ((x1[i] - bounds[0]) / voxdimx) * nx )
            py = np.floor( ((y1[i] - bounds[1]) / voxdimy) * ny )
            pz = np.floor( ((z1[i] - bounds[2]) / voxdimz) * nz )
            vox_idx[i] = int(px + nx * py + nx * ny * pz)   
        
        if x == nx:
            x -= 1
        if y == ny:
            y -= 1           
        if z == nz:
            z -= 1
         
        if dx > 0:
            tVoxelX = (x + 1) / nx
            stepX = 1
        elif dx < 0:
            tVoxelX = x / nx
            stepX = -1
        else:
            tVoxelX = (x + 1) / nx
            stepX = 0
        
        if dy > 0:
            tVoxelY = (y + 1) / ny
            stepY = 1
        elif dy < 0:
            tVoxelY = y / ny
            stepY = -1
        else:
            tVoxelY = (y + 1) / ny
            stepY = 0  
        
        if dz > 0:
            tVoxelZ = (z + 1) / nz
            stepZ = 1
        elif dz < 0:
            tVoxelZ = z / nz
            stepZ = -1
        else:
            tVoxelZ = (z + 1) / nz
            stepZ = 0            
        
        voxelMaxX = bounds[0] + tVoxelX * voxdimx
        voxelMaxY = bounds[1] + tVoxelY * voxdimy
        voxelMaxZ = bounds[2] + tVoxelZ * voxdimz
        
        if dx == 0:
            tMaxX = tmax
            tDeltaX = tmax
        else:
            tMaxX = tmin + (voxelMaxX - startX) / dx
            tDeltaX = voxelsize / abs(dx)
            
        if dy == 0:    
            tMaxY = tmax
            tDeltaY = tmax
        else:
            tMaxY = tmin + (voxelMaxY - startY) / dy
            tDeltaY = voxelsize / abs(dy)

        if dz == 0:
            tMaxZ = tmax
            tDeltaZ = tmax
        else:
            tMaxZ = tmin + (voxelMaxZ - startZ) / dz
            tDeltaZ = voxelsize / abs(dz)

        wmiss = 1.0
        woccl = 0.0
        if target_count > 0:
            w = 1.0 / target_count
        else:
            w = 0.0
       
        startR = min(tMaxX, tMaxY, tMaxZ)

        last_voxel_hit = False
        last_canopy_hit_num = 0.0
        while (x < nx) and (x >= 0) and (y < ny) and (y >= 0) and (z < nz) and (z >= 0):
            
            vidx = int(x + nx * y + nx * ny * z)
            zeni[vidx] += theta 
            
            if tMaxX < tMaxY:
                if tMaxX < tMaxZ:
                    x += stepX
                    tMaxX += tDeltaX
                else:
                    z += stepZ
                    tMaxZ += tDeltaZ
            else:
                if tMaxY < tMaxZ:
                    y += stepY
                    tMaxY += tDeltaY
                else:
                    z += stepZ
                    tMaxZ += tDeltaZ

            endR = min(tMaxX, tMaxY, tMaxZ)
            plen = endR - startR

            vg_num = 0
            gnd_num = 0
            for i in range(target_count):
                if (vidx == vox_idx[i]):
                    if (gnd[i] == 0):
                        vg_num += 1
                    else:
                        gnd_num += 1

            if last_voxel_hit:
                woccl += w * last_canopy_hit_num
                last_voxel_hit = False
                last_canopy_hit_num = 0.0

            if vg_num + gnd_num > 0:
                hits[vidx] += w * vg_num
                phit[vidx] += plen * vg_num
                wmiss -= w * (vg_num + gnd_num)
                woccl += w * gnd_num
                last_voxel_hit = vg_num > 0
                last_canopy_hit_num = vg_num
            else:
                pmiss[vidx] += plen

            occl[vidx] += woccl
            miss[vidx] += wmiss

            startR = endR


@njit
def grid_intersection(x0, y0, z0, dx, dy, dz, bounds):
    """
    Voxel grid intersection test using Smits algorithm
    Inputs:
       x0, y0, z0
       dz, dy, dz
       bounds
    Outputs:
       intersect: 0 = no intersection, 1 = intersection
       tmin: min distance from the beam origin
       tmax: max distance from the beam origin
    """
    if dx != 0:
        divX = 1.0 / dx
    else:
        divX = 1.0
    
    if divX >= 0:
    	tmin = (bounds[0] - x0) * divX
    	tmax = (bounds[3] - x0) * divX
    else:
    	tmin = (bounds[3] - x0) * divX
    	tmax = (bounds[0] - x0) * divX
      
    if dy != 0:
        divY = 1.0 / dy
    else:
        divY = 1.0
    
    if divY >= 0:
        tymin = (bounds[1] - y0) * divY
        tymax = (bounds[4] - y0) * divY
    else:
    	tymin = (bounds[4] - y0) * divY
    	tymax = (bounds[1] - y0) * divY
    
    if (tmin > tymax) or (tymin > tmax):
        intersect = False
        tmin = -1.0
    else:
        if tymin > tmin:
            tmin = tymin
        if tymax < tmax:
            tmax = tymax

        if dz != 0:
            divZ = 1.0 / dz
        else:
            divZ = 1.0
        
        if divZ >= 0:
            tzmin = (bounds[2] - z0) * divZ
            tzmax = (bounds[5] - z0) * divZ
        else:
            tzmin = (bounds[5] - z0) * divZ
            tzmax = (bounds[2] - z0) * divZ

        if (tmin > tzmax) or (tzmin > tmax):
            intersect = False
            tmin = -1.0
        else:
            if tzmin > tmin:
                tmin = tzmin
            if tzmax < tmax:
                tmax = tzmax
            intersect = True
    
    return intersect,tmin,tmax


@njit
def run_linear_model_numba(zenith, pgap, weights, null=-9999, min_n=3):
    """
    Numba function to solve the linear model of Jupp et al. (2009)
    across many voxels
    """
    sh = (pgap.shape[1],pgap.shape[2])
    paiv = np.full(sh, null, dtype=np.float32)
    paih = np.full(sh, null, dtype=np.float32)
    
    for x in range(pgap.shape[2]):
        for y in range(pgap.shape[1]):

            valid = ((zenith[:,y,x] != null) & 
                     (pgap[:,y,x] != null) & 
                     (weights[:,y,x] != null))
            n = np.count_nonzero(valid)
            if n >= min_n:

                zenith_i = zenith[valid,y,x]
                pgap_i = pgap[valid,y,x]
                weights_i = weights[valid,y,x]

                kthetal = np.full(n, np.log(1e-5), dtype=np.float32)
                for j in range(n):
                    if pgap_i[j] > 0:
                        kthetal[j] = np.log(pgap_i[j])
                xtheta = np.abs(2 * np.tan(zenith_i) / np.pi)
            
                W = np.sqrt(np.diag(weights_i))
                A = np.ones((n,2), dtype=np.float32)
                A[:,1] = xtheta
                A = np.dot(W, A)
                B = np.dot(-kthetal,W)
                
                result,resid,rank,s = np.linalg.lstsq(A, B)
                paiv[y,x] = result[0]
                paih[y,x] = result[1]

                if result[0] < 0:
                    paih[y,x] = np.mean(-kthetal)
                    paiv[y,x] = 0.0
                if result[1] < 0:
                    paiv[y,x] = np.mean(-kthetal / xtheta)
                    paih[y,x] = 0.0
    
    return paiv,paih


def write_voxelgrid(vmodel, data, filename):
    """
    Write a voxelgrid to file
    """
    profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': vmodel.nodata,
               'width': vmodel.nx, 'height': vmodel.ny, 'count': vmodel.nz,
               'transform': Affine(vmodel.resolution, 0.0, vmodel.bounds[0], 0.0, 
                                   -vmodel.resolution, vmodel.bounds[4]),
               'blockxsize': 256, 'blockysize': 256, 'tiled': True,
               'compress': 'deflate', 'interleave': 'pixel'}
    elev = vmodel.bounds[2] + vmodel.nz * vmodel.resolution
    with rio.Env():
        with rio.open(filename, 'w', **profile) as dst:
            dst.write(data)
            dst.build_overviews([2,4,8], Resampling.average)
            for i in range(vmodel.nz):
                description = f'{elev[i]:.02f}m'
                dst.set_band_description(i+1, description)


def multivoxel_worker(inputs):
    """
    Run the voxelization for a single scan
    """
    config,rxp_fn,transform_fn,rdbx_fn,prefix = inputs
    vgrid = voxelization.VoxelGrid(dtm_filename=config['dtm'])
    vgrid.add_riegl_scan_position(rxp_fn, transform_fn, rdbx_file=rdbx_fn)
    vgrid.voxelize_scan(config['bounds'], config['voxelsize'], save_counts=True)
    vgrid.write_grids(prefix)
    return vgrid.filenames


def run_multivoxel_grids(rxp_fn, transform_fn, config,
    rdbx_file=None, outdir=None, ncpu=1):
    """
    Multiprocessing of multiple scans on a voxel grid
    """
    npos = len(rxp_fn)
    progress_bar = tqdm(total=npos)

    def update_multivoxel_outputs(result):
        """
        Callback function for parallelized voxel processing
        """
        if result is not None:
            name = result[0]
            filenames = result[1]
            config['positions'][name] = filenames
        progress_bar.update()

    pool = mp.Pool(ncpu)
    for i in range(npos):
        fn = os.path.basename(rxp_fn[i])
        rdbx_fn = None if rdbx_file is None else rdbx_file[i]
        name = os.path.splitext(fn)[0]
        prefix = f'{outdir}/{name}'
        pool.apply_async(multivoxel_worker, 
                         (config,rxp_fn[i],transform_fn[i],rdbx_fn,prefix), 
                         callback=update_multivoxel_outputs)
    pool.close()
    pool.join()


def voxel_bounds_from_file(path):
    """
    Read bounding box from LAZ header (no point data loaded).
 
    Returns
    -------
    (xmin, ymin, zmin, xmax, ymax, zmax)
    """
    with laspy.open(path) as f:
        hdr = f.header
    mins = np.asarray(hdr.min)
    maxs = np.asarray(hdr.max)
    return (mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2])
 
 
def _normalize(vx, vy, vz):
    norm = np.sqrt(vx**2 + vy**2 + vz**2)
    return vx / norm, vy / norm, vz / norm