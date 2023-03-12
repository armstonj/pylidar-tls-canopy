#!/usr/bin/env python3
"""
pylidar_tls_canopy

Code for gridding scans

John Armston
University of Maryland
October 2022
"""

import sys
import numpy as np
from numba import njit

import rasterio as rio
from rasterio.enums import Resampling
from affine import Affine

from . import riegl_io
from . import leaf_io
from . import RIO_DEFAULT_PROFILE


class LidarGrid:
    def __init__(self, ncols, nrows, xmin, ymax, count=1, nodata=-9999,
        resolution=1, init_cntgrid=False):
        self.nrows = nrows
        self.ncols = ncols
        self.xmin = xmin
        self.ymax = ymax
        self.count = count
        self.nodata = nodata
        self.resolution = resolution
        self.init_cntgrid = init_cntgrid
        self.init_grid()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.outgrid = None

    def init_grid(self, profile=RIO_DEFAULT_PROFILE, **kwargs):
        """
        Initialize output grid
        """
        self.profile = profile
        self.profile.update(height=self.nrows, width=self.ncols,
            transform=Affine(self.resolution, 0.0, self.xmin, 0.0, 
            -self.resolution, self.ymax), count=self.count, nodata=self.nodata)

        for key,value in kwargs.items():
            if key in self.profile:
                self.profile[key] = value

        self.outgrid = np.full((self.count,self.nrows,self.ncols), 
            self.profile['nodata'], dtype=self.profile['dtype'])
        if self.init_cntgrid:
            self.cntgrid = np.zeros((self.count,self.nrows,self.ncols), dtype=np.uint32)

    def insert_values(self, values, xidx, yidx, zidx):
        """
        Insert values to indexed location on grid
        """
        if np.isscalar(zidx):
            zidx = np.full(yidx.shape, zidx, dtype=yidx.dtype)
        i = (xidx >= 0) & (yidx >= 0) & (zidx >= 0)
        i &= (xidx < self.outgrid.shape[2]) & (yidx < self.outgrid.shape[1]) & (zidx < self.outgrid.shape[0]) 
        self.outgrid[zidx[i],yidx[i],xidx[i]] = values[i]

    def add_values(self, values, xidx, yidx, zidx, method='MEAN'):
        """
        Add values to indexed location on grid
        """
        if np.isscalar(zidx):
            zidx = np.full(yidx.shape, zidx, dtype=yidx.dtype)

        i = (xidx >= 0) & (yidx >= 0) & (zidx >= 0)
        i &= (xidx < self.outgrid.shape[2]) & (yidx < self.outgrid.shape[1]) & (zidx < self.outgrid.shape[0])

        add_by_idx(values[i], xidx[i], yidx[i], zidx[i], self.profile['nodata'], 
            self.outgrid, self.cntgrid, method=method)

    def add_column(self, values, xidx, zidx=0, method='MEAN'):
        """
        Add a profile to indexed column on grid
        """
        yidx = np.arange(values.shape[0], dtype=int)
        if np.isscalar(xidx):
            xidx = np.full(values.shape[0], xidx, dtype=int)
        if np.isscalar(zidx):
            zidx = np.full(values.shape[0], zidx, dtype=int)
        invalid = np.isnan(values) | (values == self.profile['nodata'])
        invalid &= (xidx < 0) | (xidx >= self.outgrid.shape[2]) 
        add_by_idx(values, xidx[~invalid], yidx[~invalid], zidx[~invalid], 
            self.profile['nodata'], self.outgrid, self.cntgrid, method=method)

    def finalize_grid(self, method='MEAN'):
        """
        Finalize grid
        """
        if method == 'MEAN':
            tmp = np.full((self.count,self.nrows,self.ncols),
                self.profile['nodata'], dtype=self.profile['dtype'])
            np.divide(self.outgrid, self.cntgrid, out=tmp, where=self.cntgrid > 0, dtype=np.float32)
            self.outgrid = tmp

    def get_grid(self):
        """
        Return a copy of the grid
        """
        return self.outgrid.copy()

    def write_grid(self, filename, descriptions=None):
        """
        Write the results to file
        """
        with rio.Env():
            with rio.open(filename, 'w', **self.profile) as dst:
                dst.write(self.outgrid)
                dst.build_overviews([2,4,8], Resampling.average)
                if descriptions is not None:
                    for i in range(self.outgrid.shape[0]):
                        dst.set_band_description(i+1, descriptions[i])


@njit
def add_by_idx(values, xidx, yidx, zidx, nodata, outgrid, cntgrid, method='SUM'):
    """
    Numba function to add values over a grid
    """
    for i in range(values.shape[0]):
        if outgrid[zidx[i],yidx[i],xidx[i]] != nodata:
            if method in ('MEAN','SUM'):
                outgrid[zidx[i],yidx[i],xidx[i]] += values[i]
            elif method == 'MAX':
                if values[i] > outgrid[zidx[i],yidx[i],xidx[i]]:
                    outgrid[zidx[i],yidx[i],xidx[i]] = values[i]
            elif method == 'MIN':
                if values[i] < outgrid[zidx[i],yidx[i],xidx[i]]:
                    outgrid[zidx[i],yidx[i],xidx[i]] = values[i]
        else:
            outgrid[zidx[i],yidx[i],xidx[i]] = values[i]
        cntgrid[zidx[i],yidx[i],xidx[i]] += 1


def grid_rdbx_cartesian(rdbx_list, transform_list, res, attribute='z', method='MAX', extent=[50,50], 
    ulc=[-25,25], planefit=None):
    """
    Wrapper function to grid the REIGL point data on a cartesian grid
    """
    if isinstance(rdbx_list, str):
        rdbx_list = [rdbx_list]
        transform_list = [transform_list]
    ncols = int( extent[0] // res + 1 )
    nrows = int( extent[1] // res + 1 )
    with LidarGrid(ncols, nrows, ulc[0], ulc[1], resolution=res, init_cntgrid=True) as grd:    
        for rdbx_fn,transform_fn in zip(rdbx_list,transform_list):
            with riegl_io.RDBFile(rdbx_fn, transform_file=transform_fn) as rdb:
                while rdb.point_count_current < rdb.point_count_total:
                    rdb.read_next_chunk()
                    if rdb.point_count > 0:
                        xidx = (rdb.get_chunk('x') - ulc[0]) // res
                        yidx = (ulc[1] - rdb.get_chunk('y')) // res
                        if planefit is not None:
                            vals = rdb.get_chunk('z') - (planefit['Parameters'][1] * rdb.get_chunk('x') +
                                planefit['Parameters'][2] * rdb.get_chunk('y') + planefit['Parameters'][0]) 
                        else:
                            vals = rdb.get_chunk(attribute)
                        valid = (xidx >= 0) & (xidx < ncols) & (yidx >= 0) & (yidx < nrows)
                        grd.add_values(vals[valid], np.uint16(xidx[valid]), np.uint16(yidx[valid]), 
                            0, method=method)
        grd.finalize_grid(method=method)
        scan_grid = grd.get_grid()
    return scan_grid


def grid_rxp_cartesian(rxp_list, transform_list, res, attribute='z', method='MAX', extent=[50,50], 
    ulc=[-25,25], planefit=None):
    """
    Wrapper function to grid the RIEGL pulse data on a cartesian grid
    """
    if isinstance(rxp_list, str):
        rxp_list = [rxp_list]
        transform_list = [transform_list]
    ncols = int( extent[0] // res + 1 )
    nrows = int( extent[1] // res + 1 )
    with LidarGrid(ncols, nrows, ulc[0], ulc[1], resolution=res, init_cntgrid=True) as grd:
        for rxp_fn,transform_fn in zip(rxp_list,transform_list):
            with riegl_io.RXPFile(rxp_fn, transform_file=transform_fn) as rxp:
                if attribute in rxp.pulses:
                    return_as_point_attribute = False
                else:
                    return_as_point_attribute = True
                xidx = (rxp.get_data('x', return_as_point_attribute=return_as_point_attribute) - ulc[0]) // res
                yidx = (ulc[1] - rxp.get_data('y', return_as_point_attribute=return_as_point_attribute)) // res
                if planefit is not None:
                    vals = rxp.get_data('z', return_as_point_attribute=return_as_point_attribute) - (planefit['Parameters'][0] + 
                        planefit['Parameters'][1] * rxp.get_data('x', return_as_point_attribute=return_as_point_attribute) + 
                        planefit['Parameters'][2] * rxp.get_data('y', return_as_point_attribute=return_as_point_attribute))
                else:
                    vals = rxp.get_data(attribute, return_as_point_attribute=return_as_point_attribute)
                valid = (xidx >= 0) & (xidx < ncols) & (yidx >= 0) & (yidx < nrows)
                grd.add_values(vals[valid], np.uint16(xidx[valid]), np.uint16(yidx[valid]), 
                    0, method=method)
        grd.finalize_grid(method=method) 
        scan_grid = grd.get_grid()
    return scan_grid


def grid_leaf_spherical(leaf_fn, resolution, attribute='range',
    method='MEAN', sensor_height=1.5, transform=True):
    """
    Wrapper function to grid the LEAF point data on a spherical grid
    """
    res = np.radians(resolution)
    ncols = int( (2 * np.pi) // res + 1 )
    nrows = int( np.pi // res + 1 )
    with leaf_io.LeafScanFile(leaf_fn, sensor_height=sensor_height, transform=transform) as leaf:
        if leaf.data.empty:
            return None
        with LidarGrid(ncols, nrows, 0, np.pi, resolution=res, init_cntgrid=True) as grd:
            xidx = np.int16(leaf.data.azimuth.to_numpy() // res)
            yidx = np.int16(leaf.data.zenith.to_numpy() // res)
            vals = leaf.data[attribute].to_numpy()
            valid = (xidx >= 0) & (xidx < ncols) & (yidx >= 0) & (yidx < nrows)
            grd.add_values(vals[valid], xidx[valid], yidx[valid], 0, method=method)
            grd.finalize_grid(method=method)
            scan_grid = grd.get_grid()
    return scan_grid


def grid_rdbx_spherical(rdbx_fn, transform_fn, resolution, attribute='range', 
    first_only=False, method='MEAN'):
    """
    Wrapper function to grid the REIGL point data on a spherical grid
    """
    res = np.radians(resolution)
    ncols = int( (2 * np.pi) // res + 1 )
    nrows = int( np.pi // res + 1 )
    with riegl_io.RDBFile(rdbx_fn, transform_file=transform_fn, first_only=first_only) as rdb:
        with LidarGrid(ncols, nrows, 0, np.pi, resolution=res, init_cntgrid=True) as grd:
            while rdb.point_count_current < rdb.point_count_total:
                rdb.read_next_chunk()
                if rdb.point_count > 0:
                    xidx = np.int16(rdb.get_chunk('azimuth') // res)
                    yidx = np.int16(rdb.get_chunk('zenith') // res)
                    vals = rdb.get_chunk(attribute)
                    valid = (xidx >= 0) & (xidx < ncols) & (yidx >= 0) & (yidx < nrows)
                    grd.add_values(vals[valid], xidx[valid], yidx[valid], 0, method=method)
            grd.finalize_grid(method=method)
            scan_grid = grd.get_grid()
    return scan_grid


def grid_rxp_spherical(rxp_fn, transform_fn, resolution, attribute='zenith', 
    first_only=False, method='MEAN'):
    """
    Wrapper function to grid the REIGL pulse data on a spherical grid
    """
    res = np.radians(resolution)
    ncols = int( (2 * np.pi) // res + 1 )
    nrows = int( np.pi // res + 1 )
    with riegl_io.RXPFile(rxp_fn, transform_file=transform_fn) as rxp:
        if attribute in rxp.pulses: 
            return_as_point_attribute = False
        else:
            return_as_point_attribute = True
        with LidarGrid(ncols, nrows, 0, np.pi, resolution=res, init_cntgrid=True) as grd:
            xidx = rxp.get_data('azimuth', return_as_point_attribute=return_as_point_attribute) // res
            yidx = rxp.get_data('zenith', return_as_point_attribute=return_as_point_attribute) // res
            vals = rxp.get_data(attribute, return_as_point_attribute=return_as_point_attribute)
            valid = (xidx >= 0) & (xidx < ncols) & (yidx >= 0) & (yidx < nrows)
            if first_only:
                index = rxp.get_data('target_index', return_as_point_attribute=return_as_point_attribute)
                valid &= (index == 1)
            grd.add_values(vals[valid], np.uint16(xidx[valid]), np.uint16(yidx[valid]), 
                0, method=method)
            grd.finalize_grid(method=method)
            scan_grid = grd.get_grid()
    return scan_grid


def grid_rdbx_scan(rdbx_fn, transform_fn=None, attribute='reflectance'):
    """
    Wrapper function to grid the REIGL point data on a scan grid
    """
    with riegl_io.RDBFile(rdbx_fn, chunk_size=100000, transform_file=transform_fn) as rdb:
        with LidarGrid(rdb.maxc+1, rdb.maxr+1, 0, rdb.maxr, count=rdb.max_target_count) as grd:
            while rdb.point_count_current < rdb.point_count_total:
                rdb.read_next_chunk()
                if rdb.point_count > 0:
                    xidx = rdb.get_chunk('scanline')
                    yidx = rdb.get_chunk('scanline_idx')
                    zidx = rdb.get_chunk('target_index') - 1
                    vals = rdb.get_chunk(attribute)
                    grd.insert_values(vals, xidx, yidx, zidx)
            scan_grid = grd.get_grid()
    return scan_grid


def grid_rxp_scan(rxp_fn, transform_fn=None, attribute='target_count'):
    """
    Wrapper function to grid the REIGL pulse data on a scan grid
    """
    with riegl_io.RXPFile(rxp_fn, transform_file=transform_fn) as rxp:
        if attribute in rxp.pulses:
            return_as_point_attribute = False
            nvars = 1
            zidx = 0
        else:
            return_as_point_attribute = True
            nvars = rxp.max_target_count
            zidx = rxp.get_data('target_index') - 1
        with LidarGrid(rxp.maxc+1, rxp.maxr+1, 0, rxp.maxr, count=nvars) as grd:
            xidx = rxp.get_data('scanline', return_as_point_attribute=return_as_point_attribute)
            yidx = rxp.get_data('scanline_idx', return_as_point_attribute=return_as_point_attribute)
            vals = rxp.get_data(attribute, return_as_point_attribute=return_as_point_attribute)
            grd.insert_values(vals, xidx, yidx, zidx)
            scan_grid = grd.get_grid()
    return scan_grid

