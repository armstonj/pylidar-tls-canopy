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


def grid_riegl_cartesian(riegl_list, transform_list, res, attribute='z', method='MAX', extent=[50,50], 
    ulc=[-25,25], planefit=None, query_str=None, driver='rdbx'):
    """
    Wrapper function to grid the RIEGL pulse data on a cartesian grid
    """
    if isinstance(riegl_list, str):
        riegl_list = [riegl_list]
        transform_list = [transform_list]
    ncols = int( extent[0] // res + 1 )
    nrows = int( extent[1] // res + 1 )
    with LidarGrid(ncols, nrows, ulc[0], ulc[1], resolution=res, init_cntgrid=True) as grd:
        for fn,transform_fn in zip(riegl_list,transform_list):
            data = {}
            var = ('x','y','z',attribute)
            if driver == 'rxp':
                with riegl_io.RXPFile(fn, transform_file=transform_fn, query_str=query_str) as rxp:
                    return_as_point_attribute = False if attribute in rxp.pulses else True
                    for col in set(var):
                        data[col] = rxp.get_data(col, return_as_point_attribute=return_as_point_attribute)
            elif driver == 'rdbx':
                with riegl_io.RDBFile(fn, transform_file=transform_fn, query_str=query_str) as rdb:
                    for col in set(var):
                        data[col] = rdb.get_data(col)
            else:
                msg = f'{driver} is not a valid driver name'
                raise ValueError(msg)

            xidx = (data['x'] - ulc[0]) // res
            yidx = (ulc[1] - data['y']) // res
            if planefit is not None:
                vals = data['z'] - (planefit['Parameters'][0] + planefit['Parameters'][1] * data['x'] + 
                    planefit['Parameters'][2] * data['y'])
            else:
                vals = data[attribute]
            valid = (xidx >= 0) & (xidx < ncols) & (yidx >= 0) & (yidx < nrows)
            grd.add_values(vals[valid], np.uint16(xidx[valid]), np.uint16(yidx[valid]), 0, method=method)
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


def grid_riegl_spherical(fn, transform_fn, resolution, attribute='zenith', method='MEAN', 
    query_str=None, driver='rdbx'):
    """
    Wrapper function to grid the REIGL pulse data on a spherical grid
    """
    res = np.radians(resolution)
    ncols = int( (2 * np.pi) // res + 1 )
    nrows = int( np.pi // res + 1 )

    data = {}
    var = ('azimuth','zenith',attribute)
    if driver == 'rxp':
        with riegl_io.RXPFile(fn, transform_file=transform_fn, query_str=query_str) as rxp:
            return_as_point_attribute = False if attribute in rxp.pulses else True
            for col in set(var):
                data[col] = rxp.get_data(col, return_as_point_attribute=return_as_point_attribute)
    elif driver == 'rdbx':
        with riegl_io.RDBFile(fn, transform_file=transform_fn, query_str=query_str) as rdb:
            for col in set(var):
                data[col] = rdb.get_data(col)
    else:
        msg = f'{driver} is not a valid driver name'
        raise ValueError(msg)

    with LidarGrid(ncols, nrows, 0, np.pi, resolution=res, init_cntgrid=True) as grd:
        xidx = data['azimuth'] // res
        yidx = data['zenith'] // res
        valid = (xidx >= 0) & (xidx < ncols) & (yidx >= 0) & (yidx < nrows)
        grd.add_values(data[attribute][valid], np.uint16(xidx[valid]), 
            np.uint16(yidx[valid]), 0, method=method)
        grd.finalize_grid(method=method)
        scan_grid = grd.get_grid()
    return scan_grid


def grid_riegl_scan(fn, transform_fn=None, attribute='reflectance', query_str=None, driver='rdbx'):
    """
    Wrapper function to grid the REIGL pulse data on a scan grid
    """
    data = {}
    var = ('scanline','scanline_idx',attribute)
    if driver == 'rxp':
        with riegl_io.RXPFile(fn, transform_file=transform_fn, query_str=query_str) as rxp:
            if attribute in rxp.pulses:
                return_as_point_attribute = False
                nvars = 1
                zidx = 0 
            else:
                return_as_point_attribute = True
                nvars = rxp.max_target_count
                zidx = rxp.get_data('target_index') - 1
            maxc = rxp.maxc
            maxr = rxp.maxr
            for col in set(var):
                data[col] = rxp.get_data(col, return_as_point_attribute=return_as_point_attribute)
    elif driver == 'rdbx':
        with riegl_io.RDBFile(fn, transform_file=transform_fn, query_str=query_str) as rdb:
            maxc = rdb.maxc
            maxr = rdb.maxr
            nvars = rdb.max_target_count
            for col in set(var):
                data[col] = rdb.get_data(col)
            zidx = rdb.get_data('target_index') - 1
    else:
        msg = f'{driver} is not a valid driver name'
        raise ValueError(msg)

    with LidarGrid(maxc+1, maxr+1, 0, maxr, count=nvars) as grd:
        grd.insert_values(data[attribute], data['scanline'], data['scanline_idx'], zidx)
        scan_grid = grd.get_grid()

    return scan_grid
