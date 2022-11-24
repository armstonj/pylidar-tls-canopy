#!/usr/bin/env python3
"""
riegl_canopy

Code for gridding scans

John Armston
University of Maryland
October 2022
"""


import numpy as np
from numba import njit

import rasterio as rio
from rasterio.enums import Resampling
from affine import Affine

from . import RIO_DEFAULT_PROFILE


class RIEGLGrid:
    def __init__(self, ncols, nrows, xmin, ymax, nvars=1, 
        resolution=1, init_cntgrid=False, profile=RIO_DEFAULT_PROFILE):
        self.profile = profile
        self.nrows = nrows
        self.ncols = ncols
        self.xmin = xmin
        self.ymax = ymax
        self.nvars = nvars
        self.resolution = resolution
        self.init_cntgrid = init_cntgrid

    def __enter__(self):
        self.init_grid(init_cntgrid=self.init_cntgrid, count=self.nvars)
        return self

    def __exit__(self, type, value, traceback):
        self.outgrid = None

    def init_grid(self, **kwargs):
        """
        Initialize output grid
        """
        self.profile.update(height=self.nrows, width=self.ncols,
            transform=Affine(self.resolution, 0.0, self.xmin, 0.0, 
            -self.resolution, self.ymax))

        for key,value in kwargs.items():
            if key in self.profile:
                self.profile[key] = value

        self.outgrid = np.full((self.nvars,self.nrows,self.ncols), 
            self.profile['nodata'], dtype=self.profile['dtype'])
        if self.init_cntgrid:
            self.cntgrid = np.zeros((self.nvars,self.nrows,self.ncols), dtype=np.uint32)

    def insert_values(self, values, xidx, yidx, zidx):
        """
        Insert values to indexed location on grid
        """
        self.outgrid[zidx,yidx,xidx] = values

    def add_values(self, values, xidx, yidx, zidx):
        """
        Add values to indexed location on grid
        """
        sum_by_idx(values, xidx, yidx, zidx, self.profile['nodata'], 
            self.outgrid, self.cntgrid)

    def finalize_grid(self):
        """
        Finalize grid
        """
        if self.init_cntgrid:
            tmp = np.full((self.nvars,self.nrows,self.ncols),
            self.profile['nodata'], dtype=self.profile['dtype'])
            np.divide(self.outgrid, self.cntgrid, out=tmp, where=self.cntgrid > 0, dtype=np.float32)
            self.outgrid = tmp

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
def sum_by_idx(values, xidx, yidx, zidx, nodata, outgrid, cntgrid):
    """
    Numba function to sum values over a grid
    """
    for i in range(values.shape[0]):
        if outgrid[zidx[i],yidx[i],xidx[i]] != nodata:
            outgrid[zidx[i],yidx[i],xidx[i]] += values[i]
        else:
            outgrid[zidx[i],yidx[i],xidx[i]] = values[i]
        cntgrid[zidx[i],yidx[i],xidx[i]] += 1

