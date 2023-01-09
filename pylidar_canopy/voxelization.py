#!/usr/bin/env python3
"""
pylidar_canopy

Code for voxelization

John Armston
University of Maryland
December 2022
"""

import sys
import numpy as np

import numba as nb
from numba import njit

import rasterio as rio
from rasterio.enums import Resampling
from affine import Affine

from . import riegl_io
from . import RIO_DEFAULT_PROFILE


class VoxelGrid:
    """
    Class for a voxel grid of a TLS scan
    """
    def __init__(self, dtm_filename=None, nodata=-9999, profile=RIO_DEFAULT_PROFILE):
        self.dtm_filename = dtm_filename
        self.nodata = nodata
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

    def add_riegl_scan_position(self, rxp_file, transform_file, rdbx_file=None):
        """
        Add a scan position using rdbx and rxp
        """
        if rdbx_file is None:
            self.add_riegl_scan_position_rxp(rxp_file, transform_file, points=True)
        else:
            self.add_riegl_scan_position_rxp(rxp_file, transform_file, points=False)
            self.add_riegl_scan_position_rdbx(rdbx_file, transform_file)

        self.classify_ground(self.points['x'], self.points['y'],
                    self.points['z'], self.count)

    def add_riegl_scan_position_rdbx(self, rdbx_file, transform_file):
        """
        Add a scan position point data using rdbx
        """
        rdb_attributes = {'riegl.xyz': 'riegl_xyz','riegl.target_index': 'target_index',
            'riegl.target_count': 'target_count', 'riegl.scan_line_index': 'scanline', 
            'riegl.shot_index_line': 'scanline_idx'}
        with riegl_io.RDBFile(rdbx_file, chunk_size=100000, attributes=rdb_attributes,
            transform_file=transform_file) as rdb:

            dtype_list = []
            for name in ('x','y','z'):
                dtype_list.append((str(name), '<f8', rdb.max_target_count))
            npulses = self.count.shape[0]
            self.points = np.empty(npulses, dtype=dtype_list)

            while rdb.point_count_current < rdb.point_count_total:
                rdb.read_next_chunk()
                if rdb.point_count > 0:
                    index = rdb.get_chunk('target_index')
                    scanline = rdb.get_chunk('scanline')
                    scanline_idx = rdb.get_chunk('scanline_idx')
                    for name in ('x','y','z'):
                        tmp = rdb.get_chunk(name)
                        riegl_io.get_rdbx_points_by_rxp_pulse(tmp, index, scanline, scanline_idx,
                            self.pulse_id, self.pulse_scanline, self.pulse_scanline_idx, self.points[name])

    def add_riegl_scan_position_rxp(self, rxp_file, transform_file, points=True):
        """
        Add a scan position using rxp
        VZ400 and/or pulse rate <=300 kHz only
        """
        with riegl_io.RXPFile(rxp_file, transform_file=transform_file) as rxp:
            self.count = rxp.get_data('target_count')
            
            self.dx = rxp.get_data('beam_direction_x')
            self.dy = rxp.get_data('beam_direction_y')
            self.dz = rxp.get_data('beam_direction_z')
            
            self.x0 = rxp.transform[3,0] 
            self.y0 = rxp.transform[3,1]
            self.z0 = rxp.transform[3,2]

            if points:
                self.points = rxp.get_points_by_pulse(['x','y','z'])
            else:
                self.pulse_id = rxp.get_data('pulse_id')
                self.pulse_scanline = rxp.get_data('scanline')
                self.pulse_scanline_idx = rxp.get_data('scanline_idx')

    def _init_voxel_grid(self, bounds, voxelsize):
        """
        Initialize the voxel grid
        """
        self.bounds = nb.typed.List(bounds)
        self.voxelsize = voxelsize

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

    def voxelize_scan(self, bounds, voxelsize):
        """
        Voxelize the scan data
        """
        self._init_voxel_grid(bounds, voxelsize)
        hits = np.zeros(self.nvox, dtype=float)
        miss = np.zeros(self.nvox, dtype=float)
        occl = np.zeros(self.nvox, dtype=float)
        zeni = np.zeros(self.nvox, dtype=float)

        run_traverse_voxels(self.x0, self.y0, self.z0, self.ground, self.points['x'].data, 
            self.points['y'].data, self.points['z'].data, self.dx, self.dy, self.dz, 
            self.count, self.voxdimx, self.voxdimy, self.voxdimz, self.nx, self.ny, 
            self.nz, self.bounds, self.voxelsize, hits, miss, occl, zeni)

        self.voxelgrids = {}
        nshots = hits + miss
        nbeams = nshots + occl

        self.voxelgrids['vwts'] = np.full(self.nvox, self.nodata, dtype=float)
        np.divide(nshots, nbeams, out=self.voxelgrids['vwts'], where=nbeams > 0)

        self.voxelgrids['pgap'] = np.full(self.nvox, self.nodata, dtype=float)
        np.divide(miss, nshots, out=self.voxelgrids['pgap'], where=nshots > 0)

        self.voxelgrids['zeni'] = np.full(self.nvox, self.nodata, dtype=float)
        np.divide(zeni, nbeams, out=self.voxelgrids['zeni'], where=nbeams > 0)

        self.voxelgrids['vcls'] = self.classify_voxels(hits, miss, occl)

        self.voxelgrids['hits'] = hits
        self.voxelgrids['miss'] = miss
        self.voxelgrids['occl'] = occl

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
def run_traverse_voxels(x0, y0, z0, gnd, x, y, z, dx, dy, dz, target_count, voxdimx, voxdimy, voxdimz,
                        nx, ny, nz, bounds, voxelsize, hits, miss, occl, zeni):
    """
    Loop through each pulse and run voxel traversal
    """
    max_nreturns = np.max(target_count)
    vox_idx = np.empty(max_nreturns, dtype=np.uint32)
    for i in range(target_count.shape[0]):        
        traverse_voxels(x0, y0, z0, gnd[i,:], x[i,:], y[i,:], z[i,:], dx[i], dy[i], dz[i],
            nx, ny, nz, voxdimx, voxdimy, voxdimz, bounds, voxelsize, target_count[i],
            hits, miss, occl, zeni, vox_idx)
    

@njit
def traverse_voxels(x0, y0, z0, gnd, x1, y1, z1, dx, dy, dz, nx, ny, nz, voxdimx, voxdimy, voxdimz,
                    bounds, voxelsize, target_count, hits, miss, occl, zeni, vox_idx):
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
        
        while (x < nx) and (x >= 0) and (y < ny) and (y >= 0) and (z < nz) and (z >= 0):
            
            vidx = int(x + nx * y + nx * ny * z)
            zeni[vidx] += theta 

            for i in range(target_count):
                if (vidx == vox_idx[i]) and (gnd[i] == 0):
                    hits[vidx] += w
                    woccl += w
                    wmiss -= w
            
            occl[vidx] += woccl
            miss[vidx] += wmiss
             
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
