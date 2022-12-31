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
    def __init__(self, bounds, voxelsize, nodata=-9999, profile=RIO_DEFAULT_PROFILE):
        self.bounds = bounds
        self.voxelsize = voxelsize
        self.nodata = nodata
        self.profile = profile

        self.nx = int( (bounds[3] - bounds[0]) // voxelsize)
        self.ny = int( (bounds[4] - bounds[1]) // voxelsize)
        self.nz = int( (bounds[5] - bounds[2]) // voxelsize)
        
        self.voxdimx = bounds[3] - bounds[0]
        self.voxdimy = bounds[4] - bounds[1]
        self.voxdimz = bounds[5] - bounds[2]
        
        self.nvox = int(self.nx * self.ny * self.nz)

        self.profile.update(height=self.ny, width=self.nx, count=self.nz,
            transform=Affine(self.voxelsize, 0.0, self.bounds[0], 0.0,
            -self.voxelsize, self.bounds[4]), nodata=self.nodata)

    def add_riegl_scan_position_rxp(self, rxp_file, transform_file):
        """
        Add a scan position using rxp
        VZ400 and/or pulse rate <=300 kHz only
        """
        with riegl_io.RXPFile(rxp_file, transform_file=transform_file) as rxp:
            self.x = rxp.get_data('x', return_points_by_pulse=True)
            self.y = rxp.get_data('y', return_points_by_pulse=True)
            self.z = rxp.get_data('z', return_points_by_pulse=True)
            
            self.count = rxp.get_data('target_count')
            self.azimuth = rxp.get_data('azimuth')
            self.zenith = rxp.get_data('zenith')
            
            self.x0 = rxp.transform[3,0] 
            self.y0 = rxp.transform[3,1]
            self.z0 = rxp.transform[3,2]

    def voxelize_scan(self):
        """
        Voxelize the scan data
        """
        hits = np.zeros(self.nvox, dtype=float)
        miss = np.zeros(self.nvox, dtype=float)
        occl = np.zeros(self.nvox, dtype=float)
        plen = np.ones(self.nvox, dtype=float)

        dx,dy,dz = self.get_direction_vector(self.zenith, self.azimuth)

        bounds = nb.typed.List(self.bounds)
        run_traverse_voxels(self.x0, self.y0, self.z0, self.x, self.y, self.z, dx, dy, dz, self.count, 
            self.voxdimx, self.voxdimy, self.voxdimz, self.nx, self.ny, self.nz, 
            bounds, self.voxelsize, hits, miss, occl, plen)

        self.voxelgrids = {}
        nshots = hits + miss

        self.voxelgrids['vwts'] = np.zeros(self.nvox, dtype=float)
        np.divide(nshots * plen, nshots + occl, out=self.voxelgrids['vwts'], where=nshots > 0)

        self.voxelgrids['pgap'] = np.full(self.nvox, self.nodata, dtype=float)
        np.divide(miss, nshots, out=self.voxelgrids['pgap'], where=nshots > 0)

        self.voxelgrids['vcls'] = self.classify_voxels(hits, miss, occl)

    def get_direction_vector(self, zenith, azimuth):
        """
        Calculate the direction vector
        """
        dx = np.sin(zenith) * np.sin(azimuth)
        dy = np.sin(zenith) * np.cos(azimuth)
        dz = np.cos(zenith)
        return dx,dy,dz

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
        classification = np.zeros_like(hits)

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
        new_shape = (self.nz, self.ny, self.nx)
        for k in self.voxelgrids:
            with rio.Env():
                filename = f'{prefix}_{k}.tif'
                with rio.open(filename, 'w', **self.profile) as dst:
                    dst.write(self.voxelgrids[k].reshape(new_shape))
                    dst.build_overviews([2,4,8], Resampling.average)
                    for i in range(self.nz):
                        height = self.bounds[2] + i * self.voxelsize
                        description = f'{height:.02f}m'
                        dst.set_band_description(i+1, description)


@njit
def run_traverse_voxels(x0, y0, z0, x, y, z, dx, dy, dz, target_count, voxdimx, voxdimy, voxdimz,
                        nx, ny, nz, bounds, voxelsize, hits, miss, occl, plen):
    """
    Loop through each pulse and run voxel traversal
    """
    max_nreturns = np.max(target_count)
    vox_idx = np.empty(max_nreturns, dtype=np.uint32)
    for i in range(target_count.shape[0]):        
        traverse_voxels(x0, y0, z0, x[:,i], y[:,i], z[:,i], dx[i], dy[i], dz[i],
            nx, ny, nz, voxdimx, voxdimy, voxdimz, bounds, voxelsize, target_count[i],
            hits, miss, occl, plen, vox_idx)
    

@njit
def traverse_voxels(x0, y0, z0, x1, y1, z1, dx, dy, dz, nx, ny, nz, voxdimx, voxdimy, voxdimz,
                    bounds, voxelsize, target_count, hits, miss, occl, plen, vox_idx):
    """
    A fast and simple voxel traversal algorithm through a 3D voxel space (J. Amanatides and A. Woo, 1987)
    Inputs:
       x0, y0, z0
       x1, y1, z1
       dx, dy, dz
       nX, nY, nZ
       bounds
       voxelSize
       number_of_returns
       voxIdx
    Outputs:
       hits
       miss
       occl
       plen
    """
    intersect, tmin, tmax = grid_intersection(x0, y0, z0, dx, dy, dz, bounds)    
    if intersect:
        
        tmin = max(0, tmin)
        tmax = min(1, tmax)

        startX = x0 + tmin * dx
        startY = y0 + tmin * dy
        startZ = z0 + tmin * dz
        
        x = int( (startX - bounds[0]) // voxdimx) * nx
        y = int( (startY - bounds[1]) // voxdimy) * ny
        z = int( (startZ - bounds[2]) // voxdimz) * nz               
        
        for i in range(target_count):
            px = int( (x1[i] - bounds[0]) // voxdimx) * nx
            py = int( (y1[i] - bounds[1]) // voxdimy) * ny
            pz = int( (z1[i] - bounds[2]) // voxdimz) * nz
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
            
            for i in range(target_count):
                if vidx == vox_idx[i]:
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
