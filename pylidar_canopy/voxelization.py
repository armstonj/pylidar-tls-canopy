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

        self.nx = (bounds[3] - bounds[0]) // voxelsize[0]
        self.ny = (bounds[4] - bounds[1]) // voxelsize[1]
        self.nz = (bounds[5] - bounds[2]) // voxelsize[2]
        
        self.voxdimx = bounds[3] - bounds[0]
        self.voxdimy = bounds[4] - bounds[1]
        self.voxdimz = bounds[5] - bounds[2]
        
        self.nvox = self.nx * self.ny * self.nz

        self.profile.update(height=self.ny, width=self.nx, count=self.nz,
            transform=Affine(self.voxelsize, 0.0, self.bounds[0], 0.0,
            -self.voxelsize, self.bounds[4]), nodata=self.nodata)

    def add_riegl_scan_position_rxp(self, rxp_file, transform_file):
        """
        Add a scan position using rxp
        VZ400 and/or pulse rate <=300 kHz only
        """
        with riegl_io.RXPFile(rxp_file, transform_file=transform_file) as rxp:
            self.x = rxp.get_data('x', return_point_by_pulse=True)
            self.y = rxp.get_data('y', return_point_by_pulse=True)
            self.z = rxp.get_data('z', return_point_by_pulse=True)
            self.count = rxp.get_data('target_count')
            self.azimuth = rxp.get_data('azimuth')
            self.zenith = rxp.get_data('zenith')
            self.x0 = rxp.transform[0,4] 
            self.y0 = rxp.transform[1,4]
            self.z0 = rxp.transform[2,4]

    def voxelize_scan(self):
        """
        Voxelize the scan data
        """
        hits = np.zeros(self.nvox, dtype=float)
        miss = np.zeros(self.nvox, dtype=float)
        occl = np.zeros(self.nvox, dtype=float)
        plen = np.ones(self.nvox, dtype=float)

        dx,dy,dz = self.get_direction_vector(self.zenith, self.azimuth)

        run_traverse_voxels(self.x0, self.y0, self.z0, self.x, self.y, self.z, dx, dy, dz, self.count, 
            self.voxdimx, self.voxdimy, self.voxdimz, self.nx, self.ny, self.nz, 
            self.bounds, self.voxelsize, hits, miss, occl, plen)

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
    vox_idx = numpy.empty(max_nreturns, dtype=int)
    for i in range(number_of_returns.shape[0]):        
        traverse_voxels(x0, y0, z0, x[:,i], y[:,i], z[:,i], dx[i], dy[i], dz[i],
            nx, ny, nz, voxdimx, voxdimy, voxdimz, bounds, voxelsize, target_count[i],
            hits, miss, occl, plen, vox_idx)
    

@njit
def traverse_voxels(x0, y0, z0, x1, y1, z1, dx, dy, dz, nX, nY, nZ, voxDimX, voxDimY, voxDimZ,
                    bounds, voxelsize, target_count, hitsArr, missArr, occlArr, plenArr, voxIdx):
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
       hitsArr
       missArr
       occlArr
       plenArr
    """
    intersect, tmin, tmax = grid_intersection(x0, y0, z0, dx, dy, dz, bounds)    
    if intersect:
        
        tmin = max(0, tmin)
        tmax = min(1, tmax)

        startX = x0 + tmin * dx
        startY = y0 + tmin * dy
        startZ = z0 + tmin * dz
        
        x = (startX - bounds[0]) // voxDimX * nX
        y = (startY - bounds[1]) // voxDimY * nY
        z = (startZ - bounds[2]) // voxDimZ * nZ               
        
        for i in range(number_of_returns):
            px = (x1[i] - bounds[0]) // voxDimX * nX
            py = (y1[i] - bounds[1]) // voxDimY * nY
            pz = (z1[i] - bounds[2]) // voxDimZ * nZ
            voxIdx[i] = int(px + nX * py + nX * nY * pz)   
        
        if x == nX:
            x -= 1
        if y == nY:
            y -= 1           
        if z == nZ:
            z -= 1
         
        if dx > 0:
            tVoxelX = (x + 1) / nX
            stepX = 1
        elif dx < 0:
            tVoxelX = x / nX
            stepX = -1
        else:
            tVoxelX = (x + 1) / nX
            stepX = 0
        
        if dy > 0:
            tVoxelY = (y + 1) / nY
            stepY = 1
        elif dy < 0:
            tVoxelY = y / nY
            stepY = -1
        else:
            tVoxelY = (y + 1) / nY
            stepY = 0  
        
        if dz > 0:
            tVoxelZ = (z + 1) / nZ
            stepZ = 1
        elif dz < 0:
            tVoxelZ = z / nZ
            stepZ = -1
        else:
            tVoxelZ = (z + 1) / nZ
            stepZ = 0            
        
        voxelMaxX = bounds[0] + tVoxelX * voxDimX
        voxelMaxY = bounds[1] + tVoxelY * voxDimY
        voxelMaxZ = bounds[2] + tVoxelZ * voxDimZ
        
        if dx == 0:
            tMaxX = tmax
            tDeltaX = tmax
        else:
            tMaxX = tmin + (voxelMaxX - startX) / dx
            tDeltaX = voxelSize[0] / abs(dx)
            
        if dy == 0:    
            tMaxY = tmax
            tDeltaY = tmax
        else:
            tMaxY = tmin + (voxelMaxY - startY) / dy
            tDeltaY = voxelSize[1] / abs(dy)
            
        if dz == 0:
            tMaxZ = tmax
            tDeltaZ = tmax
        else:
            tMaxZ = tmin + (voxelMaxZ - startZ) / dz
            tDeltaZ = voxelSize[2] / abs(dz)
        
        wmiss = 1.0
        woccl = 0.0
        if number_of_returns > 0:
            w = 1.0 / number_of_returns
        else:
            w = 0.0
        
        while (x < nX) and (x >= 0) and (y < nY) and (y >= 0) and (z < nZ) and (z >= 0):
            
            vidx = int(x + nX * y + nX * nY * z)
            
            for i in range(number_of_returns):
                if vidx == voxIdx[i]:
                    hitsArr[vidx] += w
                    woccl += w
                    wmiss -= w
            
            occlArr[vidx] += woccl
            missArr[vidx] += wmiss 
                        
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
