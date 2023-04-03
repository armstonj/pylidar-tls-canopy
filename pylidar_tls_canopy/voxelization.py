#!/usr/bin/env python3
"""
pylidar_tls_canopy

Code for voxelization

John Armston
University of Maryland
December 2022
"""

import sys
import json
import numpy as np

import numba as nb
from numba import njit

import rasterio as rio
from rasterio.enums import Resampling
from affine import Affine

from . import riegl_io
from . import RIO_DEFAULT_PROFILE


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

    def read_voxelgrids(self, z=0):
        """
        Read all the data required to invert the Pgap model
        """
        sh = (self.npos, self.ny, self.nx)
        self.voxelgrids = {}
        for k in ('pgap','zeni','vwts'):
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
            self.read_voxelgrids(z=i)
            if weights:
                w = self.voxelgrids['vwts']
            else:
                w = np.ones(self.voxelgrids['vwts'].shape, dtype=np.float32)
            nscans[i] = np.sum(self.voxelgrids['vwts'] > 0, axis=0, dtype=np.uint8)
            paiv[i],paih[i] = run_linear_model_numba(self.voxelgrids['zeni'], 
                self.voxelgrids['pgap'], w, null=self.nodata, min_n=min_n)

        return paiv,paih,nscans        

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


class VoxelGrid:
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

    def add_riegl_scan_position(self, rxp_file, transform_file, rdbx_file=None, 
        chunk_size=10000000):
        """
        Add a scan position using rdbx and rxp
        """
        if rdbx_file is None:
            self.add_riegl_scan_position_rxp(rxp_file, transform_file, points=True)
        else:
            self.add_riegl_scan_position_rxp(rxp_file, transform_file, points=False)
            self.add_riegl_scan_position_rdbx(rdbx_file, transform_file, chunk_size=chunk_size)

        self.classify_ground(self.points['x'], self.points['y'],
                    self.points['z'], self.count)

    def add_riegl_scan_position_rdbx(self, rdbx_file, transform_file, chunk_size=10000000):
        """
        Add a scan position point data using rdbx
        """
        rdb_attributes = {'riegl.xyz': 'riegl_xyz','riegl.target_index': 'target_index',
            'riegl.target_count': 'target_count', 'riegl.scan_line_index': 'scanline', 
            'riegl.shot_index_line': 'scanline_idx'} 
        with riegl_io.RDBFile(rdbx_file, chunk_size=chunk_size, attributes=rdb_attributes,
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
                            self.pulse_scanline, self.pulse_scanline_idx, self.points[name])

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
                self.pulse_scanline = rxp.get_data('scanline')
                self.pulse_scanline_idx = rxp.get_data('scanline_idx')

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

        run_traverse_voxels(self.x0, self.y0, self.z0, self.ground, self.points['x'].data, 
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
def run_traverse_voxels(x0, y0, z0, gnd, x, y, z, dx, dy, dz, target_count, voxdimx, voxdimy, voxdimz,
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

            missed = False
            for i in range(target_count):
                if (vidx == vox_idx[i]) and (gnd[i] == 0):
                    hits[vidx] += w
                    phit[vidx] += plen
                    woccl += w
                    wmiss -= w
                elif (gnd[i] == 1):
                    woccl += w
                    wmiss -= w
                else:
                    missed = True

            occl[vidx] += woccl
            miss[vidx] += wmiss
            if missed:
                pmiss[vidx] += plen

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

