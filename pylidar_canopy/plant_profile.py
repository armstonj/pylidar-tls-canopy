#!/usr/bin/env python3
"""
pylidar_canopy

Code for vertical profiles

John Armston
University of Maryland
October 2022
"""

import sys
import warnings
import numpy as np
import pandas as pd
from numba import njit

import statsmodels.api as sm

from . import riegl_io
from . import leaf_io


class Jupp2009:
    """
    Class for Foliage Profiles from Jupp 2009
    """
    def __init__(self, hres=0.5, zres=5, ares=45, min_z=0, max_z=70, 
                 min_h=0, max_h=50):
        self.hres = hres
        self.zres = zres
        self.ares = ares
        self.min_z = min_z
        self.max_z = max_z
        self.min_h = min_h
        self.max_h = max_h

        self.min_z_r = np.radians(min_z)
        self.zres_r = np.radians(zres)
        self.ares_r = np.radians(ares)

        self.height_bin = np.arange(self.min_h, self.max_h, self.hres)
        self.zenith_bin = np.arange(self.min_z, self.max_z, self.zres) + self.zres / 2
        self.azimuth_bin = np.arange(0, 360, self.ares) + self.ares / 2 

        nhbins = int( (max_h - min_h) // hres )
        nzbins = int( (max_z - min_z) // zres )
        nabins = int(360 // ares)        

        self.target_output = np.zeros((nzbins, nabins, nhbins), dtype=np.float32)
        self.shot_output = np.zeros((nzbins,nabins,1), dtype=np.float32)

    def add_targets(self, target_height, target_index, target_count, target_zenith, 
        target_azimuth, method='WEIGHTED'):
        """
        Add targets
        """
        h_idx = np.int16((target_height - self.min_h) // self.hres)
        z_idx = np.int16((target_zenith - self.min_z_r) // self.zres_r)
        a_idx = np.int16(target_azimuth // self.ares_r)
        if method == 'WEIGHTED':
            w = 1 / target_count
            sum_by_index_3d(w, z_idx, a_idx, h_idx, self.target_output)
        elif method == 'FIRSTLAST':
            w = np.full(target_count.shape[0], 0.5, dtype=np.float32)
            sum_by_index_3d(w, z_idx, a_idx, h_idx, self.target_output)
        elif method == 'ALL':
            w = np.ones(target_height.shape[0], dtype=np.float32)
            sum_by_index_3d(w, z_idx, a_idx, h_idx, self.target_output)
        elif method == 'FIRST':
            idx = (target_index == 1)
            if np.any(idx):
                w = np.ones(np.count_nonzero(idx), dtype=np.float32)
                sum_by_index_3d(w, z_idx[idx], a_idx[idx], h_idx[idx], 
                    self.target_output)
        else:
            print(f'{method} is not a recognized target weighting method')
            sys.exit()

    def add_shots(self, target_count, shot_zenith, shot_azimuth, method='WEIGHTED'):
        """
        Add shots
        """
        z_idx = np.int16((shot_zenith - self.min_z_r) // self.zres_r)
        a_idx = np.int16(shot_azimuth // self.ares_r)
        if method in ('WEIGHTED','FIRST','FIRSTLAST'):
            shot_cnt = np.ones(target_count.shape[0], dtype=np.float32)
        elif method == 'ALL':
            shot_cnt = target_count.astype(np.float32)
        else:
            print(f'{method} is not a recognized target weighting method')
            sys.exit()
        sum_by_index_2d(shot_cnt, z_idx, a_idx, self.shot_output)

    def add_riegl_scan_position(self, rxp_file, transform_file, planefit, rdbx_file=None,
        method='WEIGHTED', min_zenith=5, max_zenith=70, max_hr=None):
        """
        Add a scan position to the profile
        """
        if rdbx_file is None:
            self.add_riegl_scan_position_rxp(rxp_file, transform_file, planefit,
                method=method, min_zenith=min_zenith, max_zenith=max_zenith, max_hr=max_hr)
        else:
            self.add_riegl_scan_position_rdbx(rdbx_file, rxp_file, transform_file, planefit,
                method=method, min_zenith=min_zenith, max_zenith=max_zenith, max_hr=max_hr)

    def add_riegl_scan_position_rdbx(self, rdbx_file, rxp_file, transform_file, planefit,
        method='WEIGHTED', min_zenith=5, max_zenith=70, max_hr=None):
        """
        Add a scan position to the profile using rdbx
        """
        min_zenith_r = np.radians(min_zenith)
        max_zenith_r = np.radians(max_zenith)
        
        rdb_attributes = {'riegl.xyz': 'riegl_xyz','riegl.target_index': 'target_index',
            'riegl.target_count': 'target_count'}        
        with riegl_io.RDBFile(rdbx_file, chunk_size=100000, attributes=rdb_attributes,
            transform_file=transform_file) as rdb:
            while rdb.point_count_current < rdb.point_count_total:
                rdb.read_next_chunk()
                if rdb.point_count > 0:
                    zenith = rdb.get_chunk('zenith')
                    azimuth = rdb.get_chunk('azimuth')
                    index = rdb.get_chunk('target_index')
                    count = rdb.get_chunk('target_count')
                    x = rdb.get_chunk('x')
                    y = rdb.get_chunk('y')
                    z = rdb.get_chunk('z')
                    height = z - (planefit['Parameters'][1] * x +
                        planefit['Parameters'][2] * y + planefit['Parameters'][0])
                    idx = (zenith >= min_zenith_r) & (zenith < max_zenith_r)
                    if max_hr is not None:
                        hr = rdb.get_chunk('range') * np.sin(zenith)
                        idx &= hr < max_hr
                    if np.any(idx):
                        self.add_targets(height[idx], index[idx], count[idx], zenith[idx],
                        azimuth[idx], method=method)

        with riegl_io.RXPFile(rxp_file, transform_file=transform_file) as rxp:
            azimuth = rxp.get_data('azimuth', return_as_point_attribute=False)
            zenith = rxp.get_data('zenith', return_as_point_attribute=False)
            count = rxp.get_data('target_count', return_as_point_attribute=False)
            idx = (zenith >= min_zenith_r) & (zenith < max_zenith_r)
            if np.any(idx):
                self.add_shots(count[idx], zenith[idx], azimuth[idx], method=method)

    def add_riegl_scan_position_rxp(self, rxp_file, transform_file, planefit,
        method='WEIGHTED', min_zenith=5, max_zenith=70, max_hr=None):
        """
        Add a scan position to the profile using rxp
        VZ400 and/or pulse rate <=300 kHz only
        """
        min_zenith_r = np.radians(min_zenith)
        max_zenith_r = np.radians(max_zenith)

        with riegl_io.RXPFile(rxp_file, transform_file=transform_file) as rxp:
            # Point data
            azimuth = rxp.get_data('azimuth', return_as_point_attribute=True)
            zenith = rxp.get_data('zenith', return_as_point_attribute=True)
            index = rxp.get_data('target_index', return_as_point_attribute=True)
            count = rxp.get_data('target_count', return_as_point_attribute=True)
            x = rxp.get_data('x', return_as_point_attribute=True)
            y = rxp.get_data('y', return_as_point_attribute=True)
            z = rxp.get_data('z', return_as_point_attribute=True)
            height = z - (planefit['Parameters'][1] * x +
                planefit['Parameters'][2] * y + planefit['Parameters'][0])
            idx = (zenith >= min_zenith_r) & (zenith < max_zenith_r)
            if max_hr is not None:
                hr = rxp.get_data('range') * np.sin(zenith)
                idx &= hr < max_hr
            if np.any(idx):
                self.add_targets(height[idx], index[idx], count[idx], zenith[idx],
                azimuth[idx], method=method)

            # Pulse data
            azimuth = rxp.get_data('azimuth', return_as_point_attribute=False)
            zenith = rxp.get_data('zenith', return_as_point_attribute=False)
            count = rxp.get_data('target_count', return_as_point_attribute=False)
            idx = (zenith >= min_zenith_r) & (zenith < max_zenith_r)
            if np.any(idx):
                self.add_shots(count[idx], zenith[idx], azimuth[idx], method=method)

    def add_leaf_scan_position(self, leaf_file, method='WEIGHTED', min_zenith=5, 
        max_zenith=70, sensor_height=None):
        """
        Add a leaf scan position to the profile
        """
        min_zenith_r = np.radians(min_zenith)
        max_zenith_r = np.radians(max_zenith)

        with leaf_io.LeafScanFile(leaf_file, sensor_height=sensor_height) as leaf:
            self.datetime = leaf.datetime
            if not leaf.data.empty:
                # Point data
                azimuth = leaf.data['azimuth'].to_numpy()
                zenith = leaf.data['zenith'].to_numpy()
                count = leaf.data['target_count'].to_numpy()
                for n in (1,2):
                    height = leaf.data[f'h{n:d}'].to_numpy()
                    index = np.full(height.shape, n, dtype=np.uint8)
                    idx = (zenith >= min_zenith_r) & (zenith < max_zenith_r) & ~np.isnan(height)
                    if np.any(idx):
                        self.add_targets(height[idx], index[idx], count[idx], zenith[idx],
                        azimuth[idx], method=method)

                # Pulse data
                idx = (zenith >= min_zenith_r) & (zenith < max_zenith_r)
                if np.any(idx):
                    self.add_shots(count[idx], zenith[idx], azimuth[idx], method=method)

                return True
            else:
                return False

    def get_pgap_theta_z(self, min_azimuth=0, max_azimuth=360, invert=False):
        """
        Get the Pgap by zenith and height bin for a given azimuth bin range
        """
        cover_theta_z = np.full(self.target_output.shape, np.nan, dtype=float)
        np.divide(np.cumsum(self.target_output,axis=2), self.shot_output, out=cover_theta_z, 
            where=self.shot_output > 0)
        
        mina = min_azimuth - self.ares / 2
        maxa = max_azimuth + self.ares / 2
        idx = (self.azimuth_bin >= mina) & (self.azimuth_bin < maxa)
        if invert:
            idx = ~idx

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            cover_theta_z = np.nanmean(cover_theta_z[:,idx,:], axis=1)
        
        self.pgap_theta_z = 1 - cover_theta_z 

    def calcLinearPlantProfiles(self):
        """
        Calculate the linear model PAI (see Jupp et al., 2009)
        """
        zenith_bin_r = np.radians(self.zenith_bin)
        kthetal = np.full(self.pgap_theta_z.shape, np.log(1e-5), dtype=float)
        np.log(self.pgap_theta_z, out=kthetal, where=self.pgap_theta_z>0)
        xtheta = np.abs(2 * np.tan(zenith_bin_r) / np.pi)
        paiv = np.zeros(self.pgap_theta_z.shape[1], dtype=np.float32)
        paih = np.zeros(self.pgap_theta_z.shape[1], dtype=np.float32)
        for i,h in enumerate(self.height_bin):
            a = np.vstack([xtheta, np.ones(xtheta.shape[0])]).T
            y = -kthetal[:,i]
            valid = ~np.isnan(y)
            if np.count_nonzero(valid) > 2:
                result,resid,rank,s = np.linalg.lstsq(a[valid,:], y[valid], rcond=None)
                paiv[i] = result[0]
                paih[i] = result[1]
                if result[0] < 0:
                    paih[i] = np.nanmean(y)
                    paiv[i] = 0.0
                if result[1] < 0:
                    paiv[i] = np.mean(y[valid] / xtheta[valid])
                    paih[i] = 0.0

        pai = paiv + paih
        mla = np.degrees( np.arctan2(paiv,paih) )
        
        return pai,mla

    def calcHingePlantProfiles(self):
        """
        Calculate the hinge angle PAI (see Jupp et al., 2009)
        """
        zenith_bin_r = np.radians(self.zenith_bin)
        hingeindex = np.argmin(np.abs(zenith_bin_r - np.arctan(np.pi / 2)))
        
        pai_lim = np.log(1e-5)
        tmp = np.full(self.pgap_theta_z.shape[1], pai_lim)
        np.log(self.pgap_theta_z[hingeindex,:], out=tmp, 
            where=self.pgap_theta_z[hingeindex,:] > 0)
        
        pai = -1.1 * tmp

        return pai

    def calcSolidAnglePlantProfiles(self, total_pai=None):
        """
        Calculate the Jupp et al. (2009) solid angle weighted PAI
        """
        zenith_bin_r = np.radians(self.zenith_bin)
        zenith_bin_size = zenith_bin_r[1] - zenith_bin_r[0]
        w = 2 * np.pi * np.sin(zenith_bin_r) * zenith_bin_size
        wn = w / np.sum(w[self.pgap_theta_z[:,-1] < 1])
        ratio = np.zeros(self.pgap_theta_z.shape[1])

        pai_lim = np.log(1e-5)
        for i in range(zenith_bin_r.shape[0]):
            if (self.pgap_theta_z[i,-1] < 1):
                num = np.full(self.pgap_theta_z.shape[1], pai_lim)
                np.log(self.pgap_theta_z[i,:], out=num, where=self.pgap_theta_z[i,:] > 0)
                den = np.log(self.pgap_theta_z[i,-1]) if (self.pgap_theta_z[i,-1] > 0) else pai_lim
                ratio += wn[i] * num / den

        if total_pai is None:
            hpp_pai = self.calcHingePlantProfiles()
            total_pai = np.max(hpp_pai)
        elif not isinstance(total_pai, float):
            print('Total PAI has not been defined')
            sys.exit()
         
        pai = total_pai * ratio

        return pai

    def get_pavd(self, pai_z):
        """
        Get the PAVD using central differences
        """
        pavd = np.gradient(pai_z, self.height_bin)
        return pavd

    def exportPlantProfiles(self, outfile=None):
        """
        Write out the vertical plant profiles to file
        """
        linear_pai,linear_mla = self.calcLinearPlantProfiles()
        plant_profiles = {'Height': self.height_bin,
                          'HingePAI': self.calcHingePlantProfiles(),
                          'LinearPAI': linear_pai,
                          'LinearMLA': linear_mla,       
                          'WeightedPAI': self.calcSolidAnglePlantProfiles()}
        
        plant_profiles['HingePAVD'] = self.get_pavd(plant_profiles['HingePAI'])
        plant_profiles['LinearPAVD'] = self.get_pavd(plant_profiles['LinearPAI'])
        plant_profiles['WeightedPAVD'] = self.get_pavd(plant_profiles['WeightedPAI'])

        df = pd.DataFrame.from_dict(plant_profiles)
        if outfile is None:
            return df
        else:
            df.to_csv(outfile, float_format='%.05f', index=False)

    def exportPgapProfiles(self, outfile=None):
        """
        Write out the Pgap profiles to file
        """
        pgap_profiles = {'Height': self.height_bin}
        for i in range(self.pgap_theta_z.shape[0]):
            name = f'Zenith{self.zenith_bin[i]*10:04.0f}'
            pgap_profiles[name] = self.pgap_theta_z[i,:]

        df = pd.DataFrame.from_dict(pgap_profiles)
        if outfile is None:
            return df
        else:
            df.to_csv(outfile, float_format='%.05f', index=False)


def get_min_z_grid(input_files, transform_files, grid_extent, grid_resolution, rxp=False):
    """
    Wrapper function a minimum z grid for input to the ground plane fitting
    """
    if not rxp:
        x,y,z,r = get_min_z_grid_rdbx(input_files, transform_files, 
            grid_extent, grid_resolution)
    else:
        x,y,z,r = get_min_z_grid_rxp(input_files, transform_files, 
            grid_extent, grid_resolution)
    return x,y,z,r


def get_min_z_grid_rdbx(rdbx_files, transform_files, grid_extent, grid_resolution):
    """
    Wrapper function a minimum z grid for input to the ground plane fitting
    """
    rdb_attributes = {'riegl.xyz': 'riegl_xyz'}
    ncols = nrows = int(grid_extent // grid_resolution) + 1
    outgrid = np.empty((4,nrows,ncols), dtype=np.float32)
    valid = np.zeros((nrows,ncols), dtype=bool)
    for i,fn in enumerate(rdbx_files):
        with riegl_io.RDBFile(fn, chunk_size=100000, attributes=rdb_attributes,
            transform_file=transform_files[i]) as rdb:
            while rdb.point_count_current < rdb.point_count_total:
                rdb.read_next_chunk()
                if rdb.point_count > 0:
                    x = rdb.get_chunk('x')
                    y = rdb.get_chunk('y')
                    z = rdb.get_chunk('z')
                    r = rdb.get_chunk('range')
                    min_z_grid(x, y, z, r, -grid_extent/2, grid_extent/2,
                        grid_resolution, outgrid, valid)
    x,y,z,r = (outgrid[0,:,:][valid], outgrid[1,:,:][valid], 
               outgrid[2,:,:][valid], outgrid[3,:,:][valid])
    return x,y,z,r


def get_min_z_grid_rxp(rxp_files, transform_files, grid_extent, grid_resolution):
    """
    Wrapper function a minimum z grid for input to the ground plane fitting
    Using RXP only as input
    """
    ncols = nrows = int(grid_extent // grid_resolution) + 1
    outgrid = np.empty((4,nrows,ncols), dtype=np.float32)
    valid = np.zeros((nrows,ncols), dtype=bool)
    for i,fn in enumerate(rxp_files):
        with riegl_io.RXPFile(fn, transform_file=transform_files[i]) as rxp:
            x = rxp.get_data('x', return_as_point_attribute=True)
            y = rxp.get_data('y', return_as_point_attribute=True)
            z = rxp.get_data('z', return_as_point_attribute=True)
            r = rxp.get_data('range', return_as_point_attribute=True)
            min_z_grid(x, y, z, r, -grid_extent/2, grid_extent/2,
                grid_resolution, outgrid, valid)
    x,y,z,r = (outgrid[0,:,:][valid], outgrid[1,:,:][valid],
               outgrid[2,:,:][valid], outgrid[3,:,:][valid])
    return x,y,z,r


@njit 
def sum_by_index_2d(values, idx1, idx2, output):
    """
    Sum point values by two indices
    """
    inbounds = ( (idx1 >= 0) & (idx1 < output.shape[0]) &
                 (idx2 >= 0) & (idx2 < output.shape[1]) )
    for i in range(values.shape[0]):
        if inbounds[i]:
            y = int(idx1[i])
            x = int(idx2[i])
            output[y,x] += values[i]


@njit
def sum_by_index_1d(values, idx, output):
    """
    Sum point values by one index
    """
    inbounds = (idx >= 0) & (idx < output.shape[0])
    for i in range(values.shape[0]):
        if inbounds[i]:
            y = int(idx[i])
            output[y] += values[i]


@njit
def sum_by_index_3d(values, idx1, idx2, idx3, output):
    """
    Sum point values by one index
    """
    inbounds = ( (idx1 >= 0) & (idx1 < output.shape[0]) &
                 (idx2 >= 0) & (idx2 < output.shape[1]) &
                 (idx3 >= 0) & (idx3 < output.shape[2]))
    for i in range(values.shape[0]):
        if inbounds[i]:
            z = int(idx1[i])
            y = int(idx2[i])
            x = int(idx3[i])
            output[z,y,x] += values[i]


def calcGroundPlane(x, y, z, r, resolution=10, reportfile=None):
    """
    Approximate the ground as a plane following Calders et al. (2014)
    """
    minx = np.min(x)
    maxy = np.max(y)
    ncols = int( (np.max(x) - minx) // resolution ) + 1
    nrows = int( (maxy - np.min(y)) // resolution ) + 1

    outgrid = np.empty((4,nrows,ncols), dtype=np.float32)
    valid = np.zeros((nrows,ncols), dtype=bool)
    min_z_grid(x, y, z, r, minx, maxy, resolution, outgrid, valid)

    xgrid = outgrid[0,:,:][valid]
    ygrid = outgrid[1,:,:][valid]
    zgrid = outgrid[2,:,:][valid]
    rgrid = outgrid[3,:,:][valid]
    result = plane_fit_hubers(xgrid, ygrid, zgrid, w=rgrid, reportfile=reportfile)

    return result


@njit 
def min_z_grid(x, y, z, r, minx, maxy, resolution, outgrid, valid):
    """
    Get coordinates of points on minimum elevation grid
    Following Calders et al. (2013)
    """
    xidx = (x - minx) // resolution
    yidx = (maxy - y) // resolution
    inbounds = ( (xidx >= 0) & (xidx < valid.shape[1]) &
                 (yidx >= 0) & (yidx < valid.shape[0]) )
    for i in range(z.shape[0]):
        if inbounds[i]:
            xi = int(xidx[i])
            yi = int(yidx[i])
            if valid[yi,xi] > 0:
                if z[i] < outgrid[2,yi,xi]:
                    outgrid[0,yi,xi] = x[i]
                    outgrid[1,yi,xi] = y[i]
                    outgrid[2,yi,xi] = z[i]
                    outgrid[3,yi,xi] = r[i]
            else:
                outgrid[0,yi,xi] = x[i]
                outgrid[1,yi,xi] = y[i]
                outgrid[2,yi,xi] = z[i]
                outgrid[3,yi,xi] = r[i]
                valid[yi,xi] = True


def plane_fit_hubers(x, y, z, w=None, reportfile=None):
    """
    Plane fitting (Huber's T norm with median absolute deviation scaling)
    Prior weights are set to 1 / point range
    """
    if w is None:
        w = np.ones(z.shape, dtype=np.float32)
    wz = w * z
    wxy = np.vstack((w,x*w,y*w)).T
    huber_t = sm.RLM(wz, wxy, M=sm.robust.norms.HuberT())
    huber_results = huber_t.fit()
    
    output = {}
    output['Parameters'] = huber_results.params
    output['Summary'] = huber_results.summary(yname='Z', xname=['Intercept','X','Y'])
    output['Slope'] = np.degrees( np.arctan(np.sqrt(output['Parameters'][1]**2 + output['Parameters'][2]**2)) )
    output['Aspect'] = np.degrees( np.arctan(output['Parameters'][1] / output['Parameters'][2]) )
    
    if reportfile is not None:
        with open(reportfile,'w') as f:
            for k,v in output.items():
                if k != 'Parameters':
                    f.write(f'{k:}:\n{v:}\n')
    
    return output

