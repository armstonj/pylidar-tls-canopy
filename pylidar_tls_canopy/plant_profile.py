#!/usr/bin/env python3
"""
pylidar_tls_canopy

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
                 min_h=0, max_h=50, ground_plane=[0.0,0.0,0.0]):
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

        self.ground_plane = ground_plane

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

    def add_riegl_scan_position(self, rxp_file, transform_file, rdbx_file=None, sensor_height=None,
        method='WEIGHTED', min_zenith=5, max_zenith=70, max_hr=None, query_str=None):
        """
        Add a RIEGL scan position to the profile
        """
        min_zenith_r = np.radians(min_zenith)
        max_zenith_r = np.radians(max_zenith)
        pulse_cols = ['zenith','azimuth','target_count']
        point_cols = ['x','y','z','range','target_index',
                      'zenith','azimuth','target_count']

        pulses = {}
        with riegl_io.RXPFile(rxp_file, transform_file=transform_file, query_str=query_str) as rxp:
            for col in pulse_cols:
                pulses[col] = rxp.get_data(col, return_as_point_attribute=False)
            idx = (pulses['zenith'] >= min_zenith_r) & (pulses['zenith'] < max_zenith_r)
            if np.any(idx):
                self.add_shots(pulses['target_count'][idx], pulses['zenith'][idx],
                    pulses['azimuth'][idx], method=method)

            points = {}
            if rdbx_file:
                with riegl_io.RDBFile(rdbx_file, transform_file=transform_file, query_str=query_str) as f:
                    for col in point_cols:
                        points[col] = f.get_data(col)
            else:
                for col in point_cols:
                    points[col] = rxp.get_data(col, return_as_point_attribute=True)

            if sensor_height is not None:
                zoffset = rxp.transform[3,2] - sensor_height
            else:
                zoffset = self.ground_plane[0]
        
        height = points['z'] - (self.ground_plane[1] * points['x'] +
            self.ground_plane[2] * points['y'] + zoffset)
        
        idx = (points['zenith'] >= min_zenith_r) & (points['zenith'] < max_zenith_r)
        if max_hr is not None:
            hr = points['range'] * np.sin(points['zenith'])
            idx &= hr < max_hr
        if np.any(idx):
            self.add_targets(height[idx], points['target_index'][idx], 
                points['target_count'][idx], points['zenith'][idx],
                points['azimuth'][idx], method=method)

    def add_leaf_scan_position(self, leaf_file, method='FIRSTLAST', min_zenith=5, 
        max_zenith=70, sensor_height=None):
        """
        Add a leaf scan position to the profile
        """
        min_zenith_r = np.radians(min_zenith)
        max_zenith_r = np.radians(max_zenith)
        cols = ['zenith','azimuth','target_count','h1','h2']

        with leaf_io.LeafScanFile(leaf_file, sensor_height=sensor_height) as leaf:
            self.datetime = leaf.datetime
            data = {}
            if not leaf.data.empty:
                for col in cols:
                    data[col] = leaf.data[col].to_numpy()
                for n,height in enumerate(['h1','h2'], start=1):
                    target_index = np.full(data[height].shape, n, dtype=np.uint8)
                    idx = ((data['zenith'] >= min_zenith_r) & 
                           (data['zenith'] < max_zenith_r) & 
                           ~np.isnan(data[height]))
                    if np.any(idx):
                        self.add_targets(data[height][idx], target_index[idx], 
                            data['target_count'][idx], data['zenith'][idx], 
                            data['azimuth'][idx], method=method)

                idx = (data['zenith'] >= min_zenith_r) & (data['zenith'] < max_zenith_r)
                if np.any(idx):
                    self.add_shots(data['target_count'][idx], data['zenith'][idx], 
                        data['azimuth'][idx], method=method)

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

    def calcLinearPlantProfiles(self, calc_mla=False):
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
        if calc_mla:
            mla = np.degrees( np.arctan2(paiv,paih) )
            return pai,mla
        else:
            return pai        

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
        linear_pai,linear_mla = self.calcLinearPlantProfiles(calc_mla=True)
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


def get_min_z_grid(riegl_files, transform_files, grid_extent, grid_resolution, 
    grid_origin=[0.0,0.0], query_str=None, rxp=False):
    """
    Wrapper function a minimum z grid for input to the ground plane fitting
    """
    ncols = nrows = int(grid_extent // grid_resolution) + 1
    outgrid = np.empty((4,nrows,ncols), dtype=np.float32)
    valid = np.zeros((nrows,ncols), dtype=bool)
    for i,fn in enumerate(riegl_files):
        if rxp:
            with riegl_io.RXPFile(fn, transform_file=transform_files[i], query_str=query_str) as f:
                x = f.get_data('x', return_as_point_attribute=True)
                y = f.get_data('y', return_as_point_attribute=True)
                z = f.get_data('z', return_as_point_attribute=True)
                r = f.get_data('range', return_as_point_attribute=True)
        else:
            with riegl_io.RDBFile(fn, transform_file=transform_files[i], query_str=query_str) as f:
                x = f.get_data('x')
                y = f.get_data('y')
                z = f.get_data('z')
                r = f.get_data('range')
        minx = grid_origin[0] - grid_extent / 2
        maxy = grid_origin[1] + grid_extent / 2
        min_z_grid(x, y, z, r, minx, maxy, grid_resolution, outgrid, valid)
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

