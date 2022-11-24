
import numpy as np
import pandas as pd
from numba import njit

import statsmodels.api as sm


class Jupp2009:
    """
    Class for Foliage Profiles from Jupp 2009
    """
    def __init__(self, hres=0.5, zres=5, min_z=0, max_z=70, min_h=0, max_h=50):
        self.hres = hres
        self.zres = zres
        self.min_z = min_z
        self.max_z = max_z
        self.min_h = min_h
        self.max_h = max_h

        self.zres_r = np.radians(zres)
        ncols = int( (max_h - min_h) // hres )
        nrows = int( (max_z - min_z) // zres )
        self.target_output = np.zeros((nrows, ncols), dtype=np.float32)
        self.shot_output = np.zeros((nrows,1), dtype=np.float32)

    def add_targets(self, target_height, target_index, target_count, target_zenith, 
        method='WEIGHTED'):
        """
        Add targets
        """
        h_idx = np.uint16(target_height // self.hres)
        z_idx = np.uint16(target_zenith // self.zres_r)
        if method == 'WEIGHTED':
            w = 1 / target_count
            sum_by_index_2d(w, z_idx, h_idx, self.target_output)
        elif method == 'ALL':
            w = np.ones(target_height.shape[0], dtype=np.float32)
            sum_by_index_2d(w, z_idx, h_idx, self.target_output)
        elif method == 'FIRST':
            idx = (target_index == 1)
            w = np.ones(np.count_nonzero(idx), dtype=np.float32)
            sum_by_index_2d(w, z_idx[idx], h_idx[idx], self.target_output)
        else:
            print(f'{method} is not a recognized target weighting method')
            exit(1)

    def add_shots(self, shot_zenith):
        """
        Add shots
        """
        idx = np.uint16(shot_zenith // self.zres_r)
        shot_cnt = np.ones(shot_zenith.shape[0], dtype=np.float32)
        sum_by_index_1d(shot_cnt, idx, self.shot_output)
        
    def get_pgap_theta_z(self):
        """
        Get the Pgap by zenith and height bin
        """
        self.height_bin = np.arange(self.min_h, self.max_h, self.hres)
        self.zenith_bin = np.arange(self.min_z, self.max_z, self.zres) + self.zres / 2
        
        cover_theta_z = np.full(self.target_output.shape, np.nan, dtype=float)
        np.divide(np.cumsum(self.target_output,axis=1), self.shot_output, out=cover_theta_z, 
            where=self.shot_output > 0)

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
        pai = -1.1 * np.log(self.pgap_theta_z[hingeindex,:])

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
            exit(1)
         
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


@njit 
def sum_by_index_2d(values, idx1, idx2, output):
    """
    Sum point values by two indices
    """
    inbounds = ( (idx1 >= 0) & (idx1 < output.shape[0]) &
                 (idx2 >= 0) & (idx2 < output.shape[1]) )
    for v,y,x,b in zip(values, idx1, idx2, inbounds):
        if b:
            output[y,x] += v


@njit
def sum_by_index_1d(values, idx, output):
    """
    Sum point values by one index
    """
    inbounds = (idx >= 0) & (idx < output.shape[0])
    for v,i,b in zip(values, idx, inbounds):
        if b:
            output[i] += v


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

    # https://stackoverflow.com/questions/45532967/statsmodels-weights-in-robust-linear-regression

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

