#!/usr/bin/env python3
"""
pylidar_canopy

Drivers for handling LEAF files

John Armston
University of Maryland
December 2022
"""

import sys
import re
import os
import ast
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class LeafScanFile:
    def __init__(self, filename, sensor_height=None):
        self.filename = filename
        self.sensor_height = sensor_height
        pattern = re.compile(r'(\w{8})_(\d{4})_(hemi|hinge|ground)_(\d{8})-(\d{6})Z_(\d{4})_(\d{4})\.csv')
        fileinfo = pattern.fullmatch( os.path.basename(filename) )
        if fileinfo:
            self.serial_number = fileinfo[1]
            self.scan_count = int(fileinfo[2])
            self.scan_type = fileinfo[3]
            self.datetime = datetime.strptime(f'{fileinfo[4]}{fileinfo[5]}', '%Y%m%d%H%M%S')
            self.zenith_shots = int(fileinfo[6])
            self.azimuth_shots = int(fileinfo[7])
        else:
            print(f'{filename} is not a recognized LEAF scan file')
        self.read_meta()
        self.read_data()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def read_meta(self):
        """
        Read the file header
        """
        with open(self.filename, 'r') as f:
            self.header = {}
            self.footer = {}
            in_header = True
            for line in f:
                header = True
                if line.startswith('#'):
                    if 'Finished' in line:
                        lparts = line.strip().split()
                        self.duration = float(lparts[2])
                    else:
                        lparts = line.strip().split(':')
                        key = lparts[0][1::].strip()
                        val = lparts[1].strip()
                        try:
                            val = ast.literal_eval(val)
                        except:
                            pass
                        if in_header:
                            self.header[key] = val
                        else:
                            self.footer[key] = val
                else:
                    in_header = False

    def read_data(self, transform=True):
        """
        Read file
        """
        self.data = pd.read_csv(self.filename, comment='#', na_values=-1.0,
            names=['sample_count','scan_encoder','rotary_encoder','range1',
                   'intensity1','range2','sample_time'])

        if self.data.empty:
            return

        self.data['target_count'] = 2 - (np.isnan(self.data['range1']).astype(int) +
            np.isnan(self.data['range2']).astype(int))

        self.data['datetime'] = [self.datetime + timedelta(milliseconds=s)
            for s in self.data['sample_time'].cumsum()]

        self.data['zenith'] = self.data['scan_encoder'] / 1e4 * 2 * np.pi
        self.data['azimuth'] = self.data['rotary_encoder'] / 2e4 * 2 * np.pi

        if transform:
            dx,dy,dz = (d / 1024 for d in self.header['Tilt'])
            r,theta,phi = xyz2rza(dx, dy, dz)
            self.data['zenith'] += theta
            self.data['azimuth'] += phi

        if self.scan_type == 'hemi':
            idx = self.data['zenith'] < np.pi 
            self.data.loc[idx,'azimuth'] = self.data.loc[idx,'azimuth'] + np.pi
        idx = self.data['azimuth'] > (2 * np.pi)
        self.data.loc[idx,'azimuth'] = self.data.loc[idx,'azimuth'] - (2 * np.pi)
        idx = self.data['azimuth'] < 0
        self.data.loc[idx,'azimuth'] = self.data.loc[idx,'azimuth'] + (2 * np.pi)
        self.data['zenith'] = np.abs(self.data['zenith'] - np.pi)

        for i,name in enumerate(['range1','range2']):
            n = i + 1
            x,y,z = rza2xyz(self.data[name], self.data['zenith'], self.data['azimuth'])
            self.data[f'x{n:d}'] = x
            self.data[f'y{n:d}'] = y
            self.data[f'z{n:d}'] = z
            if self.sensor_height:
                self.data[f'h{n:d}'] = z + self.sensor_height


class LeafPowerFile:
    def __init__(self, filename):
        self.filename = filename
        pattern = re.compile(r'(\w{8})_pwr_(\d{8})\.csv')
        fileinfo = pattern.fullmatch( os.path.basename(filename) )
        if fileinfo:
            self.serial_number = fileinfo[1]
            self.datetime = datetime.strptime(fileinfo[2], '%Y%m%d')
        else:
            print(f'{filename} is not a recognized LEAF power file')

    def __enter__(self):
        self.read_data()
        return self

    def __exit__(self, type, value, traceback):
        pass

    def read_data(self):
        """
        Read file
        """
        self.data = pd.read_csv(self.filename,
            names=['datetime','battery_voltage','current',
                   'unknown1','unknown2'])

        self.data['datetime'] = pd.to_datetime(self.data['datetime'],
            format='%Y%m%d-%H%M%S')


def rza2xyz(r, theta, phi):
    """
    Calculate xyz coordinates from the spherical data
    Right-hand coordinate system
    """
    x = r * np.sin(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.cos(theta)

    return x,y,z


def xyz2rza(x, y, z):
    """
    Calculate spherical coordinates from the xyz data
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(x, y)
    if np.isscalar(phi):
        phi = phi+2*np.pi if x < 0 else phi
        phi = phi-2*np.pi if x > (2*np.pi) else phi
    else:
        np.add(phi, 2*np.pi, out=phi, where=x < 0)
        np.subtract(phi, 2*np.pi, out=phi, where=x > (2*np.pi))

    return r, theta, phi

