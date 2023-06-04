#!/usr/bin/env python3
"""
pylidar_tls_canopy

Drivers for handling RIEGL rdbx and rxp files

John Armston
University of Maryland
October 2022
"""

try:
    import riegl_rdb
except ImportError:
    print('RIEGL RDBlib is not available')

try:
    import riegl_rxp
except ImportError:
    print('RIEGL RiVlib is not available')

import re
import sys
import json
import numpy as np
from numba import njit

from . import DEFAULT_RDB_ATTRIBUTES
from . import PRR_MAX_TARGETS 


class RDBFile:
        def __init__(self, filename, transform_file=None, pose_file=None, query_str=None):
        self.filename = filename
        if transform_file is not None:
            self.transform = read_transform_file(transform_file)
        elif pose_file is not None:
            with open(pose_file,'r') as f:
                pose = json.load(f)
            self.transform = calc_transform_matrix(pose['pitch'], pose['roll'], pose['yaw'])
        else:
            self.transform = None
        self.query_str = query_str
        self.read_file()

    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        pass
            
    def run_query(self):
        """
        Query the points array
        """
        if not isinstance(query_str, list):
            query_str = [query_str]

        r = re.compile(r'((?:\(|))\s*([a-z]+)\s*(<|>|==|>=|<=|!=)\s*([-+]?\d*\.\d+|\d+)\s*((?:\)|))')
        valid = np.ones(self.points.shape[0], dtype=bool)
        for q in query_str:
            m = r.match(q)
            if m is not None:
                if m.group(2) in self.points:
                    q_str = f'self.points[{m.group(2)}] {m.group(3)} {m.group(4)}'
                else:
                    msg = f'{m.group(2)} is not a valid point attribute name'
                    print(msg)
                    return
                try:
                    valid &= eval(q_str)
                except SyntaxError:
                    msg = f'{q} is not a valid query string'
                    print(msg)

        return valid

    def read_file(self):
        """
        Read file and get global stats
        """
        self.meta, points = riegl_rdb.readFile(self.filename)

        self.points = {}
        if self.query_str is not None:
            target_count = np.repeat(pulses['target_count'], pulses['target_count'])
            scanline = np.repeat(pulses['scanline'], pulses['target_count'])
            scanline_idx = np.repeat(pulses['scanline_idx'], pulses['target_count'])
            valid = self.run_query()
            points = points[valid]
            self.points['target_index'],self.points['target_count'] = reindex_targets(points['target_index'],
                points['target_count'], points['scanline'], points['scanline_idx'])

        if self.transform_file is not None:
            self.transform = read_transform_file(self.transform_file)
        else:
            pose = self.get_meta('riegl.pose_estimation')
            self.transform = calc_transform_matrix(pose['orientation']['pitch'],
                pose['orientation']['roll'], pose['orientation']['yaw'])

        if self.transform is not None:
            xyz = np.vstack((points['x'], points['y'], points['z'])).T
            x_t,y_t,z_t = apply_transformation(points['x'], points['y'], points['z'],
                points['x'].shape[0], self.transform, translate=True)
            self.points['x'] = x_t
            self.points['y'] = y_t
            self.points['z'] = z_t
        self.points['valid'] = (points['scanline'] >= 0)

        for name in points.dtype.names:
            if name not in self.points:
                self.points[name] = points[name]

        self.minc = 0
        self.maxc = np.max(self.points['scanline'])
        self.minr = 0
        self.maxr = np.max(self.points['scanline_idx'])
        self.max_range = np.max(self.points['range'])
        self.max_target_count = np.max(self.points['target_count'])

    def get_data(self, name):
        """
        Get a point attribute
        """
        if name in self.points:
            data = self.points[name]
            valid = self.points['valid']
        else:
            print(f'{name:} is not a point attribute')
            sys.exit()

        return data[valid]

    def get_meta(self, key):
        """
        Get an individual metadata item
        'riegl.pose_estimation', 'riegl.geo_tag', 'riegl.notch_filter', 'riegl.window_echo_correction',
        'riegl.detection_probability', 'riegl.angular_notch_filter', 'riegl.pulse_position_modulation',
        'riegl.noise_estimates', 'riegl.device', 'riegl.atmosphere', 'riegl.near_range_correction', 
        'riegl.time_base', 'riegl.scan_pattern', 'riegl.device_geometry', 'riegl.range_statistics', 
        'riegl.beam_geometry', 'riegl.reflectance_calculation', 'riegl.window_analysis',
        'riegl.mta_settings', 'riegl.point_attribute_groups'        
        """
        return json.loads(self.meta[key])

    def get_points_by_pulse(self, colnames, pulse_scanline, pulse_scanline_idx):
        """
        Function to reorder rdbx sourced point data according to rxp sourced pulse data
        """
        dtype_str = {'x':'<f8', 'y':'<f8', 'z':'<f8', 
                     'range':'<f8'}
        dtype_list = []
        for name in colnames:
            dtype_list.append((str(name), dtype_str[name], self.max_target_count))

        pulse_id_rxp = pulse_scanline * np.max(pulse_scanline_idx) + pulse_scanline_idx
        pulse_id_rdb = self.get_data('scanline') * np.max(pulse_scanline_idx) + self.get_data('scanline_idx')
        
        pulse_sort_idx_rxp = np.argsort(pulse_id_rxp)
        pulse_sort_idx_rdb = np.argsort(pulse_id_rdb)
        idx = np.searchsorted(pulse_id_rxp, pulse_id_rdb[pulse_sort_idx_rdb], sorter=pulse_sort_idx_rxp)

        output = np.empty(pulse_scanline.shape[0], dtype=dtype_list)
        for name in colnames:
            output[idx,target_index-1] = self.get_data(name)[pulse_sort_idx_rdb]

        return output


class RXPFile:
    def __init__(self, filename, transform_file=None, pose_file=None, query_str=None):
        self.filename = filename
        if transform_file is not None:
            self.transform = read_transform_file(transform_file)
        elif pose_file is not None:
            with open(pose_file,'r') as f:
                pose = json.load(f)
            self.transform = calc_transform_matrix(pose['pitch'], pose['roll'], pose['yaw'])
        else:
            self.transform = None
        self.query_str = query_str
        self.read_file()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def run_query(self):
        """
        Query the points array
        """
        if not isinstance(query_str, list):
            query_str = [query_str]

        r = re.compile(r'((?:\(|))\s*([a-z]+)\s*(<|>|==|>=|<=|!=)\s*([-+]?\d*\.\d+|\d+)\s*((?:\)|))')
        valid = np.ones(self.points.shape[0], dtype=bool)
        for q in query_str:
            m = r.match(q)
            if m is not None:
                if m.group(2) in self.points:
                    q_str = f'self.points[{m.group(2)}] {m.group(3)} {m.group(4)}' 
                else:
                    msg = f'{m.group(2)} is not a valid point attribute name'
                    print(msg)
                    return
                try:
                    valid &= eval(q_str)
                except SyntaxError:
                    msg = f'{q} is not a valid query string'
                    print(msg)
        
        return valid

    def read_file(self):
        """
        Read file and get global stats
        """
        self.meta, points, pulses = riegl_rxp.readFile(self.filename)

        self.points = {}
        self.pulses = {}
        if self.query_str is not None:
            valid = self.run_query()
            points = points[valid]
           
            target_count = np.repeat(pulses['target_count'], pulses['target_count'])
            scanline = np.repeat(pulses['scanline'], pulses['target_count'])
            scanline_idx = np.repeat(pulses['scanline_idx'], pulses['target_count']) 
            self.points['target_index'],new_target_count = reindex_targets(points['target_index'], 
                target_count[valid], scanline[valid], scanline_idx[valid])

            pulse_id = pulses['scanline'] * np.max(pulses['scanline_idx']) + pulses['scanline_idx']
            point_id = scanline[valid] * np.max(pulses['scanline_idx']) + scanline_idx[valid]

            pulse_sort_idx = np.argsort(pulse_id)
            point_sort_idx = np.argsort(point_id)
            first = (self.points['target_index'][point_sort_idx] == 1)
            idx = np.searchsorted(pulse_id, point_id[point_sort_idx][first], sorter=pulse_sort_idx)

            self.pulses['target_count'] = np.zeros(pulses.shape[0], dtype=np.uint8)         
            self.pulses['target_count'][idx] = new_target_count[point_sort_idx][first]

        if self.transform is None:
            if 'PITCH' in self.meta:
                self.transform = calc_transform_matrix(self.meta['PITCH'], self.meta['ROLL'], self.meta['YAW'])
        
        if self.transform is not None:
            x_t,y_t,z_t = apply_transformation(pulses['beam_direction_x'], pulses['beam_direction_y'], 
                pulses['beam_direction_z'], pulses['beam_direction_x'].shape[0], self.transform)
            self.pulses['beam_direction_x'] = x_t
            self.pulses['beam_direction_y'] = y_t
            self.pulses['beam_direction_z'] = z_t
            _, self.pulses['zenith'], self.pulses['azimuth'] = xyz2rza(x_t, y_t, z_t)
        self.pulses['valid'] = pulses['scanline'] >= 0

        if self.transform is not None:
            xyz = np.vstack((points['x'], points['y'], points['z'])).T
            x_t,y_t,z_t = apply_transformation(points['x'], points['y'], points['z'], 
                points['x'].shape[0], self.transform, translate=True)
            self.points['x'] = x_t
            self.points['y'] = y_t
            self.points['z'] = z_t
        self.points['valid'] = np.repeat(pulses['scanline'], pulses['target_count']) >= 0

        for name in pulses.dtype.names:
            if name not in self.pulses:
                self.pulses[name] = pulses[name]
            
        for name in points.dtype.names:
            if name not in self.points:
                self.points[name] = points[name]

        self.minc = 0
        self.maxc = np.max(self.pulses['scanline'])
        self.minr = 0
        self.maxr = np.max(self.pulses['scanline_idx'])
        self.max_range = np.max(self.points['range'])
        self.max_target_count = np.max(self.pulses['target_count'])

    def get_points_by_pulse(self, names):
        """
        Reshape data as a number of pulses by max_target_count array
        Multiple point attributes are handled using a structured array
        """
        dtype_list = []
        for name in names:
            t = self.points[name].dtype.str
            dtype_list.append((str(name), t, self.max_target_count))
        
        pulse_id = np.repeat(self.pulses['pulse_id'] - 1, self.pulses['target_count'])
        
        npulses = self.pulses['pulse_id'].shape[0]
        data = np.empty(npulses, dtype=dtype_list)
        for i in range(self.max_target_count):
            point_idx = self.points['target_index'] == i + 1
            for name,t,s in dtype_list:
                idx = pulse_id[point_idx]
                data[name][idx,i] = self.points[name][point_idx]

        return data[self.pulses['valid']]

    def get_data(self, name, return_as_point_attribute=False):
        """
        Get a pulse or point attribute
        """
        if name in self.pulses:
            if return_as_point_attribute:
                data = np.repeat(self.pulses[name], self.pulses['target_count'])
                valid = self.points['valid']
            else:
                data = self.pulses[name]
                valid = self.pulses['valid']
        elif name in self.points:
            data = self.points[name]
            valid = self.points['valid']
        else:
            print(f'{name:} is not a pulse or point attribute')
            sys.exit()
        
        return data[valid]


@njit
def reindex_targets(target_index, target_count, scanline, scanline_idx):
    """
    Reindex the target index and count
    Assumes the input data are time-sequential
    """
    new_target_index = np.ones_like(target_index)
    new_target_count = np.ones_like(target_count)

    n = 1
    for i in range(1, target_index.shape[0], 1):
        same_pulse = (scanline[i] == scanline[i-1]) & (scanline_idx[i] == scanline_idx[i-1])
        if same_pulse:
            new_target_index[i] = new_target_index[i-1] + 1 
            new_target_count[i-n:i+1] = n + 1
            n += 1
        else:
            n = 1
        
    return new_target_index,new_target_count


def calc_transform_matrix(pitch, roll, yaw):
    """
    Get transform matrix
    Set compass reading to zero if nan
    """
    pitch = np.radians(pitch)
    pitch_mat = np.identity(4)
    pitch_mat[0,0] = np.cos(pitch)
    pitch_mat[0,2] = np.sin(pitch)
    pitch_mat[2,0] = -np.sin(pitch)
    pitch_mat[2,2] = np.cos(pitch)

    roll = np.radians(roll)
    roll_mat = np.identity(4)
    roll_mat[1,1] = np.cos(roll)
    roll_mat[1,2] = -np.sin(roll)
    roll_mat[2,1] = np.sin(roll)
    roll_mat[2,2] = np.cos(roll)

    yaw = np.radians(yaw)
    if np.isnan(yaw):
        yaw = 0.0
    yaw_mat = np.identity(4)
    yaw_mat[0,0] = np.cos(yaw)
    yaw_mat[0,1] = -np.sin(yaw)
    yaw_mat[1,0] = np.sin(yaw)
    yaw_mat[1,1] = np.cos(yaw)

    tmp_mat = yaw_mat.dot(pitch_mat)
    transform = tmp_mat.dot(roll_mat)

    return transform


def apply_transformation(x, y, z, size, transform_matrix, translate=False):
    """
    Apply transformation
    d: apply transformation (1) or rotation only (0)
    """
    xyz = np.vstack((x, y, z)).T
    if translate:
        t = np.ones((size,1))
    else:
        t = np.zeros((size,1))

    xyz = np.concatenate((xyz, t), 1)
    xyz_t = np.dot(xyz, transform_matrix)

    return xyz_t[:,0],xyz_t[:,1],xyz_t[:,2]


def xyz2rza(x, y, z):
    """
    Calculate spherical coordinates from the xyz data
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(x, y)
    np.add(phi, 2*np.pi, out=phi, where=x < 0)

    return r, theta, phi


def read_transform_file(fn):
    """
    Read the transform matrix (rotation + translation) 
    """
    with open(fn, 'rb') as f:
        transform = np.loadtxt(f, delimiter=' ', dtype=np.float32)
    return transform.T

