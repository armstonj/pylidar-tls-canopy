#!/usr/bin/env python3

DESCRIPTION='''
pyidar_tls_canopy: Create a grid of the scan using spherical coordinates

John Armston
University of Maryland
October 2022
'''

from pylidar_tls_canopy import riegl_io
from pylidar_tls_canopy import grid

import os
import argparse
import numpy as np
from tqdm import tqdm


RDB_ATTRIBUTES = {'riegl.id': 'point_id', 'riegl.timestamp': 'timestamp', 'riegl.xyz': 'riegl_xyz',
                  'riegl.target_index': 'target_index', 'riegl.target_count': 'target_count',
                  'riegl.reflectance': 'reflectance','riegl.amplitude': 'amplitude',
                  'riegl.deviation': 'deviation'}


def get_args():
    """
    Get the command line arguments
    """
    argparser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument('-i','--input', metavar='FILE', type=str,
        help='Input RIEGL filename')
    argparser.add_argument('-o','--output', metavar='FILE', type=str, default=None,
        help='Output filename')
    argparser.add_argument('-a','--attribute', metavar='STR', type=str, default='range',
        help='Pulse or point attribute to grid')
    argparser.add_argument('-q','--query', metavar='STR', type=str, default=None,
        help='Query of points to include in grid')
    argparser.add_argument('-c','--chunk_size', metavar='INT', type=int, default=100000,
        help='Chunksize for reading rdbx files')
    argparser.add_argument('-r','--resolution', metavar='FLOAT', type=float, default=1,
        help='Angular resolution (degrees)')
    argparser.add_argument('-R','--range_resolution', metavar='FLOAT', type=float, default=None,
        help='Range resolution (meters)')
    argparser.add_argument('-p','--pose_file', metavar='FILE', type=str,
        help='Input RIEGL pose filename (for RXP file processing')
    argparser.add_argument('-t','--transform_file', metavar='FILE', type=str,
        help='Input RIEGL transform dat filename')
    argparser.add_argument('-m','--method', metavar='STR', type=str, choices=['MEAN','MAX','MIN','SUM'], default='MEAN',
        help='Method to apply per grid cell')
    args = argparser.parse_args()

    return args


def run():

    args = get_args()
    if args.input is None:
        print('Run "pylidar_sphericalgrid -h" for command line arguments')
        return

    if args.output is None:
        fn = os.path.basename(args.input)
        prefix,suffix = os.path.splitext(fn)
        args.output = f'{prefix:}_{suffix[1::]:}_{args.attribute:}_spherical.tif'

    res = np.radians(args.resolution)
    ncols = int( (2 * np.pi) // res + 1 )
    nrows = int( np.pi // res + 1 )

    if args.input.endswith('.rdbx'):

        with riegl_io.RDBFile(args.input, chunk_size=args.chunk_size, query_str=args.query, 
            attributes=RDB_ATTRIBUTES, transform_file=args.transform_file) as rdb:
            if args.range_resolution is not None:
                nvars = rdb.max_range // args.range_resolution + 1
            else:
                nvars = 1
                args.range_resolution = rdb.max_range
            with grid.LidarGrid(ncols, nrows, 0, np.pi, nvars=nvars, resolution=res, init_cntgrid=True) as grd:
                with tqdm(total=rdb.point_count_total) as pbar:
                    while rdb.point_count_current < rdb.point_count_total:
                        rdb.read_next_chunk()
                        if rdb.point_count > 0:
                            xidx = np.uint16(rdb.get_chunk('azimuth') // res)
                            yidx = np.uint16(rdb.get_chunk('zenith') // res)
                            zidx = np.uint16(rdb.get_chunk('range') // args.range_resolution)
                            vals = rdb.get_chunk(args.attribute)
                            grd.add_values(vals, xidx, yidx, zidx, method=args.method)
                        pbar.update(rdb.point_count)
                grd.finalize_grid()
                descriptions = [f'Range {i*args.range_resolution:.2f}' for i in range(nvars)]
                grd.write_grid(args.output, descriptions=descriptions)

    elif args.input.endswith('.rxp'):

        with riegl_io.RXPFile(args.input, transform_file=args.transform_file, pose_file=args.pose_file) as rxp:
            if args.attribute in rxp.pulses:
                return_as_point_attribute = False
                nvars = 1
                zidx = 0
            else:
                return_as_point_attribute = True
                if args.range_resolution is not None:
                    nvars = rxp.max_range // args.range_resolution + 1
                else:
                    nvars = 1
                    args.range_resolution = rxp.max_range
                zidx = rxp.get_data('range')  // args.range_resolution
            with grid.LidarGrid(ncols, nrows, 0, np.pi, nvars=nvars, resolution=res, init_cntgrid=True) as grd:
                xidx = rxp.get_data('azimuth', return_as_point_attribute=return_as_point_attribute) // res
                yidx = rxp.get_data('zenith', return_as_point_attribute=return_as_point_attribute) // res
                vals = rxp.get_data(args.attribute, return_as_point_attribute=return_as_point_attribute)
                grd.add_values(vals, np.uint16(xidx), np.uint16(yidx), np.uint16(zidx), method=args.method)
                grd.finalize_grid()
                descriptions = [f'Range {i*args.range_resolution:.2f}' for i in range(nvars)]
                grd.write_grid(args.output, descriptions=descriptions)

    else:
        print(f'{args.input:} not a recognized RIEGL file')

