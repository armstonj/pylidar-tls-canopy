#!/usr/bin/env python3

DESCRIPTION='''
pylidar_tls_canopy: Create a grid of the scan using cartesian coordinates

John Armston
University of Maryland
November 2022
'''

from pylidar_tls_canopy import riegl_io
from pylidar_tls_canopy import grid

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm


def get_args():
    """
    Get the command line arguments
    """
    argparser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument('-i','--input', metavar='FILE', type=str, nargs='+',
        help='Input RIEGL filenames')
    argparser.add_argument('-o','--output', metavar='FILE', type=str, default=None,
        help='Output filename')
    argparser.add_argument('-a','--attribute', metavar='STR', type=str, default='z',
        help='Pulse or point attribute to grid')
    argparser.add_argument('-c','--chunk_size', metavar='INT', type=int, default=100000,
        help='Chunksize for reading rdbx files')
    argparser.add_argument('-r','--resolution', metavar='FLOAT', type=float, default=0.5,
        help='Spatial resolution (meters)')
    argparser.add_argument('-e','--extent', metavar='FLOAT', type=float, default=[50,50], nargs=2,
        help='Spatial extent [X,Y] (meters)')
    argparser.add_argument('-u','--ulc', metavar='FLOAT', type=float, default=[-25,25], nargs=2,
        help='Upper left corner coordinate (meters)')
    argparser.add_argument('-p','--pose_file', metavar='FILE', type=str, nargs='+',
        help='Input RIEGL pose filenames (for RXP file processing if no transform_file')
    argparser.add_argument('-t','--transform_file', metavar='FILE', type=str,
        help='Input RIEGL transform dat filenames')
    argparser.add_argument('-m','--method', metavar='STR', type=str, choices=['MEAN','MAX','MIN','SUM'], default='MAX',
        help='Method to apply per grid cell')
    argparser.add_argument('-q','--query_str', metavar='STR', type=str, default=None,
        help='Conditional statements for querying a point cloud subset')
    args = argparser.parse_args()

    return args


def run():

    args = get_args()
    if args.input is None:
        print('Run "pylidar_cartesiangrid -h" for command line arguments')
        return

    if args.output is None:
        fn = os.path.basename(args.input[0])
        prefix,suffix = os.path.splitext(fn)
        args.output = f'{prefix:}_{suffix[1::]:}_{args.attribute:}_cartesian.tif'

    ncols = int( args.extent[0] // args.resolution + 1 )
    nrows = int( args.extent[1] // args.resolution + 1 )

    with grid.LidarGrid(ncols, nrows, -args.extent[0]/2, args.extent[1]/2, resolution=args.resolution, init_cntgrid=True) as grd:
        for fn, transform_fn in zip(args.input,args.transform_file):

            if fn.endswith('.rdbx'):
                
                with riegl_io.RDBFile(fn, transform_file=transform_fn, query_str=args.query_str) as rdb:
                    xidx = (rdb.get_data('x') - args.ulc[0]) // args.resolution
                    yidx = (args.ulc[1] - rdb.get_data('y')) // args.resolution
                    vals = rdb.get_data(attribute)
                    valid = (xidx >= 0) & (xidx < ncols) & (yidx >= 0) & (yidx < nrows)
                    grd.add_values(vals[valid], np.uint16(xidx[valid]), np.uint16(yidx[valid]), 0, method=args.method)

            elif fn.endswith('.rxp'):
                with riegl_io.RXPFile(fn, transform_file=transform_fn, query_str=args.query_str) as rxp:
                    if args.attribute in rxp.pulses:
                        return_as_point_attribute = False
                    else:
                        return_as_point_attribute = True
                    xidx = (rxp.get_data('x', return_as_point_attribute=return_as_point_attribute) - args.ulc[0]) // args.resolution
                    yidx = (args.ulc[1] - rxp.get_data('y', return_as_point_attribute=return_as_point_attribute)) // args.resolution
                    vals = rxp.get_data(attribute, return_as_point_attribute=return_as_point_attribute)
                    valid = (xidx >= 0) & (xidx < ncols) & (yidx >= 0) & (yidx < nrows)
                    grd.add_values(vals[valid], np.uint16(xidx[valid]), np.uint16(yidx[valid]), 0, method=args.method)

            else:
                print(f'{fn} is not a recognized RIEGL file')
                sys.exit()
        
        grd.finalize_grid(method=args.method)
        grd.write_grid(args.output, descriptions=[f'{args.attribute} {args.method}'])

