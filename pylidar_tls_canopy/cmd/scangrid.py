#!/usr/bin/env python3

DESCRIPTION='''
pylidar_tls_canopy: Create a grid of the scan

John Armston
University of Maryland
October 2022
'''

from pylidar_tls_canopy import riegl_io
from pylidar_tls_canopy import grid

import os
import argparse
from tqdm import tqdm


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
    argparser.add_argument('-p','--pose_file', metavar='FILE', type=str,
        help='Input RIEGL pose filename (for RXP file processing')
    argparser.add_argument('-t','--transform_file', metavar='FILE', type=str,
        help='Input RIEGL transform dat filename')
    args = argparser.parse_args()

    return args


def run():

    args = get_args()
    if args.input is None:
        print('Run "pylidar_scangrid -h" for command line arguments')
        return

    if args.output is None:
        fn = os.path.basename(args.input)
        prefix,suffix = os.path.splitext(fn)
        args.output = f'{prefix:}_{suffix[1::]:}_{args.attribute:}_scan.tif'

    if args.input.endswith('.rdbx'):

        with riegl_io.RDBFile(args.input, chunk_size=args.chunk_size, query_str=args.query, transform_file=args.transform_file) as rdb:
            with grid.LidarGrid(rdb.maxc+1, rdb.maxr+1, 0, rdb.maxr, nvars=rdb.max_target_count) as grd:
                with tqdm(total=rdb.point_count_total) as pbar:
                    while rdb.point_count_current < rdb.point_count_total:
                        rdb.read_next_chunk()
                        if rdb.point_count > 0:
                            xidx = rdb.get_chunk('scanline')
                            yidx = rdb.get_chunk('scanline_idx') 
                            zidx = rdb.get_chunk('target_index') - 1
                            vals = rdb.get_chunk(args.attribute)
                            grd.insert_values(vals, xidx, yidx, zidx)
                        pbar.update(rdb.point_count)
                descriptions = [f'Return {i+1:d}' for i in range(rdb.max_target_count)]
                grd.write_grid(args.output, descriptions=descriptions)

    elif args.input.endswith('.rxp'):

        with riegl_io.RXPFile(args.input, transform_file=args.transform_file, pose_file=args.pose_file) as rxp:
            if args.attribute in rxp.pulses:
                return_as_point_attribute = False
                nvars = 1
                zidx = 0
            else:
                return_as_point_attribute = True
                nvars = rxp.max_target_count
                zidx = rxp.get_data('target_index') - 1
            with grid.LidarGrid(rxp.maxc+1, rxp.maxr+1, 0, rxp.maxr, nvars=nvars) as grd:
                xidx = rxp.get_data('scanline', return_as_point_attribute=return_as_point_attribute)
                yidx = rxp.get_data('scanline_idx', return_as_point_attribute=return_as_point_attribute)
                vals = rxp.get_data(args.attribute, return_as_point_attribute=return_as_point_attribute)
                grd.insert_values(vals, xidx, yidx, zidx)
                descriptions = [f'Return {i+1:d}' for i in range(rxp.max_target_count)]
                grd.write_grid(args.output, descriptions=descriptions)

    else:
        print(f'{args.input:} not a recognized RIEGL file')

