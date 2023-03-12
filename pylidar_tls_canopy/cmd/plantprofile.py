#!/usr/bin/env python3

DESCRIPTION='''
pylidar_tls_canopy: Create a vertical plant profile

John Armston
University of Maryland
October 2022
'''

from pylidar_tls_canopy import riegl_io
from pylidar_tls_canopy import leaf_io
from pylidar_tls_canopy import plant_profile

import os
import argparse
import numpy as np
from tqdm import tqdm


RDB_ATTRIBUTES = {'riegl.xyz': 'riegl_xyz','riegl.target_index': 'target_index', 
                  'riegl.target_count': 'target_count'}


def get_args():
    """
    Get the command line arguments
    """
    argparser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument('-i','--input_file', metavar='FILE', type=str, nargs='+',
        help='Input RIEGL rxp filenames (typically upright then tilt scans or LEAF csv filename)')
    argparser.add_argument('-d','--rdbx_input', metavar='FILE', type=str, nargs='+', default=None,
        help='Input RIEGL rdbx filenames (must match rxp filename order)')
    argparser.add_argument('-o','--outfile', metavar='FILE', type=str, default=None,
        help='Output plant profile filename (CSV)')
    argparser.add_argument('-p','--pgap_outfile', metavar='FILE', type=str, default=None,
        help='Output Pgap profile filename (CSV)')
    argparser.add_argument('-c','--chunk_size', metavar='INT', type=int, default=100000,
        help='Chunksize for reading rdbx files')
    argparser.add_argument('--height_resolution', metavar='FLOAT', type=float, default=0.5,
        help='Vertical resolution (height)')
    argparser.add_argument('--zenith_resolution', metavar='FLOAT', type=float, default=5,
        help='Zenith resolution (degrees)')
    argparser.add_argument('--azimuth_resolution', metavar='FLOAT', type=float, default=45,
        help='Azimuth resolution (degrees)')
    argparser.add_argument('--min_zenith', metavar='FLOAT', type=float, default=None, nargs='+',
        help='Minimum zenith angle to use for input file')
    argparser.add_argument('--max_zenith', metavar='FLOAT', type=float, default=None, nargs='+',
        help='Maximum zenith angle to use for input file')
    argparser.add_argument('--min_azimuth', metavar='FLOAT', type=float, default=0, 
        help='Minimum azimuth angle to use for input file')
    argparser.add_argument('--max_azimuth', metavar='FLOAT', type=float, default=360,
        help='Maximum azimuth angle to use for input file')
    argparser.add_argument('--invert_azimuth', action='store_true', default=False,
        help='Exclude rather than include specified azimuth angle range')
    argparser.add_argument('--max_height', metavar='FLOAT', type=float, default=50,
        help='Maximum height above ground to use for the vertical profile')
    argparser.add_argument('--grid_extent', metavar='FLOAT', type=float, default=60,
        help='Plane fit grid extent (m)')
    argparser.add_argument('--grid_resolution', metavar='FLOAT', type=float, default=10,
        help='Plane fit grid resolution (m)')
    argparser.add_argument('-s','--sensor_height', metavar='FLOAT', type=float, default=1.6,
        help='Height above ground of LEAF sensor origin (m)')
    argparser.add_argument('-t','--transform_file', metavar='FILE', type=str, nargs='+',
        help='Input RIEGL transform dat filenames (must match rxp filename order)')
    argparser.add_argument('-R','--reportfile', metavar='FILE', type=str,
        help='Plane fitting report file')
    argparser.add_argument('-m','--method', metavar='STR', type=str, default='WEIGHTED', 
        choices=('WEIGHTED','FIRST','ALL','FIRSTLAST'),
        help='Method for calculating Pgap from the point clouds')
    argparser.add_argument('-l','--leaf', action='store_true', default=False,
        help='--input_file are LEAF instrument files')
    args = argparser.parse_args()

    return args


def run():

    args = get_args()
    if args.input_file is None:
        print('Run "pylidar_plantprofile -h" for command line arguments')
        return

    if args.outfile is None:
        args.outfile = 'plant_profiles.csv'

    # Check if RDBX files are available
    rxp = False
    if args.rdbx_input is None:
        args.rdbx_input = [None] * len(args.input_file)
        rxp = True

    # Get the ground plane
    ground_plane = None
    if not args.leaf:
        print('Fitting the ground plane')
        x,y,z,r = plant_profile.get_min_z_grid(args.rdbx_input, args.transform_file,
            args.grid_extent, args.grid_resolution, rxp=rxp)
        planefit = plant_profile.plane_fit_hubers(x, y, z, w=1/r, reportfile=args.reportfile)
        ground_plane = planefit['Parameters']

    # Initialize the profile
    min_zenith_r = np.radians(args.min_zenith)
    max_zenith_r = np.radians(args.max_zenith)
    vpp = plant_profile.Jupp2009(hres=args.height_resolution, zres=args.zenith_resolution, ares=args.azimuth_resolution, 
        min_z=min(args.min_zenith), max_z=max(args.max_zenith), min_h=0, max_h=args.max_height, ground_plane=ground_plane) 

    # Read the data
    n_pos = len(args.transform_file)
    for i in range(n_pos):
        print(f'Reading scan position {i+1:d}')
        if not args.leaf:
            vpp.add_riegl_scan_position(args.input_file[i], args.transform_file[i], rdbx_file=args.rdbx_input[i],
                method=args.method, min_zenith=args.min_zenith[i], max_zenith=args.max_zenith[i], max_hr=None, 
                sensor_height=args.sensor_height)
        else:
            vpp.add_leaf_scan_position(args.input_file[i], method=args.method, min_zenith=args.min_zenith[i],
                max_zenith=args.max_zenith[i], sensor_height=args.sensor_height)

    # Run and export the profiles
    print('Creating the vertical plant profiles')
    vpp.get_pgap_theta_z(min_azimuth=args.min_azimuth, max_azimuth=args.max_azimuth, invert=args.invert_azimuth)
    vpp.exportPlantProfiles(outfile=args.outfile)
    if args.pgap_outfile is not None:
        vpp.exportPgapProfiles(outfile=args.pgap_outfile)

