#!/usr/bin/env python3

DESCRIPTION='''
pylidar_tls_canopy: Voxelization of TLS scans

John Armston
University of Maryland
October 2022
'''

from pylidar_tls_canopy import riegl_io
from pylidar_tls_canopy import voxelization

import os
import json
import argparse
import numpy as np
import rasterio as rio
from tqdm import tqdm


def get_args():
    """
    Get the command line arguments
    """
    argparser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument('-i','--rxp_fn', metavar='FILE', type=str, nargs='+',
        help='Input RIEGL rxp filenames (typically upright then tilt scans or LEAF csv filename)')
    argparser.add_argument('-d','--rdbx_fn', metavar='FILE', type=str, nargs='+', default=None,
        help='Input RIEGL rdbx filenames (must match rxp filename order)')
    argparser.add_argument('-n','--name', metavar='STR', type=str, default='voxelgrid',
        help='Output voxel grid name')
    argparser.add_argument('-c','--chunk_size', metavar='INT', type=int, default=1000000,
        help='Chunksize for reading rdbx files')
    argparser.add_argument('-v','--voxelsize', metavar='FLOAT', type=float, default=1.0,
        help='Voxel size (m)')
    argparser.add_argument('-b','--bounds', metavar='FLOAT', type=float, default=[-25,-25,-15,25,25,45], nargs=6,
        help='Voxel grid bounds [mix,miny,minz,maxx,maxy,maxz] (m)')
    argparser.add_argument('-m','--min_n', metavar='INT', type=int, default=3,
        help='Mimimum views of voxel required to fit the Jupp (2009) model')
    argparser.add_argument('-t','--transform_file', metavar='FILE', type=str, nargs='+',
        help='Input RIEGL transform dat filenames (must match rxp filename order)')
    argparser.add_argument('-g','--dtm', metavar='FILE', type=str,
        help='Path to a DTM for the plot in GeoTiff format')
    argparser.add_argument('-w','--weights', action='store_true', default=False,
        help='Use the total path length as a weight in model fitting')
    argparser.add_argument('-N','--nodata', metavar='FLOAT', type=float, default=-9999,
        help='Voxelgrid nodata value')
    argparser.add_argument('-o','--outdir', metavar='DIR', type=str, default='.',
        help='Path to directory to write outputs')
    args = argparser.parse_args()

    return args


def run():

    # Parse command line arguments
    args = get_args()
    if args.rxp_fn is None:
        print('Run "pylidar_voxelization -h" for command line arguments')
        return

    # Check if RDBX files are available
    npos = len(args.rxp_fn)
    if args.rdbx_fn is None:
        args.rdbx_fn = [None] * npos

    # Initialize the configuration file
    config = {}
    config['bounds'] = args.bounds
    config['resolution'] = args.voxelsize
    config['nx'] = int( (bounds[3] - bounds[0]) // voxelsize)
    config['ny'] = int( (bounds[4] - bounds[1]) // voxelsize)
    config['nz'] = int( (bounds[5] - bounds[2]) // voxelsize)
    config['nodata'] = args.nodata
    config['dtm'] = args.dtm
    config['positions'] = {}

    # Loop through the scans and add them to the voxel grid
    pbar = tqdm(args.rxp_fn)
    for i,rxp_fn in enumerate(pbar):
        fn = os.path.basename(rxp_fn)
        pbar.set_description(f'Voxelizing {fn}')
        name = f'{args.name}_{os.path.splitext(fn)[0]}'
        vgrid_i = voxelization.VoxelGrid(dtm_filename=config['dtm'])
        vgrid_i.add_riegl_scan_position(rxp_fn, args.transform_fn[i], 
            rdbx_file=args.rdbx_fn[i], chunk_size=args.chunk_size)
    vgrid_i.voxelize_scan(config['bounds'], config['voxelsize'], save_counts=True)
    prefix = f'{args.outdir}/{name}'
    vgrid_i.write_grids(prefix)
    config['positions'][name] = vgrid_i.filenames

    # Write the configuration to file
    config_fn = f'{args.outdir}/{args.name}_config.json'
    with open(config_fn, 'w') as f:
        tmp = json.dumps(config, indent=4)
        f.write(tmp)

    # Derive PAI and the vertical canopy cover profile
    vmodel = voxelization.VoxelModel(config_fn)
    paiv,paih,nscans = vmodel.run_linear_model(min_n=args.min_n, weights=args.weights)
    cover_z = vmodel.get_cover_profile(paiv)

    # Write the outputs to file
    names = ['paiv','paih','nscans','coverz']
    for i,grid in enumerate([paiv,paih,nscans,cover_z]):
        output_fn = f'{args.outdir}/{args.name}_{names[i]}.tif'
        voxelization.write_voxelgrid(vmodel, grid, output_fn)

