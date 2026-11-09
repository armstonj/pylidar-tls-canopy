#!/usr/bin/env python3

DESCRIPTION='''
pylidar_tls_canopy: Voxelization of drone lidar (DLS) LAZ flightlines

Each flightline is voxelized one gps_time bucket at a time and written as a
set of GeoTIFF grids. A VoxelModel configuration file listing all flightlines
is written to the output directory.

John Armston
University of Maryland
November 2026
'''

import os
import json
import glob
import argparse
import multiprocessing as mp


def get_args():
    """
    Get the command line arguments
    """
    argparser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument('-i','--laz', metavar='FILE', type=str, nargs='+', required=True,
        help='Input LAZ flightlines (files or glob patterns)')
    argparser.add_argument('-b','--bounds', metavar='FLOAT', type=float, nargs=6, required=True,
        help='Voxel grid bounds [minx,miny,minz,maxx,maxy,maxz] (m)')
    argparser.add_argument('-v','--voxelsize', metavar='FLOAT', type=float, default=0.5,
        help='Voxel size (m)')
    argparser.add_argument('-g','--dtm', metavar='FILE', type=str, default=None,
        help='DTM (GeoTiff, same CRS as the LAZ) for ground and noise classification')
    argparser.add_argument('-t','--ztop', metavar='FLOAT', type=float, default=None,
        help='Canopy top elevation used for the crop buffer and noise ceiling (default: maxz)')
    argparser.add_argument('-z','--max_zenith', metavar='FLOAT', type=float, default=70.0,
        help='Maximum view zenith of pulses (degrees)')
    argparser.add_argument('-u','--zbelow', metavar='FLOAT', type=float, default=5.0,
        help='Depth below minz from which returns are read (m)')
    argparser.add_argument('-a','--assume_nadir', action='store_true', default=False,
        help='Treat every pulse as vertical')
    argparser.add_argument('-s','--spill_dir', metavar='DIR', type=str, required=True,
        help='Directory for temporary gps_time bucket files')
    argparser.add_argument('-d','--bucket_seconds', metavar='FLOAT', type=float, default=2.0,
        help='gps_time bucket length (s)')
    argparser.add_argument('-n','--name', metavar='STR', type=str, default='dls',
        help='Name of the configuration file')
    argparser.add_argument('-o','--outdir', metavar='DIR', type=str, default='.',
        help='Output directory')
    argparser.add_argument('-p','--nproc', metavar='INT', type=int, default=1,
        help='Number of flightlines processed in parallel')
    argparser.add_argument('-r','--threads', metavar='INT', type=int, default=4,
        help='LAZ decompression threads per process')
    return argparser.parse_args()


def _worker(task):
    laz, args = task
    os.environ['RAYON_NUM_THREADS'] = str(args.threads)
    from pylidar_tls_canopy import voxelization
    name = os.path.splitext(os.path.basename(laz))[0]
    prefix = os.path.join(args.outdir, name)
    g = voxelization.DLSVoxelGrid()
    g.voxelize_flightline(laz, args.bounds, args.voxelsize, spill_dir=args.spill_dir,
        bucket_seconds=args.bucket_seconds, dtm_filename=args.dtm, ztop=args.ztop,
        max_zenith=args.max_zenith, zbelow=args.zbelow, assume_nadir=args.assume_nadir)
    if g.summary.get('n_pulses', 0) == 0:
        return name, None, None, None
    g.write_grids(prefix)
    g.write_summary(prefix)
    return name, g.filenames, [int(n) for n in (g.nx, g.ny, g.nz)], g.bounds.tolist()


def run():
    args = get_args()
    files = sorted(f for p in args.laz for f in glob.glob(p))
    if not files:
        raise SystemExit('No input LAZ files')
    os.makedirs(args.outdir, exist_ok=True)

    config = {'positions': {}, 'resolution': args.voxelsize, 'nodata': -9999.0, 'dtm': args.dtm}
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.nproc) as pool:
        for i, (name, filenames, n, bounds) in enumerate(pool.imap_unordered(_worker, [(f, args) for f in files])):
            if filenames is None:
                print(f'[{i + 1}/{len(files)}] {name}: no pulses within the crop bounds', flush=True)
                continue
            config['positions'][name] = filenames
            config['nx'], config['ny'], config['nz'] = n
            config['bounds'] = bounds
            print(f'[{i + 1}/{len(files)}] {name}', flush=True)

    config['positions'] = dict(sorted(config['positions'].items()))
    config_fn = os.path.join(args.outdir, f'{args.name}_config.json')
    with open(config_fn, 'w') as f:
        json.dump(config, f, indent=4)
    print(config_fn)


if __name__ == '__main__':
    run()
