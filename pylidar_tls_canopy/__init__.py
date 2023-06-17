
"""
Main module
"""
# This file is part of pylidar_tls_canopy
# Copyright (C) 2022

PYLIDAR_TLS_CANOPY_VERSION = '0.3'
__version__ = PYLIDAR_TLS_CANOPY_VERSION

PRR_MAX_TARGETS = {"100 kHz": 18, "300 kHz": 15, "600 kHz": 8, "1200 kHz": 4}

from affine import Affine
RIO_DEFAULT_PROFILE = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': -9999.0,
                       'width': 8984, 'height': 2539, 'count': 1,
                       'transform': Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2539),
                       'blockxsize': 256, 'blockysize': 256, 'tiled': True,
                       'compress': 'deflate', 'interleave': 'pixel'}


