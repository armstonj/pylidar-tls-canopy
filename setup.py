"""
Install script for riegl_canopy
"""

import os
import sys
import ctypes
from numpy.distutils.core import setup, Extension

NUMPY_MACROS = ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')

def getExtraCXXFlags():
    """
    Looks at the $PYLIDAR_CXX_FLAGS environment variable.
    If it exists, this function returns a list of flags
    to be passed as the extra_compile_args argument to
    the Extension constructor.
    Otherwise None.
    """
    if 'PYLIDAR_CXX_FLAGS' in os.environ:
        return os.environ['PYLIDAR_CXX_FLAGS'].split()
    else:
        return None

def addRieglRXPDriver(extModules, cxxFlags):
    """
    RIEGL RXP driver
    """
    if 'RIVLIB_ROOT' in os.environ:
        print('Building RIEGL RXP Extension...')
        rivlibRoot = os.environ['RIVLIB_ROOT']
        rivlibs = ['scanlib-mt', 'riboost_chrono-mt', 
                   'riboost_date_time-mt', 'riboost_filesystem-mt', 
                   'riboost_regex-mt', 'riboost_system-mt', 
                   'riboost_thread-mt']
        
        # on Windows the libs do not follow the normal naming convention
        # and start with 'lib'. On Linux the compiler prepends this automatically
        # but on Windows we need to do it manually
        if sys.platform == 'win32':
            rivlibs = ['lib' + name for name in rivlibs]
            
        rieglModule = Extension(name='riegl_rxp', 
                define_macros=[NUMPY_MACROS],
                sources=['src/riegl_rxp.cpp', 'src/pylidar.c'],
                include_dirs=[os.path.join(rivlibRoot, 'include')],
                extra_compile_args=cxxFlags,
                libraries=rivlibs,
                library_dirs=[os.path.join(rivlibRoot, 'lib')],
                runtime_library_dirs=[os.path.join(rivlibRoot, 'lib')])
                 
        extModules.append(rieglModule)
    else:
        print('RIEGL RXP Libraries not found.')
        print('If installed set $RIVLIB_ROOT to the install location of RiVLib')


# get any C++ flags
cxxFlags = getExtraCXXFlags()

# Are we installing the command line scripts?
# This is an experimental option for users who are
# using the Python entry point feature of setuptools and Conda instead
NO_INSTALL_CMDLINE = int(os.getenv('GEDIPY_NOCMDLINE', '0')) > 0
if NO_INSTALL_CMDLINE:
    scriptList = None
else:
    scriptList = ['bin/riegl_scangrid','bin/riegl_sphericalgrid',
                  'bin/riegl_plantprofile']

# External modules        
externalModules = []
addRieglRXPDriver(externalModules, cxxFlags)

if len(externalModules) == 0:
    raise SystemExit('No RIEGL libraries found - exiting install')

setup(name='riegl_canopy',
      version='0.1',
      packages=['riegl_canopy'],
      scripts=scriptList,
      ext_modules=externalModules,
      description='Tools for retriving canopy structure from RIEGL TLS measurements',
      classifiers=['Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9'
          ])
