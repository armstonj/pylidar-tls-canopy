"""
Install script for riegl_canopy
"""

import os
import sys
import numpy
import ctypes
import pylidar_tls_canopy
from setuptools import Extension,setup

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
                include_dirs=[os.path.join(rivlibRoot, 'include'), numpy.get_include()],
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

# External modules        
externalModules = []
addRieglRXPDriver(externalModules, cxxFlags)

if len(externalModules) == 0:
    print('No RIEGL libraries found. Only the LEAF driver will be available.')


setup(name='pylidar-tls-canopy',
      version=pylidar_tls_canopy.__version__,
      author='John Armston',
      author_email='armston@umd.edu',
      packages=['pylidar_tls_canopy','pylidar_tls_canopy.cmd'],
      entry_points={
          'console_scripts': [
              'pylidar_cartesiangrid = pylidar_tls_canopy.cmd.cartesiangrid:run',
              'pylidar_scangrid = pylidar_tls_canopy.cmd.scangrid:run',
              'pylidar_sphericalgrid = pylidar_tls_canopy.cmd.sphericalgrid:run',
              'pylidar_plantprofile = pylidar_tls_canopy.cmd.plantprofile:run',
          ],
      },
      ext_modules=externalModules,
      description='Tools for canopy gap probability modeling using RIEGL VZ and LEAF TLS measurements',
      classifiers=['Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10'
          ])
