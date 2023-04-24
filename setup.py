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
        elif sys.platform == 'darwin':
            print('RiVlib library is not available for MacOS')

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
        print('RIEGL RXP libraries not found.')
        print('If installed set $RIVLIB_ROOT to the install location of RiVLib')

def addRieglRDBDriver(extModules, cxxFlags):
    """
    Decides if the Riegl RDB driver is to be built. If so 
    adds the Extension class to extModules.
    """
    if 'RDBLIB_ROOT' in os.environ:
        print('Building RIEGL RDB Extension...')

        rdblibRoot = os.environ['RDBLIB_ROOT']
        if sys.platform == 'win32':
            rdbLibName = 'rdblib'
        else:
            rdbLibName = 'rdb'

        defines = getRieglRDBLibVersion(rdblibRoot, rdbLibName)
        defines.extend([NUMPY_MACROS])

        rieglRDBModule = Extension(name='riegl_rdb',
                define_macros=defines,
                sources=['src/riegl_rdb.cpp', 'src/pylidar.c'],
                include_dirs=[os.path.join(rdblibRoot, 'interface', 'c'), numpy.get_include()],
                extra_compile_args=cxxFlags,
                libraries=[rdbLibName],
                library_dirs=[os.path.join(rdblibRoot, 'library')],
                runtime_library_dirs=[os.path.join(rdblibRoot, 'library')])

        extModules.append(rieglRDBModule)
    else:
        print('RIEGL RDB libraries not found.')
        print('If installed set $RDBLIB_ROOT to the install location of RDBLib')

def getRieglRDBLibVersion(rdbRoot, libname):
    """
    Because we cannot distribute the rdblib library, we need
    to check that the major version at compile time matches the 
    version the user has at runtime. We do this by getting the
    version now and setting it as a #define. The library can then
    use the #define to check at runtime.
    
    Unfortunately the headers don't give us this information.
    
    """
    if sys.platform == 'win32':
        libname = os.path.join(rdbRoot, 'library', libname + '.dll')
    elif sys.platform == 'darwin':
        libname = os.path.join(rdbRoot, 'library', 'lib' + libname + '.dylib')
    else:
        libname = os.path.join(rdbRoot, 'library', 'lib' + libname + '.so')
    rdb = ctypes.cdll.LoadLibrary(libname)

    context = ctypes.c_void_p()
    logLevel = ctypes.c_char_p(b"NONE")
    logPath = ctypes.c_char_p(b"")
    rdb.rdb_context_new(ctypes.byref(context), logLevel, logPath)

    version = ctypes.c_char_p()
    rdb.rdb_library_version(context, ctypes.byref(version))
    versionString = version.value
    if sys.version_info[0] >= 3:
        versionString = versionString.decode()

    rdb.rdb_context_delete(ctypes.byref(context))

    # versionString is quite specific  -something like:
    # 2.2.1-2094 (x86_64-linux, Jul 11 2019, 13:10:32)
    # we probably don't have to have the exact same string
    # so just extract major and minor version numbers
    arr = versionString.split('.')
    major = arr[0]
    minor = arr[1]

    return [("RIEGL_RDB_MAJOR", major),
            ("RIEGL_RDB_MINOR", minor)]

# get any C++ flags
cxxFlags = getExtraCXXFlags()

# External modules        
externalModules = []
addRieglRXPDriver(externalModules, cxxFlags)
addRieglRDBDriver(externalModules, cxxFlags)

if len(externalModules) == 0:
    print('RiVLib and/or RDBLib not found. Only the LEAF and RDBX (if installed) drivers will be available.')

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
              'pylidar_voxelization = pylidar_tls_canopy.cmd.voxelization:run',
          ],
      },
      ext_modules=externalModules,
      description='Tools for canopy gap probability modeling using RIEGL VZ and LEAF TLS measurements',
      classifiers=['Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10'
          ])
