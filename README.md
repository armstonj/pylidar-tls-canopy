# riegl-tls-canopy

This is a lite version of [pylidar canopy tools](http://www.pylidar.org/en/latest/commandline_canopy.html) specifically for proprietary format TLS data files from RIEGL VZ series instruments.

The use of RIEGL proprietary formats require the environment variables RIVLIB_ROOT and RDBLIB_ROOT to be set to the root installations of RiVLIB and RDBLib, respectively, before building.

RiVLib (version >=2.6.0 required) and RDBLib (version >=2.4.0 required) can be downloaded from the RIEGL members area at http://www.riegl.com/members-area/software-downloads/libraries/

You also need to install the Python API for RDBLib before building (see RDBLib documentation for instructions).

This software also assumes RDBX files have been generated using RiScanPro version >= 2.15.

To build and install:
```
python setup.py build install
```

