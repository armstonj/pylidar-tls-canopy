# riegl-tls-canopy

This is a lite version of pylidar canopy tools specifically for proprietary format TLS data files from RIEGL VZ series instruments.

The use of RIEGL proprietary formats require the environment variables RIVLIB_ROOT and RDBLIB_ROOT to be set to the root installations of RiVLIB and RDBLib, respectively, before building.

RiVLib and RDBLib can be downloaded from the RIEGL members area at http://www.riegl.com/members-area/software-downloads/libraries/

You also need to install the Python API for RDBLib before building (see RDBLib documentation for instructions).

To build and install:
```
python setup.py build install
```

