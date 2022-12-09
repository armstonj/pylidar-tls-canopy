# pylidar-tls-canopy

This is a lite version of [pylidar canopy tools](http://www.pylidar.org/en/latest/commandline_canopy.html) specifically for gap probability modeling using data acquired by [RIEGL VZ series](http://www.riegl.com/nc/products/terrestrial-scanning/) and [LEAF](https://www.sensingsystems.com.au/) instruments.

The use of RIEGL proprietary formats require the environment variables RIVLIB_ROOT and RDBLIB_ROOT to be set to the root directory of your RiVLIB and RDBLib installations, respectively, before building.

RiVLib (version >=2.6.0 required) and RDBLib (version >=2.4.0 required) can be downloaded from the RIEGL members area at http://www.riegl.com/members-area/software-downloads/libraries/

You also need to install the Python API for RDBLib before building. Read the README.TXT file in the ```interface/python``` directory of your RDBLib installation.

This software also assumes RDBX files have been generated using RiScanPro version >= 2.15.

To build and install:
```
python setup.py build install
```

See the Jupyter Notebooks for gridding and vertical profile examples.

Run the following to see the arguments for command line scripts:
```
pylidar_cartesiangrid -h
pylidar_scangrid -h
pylidar_sphericalgrid -h
pylidar_plantprofile -h
```

