# pylidar-tls-canopy

This is a lite version of [pylidar canopy tools](http://www.pylidar.org/en/latest/commandline_canopy.html) specifically for gap probability modeling using data acquired by [RIEGL VZ series](http://www.riegl.com/nc/products/terrestrial-scanning/) and [LEAF](https://www.sensingsystems.com.au/) instruments.

The use of RIEGL proprietary formats requires the following environment variables to be set before building:
  - RIVLIB_ROOT to the root directory of your unzipped RiVLIB download
  - PYLIDAR_CXX_FLAGS to -std=c++11

You also need to install the Python API for RDBLib before building. A pip wheel is available in the RIEGL RDBLib download (read the README.TXT file in the ```interface/python``` directory of your RDBLib download).

RiVLib (version >=2.6.0 required) and RDBLib (version >=2.4.0 required) can be downloaded from the RIEGL members area at http://www.riegl.com/members-area/software-downloads/libraries/. 
If you don't have a RIEGL members area login, you can also find the appropriate RIEGL library downloads for your system [here](https://drive.google.com/drive/folders/1ORdOxGI23D_uB0f4iVsyvVCldk1zekle?usp=sharing). This link will disappear once this repository is made public.

This software also assumes RDBX files have been generated using RiScanPro version >= 2.15.

To build and install:
```
python setup.py build install
```

Alternatively, you can create a conda environment for pylidar-tls-canopy using the provided environment.yml file:
```
cd pylidar-tls-canopy
conda env update -f environment.yml
conda activate pylidar-tls-canopy
```
Before creating the conda environment, you will need to edit the environment.yml file to modify the paths to the RDBLib pip wheel and the RiVLIB download root directory on your local system.

See the Jupyter Notebooks for gridding, vertical profile and LEAF time-series analysis examples.

Run the following to see the arguments for command line scripts (currently only applicable to RIEGL data):
```
pylidar_cartesiangrid -h
pylidar_scangrid -h
pylidar_sphericalgrid -h
pylidar_plantprofile -h
```

