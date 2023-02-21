# pylidar-tls-canopy

This is a lite version of [pylidar canopy tools](http://www.pylidar.org/en/latest/commandline_canopy.html) specifically for gap probability modeling using data acquired by [RIEGL VZ series](http://www.riegl.com/nc/products/terrestrial-scanning/) and [LEAF](https://www.sensingsystems.com.au/) instruments.

The use of RIEGL proprietary formats requires the following environment variables to be set before building:
  - RIVLIB_ROOT to the root directory of your unzipped RiVLIB download
  - PYLIDAR_CXX_FLAGS to -std=c++11

For processing of RIEGL data, you will also need to install the Python API for RDBLib before building. A pip wheel is available in the RIEGL RDBLib download (read the README.TXT file in the ```interface/python``` directory of your RDBLib download).

RiVLib (version >=2.6.0 required) and RDBLib (version >=2.4.0 required) can be downloaded from the RIEGL members area at http://www.riegl.com/members-area/software-downloads/libraries/. 
If you don't have a RIEGL members area login, you can also find the appropriate RIEGL library downloads for your system [here](https://drive.google.com/drive/folders/1ORdOxGI23D_uB0f4iVsyvVCldk1zekle?usp=sharing). This link will disappear once this repository is made public or merged with pylidar.

This software also assumes RDBX files have been generated using RiScanPro version >= 2.15.

pylidar-tls-canopy will install without REIGL libraries being available, but will then only work with LEAF data files.

To build and install:
```
cd pylidar-tls-canopy
python -m pip install .
```

Alternatively, you can create a conda environment for pylidar-tls-canopy using the provided environment.yml file:
```
cd pylidar-tls-canopy
conda env update -f environment.yml
conda activate pylidar-tls-canopy
```
Before creating the conda environment, you will need to edit the environment.yml file to modify the paths to the RDBLib pip wheel and the RiVLIB download root directory on your local system. If you are only installing pylidar-tls-canopy for analysis of LEAF data then remove these lines.

See the Jupyter Notebooks for RIEGL gridding, RIEGL vertical profile, LEAF time-series, and RIEGL voxel analysis examples.

Run the following to see the arguments for command line scripts:
```
pylidar_cartesiangrid -h
pylidar_scangrid -h
pylidar_sphericalgrid -h
pylidar_plantprofile -h
```
All of these are applicable to RIEGL data. Only pylidar_plantprofile is applicable to LEAF data.
