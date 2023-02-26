# pylidar-tls-canopy

This is a lite version of [pylidar canopy tools](http://www.pylidar.org/en/latest/commandline_canopy.html) specifically for gap probability modeling using data acquired by [RIEGL VZ series](http://www.riegl.com/nc/products/terrestrial-scanning/) and [LEAF](https://www.sensingsystems.com.au/) instruments.

See INSTALL.txt for installation instructions.

See the Jupyter Notebooks for RIEGL gridding, RIEGL vertical profile, LEAF time-series, and RIEGL voxel analysis examples.

Run the following to see the arguments for command line scripts:
```
pylidar_cartesiangrid -h
pylidar_scangrid -h
pylidar_sphericalgrid -h
pylidar_plantprofile -h
```
All of these are applicable to RIEGL data. Only pylidar_plantprofile is applicable to LEAF data.
