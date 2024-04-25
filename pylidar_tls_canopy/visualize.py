#!/usr/bin/env python3
"""
pylidar_tls_canopy

Code for visualization of pylidar-tls-canopy outputs

John Armston
University of Maryland
January 2024
"""

import matplotlib
import numpy as np
import rasterio as rio

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from matplotlib import animation, rc, pyplot
from matplotlib.ticker import MaxNLocator

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from IPython.display import HTML


def plot_riegl_grid(data, label='Range (m)', clim=[0,30], figsize=(16,10), nbins=10,
                    cmap='bone', nreturns=None, extend='max', nodata=-9999, extent=None,
                    xlabel=None, ylabel=None, facecolor='white', title=False):
    """Example function to plot a RIEGL grid"""
    if nreturns is None:
        nreturns = data.shape[0]
    fig, ax = plt.subplots(ncols=1, nrows=nreturns, squeeze=False, 
                           sharex=False, sharey=False, figsize=figsize)
    with plt.style.context('seaborn-v0_8-notebook'):
        for i in range(nreturns):  
            ax[i,0].set_facecolor(facecolor)
            ax[i,0].set(adjustable='box', aspect='equal')
            ax[i,0].set(xlabel=xlabel, ylabel=ylabel)
            if title and nreturns > 1:
                ax[i,0].set_title(f'Return {i+1:d} (maximum {data.shape[0]:d})', fontsize=14)
            if extent is None:
                ax[i,0].get_xaxis().set_visible(False)
                ax[i,0].get_yaxis().set_visible(False)
            tmp = np.ma.masked_equal(data[i], nodata)
            p = ax[i,0].imshow(tmp, interpolation='nearest', clim=clim, 
                               cmap=matplotlib.cm.get_cmap(cmap,nbins),  
                               vmin=clim[0], vmax=clim[1], extent=extent)
            divider = make_axes_locatable(ax[i,0])
            cax = divider.append_axes('right', size='2%', pad=0.05)
            cbar = fig.colorbar(p, label=label, cax=cax, extend=extend)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=nbins))
    fig.tight_layout()
    plt.show()


def plot_ground_plane(x, y, z, grid_extent, grid_resolution, planefit, 
                      grid_origin=[0,0], figsize=[8,8], elev=None, azim=None):
    """Example function to plot a ground plane"""    
    xv = np.linspace(grid_origin[0]-grid_extent/2, grid_origin[0]+grid_extent/2, grid_extent//grid_resolution + 1)
    yv = np.linspace(grid_origin[1]-grid_extent/2, grid_origin[1]+grid_extent/2, grid_extent//grid_resolution + 1)
    xc,yc = np.meshgrid(xv,yv)
    zc = planefit['Parameters'][1] * xc + planefit['Parameters'][2] * yc + planefit['Parameters'][0]    
    fig = plt.figure(figsize=figsize)
    with plt.style.context('seaborn-v0_8-talk'):
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
        ax.plot_wireframe(xc, yc, zc)
        ax.scatter(x, y, z, color='Brown')
        ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
        ax.set(title=f'Extent = {grid_extent}m | Resolution = {grid_resolution}m')  
        ax.view_init(elev, azim)
    fig.tight_layout()
    plt.show()
    
    
def plot_vertical_profiles(profiles, height, labels=['Hinge','Linear','Weighted'], title=None,
                          figsize=[6,6], ylim=(0,50), xlim=None, xlabel=r'PAVD ($m^2 m^{-3}$)'):
    """Example function to plot a vertical profile"""
    fig, ax = plt.subplots(figsize=figsize, squeeze=True)
    with plt.style.context('seaborn-v0_8-talk'):
        for i,profile in enumerate(profiles):
            if labels is None:
                ax.plot(profile, height, linewidth=1.5)
            else:
                ax.plot(profile, height, label=labels[i], linewidth=1.5)
        ax.set(xlabel=xlabel, ylabel='Height (m)')
        ax.set(ylim=ylim, xlim=xlim, title=title)
        ax.set_facecolor('white')
        if labels is not None:
            ax.legend()
    fig.tight_layout() 
    plt.show()


def plot_leaf_grid(data, inset=None, label='Range (m)', clim=[0,30], figsize=(16,10), nbins=10,
                   cmap='bone', inset_cmap='Greens_r', nreturns=None, extend='max', nodata=-9999, 
                   extent=None, xlabel=None, ylabel=None, facecolor='white', title=False):
    """Example function to plot a LEAF grid"""
    if nreturns is None:
        nreturns = data.shape[0]
    fig, ax = plt.subplots(ncols=1, nrows=nreturns, squeeze=False, 
                           sharex=False, sharey=False, figsize=figsize)
    with plt.style.context('seaborn-v0_8-notebook'):
        for i in range(nreturns):  
            ax[i,0].set_facecolor(facecolor)
            ax[i,0].set(adjustable='box', aspect='equal')
            ax[i,0].set(xlabel=xlabel, ylabel=ylabel)
            if isinstance(title, bool):
                if title and nreturns > 1:
                    ax[i,0].set_title(f'Return {i+1:d} (maximum {data.shape[0]:d})', fontsize=14)
            else:
                if title:
                    ax[i,0].set_title(title, fontsize=14)
            if extent is None:
                ax[i,0].get_xaxis().set_visible(False)
                ax[i,0].get_yaxis().set_visible(False)
            tmp = np.ma.masked_equal(data[i], nodata)
            p = ax[i,0].imshow(tmp, interpolation='none', clim=clim, 
                               cmap=matplotlib.cm.get_cmap(cmap,nbins),  
                               vmin=clim[0], vmax=clim[1], extent=extent)
            if inset is not None:
                tmp = np.ma.masked_equal(inset[i], nodata)
                pi = ax[i,0].imshow(tmp, interpolation='none', clim=clim, 
                                    cmap=matplotlib.cm.get_cmap(inset_cmap,nbins),  
                                    vmin=clim[0], vmax=clim[1], extent=extent)
            divider = make_axes_locatable(ax[i,0])
            cax = divider.append_axes('right', size='2%', pad=0.05)
            cbar = fig.colorbar(p, label=label, cax=cax, extend=extend)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=nbins))
    fig.tight_layout()
    plt.show()

    
def plot_xyz(x, y, z, c, figsize=[8,8], elev=None, azim=None, cmap='viridis', xylim=(None,None)):
    """Example function to plot the LEAF point cloud""" 
    fig = plt.figure(figsize=figsize)
    with plt.style.context('seaborn-v0_8-talk'):
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
        ax.scatter(x, y, z, c=c, s=3, cmap=cmap)
        ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', xlim=xylim, ylim=xylim)
        ax.view_init(elev, azim)
    fig.tight_layout()
    plt.show()    


def plot_timseries_2d(data, clim=[0,0.2], title=None, nbins='auto', cmap='gist_earth', 
    label=r'PAVD ($m^{2} m^{-3}$)', facecolor='0.5', extend='max', xextent=[None,None], 
    yextent=[None,None], figsize=(20,10), xticks=None, xrotation=0):
    """Example function to plot a 2D histogram of the LEAF PAVD time-series""" 
    fig, ax = plt.subplots(ncols=1, nrows=1, squeeze=True, figsize=figsize)
    xextent = [mdates.date2num(d) if d is not None else None for d in xextent]
    if xticks is not None:
        xticks = [mdates.date2num(d) for d in xticks]
    if isinstance(nbins, int):
        cmap = matplotlib.cm.get_cmap(cmap,nbins)
    with plt.style.context('seaborn-v0_8-talk'):
        ax.set_facecolor(facecolor)
        ax.set(adjustable='datalim', xlabel='Date', ylabel='Height (m)')
        ax.xaxis_date()
        date_format = mdates.DateFormatter('%d-%b-%y')
        ax.xaxis.set_major_formatter(date_format)
        if xticks is not None:
            ax.set_xticks(xticks)
        if title is not None:
            ax.set_title(title)
        p = ax.imshow(data, interpolation='none', clim=clim, cmap=cmap, alpha=1.0, 
                      vmin=clim[0], vmax=clim[1], extent=xextent+yextent, aspect='auto')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=xrotation)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.05)
        cbar = fig.colorbar(p, label=label, cax=cax, extend=extend)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=nbins))
        
    fig.tight_layout()
    plt.show()

    
def plot_timseries_1d(dates, values, quality, fitted=None, xlim=[None,None], ylim=[0,None], 
    title=None, ylabel=r'PAI ($m^{2} m^{-2}$)', linestyle=None, figsize=(20,10)):
    """Example function to plot a 2D histogram of the LEAF PAVD time-series""" 
    fig, ax = plt.subplots(ncols=1, nrows=1, squeeze=True, figsize=figsize)
    with plt.style.context('seaborn-v0_8-talk'):
        ax.scatter(dates[quality], values[quality], linestyle=linestyle, color='DarkGreen', label='True')
        ax.scatter(dates[~quality], values[~quality], linestyle=linestyle, color='Brown', label='False')
        if fitted is not None:
            ax.plot(dates, fitted, linestyle=linestyle, color='Black')
        date_format = mdates.DateFormatter('%d-%b-%y')
        ax.xaxis.set_major_formatter(date_format)
        ax.set(xlabel='Date', ylabel=ylabel, xlim=xlim, ylim=ylim, title=title)
    fig.autofmt_xdate()
    plt.legend(title='Quality')
    plt.tight_layout()
    plt.show()


class VizVoxelGrid:
    """Class for creating an animation of a voxel grid"""

    def __init__(self, figsize=[15,4], ncols=3, nrows=1):
        self.fig,self.axes = plt.subplots(figsize=figsize, ncols=ncols,
            nrows=nrows, squeeze=True)

    def create_viz(self, filenames, layer=1, names=['hits','miss','occl'], voxelsize=1,
        vmin_vals=[0,0,0], vmax_vals=[1000,10000,10000], titles=None, interval=50,
        frames=57, bounds=[-25,-25,-15,25,25,45], nodata=-9999, cmap='gist_gray',
        facecolor='lightyellow'):

        self.layer = layer
        self.bounds = bounds
        self.voxelsize = voxelsize
        self.nodata = nodata
        self.cmap = cmap
        self.facecolor = facecolor

        if titles is None:
            titles = names

        self.src = [rio.open(filenames[k]) for k in sorted(filenames) if k in names]

        self.__initfig(vmin_vals, vmax_vals, titles)
        anim = animation.FuncAnimation(self.fig, self.__updatefig, interval=interval, frames=frames)
        plt.close()

        return HTML(anim.to_jshtml())

    def __initfig(self, vmin_vals, vmax_vals, titles):
        
        elev = self.bounds[2] + (self.layer - 1) * self.voxelsize
        self.fig.suptitle(f'Elevation {elev:.1f} m', fontsize=16)
        self.images = []    
        with plt.style.context('seaborn-v0_8-notebook'):    
            for i,ax in enumerate(self.axes):
                tmp = np.ma.masked_equal(self.src[i].read(self.layer), self.nodata)
                im = ax.imshow(tmp, cmap=self.cmap, animated=True, 
                               vmin=vmin_vals[i], vmax=vmax_vals[i])        
                xt = ax.get_xticks().tolist()
                yt = ax.get_yticks().tolist()
                t = [self.src[i].transform * c for c in zip(xt,yt)]
                ax.xaxis.set_major_locator(mticker.FixedLocator(xt))
                ax.set_xticklabels([f'{c[0]:.1f}' for c in t])
                ax.yaxis.set_major_locator(mticker.FixedLocator(yt))
                ax.set_yticklabels([f'{c[1]:.1f}' for c in t])
                ax.set_title(titles[i])
                ax.set_facecolor(self.facecolor)
                self.images.append(im)    

    def __updatefig(self, *args):
        elev = self.bounds[2] + (self.layer - 1) * self.voxelsize
        self.fig.suptitle(f'Elevation {elev:.1f} m', fontsize=16)
        for i,im in enumerate(self.images):
            tmp = np.ma.masked_equal(self.src[i].read(self.layer), self.nodata)
            im.set_array(tmp)
        self.layer += 1


def plot_voxel_grid(data, title=['Range (m)'], clim=[[0,30]], figsize=(16,10), nbins=[10],
                    cmap=['bone'], extend=['max'], nodata=-9999, extent=None,
                    xlabel=None, ylabel=None, facecolor='white', suptitle=None, 
                    ncols=None, nrows=1):
    """Example function to plot a voxel grid"""
    ngrids = len(data)
    if ncols is None:
        ncols = ngrids
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, squeeze=True, 
                           sharex=True, sharey=True, figsize=figsize)
    fig.suptitle(suptitle, fontsize=16)
    with plt.style.context('seaborn-v0_8-notebook'):
        for i in range(ngrids):  
            ax[i].set_facecolor(facecolor)
            ax[i].set(adjustable='box', aspect='equal')
            ax[i].set(xlabel=xlabel, ylabel=ylabel, title=title[i])
            if extent is None:
                ax[i].get_xaxis().set_visible(False)
                ax[i].get_yaxis().set_visible(False)
            tmp = np.ma.masked_equal(data[i], nodata)
            p = ax[i].imshow(tmp, interpolation='nearest', clim=clim[i], 
                             cmap=matplotlib.cm.get_cmap(cmap[i],nbins[i]),  
                             vmin=clim[i][0], vmax=clim[i][1], extent=extent)
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes('right', size='2%', pad=0.05)
            cbar = fig.colorbar(p, cax=cax, extend=extend[i])
            cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=nbins[i]))
    fig.tight_layout()
    plt.show()


def plot_voxel_profiles(x, y, figsize=[12,5], ncols=3, nrows=1, config=None):
    """Example function to plot a voxel profiles"""
    fig,ax = pyplot.subplots(figsize=figsize, ncols=ncols, nrows=nrows, squeeze=True)
 
    for i,a in enumerate(ax):
        if config is None:
            a.plot(x[i], y[i])
            a.set(ylabel='Elevation (m)')
        else:
            d = config[i]
            a.plot(x[i], y[i], color=d['color'], linestyle=d['linestyle'], label=d['label'])
            a.set(xlabel=d['xlabel'], ylabel=d['ylabel'], title=d['title'], xlim=d['xlim'])
            if d['legend']:
                a.legend()
        a.grid(False)

    fig.tight_layout()

