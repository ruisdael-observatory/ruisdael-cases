import matplotlib.pyplot as pl
import xarray as xr
import numpy as np

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def cartopy_subplot(a,b,c,label=''):
    proj = ccrs.LambertConformal(central_longitude=4.9, central_latitude=51.967)
    ax = pl.subplot(a,b,c,projection=proj)
    pl.title(label, loc='left')
    
    # Add coast lines et al.
    ax.coastlines(resolution='10m', linewidth=0.8, color='k')
    
    countries = cfeature.NaturalEarthFeature(
            category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none', zorder=100)
    ax.add_feature(countries, edgecolor='k', linewidth=0.8)
    
    lakes = cfeature.NaturalEarthFeature(
            category='physical', name='lakes', scale='10m', facecolor='none', zorder=100)
    ax.add_feature(lakes, edgecolor='k', linewidth=0.8)
    
    ax.set_extent([lon.min()-0.1, lon.max()+0.1, lat.min()-0.1, lat.max()+0.1], ccrs.PlateCarree())

    return ax


#f = xr.open_dataset('crossxy.0001.001.nc')
fqt = xr.open_dataset('crossxy.0001.qt.001.nc')
fthl = xr.open_dataset('crossxy.0001.thl.001.nc')
fu = xr.open_dataset('crossxy.0001.u.001.nc')
fv = xr.open_dataset('crossxy.0001.v.001.nc')

lon = np.load('lon_LES.npy')
lat = np.load('lat_LES.npy')

vmin_u = fu['uxy'].min()
vmax_u = fu['uxy'].max()

vmin_v = fv['vxy'].min()
vmax_v = fv['vxy'].max()

vmin_t = fthl['thlxy'].min()
vmax_t = fthl['thlxy'].max()

vmin_q = fqt['qtxy'].min()*1e3
vmax_q = fqt['qtxy'].max()*1e3

cm = pl.cm.RdBu_r

for tt,t in enumerate(range(0,fu.time.size,1)):
    print(tt)

    pl.figure(figsize=(10,7.2))
    print(lon.shape, lat.shape,  fu['uxy'][t,:,:].shape)

    ax = cartopy_subplot(2, 2, 1, 'u (m/s), t={0:04.0f} s'.format(fu['time'][t].values))
    pc=ax.pcolormesh(lon, lat, fu['uxy'][t,:,:].transpose(), transform=ccrs.PlateCarree(), cmap=cm, vmin=vmin_u, vmax=vmax_u)
    pl.colorbar(pc)
    
    ax = cartopy_subplot(2, 2, 2, 'v (m/s)')
    pc=ax.pcolormesh(lon, lat, fv['vxy'][t,:,:].transpose(), transform=ccrs.PlateCarree(), cmap=cm, vmin=vmin_v, vmax=vmax_v)
    pl.colorbar(pc)
    
    ax = cartopy_subplot(2, 2, 3, 'thl (K)')
    pc=ax.pcolormesh(lon, lat, fthl['thlxy'][t,:,:].transpose(), transform=ccrs.PlateCarree(), cmap=cm, vmin=vmin_t, vmax=vmax_t)
    pl.colorbar(pc)
    
    ax = cartopy_subplot(2, 2, 4, 'qt (k/kg)')
    pc=ax.pcolormesh(lon, lat, fqt['qtxy'][t,:,:].transpose()*1000., transform=ccrs.PlateCarree(), cmap=cm, vmin=vmin_q, vmax=vmax_q)
    pl.colorbar(pc)
    
    pl.tight_layout()

    pl.savefig('figs/fig{0:04d}.png'.format(tt))

    pl.close('all')
