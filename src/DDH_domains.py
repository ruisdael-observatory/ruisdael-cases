import numpy as np
import matplotlib.pyplot as pl

import matplotlib.patches as mpatches

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from spatial_tools import *

from matplotlib import rc
rc('text', usetex=True)
rc('font', size=13)
rc('legend', fontsize=11)
rc('text.latex', preamble=r'\usepackage{sansmathfonts}')

pl.close('all')
pl.ion()

def create_namelist(locations, sizes):
    """
    Create Perl code required by harmonie_namelists.pm
    For averaging domains, DDH requires either the four
    corner coordinates, or the NW and SE corners
    """

    i = 1
    for plane, area_size in enumerate(sizes):

        # Write input for harmonie_namelists.pm
        # BE CAREFULL: all output locations are assigned to the same plane (BDEDDH(2,X)=1).
        # If one or more output domains overlap, they need to be given a unique plane.
        # See DDH documentation for details.
        for loc in locations:

            if area_size == -1:
                print('\'BDEDDH(1,{0:})\' => \'4\',         # {1:}, single column' .format(i, loc['name']))
                print('\'BDEDDH(2,{0:})\' => \'{1:}\',         # Plane'            .format(i, plane+1))
                print('\'BDEDDH(3,{0:})\' => \'{1:.2f}\',      # Central longitude'.format(i,loc['lon']))
                print('\'BDEDDH(4,{0:})\' => \'{1:.2f}\',     # Central latitude'  .format(i,loc['lat']))

            else:
                diff_lon = dlon(area_size/2, loc['lat'])
                diff_lat = dlat(area_size/2)

                print('\'BDEDDH(1,{0:})\' => \'3\',         # {1:}, {2:.0f}x{2:.0f}km'.format(i, loc['name'], area_size/1000.))
                print('\'BDEDDH(2,{0:})\' => \'{1:}\',         # Plane'               .format(i, plane+1))
                print('\'BDEDDH(3,{0:})\' => \'{1:.2f}\',      # West longitude'      .format(i,loc['lon']-diff_lon))
                print('\'BDEDDH(4,{0:})\' => \'{1:.2f}\',     # North latitude'       .format(i,loc['lat']+diff_lat))
                print('\'BDEDDH(5,{0:})\' => \'{1:.2f}\',      # East longitude'      .format(i,loc['lon']+diff_lon))
                print('\'BDEDDH(6,{0:})\' => \'{1:.2f}\',     # South latitude'       .format(i,loc['lat']-diff_lat))

            i += 1


def get_DOWA_domains():
    # Pretty much hardcoded for the specific DOWA setup...................
    domains = [dict(name='FINO1',        lat=54.01, lon=6.59, c=pl.cm.Paired( 0), a=0.9, ls='solid'),
               dict(name='Goeree',       lat=51.93, lon=3.67, c=pl.cm.Paired( 1), a=0.9, ls='solid'),
               dict(name='Europlatform', lat=52.00, lon=3.27, c=pl.cm.Paired( 2), a=0.9, ls='solid'),
               dict(name='K13',          lat=53.22, lon=3.22, c=pl.cm.Paired( 3), a=0.9, ls='solid'),
               dict(name='HKZ',          lat=52.30, lon=4.10, c=pl.cm.Paired( 4), a=0.9, ls='solid'),
               dict(name='P11B',         lat=52.36, lon=3.34, c=pl.cm.Paired( 5), a=0.9, ls='solid'),
               dict(name='F3-FB-1',      lat=54.85, lon=4.70, c=pl.cm.Paired( 6), a=0.9, ls='solid'),
               dict(name='Cabauw',       lat=51.97, lon=4.90, c=pl.cm.Paired( 7), a=0.9, ls='dotted'),
               dict(name='Loobos',       lat=52.17, lon=5.74, c=pl.cm.Paired( 8), a=0.9, ls='dotted'),
               dict(name='Lutjewad',     lat=53.40, lon=6.35, c=pl.cm.Paired( 9), a=0.9, ls='dotted'),
               dict(name='Schiphol',     lat=52.31, lon=4.76, c=pl.cm.Paired(10), a=0.9, ls='dotted'),
               dict(name='Rotterdam',    lat=51.91, lon=4.47, c=pl.cm.Paired(11), a=0.9, ls='dotted')]

    sizes = [-1, 10000, 30000]

    return domains, sizes


def get_domain_info(domains, sizes):

    out = dict(name=[], cent_lat=[], cent_lon=[],\
               east_lon=[], west_lon=[], north_lat=[], south_lat=[])

    for plane, area_size in enumerate(sizes):
        for loc in domains:
            if area_size == -1:
                # Single column
                out['name'     ].append( '{0:}_column'.format(loc['name']) )
                out['cent_lat' ].append( loc['lat'] )
                out['cent_lon' ].append( loc['lon'] )
                out['east_lon' ].append( None )
                out['west_lon' ].append( None )
                out['north_lat'].append( None )
                out['south_lat'].append( None )
            else:
                diff_lon = dlon(area_size/2, loc['lat'])
                diff_lat = dlat(area_size/2)

                out['name'     ].append( '{0:}_{1:.0f}km'.format(loc['name'], area_size/1000.) )
                out['cent_lat' ].append( loc['lat'] )
                out['cent_lon' ].append( loc['lon'] )
                out['east_lon' ].append( loc['lon']+diff_lon )
                out['west_lon' ].append( loc['lon']-diff_lon )
                out['north_lat'].append( loc['lat']+diff_lat )
                out['south_lat'].append( loc['lat']-diff_lat )

    return out


def plot_locations(locations, size):

    pl.figure()

    # Map projection
    proj = ccrs.LambertConformal(central_longitude=4.9, central_latitude=51.967)

    # Create single axes object in Lambert projection
    ax=pl.axes(projection=proj)

    # Draw coast lines, resolution = {'10m','50m','110m'}
    ax.coastlines(resolution='10m', linewidth=0.8, color='black')

    # Load country geometries and lakes (for the IJselmeer)
    # from www.naturalearthdata.com and add to axes
    countries = cfeature.NaturalEarthFeature(
            category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
    ax.add_feature(countries, edgecolor='black', linewidth=0.8)

    lakes = cfeature.NaturalEarthFeature(
            category='physical', name='lakes', scale='10m', facecolor='none')
    ax.add_feature(lakes, edgecolor='black', linewidth=0.8)

    # Set spatial extent of map, and add grid lines
    ax.set_extent([1, 7.5, 50.5, 55.2], ccrs.PlateCarree())
    #ax.gridlines()

    # Plot some random data
    for i,loc in enumerate(locations):
        diff_lon = np.abs(dlon(size/2, loc['lat']))
        diff_lat = np.abs(dlat(size/2))

        #label='{} ({}N, {}E)'.format(loc['name'], loc['lat'], loc['lon'])
        label='{}'.format(loc['name'])

        ax.add_patch(mpatches.Rectangle(xy=[loc['lon']-diff_lon, loc['lat']-diff_lat],
                                        width=2*diff_lon, height=2*diff_lat, ls=loc['ls'],
                                        edgecolor=loc['c'], facecolor=loc['c'], alpha=loc['a'],
                                        transform=ccrs.PlateCarree(), label=label))
    pl.legend(fontsize=9, loc='upper left')

    # Remove frame around map
    ax.outline_patch.set_visible(False)
    pl.tight_layout()




if __name__ == '__main__':

    domains, sizes = get_DOWA_domains()

    if (True):
        # Get info for NetCDF files
        info = get_domain_info(domains, sizes)

    if (False):
        # Create Harmonie namelist
        create_namelist(domains, sizes)

    if (True):
        # Plot domains on map
        plot_locations(domains, 30000)
