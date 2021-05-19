from numba import jit
import numpy as np
import cdsapi
import os

def download_era5_soil(date, path):
    """
    Download ERA5 soil fields from CDS
    """
    out_file = '{0}/{1:04d}{2:02d}{3:02d}_{4:02d}_soil.nc'.format(
            path, date.year, date.month, date.day, date.hour)

    if os.path.exists(out_file):
        print('Found {} local!'.format(out_file))
    else:
        cds_dict = {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'sea_surface_temperature', 'soil_temperature_level_1',
                    'soil_temperature_level_2', 'soil_temperature_level_3',
                    'soil_temperature_level_4', 'soil_type', 'skin_temperature',
                    'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
                    'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
                    'land_sea_mask'
                ],
                'year': '{0:04d}'.format(date.year),
                'month': '{0:02d}'.format(date.month),
                'day': '{0:02d}'.format(date.day),
                'time': '{0:02d}:{1:02d}'.format(date.hour, date.minute),
                'area': [53.47, 2.92, 50.47, 6.92],
            }

        c = cdsapi.Client()
        c.retrieve('reanalysis-era5-single-levels', cds_dict, out_file)


@jit(nopython=True, nogil=True, fastmath=True)
def calc_theta_rel(theta_rel, theta, soil_index, theta_wp, theta_fc, itot, jtot, ktot):
    """
    Calculate relative soil moisture content.
    NOTE: this contains some ERA5 specific fixes/hacks...
    """

    for k in range(ktot):
        for j in range(jtot):
            for i in range(itot):
                si = soil_index[j,i]
                if si != -1:
                    theta_rel[k,j,i] = (theta[k,j,i] - theta_wp[si]) / (theta_fc[si] - theta_wp[si])
                else:
                    theta_rel[k,j,i] = -1

    # Fix cases where ERA5 has no soil type (...), by calculating
    # the relative soil moisture content as the mean of surrounding grid points.
    for k in range(ktot):
        for j in range(jtot):
            for i in range(itot):
                si = soil_index[j,i]
                if si == -1:
                    s = 0.
                    n = 0
                    for di in range(-1,2):
                        for dj in range(-1,2):
                            ii = i+di
                            jj = j+dj

                            if ii >= 0 and ii < itot and jj >= 0 and jj < jtot and soil_index[jj,ii] != -1:
                                s += theta_rel[k,jj,ii]
                                n += 1
                    if n > 0:
                        theta_rel[k,j,i] = s / n


@jit(nopython=True, nogil=True, fastmath=True)
def init_theta_soil(theta, theta_rel, soil_index, theta_wp, theta_fc, itot, jtot, ktot):
    """
    Initialise soil moisture, scaling the relative soil moisture content
    to an absolute one, given the field capacity and wilting point of
    each soil type.
    """

    for k in range(ktot):
        for j in range(jtot):
            for i in range(itot):
                si = soil_index[k,j,i]
                theta[k,j,i] = theta_wp[si] + theta_rel[k,j,i] * (theta_fc[si]-theta_wp[si])


def calc_root_fraction(a_r, b_r, zh):
    """
    Calculate root fraction using the `a_r` and `b_r` coefficients from IFS
    """

    root_frac = np.zeros(zh.size-1)
    for k in range(1, zh.size-1):
        root_frac[k] = 0.5 * (np.exp(a_r * zh[k+1]) + \
                              np.exp(b_r * zh[k+1]) - \
                              np.exp(a_r * zh[k  ]) - \
                              np.exp(b_r * zh[k  ]))

    root_frac[0] = 1-root_frac.sum()

    return root_frac


if __name__ == '__main__':
    from datetime import datetime

    date = datetime(2019,1,1,0)
    download_era5_soil(date, '.')
