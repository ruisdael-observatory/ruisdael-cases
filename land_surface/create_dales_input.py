import matplotlib.pyplot as pl
import netCDF4 as nc4
import xarray as xr
import numpy as np
import sys

from datetime import datetime

# Custom Python scripts/tools/...:
from vegetation_properties import ifs_vegetation, top10_to_ifs
from interpolate import interp_dominant, interp_soil, Interpolate_era5
from spatial_transforms import proj4_rd, proj4_hm
from bofek2012 import BOFEK_info
from lsm_input_dales import LSM_input_DALES
from era5_soil import init_theta_soil, download_era5_soil, calc_theta_rel

# -----------------------------
# Settings
# -----------------------------
# Path to directory with `BOFEK2012_010m.nc` and `top10nl_landuse_010m.nc`
spatial_data_path = '/projects/0/einf170/landuse_soil/'
#spatial_data_path = '/home/scratch1/meteo_data/LASSIE/'

# Output directory for downloaded ERA5 files
era5_path = 'ERA5_soil/'

# Output directory of DALES input files
output_path = 'DALES_input/'

# Start date/time of experiment
start_date = datetime(year=2016, month=8, day=15, hour=6)

# Ruisdael domain:
if True:
    x0 = 910000
    y0 = 940000

    # LES grid:
    itot = 864
    jtot = 576
    dx   = 200
    dy   = 200

    nprocx = 12
    nprocy = 12

# Test domain:
if False:
    x0 = 912500
    y0 = 940000

    # LES grid:
    itot = 64
    jtot = 64
    dx   = 400
    dy   = 400

    nprocx = 2
    nprocy = 2


exp_id = 1

write_binary_output = True    # Write binary input for DALES
write_netcdf_output = True    # Write LSM input as NetCDF output (for e.g. visualisation)

ktot_soil = 4
# -----------------------------
# End settings
# -----------------------------

# Blocksize of interpolations
nblockx = max(1, itot//16 + itot%16 > 0)
nblocky = max(1, jtot//16 + jtot%16 > 0)

# Number of grid points (+/-) used in "dominant" interpolation method
nn_dominant = int(dx/10/2)

xsize = itot*dx
ysize = jtot*dy

# LES grid (in HARMONIE coordinates...)
x_hm = np.arange(x0+dx/2, x0+xsize, dx)
y_hm = np.arange(y0+dy/2, y0+ysize, dy)

# LES/HARMONIE coordinates --> lat/lon --> x/y-RD
x2d_hm, y2d_hm = np.meshgrid(x_hm, y_hm)
lon_hm, lat_hm = proj4_hm(x2d_hm, y2d_hm, inverse=True)
x2d_rd, y2d_rd = proj4_rd(lon_hm, lat_hm, inverse=False)

# Instance of `LSM_input` class, which defines/writes the DALES LSM input:
lsm_input = LSM_input_DALES(itot, jtot, ktot_soil, debug=True)

# Save lat/lon coordinates
lsm_input.lat[:,:] = lat_hm
lsm_input.lon[:,:] = lon_hm

#
# ERA5-soil
#
# Download ERA5 data for initialisation soil
download_era5_soil(start_date, era5_path)

# Read ERA5 soil
e5_soil = xr.open_dataset('{0}/{1:04d}{2:02d}{3:02d}_{4:02d}_soil.nc'.format(
        era5_path, start_date.year, start_date.month, start_date.day, start_date.hour))
e5_soil = e5_soil.reindex(latitude=e5_soil.latitude[::-1])
e5_soil = e5_soil.squeeze()

# Create interpolator for ERA5 -> LES grid
interpolate_era5 = Interpolate_era5(
        lon_hm, lat_hm, e5_soil.longitude.values, e5_soil.latitude.values, itot, jtot)

# Interpolate soil temperature
interpolate_era5.interpolate(lsm_input.t_soil[0,:,:], e5_soil.stl4.values)
interpolate_era5.interpolate(lsm_input.t_soil[1,:,:], e5_soil.stl3.values)
interpolate_era5.interpolate(lsm_input.t_soil[2,:,:], e5_soil.stl2.values)
interpolate_era5.interpolate(lsm_input.t_soil[3,:,:], e5_soil.stl1.values)

# Interpolate SST
# What to do with LES grid points where ERA5's SST has no data? Extrapolate in space?
# For now, use skin temperature where SST's are missing....
sst = e5_soil.sst.values
tsk = e5_soil.skt.values
sst[np.isnan(sst)] = tsk[np.isnan(sst)]
interpolate_era5.interpolate(lsm_input.tskin_aq[:,:], sst)

# Calculate relative soil moisture content ERA5
theta_era = np.stack(
        (e5_soil.swvl4.values, e5_soil.swvl3.values, e5_soil.swvl2.values, e5_soil.swvl1.values))
theta_rel_era = np.zeros_like(theta_era)

# Fix issues arising from the ERA5/MARS interpolation from native IFS grid to regular lat/lon.
# Near the coast, the interpolation between sea grid points (theta == 0) and land (theta >= 0)
# results in too low values for theta. Divide out the land fraction to correct for this.
m = e5_soil.lsm.values > 0
theta_era[:,m] /= e5_soil.lsm.values[m]

soil_index = np.round(e5_soil.slt.values).astype(int)
soil_index -= 1     # Fortran -> Python indexing

# Read van Genuchten lookup table
ds_vg = xr.open_dataset('van_genuchten_parameters.nc')

# Calculate the relative soil moisture content
calc_theta_rel(
        theta_rel_era, theta_era, soil_index,
        ds_vg.theta_wp.values, ds_vg.theta_fc.values,
        e5_soil.dims['longitude'], e5_soil.dims['latitude'], 4)

# Limit relative soil moisture content between 0-1
theta_rel_era[theta_rel_era < 0] = 0
theta_rel_era[theta_rel_era > 1] = 1

# Interpolate relative soil moisture content onto LES grid
theta_rel = np.zeros_like(lsm_input.theta_soil)
for k in range(4):
    interpolate_era5.interpolate(theta_rel[k,:,:], theta_rel_era[k,:,:])

#
# Process spatial data.
#
# 1. Soil (BOFEK2012)
#
bf = BOFEK_info()

ds_soil = xr.open_dataset('{}/BOFEK2012_010m.nc'.format(spatial_data_path))
ds_soil = ds_soil.sel(
        x=slice(x2d_rd.min()-500, x2d_rd.max()+500),
        y=slice(y2d_rd.min()-500, y2d_rd.max()+500))

bf_code, bf_frac = interp_dominant(
        x2d_rd, y2d_rd, ds_soil.bofek_code, valid_codes=bf.soil_id,
        max_code=507, nn=nn_dominant, nblockx=nblockx, nblocky=nblocky, dx=dx)

# Depth of full level soil layers in cm:
z_soil = np.array([194.5, 64, 17.5, 3.5])

# "Interpolate" (NN) BOFEK columns onto LSM grid:
interp_soil(
        lsm_input.index_soil, z_soil, bf_code,
        bf.soil_id_lu, bf.z_mid, bf.n_layers, bf.lookup_index,
        itot, jtot, ktot_soil)

# Set missing values (sea, ...) to ECMWF medium fine type
lsm_input.index_soil[lsm_input.index_soil<=0] = 2

init_theta_soil(
        lsm_input.theta_soil, theta_rel, lsm_input.index_soil,
        ds_vg.theta_wp.values, ds_vg.theta_fc.values, itot, jtot, ktot_soil)

# Python -> Fortran indexing
lsm_input.index_soil += 1

#
# 2. Land use (Top10NL)
#
ds_lu = xr.open_dataset('{}/top10nl_landuse_010m.nc'.format(spatial_data_path))
ds_lu = ds_lu.sel(
        x=slice(x2d_rd.min()-500, x2d_rd.max()+500),
        y=slice(y2d_rd.min()-500, y2d_rd.max()+500))

low_ids_u  = np.array([9,10,11,13]+[29,30])  # with urban
low_ids_nu = np.array([9,10,11,13])          # without urban
high_ids   = np.array([1,2,3,4,5,6,7,8,27])
water_ids  = np.array([14,15,16,17,18])
urban_ids  = np.array([29,30])

lu_low, frac_low = interp_dominant(
        x2d_rd, y2d_rd, ds_lu.land_use, valid_codes=low_ids_u,
        max_code=low_ids_u.max(), nn=nn_dominant, nblockx=nblockx, nblocky=nblocky, dx=dx)

lu_high, frac_high = interp_dominant(
        x2d_rd, y2d_rd, ds_lu.land_use, valid_codes=high_ids,
        max_code=high_ids.max(), nn=nn_dominant, nblockx=nblockx, nblocky=nblocky, dx=dx)

lu_water, frac_water = interp_dominant(
        x2d_rd, y2d_rd, ds_lu.land_use, valid_codes=water_ids,
        max_code=water_ids.max(), nn=nn_dominant, nblockx=nblockx, nblocky=nblocky, dx=dx)

lu_urban, frac_urban = interp_dominant(
        x2d_rd, y2d_rd, ds_lu.land_use, valid_codes=urban_ids,
        max_code=urban_ids.max(), nn=nn_dominant, nblockx=nblockx, nblocky=nblocky, dx=dx)

lu_low_nu, frac_low_nu = interp_dominant(
        x2d_rd, y2d_rd, ds_lu.land_use, valid_codes=low_ids_nu,
        max_code=low_ids_nu.max(), nn=nn_dominant, nblockx=nblockx, nblocky=nblocky, dx=dx)

# Set vegetation fraction over Germany
de_mask = (lu_low==-1)&(lu_high==-1)&(lu_water==-1)
frac_low  [de_mask] = 0.7
frac_high [de_mask] = 0.2
frac_water[de_mask] = 0.0

# Set default values low and high vegetation, where missing
lu_low [lu_low  == -1] = 10   # 10 = grass
lu_high[lu_high == -1] = 3    # 3  = mixed forest

# Init land-surface
lsm_input.c_lv[:,:] = frac_low
lsm_input.c_hv[:,:] = frac_high
lsm_input.c_aq[:,:] = frac_water

#
# Init low vegetation
#
for vt in np.unique(lu_low):
    iv = top10_to_ifs[vt]     # Index in ECMWF lookup table
    mask = (lu_low == vt)

    lsm_input.z0m_lv      [mask] = ifs_vegetation.z0m      [iv]
    lsm_input.z0h_lv      [mask] = ifs_vegetation.z0m      [iv]
    lsm_input.lambda_s_lv [mask] = ifs_vegetation.lambda_s [iv]
    lsm_input.lambda_us_lv[mask] = ifs_vegetation.lambda_us[iv]
    lsm_input.rs_min_lv   [mask] = ifs_vegetation.rs_min   [iv]
    lsm_input.lai_lv      [mask] = ifs_vegetation.lai      [iv]
    lsm_input.ar_lv       [mask] = ifs_vegetation.a_r      [iv]
    lsm_input.br_lv       [mask] = ifs_vegetation.b_r      [iv]

    # Multiply grid point coverage with vegetation type coverage
    lsm_input.c_lv[mask] *= ifs_vegetation.c_veg[iv]

    # Bonus, for offline LSM:
    lsm_input.type_lv[mask] = iv

#
# Init high vegetation
#
for vt in np.unique(lu_high):
    iv = top10_to_ifs[vt]     # Index in ECMWF lookup table
    mask = (lu_high == vt)

    lsm_input.z0m_hv      [mask] = ifs_vegetation.z0m      [iv]
    lsm_input.z0h_hv      [mask] = ifs_vegetation.z0m      [iv]
    lsm_input.lambda_s_hv [mask] = ifs_vegetation.lambda_s [iv]
    lsm_input.lambda_us_hv[mask] = ifs_vegetation.lambda_us[iv]
    lsm_input.rs_min_hv   [mask] = ifs_vegetation.rs_min   [iv]
    lsm_input.lai_hv      [mask] = ifs_vegetation.lai      [iv]
    lsm_input.ar_hv       [mask] = ifs_vegetation.a_r      [iv]
    lsm_input.br_hv       [mask] = ifs_vegetation.b_r      [iv]
    lsm_input.gD          [mask] = ifs_vegetation.gD       [iv]

    # Multiply grid point coverage with vegetation type coverage
    lsm_input.c_hv[mask] *= ifs_vegetation.c_veg[iv]

    # Bonus, for offline LSM:
    lsm_input.type_hv[mask] = iv

#
# Init bare soil
#
iv = 7

lsm_input.z0m_bs      [:,:] = ifs_vegetation.z0m      [iv]
lsm_input.z0h_bs      [:,:] = ifs_vegetation.z0m      [iv]
lsm_input.lambda_s_bs [:,:] = ifs_vegetation.lambda_s [iv]
lsm_input.lambda_us_bs[:,:] = ifs_vegetation.lambda_us[iv]
lsm_input.rs_min_bs   [:,:] = 50.

lsm_input.c_bs[:,:] = 1.-lsm_input.c_lv[:,:]-lsm_input.c_hv[:,:]-lsm_input.c_aq[:,:]

# Bonus, for offline LSM:
lsm_input.type_bs[:,:] = iv

#
# Init water
#
lsm_input.z0m_aq[:,:] = 0.1     # ??
lsm_input.z0h_aq[:,:] = 0.1e-2  # ??

#
# Write output
#
if write_binary_output:
    # Write binary input for DALES
    lsm_input.save_binaries(nprocx=nprocx, nprocy=nprocy, exp_id=exp_id, path=output_path)

if write_netcdf_output:
    # Save NetCDF for visualisation et al.
    lsm_input.save_netcdf('{}/lsm.inp_ruisdael_{}m.nc'.format(output_path, dx))
