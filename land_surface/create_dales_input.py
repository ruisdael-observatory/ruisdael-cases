import matplotlib.pyplot as plt
#import netCDF4 as nc4
import xarray as xr
import numpy as np
#import sys
# import os
from pathlib import Path
from datetime import datetime

# Custom Python scripts/tools/...:
from vegetation_properties import ifs_vegetation, top10_to_ifs, top10_names
from interpolate import interp_dominant, interp_soil, Interpolate_era5
from spatial_transforms import proj4_rd #, proj4_hm
from bofek2012 import BOFEK_info
from lsm_input_dales import LSM_input_DALES
from era5_soil import init_theta_soil, calc_theta_rel #, download_era5_soil
from domains import domains
# from landuse_types import lu_types_basic, lu_types_build, lu_types_crop, lu_types_depac
from landuse_types import lu_types_depac

# Correction factor for aspect ratio of plots
ASPECT_CORR = 2


def init_dales_grid(domain, ktot_soil, lutypes, parnames):
    """
    Initialise a land surface grid with the dimensions of the DALES grid
    
    Parameters
    ----------
    domain : dict
        disctionary with domain size and resolution
    ktot_soil : int
        number of soil levels
    lutypes : dict
        disctionary with land use types
    parnames : list
        List of parameters to process

    Returns
    -------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    nn_dominant : int
        Number of grid points (+/-) used in "dominant" interpolation method.
    nblockx : int
        Number of blocks in x-direction.
    nblocky : int
        Number of blocks in y-direction.

    """ 
    x0 = domain['x0']
    y0 = domain['y0']
    dx = domain['dx']
    dy = domain['dy']
    itot = domain['itot']
    jtot = domain['jtot']

    # Blocksize of interpolations
    nblockx = max(1, itot//16 + itot%16 > 0)
    nblocky = max(1, jtot//16 + jtot%16 > 0)
    
    # Number of grid points (+/-) used in "dominant" interpolation method
    nn_dominant = int(dx/10/2)
    
    xsize = itot*dx
    ysize = jtot*dy
    
    # LES grid in RD coordinates
    x_rd = np.arange(x0+dx/2, x0+xsize, dx)
    y_rd = np.arange(y0+dy/2, y0+ysize, dy)
    x2d_rd, y2d_rd = np.meshgrid(x_rd, y_rd)
    lon2d, lat2d = proj4_rd(x2d_rd, y2d_rd, inverse=True)
    
    # Instance of `LSM_input` class, which defines/writes the DALES LSM input:
    lsm_input = LSM_input_DALES(itot, jtot, ktot_soil, lutypes, parnames, debug=False)
    
    # Save lat/lon coordinates
    lsm_input.lat[:,:] = lat2d
    lsm_input.lon[:,:] = lon2d

    lsm_input.x[:] = x_rd
    lsm_input.y[:] = y_rd
    
    return lsm_input, nn_dominant, nblockx, nblocky


def get_era5_data(andir, fcdir):
    """
    Get ERA5 soil and sea surface properties & variables:
    soil temperature
    skin temperature
    soil type
    soil water content
    sea surface temperature
    land sea mask
    
    This data is stored on disk in the 'leipdir'.
    Additionally, a function is available for downloading missing data
    (not tested yet)

    Parameters
    ----------
    andir : str
        Dir with ECMWF analysis data.
    fcdir : str
        Dir with ECMWF forecast data.

    Returns
    -------
    era5_stl : dict
        soil temperature: 4 layers.
    era5_swvl : dict
        volumetric soil water: 4 layers.
    era5_lsm : Xarray.Dataset
        land sea mask.
    era5_slt : Xarray.Dataset
        soil type.
    era5_skt : Xarray.Dataset
        skin temperature.
    era5_sst: Xarray.Dataset
        sea surface temperature.

    """
    
    # Download ERA5 data for initialisation soil
    #TODO: test
    #download_era5_soil(start_date, andir)
    
    #get variables:
    #      'sea_surface_temperature', 'soil_temperature_level_1',
    #      'soil_temperature_level_2', 'soil_temperature_level_3',
    #      'soil_temperature_level_4', 'soil_type', 'skin_temperature',
    #      'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
    #      'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
    #      'land_sea_mask'    
    
    
    #sst: sea surface temperature
    era5_sst = xr.open_dataset('%s/%04d/sstk_%04d%02d%02d_1h.nc' %(fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    #skt: skin temperature
    era5_skt = xr.open_dataset('%s/%04d/skt_%04d%02d%02d_1h.nc' %(fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    #swfl1-4: soil water volumetric level
    era5_swvl1 = xr.open_dataset('%s/%04d/swvl1_%04d%02d%02d_1h.nc' %(fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    era5_swvl2 = xr.open_dataset('%s/%04d/swvl2_%04d%02d%02d_1h.nc' %(fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    era5_swvl3 = xr.open_dataset('%s/%04d/swvl3_%04d%02d%02d_1h.nc' %(fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    era5_swvl4 = xr.open_dataset('%s/%04d/swvl4_%04d%02d%02d_1h.nc' %(fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    
    #stl1-4: soil temperature level
    era5_stl1 = xr.open_dataset('%s/%04d/stl1_%04d%02d%02d_1h.nc' %(fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    era5_stl2 = xr.open_dataset('%s/%04d/stl2_%04d%02d%02d_1h.nc' %(fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    era5_stl3 = xr.open_dataset('%s/%04d/stl3_%04d%02d%02d_1h.nc' %(fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    era5_stl4 = xr.open_dataset('%s/%04d/stl4_%04d%02d%02d_1h.nc' %(fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    
    
    #lsm: land sea mask
    era5_lsm = xr.open_dataset('%s/lsm.nc' % andir)
    
    #slt: soil type
    era5_slt = xr.open_dataset('%s/slt.nc' % andir)

    era5_stl  = {'era5_stl1': era5_stl1,
                 'era5_stl2': era5_stl2,
                 'era5_stl3': era5_stl3,
                 'era5_stl4': era5_stl4   } 
    era5_swvl = {'era5_swvl1': era5_swvl1,
                 'era5_swvl2': era5_swvl2,
                 'era5_swvl3': era5_swvl3,
                 'era5_swvl4': era5_swvl4   }
         
    return era5_stl, era5_swvl, era5_lsm, era5_slt, era5_skt, era5_sst


def create_interpolator(lsm_input, e5_soil):
    """
    Create interpolator for ERA5 -> LES grid

    Parameters
    ----------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    e5_soil : xarray.Dataset 
        DESCRIPTION.

    Returns
    -------
    interpolate_era5: interpolate.Interpolate_era5 object

    """
    interpolate_era5 = Interpolate_era5(lsm_input.lon, 
                                        lsm_input.lat, 
                                        e5_soil.longitude.values, 
                                        e5_soil.latitude.values, 
                                        lsm_input.itot, 
                                        lsm_input.jtot)

    return interpolate_era5


def process_era5_soiltemp(lsm_input, era5_stl, era5_swvl, era5_sst, era5_skt, era5_lsm, era5_slt):
    """
    Interpolate ERA5 variables to DALES grid
    
    Parameters
    ----------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    era5_stl : dict
        ERA5 soil temperature per layer.
    era5_swvl : dict
        ERA5 volumetric_soil_water_layer per layer.
    era5_sst : xarray.Dataset
        ERA5 sea surface temperature.
    era5_skt : xarray.Dataset
        ERA5 skin temperature.
    era5_lsm : xarray.Dataset
        ERA5 land sea mask.
    era5_slt : xarray.Dataset
        ERA5 soil type.

    Returns
    -------
    lsm_input: LSM_input_DALES class
        Class containing Dales input parameters for all LU types.
    e5_soil: xarray.Dataset 
        ERA5 soil properties

    """
    era5_stl1 = era5_stl['era5_stl1']
    e5_soil  = xr.Dataset(coords=era5_stl1.coords)
    e5_soil  = e5_soil.drop('time')
    e5_soil['stl1']  = era5_stl['era5_stl1'].isel(time=3).drop('time')['stl1']
    e5_soil['stl2']  = era5_stl['era5_stl2'].isel(time=3).drop('time')['stl2']
    e5_soil['stl3']  = era5_stl['era5_stl3'].isel(time=3).drop('time')['stl3']
    e5_soil['stl4']  = era5_stl['era5_stl4'].isel(time=3).drop('time')['stl4']
    e5_soil['swvl1'] = era5_swvl['era5_swvl1'].isel(time=3).drop('time')['swvl1']
    e5_soil['swvl2'] = era5_swvl['era5_swvl2'].isel(time=3).drop('time')['swvl2']
    e5_soil['swvl3'] = era5_swvl['era5_swvl3'].isel(time=3).drop('time')['swvl3']
    e5_soil['swvl4'] = era5_swvl['era5_swvl4'].isel(time=3).drop('time')['swvl4']
    e5_soil['sst']   = era5_sst['sst'].isel(time=3).drop('time')
    e5_soil['skt']   = era5_skt['skt'].isel(time=3).drop('time')
    
    e5_soil['lsm']   = era5_lsm['lsm'].interp_like(e5_soil, method='nearest').fillna(0)
    e5_soil['slt']   = era5_slt['slt'].interp_like(e5_soil, method='nearest').fillna(0)
    
    # Read ERA5 soil
    e5_soil = e5_soil.reindex(latitude=e5_soil.latitude[::-1])
    e5_soil = e5_soil.squeeze()
    
    interpolate_era5 = create_interpolator(lsm_input, e5_soil)
    
    # Interpolate soil temperature
    interpolate_era5.interpolate(lsm_input.t_soil[0,:,:], e5_soil.stl4.values)
    interpolate_era5.interpolate(lsm_input.t_soil[1,:,:], e5_soil.stl3.values)
    interpolate_era5.interpolate(lsm_input.t_soil[2,:,:], e5_soil.stl2.values)
    interpolate_era5.interpolate(lsm_input.t_soil[3,:,:], e5_soil.stl1.values)

    return lsm_input, e5_soil 


def process_era5_soilmoist(lsm_input, e5_soil):
    """
    Interpolate ERA5 soil moisture to DALES grid

    Parameters
    ----------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    e5_soil : xarray.Dataset 
        ERA5 soil properties

    Returns
    -------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    e5_soil : TYPE
        DESCRIPTION.
    theta_rel : numpy.array
        Relative soil moisture content per layer.
    ds_vg : xarray.Dataset
        Van Genuchten parameters.

    """
    
    interpolate_era5 = create_interpolator(lsm_input, e5_soil)
    # Interpolate SST
    # TODO: What to do with LES grid points where ERA5's SST has no data? Extrapolate in space?
    # For now, use skin temperature where SST's are missing....
    # sst = e5_soil.sst.values
    # tsk = e5_soil.skt.values
    # sst[np.isnan(sst)] = tsk[np.isnan(sst)]
    # interpolate_era5.interpolate(lsm_input.tskin_aq[:,:], sst)
    
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

    return lsm_input, e5_soil, theta_rel, ds_vg


def process_soil_map(soilfile, lsm_input, nn_dominant, nblockx, nblocky, domain, theta_rel, ds_vg):
    """
    Interpolate BOFEK2012 soil map to DALES grid
    Fill missing values to ECMWF soil type

    Parameters
    ----------
    soilfile : str
        File name of the soil map
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    nn_dominant : int
        Number of grid points (+/-) used in "dominant" interpolation method.
    nblockx : int
        Number of blocks in x-direction.
    nblocky : int
        Number of blocks in y-direction.
    domain : dict
        Dales domain settings.
    theta_rel : numpy array
        Relative soil moisture content per layer.
    ds_vg : xarray.Dataset
        Van Genuchten parameters.

    Returns
    -------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    """
    #
    # Process spatial data.
    #
    # 1. Soil (BOFEK2012)
    #
    bf = BOFEK_info()
    
    x2d_rd, y2d_rd = np.meshgrid(lsm_input.x[:], lsm_input.y[:])

    ds_soil = xr.open_dataset('{}/{}'.format(spatial_data_path, soilfile))
    ds_soil = ds_soil.sel(
            x=slice(x2d_rd.min()-500, x2d_rd.max()+500),
            y=slice(y2d_rd.min()-500, y2d_rd.max()+500))
    
    bf_code, bf_frac = interp_dominant(
            x2d_rd, y2d_rd, ds_soil.bofek_code, valid_codes=bf.soil_id,
            max_code=507, nn=nn_dominant, nblockx=nblockx, nblocky=nblocky, dx=domain['dx'])
    
    # Depth of full level soil layers in cm:
    z_soil = np.array([194.5, 64, 17.5, 3.5])
    
    # "Interpolate" (NN) BOFEK columns onto LSM grid:
    interp_soil(
            lsm_input.index_soil, z_soil, bf_code,
            bf.soil_id_lu, bf.z_mid, bf.n_layers, bf.lookup_index,
            lsm_input.itot, lsm_input.jtot, lsm_input.ktot)
    
    # Set missing values (sea, ...) to ECMWF medium fine type
    lsm_input.index_soil[lsm_input.index_soil<=0] = 2
    
    init_theta_soil(lsm_input.theta_soil, theta_rel, lsm_input.index_soil,
                    ds_vg.theta_wp.values, ds_vg.theta_fc.values, 
                    lsm_input.itot, lsm_input.jtot, lsm_input.ktot)
    
    # Python -> Fortran indexing
    lsm_input.index_soil += 1

    return lsm_input 


def process_top10NL_map(lufile, lutypes, lsm_input, nn_dominant, nblockx, nblocky, domain):
    """
    Interpolate TOP10NL land use map to DALES grid
    Find dominant LU id per LU type

    Parameters
    ----------
    lufile : str
        Filename of the land use map
    lutypes : dict
        properties of each land use type.
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    nn_dominant : int
        Number of grid points (+/-) used in "dominant" interpolation method.
    nblockx : int
        Number of blocks in x-direction.
    nblocky : int
        Number of blocks in y-direction.
    domain : dict
        Dales domain settings.

    Returns
    -------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    lutypes : dict
        properties of each land use type.
    """ 
    #TODO: generalise for arbitrary number of LU types

    #
    # 2. Land use (Top10NL)
    #

    x2d_rd, y2d_rd = np.meshgrid(lsm_input.x[:], lsm_input.y[:])

    ds_lu = xr.open_dataset('{}/{}'.format(spatial_data_path, lufile))
    ds_lu = ds_lu.sel(
            x=slice(x2d_rd.min()-500, x2d_rd.max()+500),
            y=slice(y2d_rd.min()-500, y2d_rd.max()+500))

    luname  = [x[1]['lu_long'] for x in lu_types.items()] 
    lushort = [x[1]['lu_short'] for x in lu_types.items()] 
    lveg    = [x[1]['lveg'] for x in lu_types.items()] 
    laqu    = [x[1]['laqu'] for x in lu_types.items()] 
    ilu     = np.arange(len(lu_types)) + 1
    
    setattr(lsm_input, 'luname', luname)
    setattr(lsm_input, 'lushort', lushort)
    setattr(lsm_input, 'lveg', lveg)
    setattr(lsm_input, 'laqu', laqu)
    setattr(lsm_input, 'ilu', ilu)

    # set LU cover for each grid cell    
    for lu in lutypes:
        lutypes[lu]['lu_domid'], lu_types[lu]['lu_frac'] = interp_dominant(
                x2d_rd, y2d_rd, 
                ds_lu.land_use, 
                valid_codes=lutypes[lu]['lu_ids'],
                max_code=lutypes[lu]['lu_ids'].max(), 
                nn=nn_dominant, 
                nblockx=nblockx, nblocky=nblocky, 
                dx=domain['dx'])
        if -1 in np.unique(lutypes[lu]['lu_domid']):
#            lutypes[lu]['lu_domid'][lutypes[lu]['lu_domid']==-1] = np.unique(lutypes[lu]['lu_domid'])[1]
            lutypes[lu]['lu_domid'][lutypes[lu]['lu_domid']==-1] = np.unique(lutypes[lu]['lu_domid'])[0]
            print('filling domid for', lu)
            #TODO: smarter way to fill missing values
        setattr(lsm_input, 'c_'+lu, lu_types[lu]['lu_frac']) 
    
    # #set dominant LU id for each LU type
    # for lu in lutypes.keys():
    #     domid = lutypes[lu]['lu_domid']
        
#     lu_low     = lutypes['lv']['lu_domid']
#     lu_high    = lutypes['hv']['lu_domid']
#     lu_water   = lutypes['aq']['lu_domid']
#     lu_asphalt = lutypes['ap']['lu_domid']
#     lu_baresoil= lutypes['bs']['lu_domid']
#     try:
#         lu_build   = lutypes['bu']['lu_domid']
#     except:
#         print('')
    
# #    # Set vegetation fraction over Germany
# # TODO apply to lsm_input 
# #    de_mask = (lu_low==-1)&(lu_high==-1)&(lu_water==-1)&(lu_asphalt==-1)
# #    frac_low  [de_mask] = 0.7
# #    frac_high [de_mask] = 0.2
# #    frac_water[de_mask] = 0.0
# #    frac_asphalt[de_mask] = 0.0

#     # Set default values low and high vegetation, where missing
#     lu_low [lu_low  == -1] = 10   # 10 = grass
#     lu_high[lu_high == -1] = 3    # 3  = mixed forest
#     lu_asphalt[lu_asphalt == -1] = 20    # 20 = paved road
#     lu_baresoil[lu_baresoil == -1] = 28  # 28 = fallow land
#     lu_water[lu_water == -1] = 14        # 14 = water way 
#     try:
#         lu_build[lu_build == -1] = 29    # 29 = buildings
#     except:
#         print('')

#     lutypes['lv']['lu_domid'] = lu_low
#     lutypes['hv']['lu_domid'] = lu_high
#     lutypes['aq']['lu_domid'] = lu_water
#     lutypes['ap']['lu_domid'] = lu_asphalt
#     lutypes['bs']['lu_domid'] = lu_baresoil    
#     try:
#         lutypes['bu']['lu_domid'] = lu_build    
#     except:
#         print('')

    return lsm_input, lutypes


def init_lutypes_ifs(lsm_input, lu_dict, parnames_lsm ):
    """Assign surface properties to DALES land use types based on ECMWF
       lookup table.    

    Parameters
    ----------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    lu_dict : dict
        LU type properties.
    parnames_lsm : list
        List of land use parameters to process

    Returns
    -------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.

    """
    #TODO: generalise for arbitrary number of LU types
    # all parameters should be available for all LU types
    # selection can be made in Dales / with an ugly hack in a separate function    
    
    
    #
    # Init land use
    #
    shape = (lsm_input.jtot, lsm_input.itot)
    for lu in lu_dict.keys():
        print('\n processing', lu_dict[lu]['lu_long'])
        for parname in parnames_lsm:
            if parname == 'cover' or parname == 'c_veg':
                parfield = lu_dict[lu]['lu_frac'].copy()
            else:
                # parfield = np.full(shape, np.nan)
                parfield = np.full(shape, 0.0)
    
            for vt in lu_dict[lu]['lu_ids']:
                iv = top10_to_ifs[vt]     # Index in ECMWF lookup table
                mask = (lu_dict[lu]['lu_domid'] == vt)
              
                if parname == 'cover':
                    print('LU type TOP10:', top10_names[vt])
                    parfield[mask] *= 1
                elif parname == 'c_veg':
                    # TODO Note that cveg < cover; assign cover-cveg to bare soil
                    parfield[mask] *= ifs_vegetation.c_veg[iv]
                elif parname == 'lutype':
                    # parfield[mask] = iv
                    parfield[:] = iv  # LG: Only apply mask to cover and c_veg (DALES crashes when zeros or nans are in
                                      # the array)
                elif parname == 'tskin':
                    # TODO: assign tskin only for water surfaces
                    # parfield[mask] = 273.15
                    parfield[:] = 273.15  # LG: Only apply mask to cover and c_veg (DALES crashes when zeros or nans
                                          # are in the array)
                else:
                    if parname =='ar':
                        parname_ifs = 'a_r'
                    elif parname =='br':
                        parname_ifs = 'b_r'
                    else:
                        parname_ifs = parname
                    # parfield[mask] = getattr(ifs_vegetation, parname_ifs) [iv]
                    parfield[:] = getattr(ifs_vegetation, parname_ifs) [iv]  # LG: Only apply mask to cover and c_veg

                # Multiply grid point coverage with vegetation type coverage
                #if lu == 'bs': continue
                #cover[mask] *= ifs_vegetation.c_veg[iv]
            
            setattr(lsm_input, '_'.join([parname, lu]), parfield)

    totcover = calc_totcover(lsm_input, lu_types, 'cover')
    setattr(lsm_input, 'cover_tot', totcover)   

    totcveg = calc_totcover(lsm_input, lu_types, 'c_veg')
    setattr(lsm_input, 'c_veg_tot', totcveg)   
    
    # TODO: more consistent way to check for LU type with bare soil
    bs_name = [k for k in lu_types.keys() if 'bar' in lu_types[k]['lu_long'].lower()][0] 
    lsm_input = fill_bare_soil(lsm_input, bs_name=bs_name)    
    
    #recalculate
    totcover = calc_totcover(lsm_input, lu_types, 'cover')
    setattr(lsm_input, 'cover_tot', totcover) 
    
    totcveg = calc_totcover(lsm_input, lu_types, 'c_veg')
    setattr(lsm_input, 'c_veg_tot', totcveg) 
    
    return lsm_input 


def calc_totcover(lsm_input, lu_types, ctype):
    """
    Calculate sum over cover of individual LU types to check if it sums up to 1

    Parameters
    ----------
    lsm_input : LSM_input_DALES
        Class containing Dales input parameters for all LU types.
    lu_types : dict
        LU type properties.
    ctype : str
        LU cover type to be summed.

    Returns
    -------
    totcover : np.array
        Total LU cover.

    """
    covers = [ctype + '_' + s for s in lu_types.keys()]
    totcover = np.zeros([lsm_input.jtot,lsm_input.itot])
    for c in covers:
        totcover+=getattr(lsm_input, c)
        
    return totcover


def fill_bare_soil(lsm_input, bs_name):
    # assign remaining land cover to bare soil
    cover     = getattr(lsm_input, 'cover_tot')
    cover_bs0 = getattr(lsm_input, 'cover_'+bs_name)
    # cover_bs1 = 1.-cover
    
    # cveg      = getattr(lsm_input, 'c_veg_tot')
    # cveg_bs0  = getattr(lsm_input, 'c_veg_'+bs_name)
    # cover_bs = np.round(1 - cveg + cover_bs0, 6)
    cover_bs = np.round(1 - cover + cover_bs0, 6)
    
    setattr(lsm_input, 'cover_' + bs_name, cover_bs)

    return lsm_input
    

def init_lutypes_dep(lsm_input, lu_dict, parnames_dep, depfile ):
    """Assign deposition parameter properties to DALES land use types.    

    Parameters
    ----------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    lu_dict : dict
        LU type properties.
    parnames_dep : list
        Names of deposition parameters.
    depfile : str
        Name of file with deposition parameters per LU type.

    Returns
    -------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.

    """
    # Select temperate climate zone and default vegetation type for now
    climatezone = 'tmp'
    vegetation  = 'def'

    ds_dep = xr.open_dataset('{}/{}'.format(spatial_data_path, depfile))
    ds_dep = ds_dep.sel(climatezone=climatezone)

    shape = (lsm_input.jtot, lsm_input.itot)
    for lu in lu_dict.keys():
        print(' processing', lu_dict[lu]['lu_long'])
        for parname in parnames_dep:
            # dummy value to whole field
            # parfield = np.full(shape, np.nan)
            parfield = np.full(shape, 0.0)
            # print(parname)

            for vt in lu_dict[lu]['lu_ids']:
                # mask = (lu_dict[lu]['lu_domid'] == vt)
                # TODO: get deposition parameters for each LU class
                value = np.nan
                if len(ds_dep[parname].dims) == 1:
                    value = ds_dep[parname].sel(landuse_vegetation='_'.join([lu,vegetation])).values
                    if np.isnan(value):
                        value = 0.0
                    # print(value)
                elif len(ds_dep[parname].dims) == 2:
                    print('warning: parameter dependent on species')
                    value = 0.0
                # parfield[mask] = value
                parfield[:] = value  # LG: Apply values on the complete field, to prevent DALES from crashing on
                                     # divide by zero (the fill value) or nan (which does not exist in Fortran)
            setattr(lsm_input, '_'.join([parname, lu]), parfield)

    return lsm_input 


def write_output(lsm_input, 
                 write_binary_output=False, write_netcdf_output=True,
                 nprocx=4, nprocy=4):
    """
    Write output
    Write binary input for DALES
    Save NetCDF for visualisation et al.

    Parameters
    ----------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    write_binary_output : bool, optional
        Write binary output. The default is False.
    write_netcdf_output : bool, optional
        Write netCDF output. The default is True.
    nprocx : int, optional
        Number of processors in x-direction. Default 4/
    nprocy : int, optional
        Number of processors in y-direction. Default 4/

    Returns
    -------
    None.

    """

    if write_binary_output:
        lsm_input.save_binaries(nprocx=nprocx, nprocy=nprocy, exp_id=exp_id, path=output_path)
    
    if write_netcdf_output:
        lsm_input.save_netcdf('%s\lsm.inp_%03d.nc' %(output_path, exp_id))

    return


def some_plots(lsm_input, plotvars):
    """
    Generate some standard plots of Land Surface Model input data

    Parameters
    ----------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    plotvars : list
        List of variables to plot.

    Returns
    -------
    None.

    """   
     
    data_vars = dict()
    
    for plotvar in plotvars:
        data = getattr(lsm_input, plotvar)
        data_vars[plotvar] = (['y','x'], data)
        
    coords = dict(x=(['x'], lsm_input.x),
                  y=(['y'], lsm_input.y))
    ds_lsm = xr.Dataset(data_vars=data_vars,
                        coords=coords 
                       ) 

    for plotvar in list(ds_lsm.variables):
        if plotvar == 'x' or plotvar == 'y': continue
        fig, ax = plt.subplots(1)
        ax.set_aspect(abs((lsm_input.y[-1] - lsm_input.y[0])/(lsm_input.x[-1] - lsm_input.x[0])) * ASPECT_CORR)
        ds_lsm[plotvar].plot(ax=ax, cmap='Reds', vmin=0, vmax=None)
        plt.tight_layout()
        
    plt.show()
    
    return 


def process_input(lu_types, parnames, domain, output_path, andir, fcdir, start_date, exp_id, ktot_soil, lwrite, lplot):
    """Function that connects all processing steps:
    Init DALES grid
    Get ERA5 data
    Process soil temperature
    Process soil moisture
    Process soil map
    Process land use map
    Set land use properties
    Write output files in netCDF and binary (to be phased out) format
    Make some standard plots (optional)

    Parameters
    ----------
    lu_types : dict
        LU type properties.
    parnames : list
        List of parameter names to process.
    domain : dict
        Dales domain settings.
    output_path : str
        Dir to write output to.
    andir : str
        Dir with ECMWF analysis data.
    fcdir : str
        Dir with ECMWF forecast data.        
    start_date : datetime.datetime
        Time stamp of Dales run start.
    exp_id : int
        Experiment ID.
    ktot_soil : int
        Number of soil layers.
    lwrite : bool
        Flag to write the output.
    lplot : bool
        Flag to plot the output.

    Returns
    -------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.

    """
    lsm_input, nn_dominant, nblockx, nblocky = init_dales_grid(domain, ktot_soil, lu_types, parnames)

    era5_stl, era5_swvl, era5_lsm, era5_slt, era5_skt, era5_sst = get_era5_data(andir, fcdir)

    lsm_input, e5_soil,= process_era5_soiltemp(lsm_input, era5_stl, era5_swvl, era5_sst, era5_skt, era5_lsm, era5_slt)
    lsm_input, e5_soil, theta_rel, ds_vg = process_era5_soilmoist(lsm_input, e5_soil)

    lsm_input = process_soil_map(soilfile, lsm_input, nn_dominant, nblockx, nblocky, domain, theta_rel, ds_vg)
    lsm_input, lu_dict  = process_top10NL_map(lufile, lu_types, lsm_input, nn_dominant, nblockx, nblocky, domain)

    lsm_input = init_lutypes_ifs(lsm_input, lu_dict, parnames_lsm )
    lsm_input = init_lutypes_dep(lsm_input, lu_dict, parnames_dep, depfile )
   
    if lwrite:
        write_output(lsm_input, 
                      write_binary_output=False, 
                      write_netcdf_output=True,
                      nprocx=1,
                      nprocy=1)
     
    if lplot:
        plotvars = ['cover_'+ s for s in lu_types.keys()]
        # plotvars = ['z0h_'+ s for s in lu_types.keys()]
        # plotvars = [s+'_ara' for s in parnames]
        plotvars.append('cover_tot')
        some_plots(lsm_input, plotvars)

    return lsm_input
    

if __name__ == "__main__":


    # -----------------------------
    # Settings
    # -----------------------------
    # Path to directory with `BOFEK2012_010m.nc` and `top10nl_landuse_010m.nc`
    spatial_data_path = '//tsn.tno.nl/Data/sv/sv-059025_unix/ProjectData/ERP/Climate-and-Air-Quality/users/janssenrhh/landuse_soil'
    #lufile   = 'top10nl_landuse_010m_2017_detailed.nc' # with crop types
    lufile   = 'top10nl_landuse_010m.nc'
    soilfile = 'BOFEK2012_010m.nc'
    depfile  = 'depac_landuse_parameters.nc'
    
    # =============================================================================
    # Local ECMWF data paths
    # =============================================================================
    era5_base = Path('//tsn.tno.nl/RA-Data/Express/ra_express_modasuscratch_unix/models/LEIP/europe_w30e70s5n75/ECMWF'
                     '/od/ifs/0001')
    andir    = era5_base / 'an/sfc/F640/0000'
    fcdir    = era5_base / 'fc/sfc/F1280'

    # Settings
    exp_id = 1  # experiment ID
    ktot_soil = 4  # number of soil layers
    domain_name = 'veluwe_small'
    lwrite = True
    lplot  = True

    # Start date/time of experiment
    start_date = datetime(year=2018, month=5, day=25) #, hour=4)
    # start_date = datetime(year=2018, month=11, day=21) #, hour=4)

    # Output directory of DALES input files
    cwd = Path.cwd()
    output_path = cwd / ".." / "cases" / domain_name
    output_path.mkdir(exist_ok=True)

    # domain and domain decomposition definition
    domain = domains[domain_name]
                    
    # land use types
    # lu_types = lu_types_basic
    # lu_types = lu_types_build # basic + buildings
    # lu_types  = lu_types_crop
    lu_types  = lu_types_depac
    
    # land use parameters
    parnames_lsm = ['cover','c_veg','z0m','z0h','lai','ar','br',
                    'lambda_s','lambda_us','rs_min','gD','tskin','lutype']
    parnames_dep = ['R_inc_b','R_inc_h','SAI_a','SAI_b',
                    'fmin','alpha','Tmin','Topt','Tmax','gs_max',
                    'vpd_min','vpd_max','gamma_stom','gamma_soil_c_fac',
                    'gamma_soil_default']
    parnames = parnames_lsm + parnames_dep
    
    # -----------------------------
    # End settings
    # -----------------------------

    lsm_input = process_input(lu_types, 
                              parnames, 
                              domain, 
                              output_path, 
                              andir, 
                              fcdir, 
                              start_date, 
                              exp_id, 
                              ktot_soil, 
                              lwrite, 
                              lplot)
