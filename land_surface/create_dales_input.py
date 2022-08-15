import matplotlib.pyplot as plt
#import netCDF4 as nc4
import xarray as xr
import numpy as np
#import sys
import os
from datetime import datetime

# Custom Python scripts/tools/...:
from vegetation_properties import ifs_vegetation, top10_to_ifs, top10_names
from interpolate import interp_dominant, interp_soil, Interpolate_era5
from spatial_transforms import proj4_rd #, proj4_hm
from bofek2012 import BOFEK_info
from lsm_input_dales import LSM_input_DALES
from era5_soil import init_theta_soil, calc_theta_rel #, download_era5_soil
from domains import domains
from landuse_types import lu_types_basic, lu_types_build, lu_types_crop, lu_types_depac


def init_dales_grid(domain, ktot_soil, lutypes):
    """
    Initialise a land surface grid with the properties of the DALES grid
    
    Parameters
    ----------
    domain : dict
        disctionary with domain size and resolution
    ktot_soil : int
        number of soil levels
    lutypes : dict
        disctionary with land use types

    Returns
    -------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    nn_dominant : int
        DESCRIPTION.
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
    lsm_input = LSM_input_DALES(itot, jtot, ktot_soil, lutypes, debug=True)
    
    # Save lat/lon coordinates
    lsm_input.lat[:,:] = lat2d
    lsm_input.lon[:,:] = lon2d

    lsm_input.x[:] = x_rd
    lsm_input.y[:] = y_rd
    
    return(lsm_input, nn_dominant, nblockx, nblocky)


def get_era5_data(era5_path, leipdir, andir, fcdir):
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
    era5_path : str
        DESCRIPTION.
    leipdir : str
        DESCRIPTION.
    andir : str
        DESCRIPTION.
    fcdir : str
        DESCRIPTION.

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
    #download_era5_soil(start_date, era5_path)
    
    #get variables:
    #      'sea_surface_temperature', 'soil_temperature_level_1',
    #      'soil_temperature_level_2', 'soil_temperature_level_3',
    #      'soil_temperature_level_4', 'soil_type', 'skin_temperature',
    #      'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
    #      'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
    #      'land_sea_mask'    
    
    
    #sst: sea surface temperature
    era5_sst = xr.open_dataset('%s/%s/%04d/sstk_%04d%02d%02d_1h.nc' %(leipdir, fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    #skt: skin temperature
    era5_skt = xr.open_dataset('%s/%s/%04d/skt_%04d%02d%02d_1h.nc' %(leipdir, fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    #swfl1-4: soil water volumetric level
    era5_swvl1 = xr.open_dataset('%s/%s/%04d/swvl1_%04d%02d%02d_1h.nc' %(leipdir, fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    era5_swvl2 = xr.open_dataset('%s/%s/%04d/swvl2_%04d%02d%02d_1h.nc' %(leipdir, fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    era5_swvl3 = xr.open_dataset('%s/%s/%04d/swvl3_%04d%02d%02d_1h.nc' %(leipdir, fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    era5_swvl4 = xr.open_dataset('%s/%s/%04d/swvl4_%04d%02d%02d_1h.nc' %(leipdir, fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    
    #stl1-4: soil temperature level
    era5_stl1 = xr.open_dataset('%s/%s/%04d/stl1_%04d%02d%02d_1h.nc' %(leipdir, fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    era5_stl2 = xr.open_dataset('%s/%s/%04d/stl2_%04d%02d%02d_1h.nc' %(leipdir, fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    era5_stl3 = xr.open_dataset('%s/%s/%04d/stl3_%04d%02d%02d_1h.nc' %(leipdir, fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    era5_stl4 = xr.open_dataset('%s/%s/%04d/stl4_%04d%02d%02d_1h.nc' %(leipdir, fcdir, start_date.year, start_date.year, start_date.month, start_date.day) )
    
    
    #lsm: land sea mask
    era5_lsm = xr.open_dataset('%s/%s/lsm.nc' %(leipdir, andir) )
    
    #slt: soil type
    era5_slt = xr.open_dataset('%s/%s/slt.nc' %(leipdir, andir) )

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
    #e5_soil = xr.open_dataset('{0}/{1:04d}{2:02d}{3:02d}_{4:02d}_soil.nc'.format(
    #        era5_path, start_date.year, start_date.month, start_date.day, start_date.hour))
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
    interpolate_era5 : TYPE
        DESCRIPTION.

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

    return lsm_input, e5_soil, theta_rel, ds_vg


def process_soil_map(soilfile, lsm_input, nn_dominant, nblockx, nblocky, domain, theta_rel, ds_vg):
    """
    Interpolate BOFEK2012 soil map to DALES grid
    Fill missing values to ECMWF soil type

    Parameters
    ----------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    nn_dominant : int
        DESCRIPTION.
    nblockx : int
        Number of blocks in x-direction.
    nblocky : int
        Number of blocks in y-direction.
    domain : dict
        Dales domain settings.
    theta_rel : TYPE
        DESCRIPTION.
    ds_vg : TYPE
        DESCRIPTION.

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
    lutypes : dict
        properties of each land use type.
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    nn_dominant : int
        DESCRIPTION.
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
    ilu     = np.arange(len(lu_types)) + 1
    
    setattr(lsm_input, 'luname', luname)
    setattr(lsm_input, 'lushort', lushort)
    setattr(lsm_input, 'lveg', lveg)
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
        setattr(lsm_input, 'c_'+lu, lu_types[lu]['lu_frac']) 
    
    #set dominant LU id for each LU type
    lu_low     = lutypes['lv']['lu_domid']
    lu_high    = lutypes['hv']['lu_domid']
    lu_water   = lutypes['aq']['lu_domid']
    lu_asphalt = lutypes['ap']['lu_domid']
    lu_baresoil= lutypes['bs']['lu_domid']
    try:
        lu_build   = lutypes['bu']['lu_domid']
    except:
        print('')
            
#    # Set vegetation fraction over Germany
# TODO apply to lsm_input 
#    de_mask = (lu_low==-1)&(lu_high==-1)&(lu_water==-1)&(lu_asphalt==-1)
#    frac_low  [de_mask] = 0.7
#    frac_high [de_mask] = 0.2
#    frac_water[de_mask] = 0.0
#    frac_asphalt[de_mask] = 0.0
    
    # Set default values low and high vegetation, where missing
    lu_low [lu_low  == -1] = 10   # 10 = grass
    lu_high[lu_high == -1] = 3    # 3  = mixed forest
    lu_asphalt[lu_asphalt == -1] = 20    # 20 = paved road
    lu_baresoil[lu_baresoil == -1] = 28  # 28 = fallow land
    lu_water[lu_water == -1] = 14        # 14 = water way 
    try:
        lu_build[lu_build == -1] = 29    # 29 = buildings
    except:
        print('')

    lutypes['lv']['lu_domid'] = lu_low
    lutypes['hv']['lu_domid'] = lu_high
    lutypes['aq']['lu_domid'] = lu_water
    lutypes['ap']['lu_domid'] = lu_asphalt
    lutypes['bs']['lu_domid'] = lu_baresoil    
    try:
        lutypes['bu']['lu_domid'] = lu_build    
    except:
        print('')

    return lsm_input, lutypes


def init_lutypes_ifs(lsm_input, lu_dict ):
    """Assign surface properties to DALES land use types based on ECMWF
       lookup table.    

    Parameters
    ----------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.
    lu_dict : dict
        LU type properties.

    Returns
    -------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.

    """
    #TODO: generalise for arbitrary number of LU types
        
    #
    # Init land use
    #
    shape = lu_dict['lv']['lu_frac'].shape
    for lu in lu_dict.keys():
        if lu == 'aq': continue
        print('\n processing', lu_dict[lu]['lu_long'])
        z0m      = np.full(shape, np.nan)
        z0h      = np.full(shape, np.nan)
        lambda_s = np.full(shape, np.nan)
        lambda_us= np.full(shape, np.nan)
        rs_min   = np.full(shape, np.nan)
        lai      = np.full(shape, np.nan) 
        ar       = np.full(shape, np.nan)
        br       = np.full(shape, np.nan) 
        cover    = lu_dict[lu]['lu_frac']
        lutype   = np.full(shape, np.nan)
        gD       = np.full(shape, np.nan)
    
        for vt in lu_dict[lu]['lu_ids']:

            iv = top10_to_ifs[vt]     # Index in ECMWF lookup table
            mask = (lu_dict[lu]['lu_domid'] == vt)
        
            print('LU type TOP10:', top10_names[vt])
            #print('LU type IFS  ', ifs_vegetation.name[iv])

            z0m      [mask] = ifs_vegetation.z0m      [iv]
            #print('z0m', ifs_vegetation.z0m      [iv])
            z0h      [mask] = ifs_vegetation.z0h      [iv]
            lambda_s [mask] = ifs_vegetation.lambda_s [iv]
            lambda_us[mask] = ifs_vegetation.lambda_us[iv]
            rs_min   [mask] = ifs_vegetation.rs_min   [iv]
            lai      [mask] = ifs_vegetation.lai      [iv]
            if lu == 'ap': lai[mask] = 0
            ar       [mask] = ifs_vegetation.a_r      [iv]
            br       [mask] = ifs_vegetation.b_r      [iv]
        
            if lu == 'hv':
                gD[mask] = ifs_vegetation.gD[iv]
            # Multiply grid point coverage with vegetation type coverage
            if lu == 'bs': continue
            cover[mask] *= ifs_vegetation.c_veg[iv]
            # Bonus, for offline LSM:
            lutype[mask] = iv
            
        setattr(lsm_input, 'z0m_'+lu, z0m)  
        setattr(lsm_input, 'z0h_'+lu, z0h)   
        setattr(lsm_input, 'lambda_s_'+lu, lambda_s)   
        setattr(lsm_input, 'lambda_us_'+lu, lambda_us)   
        setattr(lsm_input, 'rs_min_'+lu, rs_min)   
        setattr(lsm_input, 'lai_'+lu, lai)   
        setattr(lsm_input, 'ar_'+lu, ar)   
        setattr(lsm_input, 'br_'+lu, br)   
        setattr(lsm_input, 'c_'+lu, cover)   
        setattr(lsm_input, 'type_'+lu, lutype)  
        if lu == 'hv':
            setattr(lsm_input, 'gD', gD)   


    #
    # Init bare soil (bs)
    #
    iv = 7
    
    lsm_input.rs_min_bs   [:,:] = 50.
    
    # assign remaining land cover to bare soil 
    lsm_input.c_bs[:,:] = 1
    for lu in lu_dict.keys():
        if lu == 'bs': continue
        cover = getattr(lsm_input, 'c_'+lu)
        print(lu, cover.mean())
        lsm_input.c_bs[:,:] = lsm_input.c_bs[:,:] - cover
    
    # Bonus, for offline LSM:
    lsm_input.type_bs[:,:] = iv
    
    #
    # Init water (aq)
    #
    setattr(lsm_input, 'z0m_aq', np.full(shape, 0.1))
    setattr(lsm_input, 'z0h_aq', np.full(shape, 0.1e-2))

    return lsm_input 


def write_output(lsm_input, 
                 expname,
                 dx,
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

    Returns
    -------
    None.

    """

    if write_binary_output:
        lsm_input.save_binaries(nprocx=nprocx, nprocy=nprocy, exp_id=exp_id, path=output_path)
    
    if write_netcdf_output:
        lsm_input.save_netcdf('%s\lsm.inp_%03d_%02dm.nc' %(output_path, exp_id, dx))

    return


def some_plots(lsm_input, plotvars):
    """
    Generate some standard plots of Land Surface Model input data

    Parameters
    ----------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.

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
        ds_lsm[plotvar].plot(ax=ax, cmap='Reds',vmin=0,vmax=None)
        plt.tight_layout()
        
    plt.show()
    
    return 


def process_input(lu_types, domain, output_path, start_date, exp_id, ktot_soil):
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
    domain : dict
        Dales domain settings.
    output_path : str
        Dir to write output to.
    start_date : datetime.datetime
        Time stamp of Dales run start.
    exp_id : int
        Experiment ID.
    ktot_soil : int
        Number of soil layers.

    Returns
    -------
    lsm_input : LSM_input_DALES 
        Class containing Dales input parameters for all LU types.

    """
    lsm_input, nn_dominant, nblockx, nblocky = init_dales_grid(domain, ktot_soil, lu_types)

    era5_path = '//tsn.tno.nl/RA-Data/Express/ra_express_modasuscratch_unix/models/LEIP/europe_w30e70s5n75/ECMWF/od/ifs/0001/fc/sfc/F1280'
    leipdir  = '//tsn.tno.nl/RA-Data/Express/ra_express_modasuscratch_unix/models/LEIP'
    andir    = 'europe_w30e70s5n75/ECMWF/od/ifs/0001/an/sfc/F640/0000'
    fcdir    = 'europe_w30e70s5n75/ECMWF/od/ifs/0001/fc/sfc/F1280'
    era5_stl, era5_swvl, era5_lsm, era5_slt, era5_skt, era5_sst = get_era5_data(era5_path, leipdir, andir, fcdir)

    lsm_input, e5_soil,= process_era5_soiltemp(lsm_input, era5_stl, era5_swvl, era5_sst, era5_skt, era5_lsm, era5_slt)
    lsm_input, e5_soil, theta_rel, ds_vg = process_era5_soilmoist(lsm_input, e5_soil)

    lsm_input = process_soil_map(soilfile, lsm_input, nn_dominant, nblockx, nblocky, domain, theta_rel, ds_vg)
    lsm_input, lu_dict  = process_top10NL_map(lufile, lu_types, lsm_input, nn_dominant, nblockx, nblocky, domain)

    lsm_input = init_lutypes_ifs(lsm_input, lu_dict )
   
    if False:
        write_output(lsm_input, 
                      expname=domain['expname'],
                      dx=domain['dx'],
                      write_binary_output=False, 
                      write_netcdf_output=True,
                      nprocx=4,
                      nprocy=4)
     
    if True:
        # plotvars = ['c_lv', 'c_hv', 'c_ap','z0m_hv']
        plotvars = ['c_hv', 'z0m_hv', 'z0h_hv', 'lai_hv']
        some_plots(lsm_input, plotvars)

    return lsm_input
    

if __name__ == "__main__":


    # -----------------------------
    # Settings
    # -----------------------------
    # Path to directory with `BOFEK2012_010m.nc` and `top10nl_landuse_010m.nc`
    spatial_data_path = '//tsn.tno.nl/Data/sv/sv-059025_unix/ProjectData/ERP/Climate-and-Air-Quality/users/janssenrhh/landuse_soil'
    # lufile   = 'top10nl_landuse_010m_2017_detailed.nc'
    lufile   = 'top10nl_landuse_010m.nc'
    soilfile = 'BOFEK2012_010m.nc'
    
    # Output directory of DALES input files
    cwd = os.getcwd()
    output_path = os.path.join(cwd, 'eindhoven_small')
    
    # Start date/time of experiment
    start_date = datetime(year=2016, month=8, day=17, hour=4)
    
    # domain and domain decomposition definition    
    domain = domains['eindhoven_small']
                    
    # land use types
    lu_types = lu_types_basic
    # lu_types = lu_types_build
    #lu_types  = lu_types_crop
    #lu_types  = lu_types_depac



    # experiment ID
    exp_id = 1

    # number of soil layers
    ktot_soil = 4 
    
    # write_binary_output = False    # Write binary input for DALES
    # write_netcdf_output = True    # Write LSM input as NetCDF output (for e.g. visualisation)
    
    # -----------------------------
    # End settings
    # -----------------------------

    lsm_input = process_input(lu_types, domain, output_path, start_date, exp_id, ktot_soil)