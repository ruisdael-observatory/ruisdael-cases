## Overview options `modlsm.f90` in the `ruisdael_lsm` branch.

## Required/optional namelist settings

`isurf=11` enables the new LSM, `z0mav` and `z0hav` are not used (but required by DALES), `albedoav` is required for now, as we do not (yet) have a spatially varying albedo.

    &NAMSURFACE
    isurf = 11
    z0mav = ..
    z0hav = ..
    albedoav = ..
    /
    
The number of soil layers is flexible, and should be set in `&DOMAIN`:

    &DOMAIN
    ....
    kmax_soil = 4
    /

`&NAMLSM` contains global settings for the new LSM. With `lheterogeneous=.false.` the land surface is initialised from namelist options (see below),
with `lheterogeneous=.true.` the surface is initialised from 2D (binary) input files. `dz_soil` contains the depth of the soil layers, from bottom to top.
`iinterp_t` (temperature) and `iinterp_theta` (soil moisture) set the soil interpolation scheme, IFS/HTESSEL uses `iinterp_t=1`
(`v=0.5*(v_low+v_high)`) and `iinterp_theta=4` (`v=max(v_low,v_high)`).

    &NAMLSM
    lheterogeneous = .true.
    lfreedrainage = .true.
    dz_soil = 1.89 0.72 0.21 0.07
    iinterp_t = 1
    iinterp_theta = 4
    /
    
## Homogeneous initialisation

The surface can be initialised homogeneous through the options in `&NAMLSM_HOMOGENEOUS`:

    &NAMLSM_HOMOGENEOUS
    ! Sub-grid tile fractions, should sum to 1:
    c_low = 0.5
    c_high = 0.3
    c_bare = 0.1
    c_water = 0.1
    
    ! Roughness lengths for the sub-grid tiles:
    z0m_low = 0.075
    z0m_high = 1.
    z0m_bare = 0.01
    z0m_water = 0.001
    
    z0h_low = 0.003
    z0h_high = 1.
    z0h_bare = 0.001
    z0h_water = 0.0001
    
    ! Conductivity skin layer for stable (s) and unstable (us) conditions:
    lambda_s_low = 10.
    lambda_s_high = 15.
    lambda_s_bare = 15.
    
    lambda_us_low = 10.
    lambda_us_high = 40.
    lambda_us_bare = 15.
    
    ! Leaf area index low and high vegetation:
    lai_low = 2.5
    lai_high = 4.
    
    ! Minimum canopy or soil resistance:
    rs_min_low = 100.
    rs_min_high = 250.
    rs_min_bare = 50.
    
    ! Initial profiles of soil temperature and soil moisture (from bottom to top):
    t_soil_p = 288.44687 289.6885 289.43475 290.13162
    theta_soil_p = 0.42387354 0.38100946 0.37343964 0.3665535
    
    ! Soil index in `van_genuchten_parameters.nc`:
    soil_index_p = 4 4 4 4
    
    ! Coefficients used to calculate the root fraction.
    ar_low = 10.739
    br_low = 2.608
    
    ar_high = 4.453
    br_high = 1.631
    
    ! Coefficient for canopy resistance high vegetation
    gD_high = 0.0003
    
    ! Fixed surface temperature open water:
    tskin_water = 290
    
## Required data files

Both the homogeneous and heterogeneous LSM requires a NetCDF file `van_genuchten_parameters.nc`, which contains the van Genuchten parameters for each soil type set by `soil_index`. The NetCDF file is available in the `DALES_root/data` directory.

### Heterogeneous surfaces

TO-DO
