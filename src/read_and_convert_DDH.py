"""
Read / process all the required DDH output
Bart van Stratum (KNMI)
"""

import numpy as np
import datetime
import netCDF4 as nc4
import xarray as xr

import read_DDH as ddh
import DDH_domains as ddom
import spatial_tools as st
import read_soil as rs

# Constants, stored in dict, not to mix up Harmonie (ch) & DALES (cd) constants
# Harmonie constants are from arpifs/setup/sucst.F90
ch = dict(grav=9.80665, T0=273.16, Rd=287.06)
cd = dict(p0=1.e5, Rd=287.04, cp=1004.)


class Read_DDH_files:
    def __init__(self, path, date, t_end, step, add_soil=False, dt=60, quiet=False):
        """
        Read / process all DDH files
        """

        base_path = path    # Path of base of directory structure
        self.add_soil = add_soil

        # Path to DDH files
        path = '{0:}/{1:04d}/{2:02d}/{3:02d}/{4:02d}'.format(path, date.year, date.month, date.day, date.hour)

        # Read first DDH file to get some settings
        f = ddh.DDH_LFA('{0:}/DHFDLHARM+{1:04d}'.format(path, step))

        self.nlev  = f.attributes['doc']['nlev']  # Number of full vertical levels
        self.nlevh = self.nlev + 1                # Number of half vertical levels
        self.ndom  = f.attributes['doc']['ndom']  # Number of output domains
        self.nt    = int(t_end/step)              # Number of output time steps

        # Create empty arrays to store the individual DDH data
        self.time = np.ma.zeros(self.nt)
        self.datetime = []

        # Array dimensions
        dim3d  = (self.nt, self.ndom, self.nlev)
        dim3dh = (self.nt, self.ndom, self.nlevh)
        dim2d  = (self.nt, self.ndom)

        # Harmonie's moisture phases
        self.qtypes = {'qv':'vapor', 'ql':'liquid', 'qi':'ice', 'qr':'rain', 'qs':'snow', 'qg':'graupel'}

        # Atmospheric quantities
        # ----------------------
        self.cp   = np.ma.zeros(dim3d)  # Specific heat at const pressure (J kg-1 K-1)
        self.p    = np.ma.zeros(dim3d)  # Pressure (Pa)
        self.dp   = np.ma.zeros(dim3d)  # Pressure difference (Pa)
        self.z    = np.ma.zeros(dim3d)  # Geopotential height (m)
        self.ph   = np.ma.zeros(dim3dh) # Half level pressure (Pa)
        self.zh   = np.ma.zeros(dim3dh) # Half level geopotential height (m)

        self.u    = np.ma.zeros(dim3d)  # u-component wind (m s-1)
        self.v    = np.ma.zeros(dim3d)  # v-component wind (m s-1)
        self.T    = np.ma.zeros(dim3d)  # Absolute temperature (K)

        for q in self.qtypes.keys(): # Specific humidity (kg kg-1)
            setattr(self, q, np.ma.zeros(dim3d))

        # Surface quantities
        # ----------------------
        self.H    = np.ma.zeros(dim2d)  # Surface sensible heat flux (W m-2)
        self.LE   = np.ma.zeros(dim2d)  # Surface latent heat flux (W m-2)
        self.Tsk  = np.ma.zeros(dim2d)  # Surface temperature (K)
        self.qsk  = np.ma.zeros(dim2d)  # Surface specific humidity (kg kg-1)
        self.swds = np.ma.zeros(dim2d)  # Surface incoming shortwave radiation (W m-2)
        self.lwds = np.ma.zeros(dim2d)  # Surface incoming longwave radiation (W m-2)

        # Physics, dynamics and total tendencies
        # Units all in "... s-1"
        self.dtu_phy = np.ma.zeros(dim3d)
        self.dtv_phy = np.ma.zeros(dim3d)
        self.dtT_phy = np.ma.zeros(dim3d)

        self.dtu_dyn = np.ma.zeros(dim3d)
        self.dtv_dyn = np.ma.zeros(dim3d)
        self.dtT_dyn = np.ma.zeros(dim3d)

        self.dtu_tot = np.ma.zeros(dim3d)
        self.dtv_tot = np.ma.zeros(dim3d)
        self.dtT_tot = np.ma.zeros(dim3d)

        # Radiative quantities
        self.dtT_rad = np.ma.zeros(dim3d)   # Temperature tendency due to radiation (K s-1)
        self.lw_rad  = np.ma.zeros(dim3dh)  # Net longwave radiative flux (W m-2)
        self.sw_rad  = np.ma.zeros(dim3dh)  # Net shortwave radiative flux (W m-2)

        # Specific humidity tendencies
        for q in self.qtypes.keys():
            setattr(self, 'dt{}_tot'.format(q), np.ma.zeros(dim3d))
            setattr(self, 'dt{}_dyn'.format(q), np.ma.zeros(dim3d))
            setattr(self, 'dt{}_phy'.format(q), np.ma.zeros(dim3d))

        # Soil properties
        if add_soil:
            self.tg1 = np.ma.zeros(dim2d)
            self.tg2 = np.ma.zeros(dim2d)
            self.wg1 = np.ma.zeros(dim2d)
            self.wg2 = np.ma.zeros(dim2d)
            self.wg3 = np.ma.zeros(dim2d)

        # Read all files
        for tt in range(step, t_end+1, step):
            t = int(tt/step)-1

            if not quiet:
                print('Reading DDH file #{0:3d} (index {1:<3d})'.format(tt,t))

            f = ddh.DDH_LFA('{0:}/DHFDLHARM+{1:04d}'.format(path,tt))

            self.datetime.append(f.attributes['datetime']['forecast_date'])

            self.cp[t,:,:] = f.read_variable('VCP1') * ch['grav']
            self.p [t,:,:] = f.read_variable('VPF1') * ch['grav']
            self.z [t,:,:] = f.read_variable('VZF1')

            self.ph[t,:,1:] = f.read_variable('VPH1') * ch['grav']
            self.zh[t,:,1:] = f.read_variable('VZH1')

            # Non-accumulated variables
            self.dp[t,:,:] = f.read_variable('VPP1')

            self.u [t,:,:] = f.read_variable('VUU1') / self.dp[t,:,:]
            self.v [t,:,:] = f.read_variable('VVV1') / self.dp[t,:,:]
            self.T [t,:,:] = f.read_variable('VCT1') / self.dp[t,:,:]/self.cp[t,:,:]

            for q in self.qtypes.keys():
                getattr(self, q)[t,:,:] = f.read_variable('V{}1'.format(q.upper())) / self.dp[t,:,:]

            self.H[t,:]    = f.read_variable('FSHF')
            self.LE[t,:]   = f.read_variable('FLHF')
            self.Tsk[t,:]  = f.read_variable('VTSK1')
            self.qsk[t,:]  = f.read_variable('VQSK1')
            self.swds[t,:] = f.read_variable('FSWDODS')
            self.lwds[t,:] = f.read_variable('FLWDTHS')

            # Accumulated tendencies/variables/..
            self.dtu_phy[t,:,:] = f.read_variable('TUUPHY9')
            self.dtv_phy[t,:,:] = f.read_variable('TVVPHY9')
            self.dtT_phy[t,:,:] = f.read_variable('TCTPHY9')

            self.dtu_dyn[t,:,:] = f.read_variable('TUUDYN9')
            self.dtv_dyn[t,:,:] = f.read_variable('TVVDYN9')
            self.dtT_dyn[t,:,:] = f.read_variable('TCTDYN9')

            # Specific humidity tendencies
            for q in self.qtypes.keys():
                getattr(self, 'dt{}_dyn'.format(q))[t,:,:] = f.read_variable('T{}DYN9'.format(q.upper()))
                getattr(self, 'dt{}_phy'.format(q))[t,:,:] = f.read_variable('T{}PHY9'.format(q.upper()))

            # Radiation
            self.dtT_rad[t,:,:] = f.read_variable('TCTRADI')
            self.lw_rad [t,:,:] = f.read_variable('FCTRAYTH')
            self.sw_rad [t,:,:] = f.read_variable('FCTRAYSO')

            # Manually calculate time; DDH can't handle times < 1hour
            self.time[t] = tt/60.

        # From Python list to Numpy array..
        self.datetime    = np.array(self.datetime)
        self.hours_since = np.array([(time-datetime.datetime(2010,1,1)).total_seconds()/3600. for time in self.datetime])

        # Mask top half levels (is not in output DDH)
        self.ph[:,:,0]     = np.ma.masked
        self.zh[:,:,0]     = np.ma.masked
        self.lw_rad[:,:,0] = np.ma.masked
        self.sw_rad[:,:,0] = np.ma.masked

        # De-accumulate the tendencies
        self.deaccumulate(self.dtu_phy, step*dt)
        self.deaccumulate(self.dtv_phy, step*dt)
        self.deaccumulate(self.dtT_phy, step*dt)

        self.deaccumulate(self.dtu_dyn, step*dt)
        self.deaccumulate(self.dtv_dyn, step*dt)
        self.deaccumulate(self.dtT_dyn, step*dt)

        self.deaccumulate(self.dtqv_dyn, step*dt)
        self.deaccumulate(self.dtql_dyn, step*dt)
        self.deaccumulate(self.dtqi_dyn, step*dt)
        self.deaccumulate(self.dtqr_dyn, step*dt)
        self.deaccumulate(self.dtqs_dyn, step*dt)
        self.deaccumulate(self.dtqg_dyn, step*dt)

        self.deaccumulate(self.dtqv_phy, step*dt)
        self.deaccumulate(self.dtql_phy, step*dt)
        self.deaccumulate(self.dtqi_phy, step*dt)
        self.deaccumulate(self.dtqr_phy, step*dt)
        self.deaccumulate(self.dtqs_phy, step*dt)
        self.deaccumulate(self.dtqg_phy, step*dt)

        self.deaccumulate(self.dtT_rad, step*dt)
        self.deaccumulate(self.sw_rad,  step*dt)
        self.deaccumulate(self.lw_rad,  step*dt)

        self.deaccumulate(self.H,    step*dt)
        self.deaccumulate(self.LE,   step*dt)
        self.deaccumulate(self.swds, step*dt)
        self.deaccumulate(self.lwds, step*dt)

        # Sum of moisture and moisture tendencies
        self.q = self.qv + self.ql + self.qi + self.qr + self.qs + self.qg

        self.dtq_dyn = self.dtqv_dyn + self.dtql_dyn + self.dtqi_dyn +\
                       self.dtqr_dyn + self.dtqs_dyn + self.dtqg_dyn
        self.dtq_phy = self.dtqv_phy + self.dtql_phy + self.dtqi_phy +\
                       self.dtqr_phy + self.dtqs_phy + self.dtqg_phy
        self.dtq_tot = self.dtqv_tot + self.dtql_tot + self.dtqi_tot +\
                       self.dtqr_tot + self.dtqs_tot + self.dtqg_tot

        # Derived quantities
        self.exner  = (self.p / 1e5)**(ch['Rd'] / self.cp[t,:]) # Exner
        self.th     = self.T / self.exner                       # Potential temperature

        # Check...: sum of dyn+phys
        self.dtu_sum = self.dtu_phy + self.dtu_dyn
        self.dtv_sum = self.dtv_phy + self.dtv_dyn
        self.dtT_sum = self.dtT_phy + self.dtT_dyn
        self.dtq_sum = self.dtq_phy + self.dtq_dyn

        # Check...: offline tendency
        self.dtu_off  = self.calc_tendency(self.u,  step*dt)
        self.dtv_off  = self.calc_tendency(self.v,  step*dt)
        self.dtT_off  = self.calc_tendency(self.T,  step*dt)
        self.dtth_off = self.calc_tendency(self.th, step*dt)
        self.dtq_off  = self.calc_tendency(self.q , step*dt)

        # Read soil properties
        if add_soil:
            domains, sizes = ddom.get_DOWA_domains()
            info = ddom.get_domain_info(domains, sizes)
            soil = rs.Read_soil(base_path, date, self.datetime, info)

            self.tg1[:,:] = soil.tg1
            self.tg2[:,:] = soil.tg2
            self.wg1[:,:] = soil.wsa1
            self.wg2[:,:] = soil.wsa2
            self.wg3[:,:] = soil.wsa3

    def calc_tendency(self, array, dt):
        tend = np.zeros_like(array)
        tend[1:,:] = (array[1:,:] - array[:-1,:]) / dt
        return tend

    def deaccumulate(self, array, dt):
        array[1:,:] = (array[1:,:] - array[:-1,:]) / dt
        array[0,:]  = array[0,:] / dt

    def to_netcdf(self, file_name, add_domain_info=False):

        def add_variable(file, name, type, dims, accumulated, ncatts, data):
            v = file.createVariable(name, type, dims, fill_value=nc4.default_fillvals['f4'])
            v.setncatts(ncatts)
            if accumulated:
                v.setncattr('type', 'accumulated')
            else:
                v.setncattr('type', 'instantaneous')

            if dims[-1] in ['level', 'hlevel']:
                v[:] = data[:,:,::-1]
            else:
                v[:] = data[:]

        # Create new NetCDF file
        f = nc4.Dataset(file_name, 'w')

        # Set some global attributes
        f.setncattr('Conventions', "CF-1.4")
        f.setncattr('institute_id', "KNMI")
        f.setncattr('model_id', "harmonie-40h1.2.tg2")
        f.setncattr('domain', "NETHERLANDS")
        f.setncattr('driving_model_id', "ERA5")
        f.setncattr('experiment_id', "DOWA_40h12tg2_fERA5")
        f.setncattr('title', "Dutch Offshore Wind Atlas (DOWA) - initial & boundary conditions for LES")
        f.setncattr('project_id', "DOWA")
        f.setncattr('institution', "Royal Netherlands Meteorological Institute, De Bilt, The Netherlands")
        f.setncattr('data_contact', "Bart van Stratum, R&D Weather & Climate Models, KNMI (bart.van.stratum@knmi.nl)")

        # Create dimensions
        f.createDimension('time',   self.nt)
        f.createDimension('level',  self.nlev)
        f.createDimension('levelh', self.nlevh)
        f.createDimension('domain', self.ndom)

        # Dimensions in NetCDF file
        dim3d  = ('time', 'domain', 'level')
        dim3dh = ('time', 'domain', 'levelh')
        dim2d  = ('time', 'domain')
        dim1d  = ('time')
        dimdom = ('domain')

        # Output data type
        dtype = 'f4'

        # Domain information
        if (add_domain_info):
            domains, sizes = ddom.get_DOWA_domains()
            info = ddom.get_domain_info(domains, sizes)

            name        = f.createVariable('name',        str,  ('domain')) #; name[:] = info['name']
            central_lat = f.createVariable('central_lat', 'f4', ('domain')); central_lat[:] = info['cent_lat']
            central_lon = f.createVariable('central_lon', 'f4', ('domain')); central_lon[:] = info['cent_lon']
            west_lon    = f.createVariable('west_lon',    'f4', ('domain')); west_lon[:]    = info['west_lon']
            east_lon    = f.createVariable('east_lon',    'f4', ('domain')); east_lon[:]    = info['east_lon']
            north_lat   = f.createVariable('north_lat',   'f4', ('domain')); north_lat[:]   = info['north_lat']
            south_lat   = f.createVariable('south_lat',   'f4', ('domain')); south_lat[:]   = info['south_lat']
            for i in range(self.ndom):
                name[i] = info['name'][i]

        # Create spatial/time variables
        add_variable(f, 'time', dtype, dim1d,  False, {'units': 'hours since 2010-01-01 00:00:00', 'long_name': 'time', 'calender': 'standard'}, self.hours_since)
        add_variable(f, 'z',    dtype, dim3d,  False, {'units': 'm',  'long_name': 'Full level geopotential height'}, self.z)
        add_variable(f, 'p',    dtype, dim3d,  False, {'units': 'Pa', 'long_name': 'Full level hydrostatic pressure'}, self.p)
        add_variable(f, 'zh',   dtype, dim3dh, False, {'units': 'm',  'long_name': 'Half level geopotential height'}, self.zh)
        add_variable(f, 'ph',   dtype, dim3dh, False, {'units': 'Pa', 'long_name': 'Half level hydrostatic pressure'}, self.ph)

        # Model variables
        add_variable(f, 'T',    dtype, dim3d, False, {'units': 'K',       'long_name': 'Absolute temperature'}, self.T)
        add_variable(f, 'u',    dtype, dim3d, False, {'units': 'm s-1',   'long_name': 'Zonal wind'}, self.u)
        add_variable(f, 'v',    dtype, dim3d, False, {'units': 'm s-1',   'long_name': 'Meridional wind'}, self.v)
        add_variable(f, 'q',    dtype, dim3d, False, {'units': 'kg kg-1', 'long_name': 'Total specific humidity'}, self.q)

        # Net radiative fluxes
        add_variable(f, 'sw_net', dtype, dim3dh, True, {'units': 'W m-2', 'long_name': 'Net shortwave radiation'}, self.sw_rad)
        add_variable(f, 'lw_net', dtype, dim3dh, True, {'units': 'W m-2', 'long_name': 'Net longwave radiation'}, self.lw_rad)

        # Surface variables
        add_variable(f, 'H',       dtype, dim2d, True,  {'units': 'W m-2',   'long_name': 'Surface sensible heat flux'}, self.H)
        add_variable(f, 'LE',      dtype, dim2d, True,  {'units': 'W m-2',   'long_name': 'Surface latent heat flux'}, self.LE)
        add_variable(f, 'T_s',     dtype, dim2d, False, {'units': 'K',       'long_name': 'Absolute (sea) surface temperature'}, self.Tsk)
        add_variable(f, 'q_s',     dtype, dim2d, False, {'units': 'kg kg-1', 'long_name': 'Surface specific humidity'}, self.qsk)
        add_variable(f, 'p_s',     dtype, dim2d, False, {'units': 'Pa',      'long_name': 'Surface pressure'}, self.ph[:,:,-1])
        add_variable(f, 'lwin_s',  dtype, dim2d, True,  {'units': 'W m-2',   'long_name': 'Surface shortwave incoming radiation'}, self.lwds)
        add_variable(f, 'swin_s',  dtype, dim2d, True,  {'units': 'W m-2',   'long_name': 'Surface longwave incoming radiation'}, self.swds)

        # Soil variables
        if self.add_soil:
            add_variable(f, 'Tg1',     dtype, dim2d, False, {'units': 'K',       'long_name': 'Top soil layer temperature'}, self.tg1)
            add_variable(f, 'Tg2',     dtype, dim2d, False, {'units': 'K',       'long_name': 'Bulk soil layer temperature'}, self.tg2)
            add_variable(f, 'wg1',     dtype, dim2d, False, {'units': 'm3 m-3',  'long_name': 'Top soil layer moisture content'}, self.wg1)
            add_variable(f, 'wg2',     dtype, dim2d, False, {'units': 'm3 m-3',  'long_name': 'Bulk soil layer moisture content'}, self.wg2)
            add_variable(f, 'wg3',     dtype, dim2d, False, {'units': 'm3 m-3',  'long_name': 'Bottom soil layer moisture content'}, self.wg3)

        for qtype,qname in self.qtypes.items():
            add_variable(f, qtype, dtype, dim3d, False, {'units': 'kg kg-1', 'long_name': 'Specific humidity ({})'.format(qname)}, getattr(self, qtype))

        # Tendencies
        add_variable(f, 'dtT_phy', dtype, dim3d, True, {'units': 'K s-1',  'long_name': 'Physics temperature tendency'},  self.dtT_phy)
        add_variable(f, 'dtT_dyn', dtype, dim3d, True, {'units': 'K s-1',  'long_name': 'Dynamics temperature tendency'}, self.dtT_dyn)
        add_variable(f, 'dtT_rad', dtype, dim3d, True, {'units': 'K s-1',  'long_name': 'Radiative temperature tendency'}, self.dtT_rad)

        add_variable(f, 'dtu_phy', dtype, dim3d, True, {'units': 'm s-2',  'long_name': 'Physics zonal wind tendency'},  self.dtu_phy)
        add_variable(f, 'dtu_dyn', dtype, dim3d, True, {'units': 'm s-2',  'long_name': 'Dynamics zonal wind tendency'}, self.dtu_dyn)

        add_variable(f, 'dtv_phy', dtype, dim3d, True, {'units': 'm s-2',  'long_name': 'Physics meridional wind tendency'},  self.dtv_phy)
        add_variable(f, 'dtv_dyn', dtype, dim3d, True, {'units': 'm s-2',  'long_name': 'Dynamics meridional wind tendency'}, self.dtv_dyn)

        add_variable(f, 'dtq_phy', dtype, dim3d, True, {'units': 'kg kg-1 s-1',  'long_name': 'Physics total specific humidity tendency'},  self.dtq_phy)
        add_variable(f, 'dtq_dyn', dtype, dim3d, True, {'units': 'kg kg-1 s-1',  'long_name': 'Dynamics total specific humidity tendency'}, self.dtq_dyn)

        for qtype,qname in self.qtypes.items():
            add_variable(f, 'dt{}_phy'.format(qtype),  dtype, dim3d, True,\
                {'units': 'kg kg-1 s-1', 'long_name': 'Physics specific humidity ({}) tendency'.format(qname)},  getattr(self, 'dt{}_phy'.format(qtype)))
            add_variable(f, 'dt{}_dyn'.format(qtype),  dtype, dim3d, True,\
                {'units': 'kg kg-1 s-1', 'long_name': 'Dynamics specific humidity ({}) tendency'.format(qname)}, getattr(self, 'dt{}_dyn'.format(qtype)))

        f.close()


if (__name__ == '__main__'):
    import matplotlib.pyplot as pl
    import matplotlib.gridspec as gridspec

    pl.close('all')

    dt      = 60      # model time step (s)
    t_end   = 180     # final file to read (-)
    step    = 10      # interval to read (-)

    # ---------------------
    # Convert DDH to NetCDF
    # ---------------------
    date = datetime.datetime(2017, 1, 1, 0)

    #data_root = '/nobackup/users/stratum/DOWA/LES_forcing'     # KNMI desktop
    data_root = '/scratch/ms/nl/nkbs/DOWA/LES_forcing'          # ECMWF

    # Read DDH files and convert to NetCDF
    if 'data' not in locals():
        data = Read_DDH_files(data_root, date, t_end, step, add_soil=False)
        data.to_netcdf('example.nc', add_domain_info=True)
