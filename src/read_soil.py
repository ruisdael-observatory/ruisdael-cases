import numpy as np
import netCDF4 as nc4
import datetime

import spatial_tools as st

class Read_soil:
    def __init__(self, path, date, ddh_times, domain_info, t0_netCDF=datetime.datetime(2015,1,1)):
        nc_path = '{0:}/{1:04d}/{2:02d}/{3:02d}/00/'.format(path, date.year, date.month, date.day)
        dtg     = '{0:04d}{1:02d}{2:02d}'.format(date.year, date.month, date.day)

        # Number of times and locations in DDH file
        ntime = ddh_times.size
        nloc  = len(domain_info['name'])

        # Get time and lat/lon fields from first NetCDF file
        f    = '{0:}/{1:}_L01.P01.sfx.NETHERLANDS.DOWA_40h12tg2_fERA5_ptD.{2:}.nc'.format(nc_path, 'tg', dtg)
        nc   = nc4.Dataset(f)
        time = nc.variables['time'][:]
        lats = nc.variables['lat'][:]
        lons = nc.variables['lon'][:]

        # Convert time to datetime objects
        time = np.array([datetime.timedelta(days=t)+t0_netCDF for t in time])

        # Which DDH times are defined and undefined in NetCDF?
        use_times = np.ma.masked_all(ntime, dtype=int)

        for t in range(ntime):
            tt   = np.abs(time-ddh_times[t]).argmin()
            diff = np.abs(time[tt]-ddh_times[t]).total_seconds()
            if diff == 0:   # dangerous
                use_times[t] = tt

        # Data arrays
        self.tg1  = np.ma.masked_all((ntime, nloc))
        self.tg2  = np.ma.masked_all((ntime, nloc))
        self.wsa1 = np.ma.masked_all((ntime, nloc))
        self.wsa2 = np.ma.masked_all((ntime, nloc))
        self.wsa3 = np.ma.masked_all((ntime, nloc))

        # Read/process the data
        for var in ['tg', 'wsa']:
            levels = 3 if var=='wsa' else 2
            for level in range(1, levels+1):
                var_name = '{0:}_L{1:02d}'.format(var, level)
                var_tmp  = getattr(self, '{}{}'.format(var, level))

                # Open NetCDF files
                f  = '{0:}/{1:}.P01.sfx.NETHERLANDS.DOWA_40h12tg2_fERA5_ptD.{2:}.nc'.format(nc_path, var_name, dtg)
                nc = nc4.Dataset(f, 'r')

                # Loop over all required times
                for t in np.ma.nonzero(use_times)[0]:
                    t_nc = use_times[t]     # index in NetCDF
                    t_dh = t                # index in DDH

                    # Read entire slice to mem :-(
                    data = nc.variables[var_name][t_nc,:]

                    # Cherry-pick the correct output locations
                    for iloc in range(len(domain_info['name'])):

                        if domain_info['east_lon'][iloc] is None:     # Single column
                            j,i = st.find_nearest_latlon(lats, lons, domain_info['cent_lat'][iloc], domain_info['cent_lon'][iloc], silent=True)
                            var_tmp[t_dh, iloc] = data[j,i]
                        else:                                         # Area mean
                            mask = ((lats <= domain_info['north_lat'][iloc]) & (lats >= domain_info['south_lat'][iloc]) &\
                                    (lons <= domain_info['east_lon' ][iloc]) & (lons >= domain_info['west_lon' ][iloc]))
                            var_tmp[t_dh, iloc] = data[mask].mean()
