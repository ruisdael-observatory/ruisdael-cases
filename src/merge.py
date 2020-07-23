import xarray as xr
import datetime
import os

def read_DDH_netcdf(start, end, path, include_variables=None):
    """
    Read all converted DDH files in NetCDF format,
    from `start` to `end` date
    """

    print('Reading DDH-NetCDF from {} to {}'.format(start, end))

    nt = int((end-start).total_seconds() / 3600. / 3.)

    # Generate list of NetCDF files to read
    files = []
    for t in range(nt):
        date = start + t*datetime.timedelta(hours=3)

        f = '{0:}/{1:04d}/{2:02d}/{3:02d}/{4:02d}/LES_forcing_{1:04d}{2:02d}{3:02d}{4:02d}.nc'.\
            format(path, date.year, date.month, date.day, date.hour)

        if os.path.exists(f):
            files.append(f)
        else:
            print('Can not find {}!! Skipping..'.format(f))

    print('Reading {} DDH NetCDF files...'.format(len(files)))

    if include_variables is not None:
        # Exclude all variables which are not in `include_variables..`
        # xarray should really have an option for this..........
        tmp = xr.open_dataset(files[0])
        all_variables = tmp.variables.keys()

        exclude = []
        for var in all_variables:
            if var not in include_variables:
                exclude.append(var)

        nc = xr.open_mfdataset(files, drop_variables=exclude, concat_dim='time', autoclose=True)
    else:
        nc = xr.open_mfdataset(files, autoclose=True)

    # Read data with xarray
    return nc

class Timer:
    def __init__(self):
        self.start = datetime.datetime.now()
    def elapsed(self):
        return datetime.datetime.now() - self.start
    def reset(self):
        self.start = datetime.datetime.now()


#path  = '/scratch/ms/nl/nkbs/DOWA/LES_forcing/'
path  = '/nobackup/users/stratum/DOWA/LES_forcing/'
rvars = ['z','time','u','v','T','dtu_dyn','dtv_dyn','dtT_dyn','dtu_phy','dtv_phy','dtT_phy']
iloc  = 3+24
name  = 'K13_30km'

t = Timer()

for year in range(2016,2018):
    for month in range(1,13):
        t.reset()

        start = datetime.datetime(year, month, 1, 0)
        if month != 12:
            end   = datetime.datetime(year, month+1, 1, 0)
        else:
            end   = datetime.datetime(year+1, 1, 1, 0)

        data = read_DDH_netcdf(start, end, path, rvars)

        data = data.isel(domain=iloc)
        data.to_netcdf('{0:}/{1:}_{2:04d}{3:02d}.nc'.format(path,name,year,month))

        print('Done: {}'.format(t.elapsed()))
