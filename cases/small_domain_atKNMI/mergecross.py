import netCDF4 as nc4
import numpy as np


def read_namelist(expnr):
    """
    Read grid dimensions and MPI decomposition from namelist
    """

    itot = None
    jtot = None
    ktot = None
    npx  = 1
    npy  = 1

    f = open('namoptions.001', 'r')
    for l in f.readlines():
        ls = l.strip()
        if len(ls) > 0 and ls[0] != '!':
            if 'itot' in l:
                itot = int(l.split('=')[1])
            elif 'jtot' in l:
                jtot = int(l.split('=')[1])
            elif 'kmax' in l:
                ktot = int(l.split('=')[1])
            elif 'nprocx' in l:
                npx = int(l.split('=')[1])
            elif 'nprocy' in l:
                npy = int(l.split('=')[1])

    return itot, jtot, ktot, npx, npy


def merge_cross(base, variable, expnr, skip, t0):
    """
    Merge DALES cross-sections (written per MPI task) into one NetCDF file
    Arguments:
        base     : cross-section type (crossxy, crossxz)
        variable : variable to merge (e.g. lwp, thl, .., so minus the `crossxy` part)
        expnr    : DALES experiment number
        skip     : Write every `skip`th cross-section
    """

    # Get settings from namelist
    itot, jtot, ktot, npx, npy = read_namelist(expnr)

    # Determine some settings
    if 'xz' in base:
        mode = 'xz'
    elif 'xy' in base:
        mode = 'xy'

    if mode == 'xy':
        chx = int(itot / npx)
        chy = int(jtot / npy)
    elif mode == 'xz':
        chx = int(itot / npx)
        chy = 1


    # Process cross-sections of each MPI task
    for i in range(npx):
        n = 1 if 'span' in base else npy
        for j in range(n):
            print('Processing i={}/{}, j={}/{}'.format(i+1,npx,j+1,npy))

            # Input file name
            fname = '{0:}.x{1:03d}y{2:03d}.{3:03d}.nc'.format(base, i, j, expnr)

            # Slices in x and y dimensions
            sx = np.s_[i*chx:(i+1)*chx]
            sy = np.s_[j*chy:(j+1)*chy]

            # Create new NetCDF file if first MPI task
            if i==0 and j==0:
                src = nc4.Dataset(fname)
                dst = nc4.Dataset('{0:}.{1:}.{2:03d}.nc'.format(base, variable, expnr), 'w')

                # Copy NetCDF attributes and dimensions
                for name in src.ncattrs():
                    dst.setncattr(name, src.getncattr(name))
                for name, dim in src.dimensions.items():
                    if dim.isunlimited():
                        dst.createDimension(name, None)
                    elif name[0] == 'x':
                        dst.createDimension(name, itot)
                    elif name[0] == 'y':
                        dst.createDimension(name, jtot)
                    elif name[0] == 'z':
                        dst.createDimension(name, ktot)

                # Create variables
                for name, var in src.variables.items():
                    if name == 'time' or name[0] == 'x' or name[0] == 'y' or name[0] == 'z' or name.replace(mode, '') == variable:
                        dst.createVariable(name, var.datatype, var.dimensions)

                # Copy time
                dst.variables['time'][:]= src.variables['time'][t0::skip]

                src.close()

            # Read NetCDF file and write to merged NetCDF
            src = nc4.Dataset(fname)

            # Loop over, and copy data
            for name, var in src.variables.items():
                if name != 'time':
                    if name[0] == 'x':
                        dst.variables[name][sx] = src.variables[name][:]
                    elif name[0] == 'y':
                        dst.variables[name][sy] = src.variables[name][:]
                    elif name[0] == 'z':
                        dst.variables[name][:] = src.variables[name][:]
                    else:
                        if name.replace(mode, '') == variable:
                            if mode == 'xy':
                                dst.variables[name][:,sy,sx] = src.variables[name][t0::skip]
                            elif mode == 'xz':
                                dst.variables[name][:,:,sx] = src.variables[name][t0::skip]

        dst.sync() 

    dst.close()


if __name__ == '__main__':

    import argparse

    # Parse input arguments
    p = argparse.ArgumentParser()
    p.add_argument('crosstype', type=str, help='Cross-section type (\"crossxy\", \"crossxz\")')
    p.add_argument('crossname', type=str, help='Variable name (e.g. \"lwp\", \"thl\"), witout \"xy\", \"xz\", ..')
    p.add_argument('expnr',     type=int, help='DALES experiment number')
    p.add_argument('--t0',      type=int, help='Start processing at t=\"t0\" index')
    p.add_argument('--skip',    type=int, help='Only process every \"skip\"-th cross-section in time')
    args = p.parse_args()

    # Some default values (if missing)
    skip = 1 if args.skip is None else args.skip
    t0   = 0 if args.skip is None else args.t0

    # Merge cross-section
    merge_cross(args.crosstype, args.crossname, args.expnr, skip, t0)
