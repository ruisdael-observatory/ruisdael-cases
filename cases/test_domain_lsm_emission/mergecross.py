import netCDF4 as nc4
import numpy as np

import sys
import os

#
# Settings
#
#expnr = 1
#npx   = 24
#npy   = 24
#itot  = 840 #1680
#jtot  = 840 #1680
#ktot  = 75
#ntime = 1

expnr = 1
npx   = 2
npy   = 2
itot  = 64
jtot  = 64
ktot  = 64
ntime = 1

#
# Get base and variable name to process
#
if len(sys.argv) != 3:
    sys.exit('Provide cross-section base name (e.g. crossxy or crossxy.0001) and variable (e.g. thl) to process')
else:
    base = sys.argv[1]
    variables = sys.argv[2]

variables = variables.split(',')

#
# Set some settings
#
if 'xz' in base:
    mode = 'xz'
elif 'xy' in base or 'cape' in base or 'surfcross' in base:
    mode = 'xy'

if mode == 'xy':
    chx = int(itot / npx)
    chy = int(jtot / npy)
elif mode == 'xz':
    chx = int(itot / npx)
    chy = 1

#
## Process cross-sections of each MPI task
#
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
            out_name = '_'.join(variables)
            dst = nc4.Dataset('{0:}.{1:}.{2:03d}.nc'.format(base, out_name, expnr), 'w')

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
                if name == 'time' or name[0] == 'x' or name[0] == 'y' or name[0] == 'z' or name.replace(mode, '') in variables:
                    print('Creating variable: ', name)
                    dst.createVariable(name, var.datatype, var.dimensions)

            # Copy time
            dst.variables['time'][:]= src.variables['time'][::ntime]

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
                    if name.replace(mode, '') in variables:
                        if mode == 'xy':
                            dst.variables[name][:,sy,sx] = src.variables[name][::ntime]
                        elif mode == 'xz':
                            dst.variables[name][:,:,sx] = src.variables[name][::ntime]

dst.close()


