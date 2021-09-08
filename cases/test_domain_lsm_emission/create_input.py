import matplotlib.pyplot as pl
import numpy as np
from datetime import datetime, timedelta
import netCDF4 as nc4

pl.close('all'); pl.ion()

class Grid_linear_stretched:
    def __init__(self, ktot, dz0, alpha):
        self.ktot  = ktot
        self.z     = np.zeros(ktot)
        self.dz    = np.zeros(ktot)
        self.dz[:] = dz0 * (1 + alpha)**np.arange(ktot)
        zh         = np.zeros(ktot+1)
        zh[1:]     = np.cumsum(self.dz)
        self.z[:]  = 0.5 * (zh[1:] + zh[:-1])
        self.zsize = zh[-1]

# Define (stretched) vertical grid:
grid = Grid_linear_stretched(64, 20, 0.03)

# Horizontal grid (for emissions):
xsize = 25600
ysize = 25600
itot = 64
jtot = 64
kemis = 9

start = datetime(2016, 8, 15, 5)
end   = datetime(2016, 8, 15, 18)
nt    = int((end-start).total_seconds()//3600+1)

# Vertical profiles:
thl = np.zeros(grid.ktot)
qt  = np.zeros(grid.ktot)
u   = np.ones(grid.ktot)*2
v   = np.ones(grid.ktot)*2
ug  = np.ones(grid.ktot)*2
vg  = np.ones(grid.ktot)*2
wls = np.zeros(grid.ktot)
tke = np.ones(grid.ktot)*0.01

# Number of CO2 fields:
co2_names = ['co2sum','co2bg','co2fos','co2ags','co2veg']
co2 = np.zeros((len(co2_names), grid.ktot))
co2[1,:]  = 400000                  # Background concentration, in ppb
co2[0,:] = co2[1:,:].sum(axis=0)    # Sum of all CO2 fields, in ppb

# Boundary layer structure
zi  = 1000
thl0 = 290
dthl0 = 2
gammathl = 0.006

qt0 = 8e-3
dqt0 = -2e-3
gammaqt = -1e-6

for k in range(grid.ktot):
    if grid.z[k] < zi:
        thl[k] = thl0
        qt[k]  = qt0
    else:
        thl[k] = thl0 + dthl0 + (grid.z[k]-zi)*gammathl
        qt[k]  = qt0  + dqt0  + (grid.z[k]-zi)*gammaqt

u[:] = 2.
v[:] = 2.
ug[:] = 2.
vg[:] = 2.

# Write DALES input:
# Initial profiles
f = open('prof.inp.001', 'w')
f.write('Initial profiles\n')
f.write('{0:^20s} {1:^20s} {2:^20s} {3:^20s} {4:^20s} {5:^20s}\n'.format(
        'z','thl','qt', 'u', 'v', 'tke'))
for k in range(grid.ktot):
    f.write('{0:1.14E} {1:1.14E} {2:1.14E} {3:1.14E} {4:1.14E} {5:1.14E}\n'.format(
        grid.z[k], thl[k], qt[k], u[k], v[k], tke[k]))
f.close()

# lscale.inp
f = open('lscale.inp.001', 'w')
f.write('Large-scale forcings\n')
f.write('{0:^20s} {1:^20s} {2:^20s} {3:^20s} {4:^20s} {5:^20s} {6:^20s} {7:^20s}\n'.format(
        'z','ug','vg','wls','dqtdx','dqtdy','dqtdt','dthldt'))
for k in range(grid.ktot):
    f.write('{0:1.14E} {1:1.14E} {2:1.14E} {3:1.14E} {4:1.14E} {5:1.14E} {6:1.14E} {7:1.14E}\n'.format(
        grid.z[k], ug[k], vg[k], wls[k], 0, 0, 0, 0))
f.close()

# scalar.inp
f = open('scalar.inp.001', 'w')
f.write('Scalars\n')
f.write('{0:^20s} '.format('z'))
for i in range(len(co2_names)):
    f.write('{0:^20s} '.format(co2_names[i]))
f.write('\n')
for k in range(grid.ktot):
    f.write('{0:1.14E} '.format(grid.z[k]))
    for i in range(len(co2_names)):
        f.write('{0:1.14E} '.format(co2[i,k]))
    f.write('\n')
f.close()

# Create emission files
emission = np.zeros((nt, kemis, jtot, itot))
emission[:, :, jtot//2-5:jtot//2+6, itot//2-5:itot//2+6] = 5000

date = start
t = 0
while date <= end:
    for co2_name in co2_names:

        fname = '{0}_emis_{1:04d}{2:02d}{3:02d}{4:02d}00_3d.nc'.format(
                co2_name, date.year, date.month, date.day, date.hour)

        nc_file = nc4.Dataset(fname, mode='w', datamodel='NETCDF4', clobber=True)

        nc_file.createDimension('x', itot)
        nc_file.createDimension('y', jtot)
        nc_file.createDimension('z', kemis)

        co2 = nc_file.createVariable(co2_name, 'double', ('z', 'y', 'x'))

        if co2_name == 'co2fos':
            co2[:] = emission[t,:,:,:]
        else:
            co2[:] = 0.

        nc_file.close()

    date += timedelta(hours=1)
    t += 1















