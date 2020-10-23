#
# Script to create LES lateral boundaries from HARMONIE 3D fields
# Bart van Stratum (KNMI), Dec. 2018
#

import xarray as xr
import numpy as np
import datetime

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Custom scripts
import hybrid_sigma_grid as hsg
import interpolate as ip


class Grid:
    """
    Simple version of LES grid, without ghost cells (not needed here)
    """
    def __init__(self, xsize, ysize, zsize, itot, jtot, ktot):

        # Store input/settings
        self.xsize = xsize
        self.ysize = ysize
        self.zsize = zsize

        self.itot = itot
        self.jtot = jtot
        self.ktot = ktot

        # Calculate grid
        self.dx = xsize / itot
        self.dy = ysize / jtot
        self.dz = zsize / ktot

        self.x = np.arange(0.5*self.dx, self.xsize, self.dx)
        self.y = np.arange(0.5*self.dy, self.ysize, self.dy)
        self.z = np.arange(0.5*self.dz, self.zsize, self.dz)

        self.xh = np.arange(0, self.xsize, self.dx)
        self.yh = np.arange(0, self.ysize, self.dy)
        self.zh = np.arange(0, self.zsize, self.dz)


class Grid_stretched:
    def __init__(self, xsize, ysize, itot, jtot, ktot, dz0, alpha):

        # Store input/settings
        self.xsize = xsize
        self.ysize = ysize
        self.zsize = None

        self.itot = itot
        self.jtot = jtot
        self.ktot = ktot

        # Calculate grid
        self.dx = xsize / itot
        self.dy = ysize / jtot
        self.dz = np.zeros(ktot)

        self.x = np.arange(0.5*self.dx, self.xsize, self.dx)
        self.y = np.arange(0.5*self.dy, self.ysize, self.dy)
        self.z = np.zeros(ktot)

        self.xh = np.arange(0, self.xsize, self.dx)
        self.yh = np.arange(0, self.ysize, self.dy)
        self.zh = np.zeros(ktot+1)

        self.dz     = np.zeros(ktot)
        self.dz[:]  = dz0 * (1 + alpha)**np.arange(ktot)
        self.zh[1:] = np.cumsum(self.dz)
        self.z[:]   = 0.5 * (self.zh[1:] + self.zh[:-1])
        self.zsize  = self.zh[-1]

    def plot(self):
        pl.figure()
        pl.title('zsize = {0:.1f} m'.format(self.zsize), loc='left')
        pl.plot(self.dz, self.z, '-x')
        pl.xlabel('dz (m)')
        pl.ylabel('z (m)')


def write_LBC(u, v, thl, qt, thls, itot, jtot, nprocx, nprocy, mpiidx, hour, minutes, iexpnr, output_dir):

    # Size of MPI sub-domains
    block_x = int(itot / nprocx)
    block_y = int(jtot / nprocy)

    # For t==0 and the left and right boundary, write the full range of y-processes
    # Otherwise only write the meridional boundaries
    if (hour==0 or mpiidx==0 or mpiidx==nprocx-1):
        yprocs = np.arange(nprocy)
    else:
        yprocs = (0, nprocy-1)

    for mpiidy in yprocs:
        slice3 = np.s_[:, mpiidy*block_y:(mpiidy+1)*block_y, :]
        slice2 = np.s_[:, mpiidy*block_y:(mpiidy+1)*block_y   ]

        # Open binary file
        name = '{0:}/lbc{1:03.0f}h{2:02.0f}m_x{3:03d}y{4:03d}.{5:03d}'.format(output_dir, hour, minutes, mpiidx, mpiidy, iexpnr)
        #print('Writing {}'.format(name))
        f = open(name, 'wb+')

        # Write boundaries and close file
        np.transpose(u   [slice3]).tofile(f)
        np.transpose(v   [slice3]).tofile(f)
        np.transpose(thl [slice3]).tofile(f)
        np.transpose(qt  [slice3]).tofile(f)
        np.transpose(thls[slice2]).tofile(f)
        f.close()


def write_initial_profiles(z, u, v, thl, qt, tke, iexpnr, output_dir):
    print('Writing initial profiles')

    # Initial profiles
    f = open('{0:}/prof.inp.{1:03d}'.format(output_dir, iexpnr), 'w')
    f.write('Initial profiles\n')
    f.write('{0:^20s} {1:^20s} {2:^20s} {3:^20s} {4:^20s} {5:^20s}\n'.format('z','thl','qt', 'u', 'v', 'tke'))
    for k in range(z.size):
        f.write('{0:1.14E} {1:1.14E} {2:1.14E} {3:1.14E} {4:1.14E} {5:1.14E}\n'.format(z[k], thl[k], qt[k], u[k], v[k], tke[k]))
    f.close()

    # Dummy lscale.inp
    f = open('{0:}/lscale.inp.{1:03d}'.format(output_dir, iexpnr), 'w')
    f.write('Large-scale forcings\n')
    f.write('{0:^20s} {1:^20s} {2:^20s} {3:^20s} {4:^20s} {5:^20s} {6:^20s} {7:^20s}\n'.format('z','ug','vg','wls','dqtdx','dqtdy','dqtdt','dthldt'))
    for k in range(z.size):
        f.write('{0:1.14E} {1:1.14E} {2:1.14E} {3:1.14E} {4:1.14E} {5:1.14E} {6:1.14E} {7:1.14E}\n'.format(z[k],0,0,0,0,0,0,0))
    f.close()

    # Dummy scalar.inp
    f = open('{0:}/scalar.inp.{1:03d}'.format(output_dir, iexpnr), 'w')
    f.write('Scalars\n')
    f.write('{0:^20s} {1:^20s} {2:^20s}\n'.format('z','s1','s2'))
    for k in range(z.size):
        f.write('{0:1.14E} {1:1.14E} {2:1.14E}\n'.format(z[k],0,0))
    f.close()



if __name__ == '__main__':
    import matplotlib.pyplot as pl
    pl.close('all')

    # Settings
    iexpnr = 1

    # -----------------
    # Production domain
    # -----------------
    if False:
        # Start and end time (index in HARMONIE files)
        t0 = 8
        t1 = 12

        # Lower left corner LES domain in HARMONIE (m)
        x0 = 700000
        y0 = 1200000

        # Domain size LES (m)
        xsize = 1680*200
        ysize = 1680*200

        # Number of grid points LES
        itot = 1680
        jtot = 1680
        ktot = 128

        # Number of x,y MPI processes
        nprocx = 24
        nprocy = 24

        # Output directory (boundaries are LARGE)
        output_dir = '/scratch/ms/nl/nkbs/DALES_boundaries/20181001_08/'

        # Harmonie data path (with yyyy/mm/dd/hh directory structure underneath)
        data_path = '/scratch/ms/nl/nkbs/HARMONIE_boundaries/'

    # -----------------
    # experimental Ruisdael campaign domain
    # -----------------
    if True:
        # Start and end time (index in HARMONIE files)
        t0 = 6
        t1 = 24

        # Lower left corner LES domain in HARMONIE (m)   
        #x0 = 700000   #  0.71 E    x0 = 0 # 7.9 W 
        #y0 = 1200000  # 54    N    y0 = 0 # 42.9 N
        
        #"Cabauw": (51.971, 4.927)
        #"Loobos": (52.1666, 5.7351) 

        # quick test to convert lat1,lon1 to meter N,E from Harmonie reference point
        # not very accurate
        #C = 40075000                   # Earth circumference, m
        #lat1 = 51.8
        #lon1 = 4.8
        #y0 = (lat1 - 42.9)/360.0 * C
        #x0 = (lon1 - -7.9)/360.0 * C * np.cos(lat1 * np.pi/180)
        
        x0 = 910000
        y0 = 940000

        # Domain size LES (m)
        xsize = 864*200
        ysize = 576*200

        # Number of grid points LES
        itot = 864
        jtot = 576
        ktot = 128

        # Number of x,y MPI processes
        nprocx = 12 # 4 nodes, 72x72 points per task
        nprocy = 8

        # Output directory (boundaries are LARGE)
        #output_dir = '/nobackup/users/stratum/KNMI_testbed/cases/nudge_boundary_HARMONIE/'
        output_dir = '/projects/0/einf170/janssonf/run_large_domain_2/'

        # Harmonie data path (with yyyy/mm/dd/hh directory structure underneath)
        #data_path = '/nobackup/users/stratum/DOWA/DOWA_fulldomain/'
        data_path = '/projects/0/einf170/Harmonie_boundaries'


    # -----------------
    # Development domain
    # -----------------
    if False:
        # Start and end time (index in HARMONIE files)
        t0 = 8
        t1 = 10

        # Lower left corner LES domain in HARMONIE (m)
        x0 = 700000
        y0 = 1200000

        # Domain size LES (m)
        xsize = 64*500
        ysize = 64*500

        # Number of grid points LES
        itot = 64
        jtot = 64
        ktot = 128

        # Number of x,y MPI processes
        nprocx = 2
        nprocy = 2

        # Output directory (boundaries are LARGE)
        #output_dir = '/nobackup/users/stratum/KNMI_testbed/cases/nudge_boundary_HARMONIE/'
        output_dir = '/projects/0/einf170/janssonf/nudge_boundary_Harmonie_test/'

        # Harmonie data path (with yyyy/mm/dd/hh directory structure underneath)
        #data_path = '/nobackup/users/stratum/DOWA/DOWA_fulldomain/'
        data_path = '/projects/0/einf170/Harmonie_boundaries'



    # DALES constants (modglobal.f90)
    cd = dict(p0=1.e5, Rd=287.04, Rv=461.5, cp=1004., Lv=2.53e6)
    cd['eps'] = cd['Rv']/cd['Rd']-1.

    # LES grid
    #grid = Grid(xsize, ysize, zsize, itot, jtot, ktot)
    grid = Grid_stretched(xsize, ysize, itot, jtot, ktot, dz0=25, alpha=0.017)

    # Hybrid sigma grid tools
    grid_sig = hsg.Sigma_grid('data/H40_65lev.txt')

    # --------------------------
    # HARMONIE data
    # --------------------------
    date_hm = datetime.datetime(year=2017, month=8, day=19)
    dowa_pt = 'ptD'  # adjust according to year

    date = '{0:04d}{1:02d}{2:02d}'.format(date_hm.year, date_hm.month, date_hm.day)
    dir  = '{0:04d}/{1:02d}/{2:02d}/00'.format(date_hm.year, date_hm.month, date_hm.day)
    u  = xr.open_dataset('{0}/{1}/ua.Slev.his.NETHERLANDS.DOWA_40h12tg2_fERA5_{2}.{3}.nc' .format(data_path, dir, dowa_pt, date))
    v  = xr.open_dataset('{0}/{1}/va.Slev.his.NETHERLANDS.DOWA_40h12tg2_fERA5_{2}.{3}.nc' .format(data_path, dir, dowa_pt, date))
    T  = xr.open_dataset('{0}/{1}/ta.Slev.his.NETHERLANDS.DOWA_40h12tg2_fERA5_{2}.{3}.nc' .format(data_path, dir, dowa_pt, date))
    q  = xr.open_dataset('{0}/{1}/hus.Slev.his.NETHERLANDS.DOWA_40h12tg2_fERA5_{2}.{3}.nc'.format(data_path, dir, dowa_pt, date))
    ql = xr.open_dataset('{0}/{1}/clw.Slev.his.NETHERLANDS.DOWA_40h12tg2_fERA5_{2}.{3}.nc'.format(data_path, dir, dowa_pt, date))
    ps = xr.open_dataset('{0}/{1}/ps.his.NETHERLANDS.DOWA_40h12tg2_fERA5_{2}.{3}.nc'      .format(data_path, dir, dowa_pt, date))
    Ts = xr.open_dataset('{0}/{1}/sst.sfx.NETHERLANDS.DOWA_40h12tg2_fERA5_{2}.{3}.nc'     .format(data_path, dir, dowa_pt, date))

    # Select a sub-area around the LES domain, to speed up
    # calculations done over the entire HARMONIE grid
    def sel_sub_area(ds, x0, y0, xsize, ysize, margin=2500):
        return ds.sel(x=slice(x0-margin, x0+xsize+margin), y=slice(y0-margin, y0+ysize+margin))

    u  = sel_sub_area(u,  x0, y0, xsize, ysize)['ua' ]
    v  = sel_sub_area(v,  x0, y0, xsize, ysize)['va' ]
    T  = sel_sub_area(T,  x0, y0, xsize, ysize)['ta' ]
    q  = sel_sub_area(q,  x0, y0, xsize, ysize)['hus']
    ql = sel_sub_area(ql, x0, y0, xsize, ysize)['clw']
    ps = sel_sub_area(ps, x0, y0, xsize, ysize)['ps' ]
    Ts = sel_sub_area(Ts, x0, y0, xsize, ysize)['sst']

    # Store lat/lon on LES grid (to-do: use proj.4 to simply re-create HARMONIE projection on LES grid..)
    intp = ip.Grid_interpolator(u['x'].values, u['y'].values, None, grid.x, grid.y, None, grid.xh, grid.yh, None, x0, y0)
    lon_LES = intp.interpolate_2d(u['lon'].values, 'x', 'y')
    lat_LES = intp.interpolate_2d(u['lat'].values, 'x', 'y')
    np.save('{}/lon_LES'.format(output_dir), lon_LES)
    np.save('{}/lat_LES'.format(output_dir), lat_LES)

    
    print("corners")
    print("(%5.2f %5.2f)  (%5.2f %5.2f)"%(lat_LES[0,-1], lon_LES[0,-1], lat_LES[-1,-1], lon_LES[-1,-1]))
    print("(%5.2f %5.2f)  (%5.2f %5.2f)"%(lat_LES[0,0], lon_LES[0,0], lat_LES[-1,0], lon_LES[-1,0]))

    if (True):
        # Create hourly boundaries:
        for t in range(t0, t1+1):
            print('Processing t={0:>2d}:00 UTC'.format(t))
            start_time = datetime.datetime.now()

            # Load data from current time step (not really necessary, but otherwise
            # from here on some variable do have a time dimension, and some dont't....)
            # Also, nice time to drop the no longer necessary xarray stuff with `.values`
            u_t  = u [t,:,:,:].values
            v_t  = v [t,:,:,:].values
            T_t  = T [t,:,:,:].values
            qv_t = q [t,:,:,:].values
            ql_t = ql[t,:,:,:].values
            ps_t = ps[t,:,:  ].values
            Ts_t = Ts[t,:,:  ].values

            # Virtual temperature for height calculation
            Tv_t = T_t * (1+cd['eps']*qv_t - ql_t)

            # Calculate pressure and height on full and half HARMONIE grid levels
            ph = grid_sig.calc_half_level_pressure(ps_t)
            zh = grid_sig.calc_half_level_Zg(ph, Tv_t)
            p  = grid_sig.calc_full_level_pressure(ph)
            z  = grid_sig.calc_full_level_Zg(zh)

            # Conversions HARMONIE quantities -> LES
            exner  = (p[::-1]/cd['p0'])**(cd['Rd']/cd['cp'])
            exners = (ps_t/cd['p0'])**(cd['Rd']/cd['cp'])

            th_t   = T_t  / exner
            ths_t  = Ts_t / exners
            thl_t  = th_t - cd['Lv'] / (cd['cp'] * exner) * ql_t
            qt_t   = qv_t + ql_t

            # Mean profiles to init LES (prof.inp)
            if (t==t0):
                mean_u = np.zeros(grid.ktot, dtype=np.float)
                mean_v = np.zeros(grid.ktot, dtype=np.float)
                mean_t = np.zeros(grid.ktot, dtype=np.float)
                mean_q = np.zeros(grid.ktot, dtype=np.float)

            # Do the rest in yz slices per MPI task in the x-direction (otherwise memory -> BOEM!)
            blocksize_x = int(itot / nprocx)
            for mpiidx in range(0, nprocx):
                print('Processing mpiidx={}/{}'.format(mpiidx+1, nprocx))

                # Create the interpolator for HARMONIE -> LES
                sx = np.s_[mpiidx*blocksize_x:(mpiidx+1)*blocksize_x]
                intp  = ip.Grid_interpolator(u['x'].values, u['y'].values, z, grid.x[sx], grid.y, grid.z, grid.xh[sx], grid.yh, grid.zh, x0, y0)

                # Interpolate HARMONIE onto LES grid
                # `::-1` reverses the vertical dimension (HARMONIE's data
                # is aranged from top-to-bottom, LES from bottom-to-top
                u_LES    = intp.interpolate_3d(u_t   [::-1,:,:], 'xh', 'y',  'z')
                v_LES    = intp.interpolate_3d(v_t   [::-1,:,:], 'x',  'yh', 'z')
                thl_LES  = intp.interpolate_3d(thl_t [::-1,:,:], 'x',  'y',  'z')
                qt_LES   = intp.interpolate_3d(qt_t  [::-1,:,:], 'x',  'y',  'z')
                ths_LES  = intp.interpolate_2d(ths_t [:,:     ], 'x',  'y'      )

                # Write the LBCs in binary format for LES
                write_LBC(u_LES, v_LES, thl_LES, qt_LES, ths_LES, itot, jtot, nprocx, nprocy, mpiidx, t-t0, 0., iexpnr, output_dir)

                if (t==t0):
                    # Store mean profiles
                    mean_u[:] += np.mean(u_LES,   axis=(0,1))
                    mean_v[:] += np.mean(v_LES,   axis=(0,1))
                    mean_t[:] += np.mean(thl_LES, axis=(0,1))
                    mean_q[:] += np.mean(qt_LES,  axis=(0,1))

            if (t==t0):
                # Write initial profiles to prof.inp.expnr
                mean_u /= nprocx
                mean_v /= nprocx
                mean_t /= nprocx
                mean_q /= nprocx

                tke = 0.1 * np.ones_like(grid.z)
                write_initial_profiles(grid.z, mean_u, mean_v, mean_t, mean_q, tke, iexpnr, output_dir)

            # Statistics
            end_time = datetime.datetime.now()
            print('Elapsed = {}'.format(end_time-start_time))


    places = {"Cabauw" : (51.971, 4.927, 'k*'),
              #"deBilt" : (52.1018, 5.1780, 'rx'),
              "Deeblespoint": (13.1625, -59.4286, 'g+'),   # 13N09'45"  59W25'43" Barbados
              "Loobos": (52.1664833, 5.7435528, 'bo'),    }

    # Plot ~location of LES domain
    if True:
        fig  = pl.figure()
        proj = ccrs.LambertConformal(central_longitude=4.9, central_latitude=51.967)
        ax   = pl.axes(projection=proj)

        # Add coast lines et al.
        ax.coastlines(resolution='10m', linewidth=0.8, color='k')

        countries = cfeature.NaturalEarthFeature(
                category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none', zorder=100)
        ax.add_feature(countries, edgecolor='k', linewidth=0.8)

        lakes = cfeature.NaturalEarthFeature(
                category='physical', name='lakes', scale='50m', facecolor='none', zorder=100)
        ax.add_feature(lakes, edgecolor='k', linewidth=0.8)

        ax.set_extent([lon_LES.min()-4.1, lon_LES.max()+4.1, lat_LES.min()-4.1, lat_LES.max()+4.1], ccrs.PlateCarree())

        pc=ax.pcolormesh(u['lon'], u['lat'], u[t0,-1,:,:], transform=ccrs.PlateCarree(), cmap=pl.cm.RdBu_r)

        for n,p in places.items():
            pl.plot(p[1], p[0], p[2],  transform=ccrs.PlateCarree())

        pl.colorbar(pc)
        pl.show()


    if False:
        kLES = 0
        kHM = -1

        pl.figure()
        pl.subplot(221)
        pl.pcolormesh((u['x']-x0)/1000., (u['y']-y0)/1000., u[t,kHM,:,:], vmin=u_LES[:,:,kLES].min(), vmax=u_LES[:,:,kLES].max(), cmap=pl.cm.magma)
        pl.colorbar()
        pl.xlim(0,xsize/1000)
        pl.ylim(0,ysize/1000)

        pl.subplot(222)
        pl.pcolormesh(grid.xh/1000., grid.y/1000, u_LES[:,:,kLES].T, vmin=u_LES[:,:,kLES].min(), vmax=u_LES[:,:,kLES].max(), cmap=pl.cm.magma)
        pl.colorbar()
        pl.xlim(0,xsize/1000)
        pl.ylim(0,ysize/1000)

        pl.subplot(223)
        pl.pcolormesh((u['x']-x0)/1000., (u['y']-y0)/1000., v[t,kHM,:,:], vmin=v_LES[:,:,kLES].min(), vmax=v_LES[:,:,kLES].max(), cmap=pl.cm.magma)
        pl.colorbar()
        pl.xlim(0,xsize/1000)
        pl.ylim(0,ysize/1000)

        pl.subplot(224)
        pl.pcolormesh(grid.x/1000., grid.yh/1000, v_LES[:,:,kLES].T, vmin=v_LES[:,:,kLES].min(), vmax=v_LES[:,:,kLES].max(), cmap=pl.cm.magma)
        pl.colorbar()
        pl.xlim(0,xsize/1000)
        pl.ylim(0,ysize/1000)

        #pl.subplot(325)
        #pl.pcolormesh((u['x']-x0)/1000., (u['y']-y0)/1000., T[t,kHM,:,:], vmin=T_LES[:,:,kLES].min(), vmax=T_LES[:,:,kLES].max(), cmap=pl.cm.magma)
        #pl.colorbar()
        #pl.xlim(0,xsize/1000)
        #pl.ylim(0,ysize/1000)

        #pl.subplot(326)
        #pl.pcolormesh(grid.x/1000., grid.y/1000, T_LES[:,:,kLES].T, vmin=T_LES[:,:,kLES].min(), vmax=T_LES[:,:,kLES].max(), cmap=pl.cm.magma)
        #pl.colorbar()
        #pl.xlim(0,xsize/1000)
        #pl.ylim(0,ysize/1000)
