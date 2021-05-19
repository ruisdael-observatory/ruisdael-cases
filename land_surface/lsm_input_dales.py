import matplotlib.pyplot as pl
import netCDF4 as nc4
import numpy as np
import sys, os

class LSM_input_DALES:
    """
    Data structure for the required input for the new LSM
    """
    def __init__(self, itot, jtot, ktot, debug=False):
        dtype_float = np.float64
        dtype_int   = np.int32

        self.itot = itot
        self.jtot = jtot
        self.ktot = ktot

        # List of fields which are written to the binary input files for DALES
        self.fields = [
                'c_lv', 'c_hv', 'c_bs', 'c_aq',
                'z0m_lv', 'z0m_hv', 'z0m_bs', 'z0m_aq',
                'z0h_lv', 'z0h_hv', 'z0h_bs', 'z0h_aq',
                'lambda_s_lv', 'lambda_s_hv', 'lambda_s_bs',
                'lambda_us_lv', 'lambda_us_hv', 'lambda_us_bs',
                'lai_lv', 'lai_hv' ,'rs_min_lv', 'rs_min_hv' ,'rs_min_bs',
                'ar_lv' ,'br_lv', 'ar_hv', 'br_hv', 'gD', 'tskin_aq',
                'index_soil', 't_soil', 'theta_soil']

        # Grid
        self.x = np.zeros(itot, dtype=dtype_float)
        self.y = np.zeros(jtot, dtype=dtype_float)

        self.lat = np.zeros((jtot, itot), dtype=dtype_float)
        self.lon = np.zeros((jtot, itot), dtype=dtype_float)

        # Soil temperature, moisture content, and index in van Genuchten lookup table.
        self.t_soil     = np.zeros((ktot, jtot, itot), dtype=dtype_float)
        self.theta_soil = np.zeros((ktot, jtot, itot), dtype=dtype_float)
        self.index_soil = np.zeros((ktot, jtot, itot), dtype=dtype_int)

        # Sub-grid fraction of vegetation (-)
        self.c_lv = np.zeros((jtot, itot), dtype=dtype_float)
        self.c_hv = np.zeros((jtot, itot), dtype=dtype_float)
        self.c_bs = np.zeros((jtot, itot), dtype=dtype_float)
        self.c_aq = np.zeros((jtot, itot), dtype=dtype_float)

        # Roughness lenghts momentum (m)
        self.z0m_lv = np.zeros((jtot, itot), dtype=dtype_float)
        self.z0m_hv = np.zeros((jtot, itot), dtype=dtype_float)
        self.z0m_bs = np.zeros((jtot, itot), dtype=dtype_float)
        self.z0m_aq = np.zeros((jtot, itot), dtype=dtype_float)

        # Roughness lenghts heat/scalars (m)
        self.z0h_lv = np.zeros((jtot, itot), dtype=dtype_float)
        self.z0h_hv = np.zeros((jtot, itot), dtype=dtype_float)
        self.z0h_bs = np.zeros((jtot, itot), dtype=dtype_float)
        self.z0h_aq = np.zeros((jtot, itot), dtype=dtype_float)

        # Conductivity skin layer (stable conditions)
        self.lambda_s_lv = np.zeros((jtot, itot), dtype=dtype_float)
        self.lambda_s_hv = np.zeros((jtot, itot), dtype=dtype_float)
        self.lambda_s_bs = np.zeros((jtot, itot), dtype=dtype_float)

        # Conductivity skin layer (unstable conditions)
        self.lambda_us_lv = np.zeros((jtot, itot), dtype=dtype_float)
        self.lambda_us_hv = np.zeros((jtot, itot), dtype=dtype_float)
        self.lambda_us_bs = np.zeros((jtot, itot), dtype=dtype_float)

        # Leaf Area Index (LAI, -)
        self.lai_lv = np.zeros((jtot, itot), dtype=dtype_float)
        self.lai_hv = np.zeros((jtot, itot), dtype=dtype_float)

        # Minimum vegetation (lv, hv) or soil resistance (s m-1)
        self.rs_min_lv = np.zeros((jtot, itot), dtype=dtype_float)
        self.rs_min_hv = np.zeros((jtot, itot), dtype=dtype_float)
        self.rs_min_bs = np.zeros((jtot, itot), dtype=dtype_float)

        # `a` and `b` coefficients root profile
        self.ar_lv = np.zeros((jtot, itot), dtype=dtype_float)
        self.br_lv = np.zeros((jtot, itot), dtype=dtype_float)

        self.ar_hv = np.zeros((jtot, itot), dtype=dtype_float)
        self.br_hv = np.zeros((jtot, itot), dtype=dtype_float)

        # TMP TMP TMP TMP
        self.tskin_aq = np.zeros((jtot, itot), dtype=dtype_float)

        # gD-coefficient for high vegetation
        self.gD = np.zeros((jtot, itot), dtype=dtype_float)

        # Bonus, for offline LSM (not written to DALES input)
        self.type_lv = np.zeros((jtot, itot), dtype=dtype_int)
        self.type_hv = np.zeros((jtot, itot), dtype=dtype_int)
        self.type_bs = np.zeros((jtot, itot), dtype=dtype_int)

        if debug:
            # Init all values at large negative number
            for field in self.fields:
                data = getattr(self, field)
                data[:] = -1e9


    def save_binaries(self, nprocx, nprocy, exp_id, path='.'):
        """
        Write all required input fields in binary format
        for DALES to `lsm.inp.x000y000.001` format.
        """

        blocksize_x = self.itot//nprocx
        blocksize_y = self.jtot//nprocy

        if not os.path.exists(path):
            sys.exit('DALES LSM output path \"{}\" does not exist!'.format(path))

        for i in range(nprocx):
            for j in range(nprocy):
                f_out = '{0}/lsm.inp.x{1:03d}y{2:03d}.{3:03d}'.format(path, i, j, exp_id)

                # Numpy slices of current MPI block in global data
                ss2 = np.s_[  j*blocksize_y:(j+1)*blocksize_y, i*blocksize_x:(i+1)*blocksize_x]
                ss3 = np.s_[:,j*blocksize_y:(j+1)*blocksize_y, i*blocksize_x:(i+1)*blocksize_x]

                with open(f_out, 'wb+') as f:
                    for field in self.fields:
                        data = getattr(self, field)
                        ss = ss2 if data.ndim==2 else ss3
                        data[ss].tofile(f)


    def save_netcdf(self, nc_file):
        """
        Save to NetCDF for visualisation et al.
        """
        nc = nc4.Dataset(nc_file, 'w')

        dimx = nc.createDimension('x', self.itot)
        dimy = nc.createDimension('y', self.jtot)
        dimz = nc.createDimension('z', self.ktot)

        var_x = nc.createVariable('x', np.float, 'x')
        var_y = nc.createVariable('y', np.float, 'y')

        var_x[:] = self.x[:]
        var_y[:] = self.y[:]

        # Fields needed for offline LSM:
        bonus = ['type_lv', 'type_hv', 'type_bs', 'lon', 'lat']

        for field in self.fields + bonus:
            data = getattr(self, field)
            dims = ['y', 'x'] if data.ndim == 2 else ['z', 'y', 'x']
            var  = nc.createVariable(field, np.float, dims)
            var[:] = data[:]

        nc.close()



if __name__ == '__main__':
    """ Just for testing... """

    lsm_input = LSM_input_DALES(itot=432, jtot=288, ktot=4)
    lsm_input.x[:] = np.arange(432)
    lsm_input.y[:] = np.arange(288)

    # ... set values ...

    # Save DALES input:
    lsm_input.save_binaries(nprocx=4, nprocy=2, exp_id=1, path='tmp/')

    # Save NetCDF output (for e.g. visualisation):
    lsm_input.save_netcdf('tmp/test.nc')
