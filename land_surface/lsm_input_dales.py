#import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np
import sys, os


class LSM_input_DALES:
    """
    Data structure for the required input for the new LSM
    """
    def __init__(self, itot, jtot, ktot, lu_types, parnames, debug=False):
        dtype_float = np.float64
        dtype_int   = np.int32
        dtype_str   = object

        self.itot = itot
        self.jtot = jtot
        self.ktot = ktot
        self.nlu  = len(lu_types)

        # Grid
        self.x = np.zeros(itot, dtype=dtype_float)
        self.y = np.zeros(jtot, dtype=dtype_float)

        self.lat = np.zeros((jtot, itot), dtype=dtype_float)
        self.lon = np.zeros((jtot, itot), dtype=dtype_float)

        # Soil temperature, moisture content, and index in van Genuchten lookup table.
        self.t_soil     = np.zeros((ktot, jtot, itot), dtype=dtype_float)
        self.theta_soil = np.zeros((ktot, jtot, itot), dtype=dtype_float)
        self.index_soil = np.zeros((ktot, jtot, itot), dtype=dtype_int)
        fields = ['index_soil', 't_soil', 'theta_soil']
        
        # LU types
        self.luname  = np.empty(len(lu_types), dtype=dtype_str)
        self.lushort = np.empty(len(lu_types), dtype=dtype_str)
        self.lveg    = np.empty(len(lu_types), dtype=dtype_str)
        self.laqu    = np.empty(len(lu_types), dtype=dtype_str)
        self.ilu     = np.zeros(len(lu_types), dtype=dtype_int)
            
        # Sub-grid fraction of LU type (-)
        zeros = np.zeros((jtot, itot), dtype=dtype_float)
        for lu in lu_types:
            for parname in parnames:
                fields = self._setpar(fields, parname, lu, zeros)
        
        # total land use cover
        self.c_tot = np.zeros((jtot, itot), dtype=dtype_float)
        fields.append('cover_tot')
        self.c_veg_tot = np.zeros((jtot, itot), dtype=dtype_float)
        fields.append('c_veg_tot')
 
        # List of fields which are written to the binary input files for DALES
        self.fields = sorted(fields)
 
        # Bonus, for offline LSM (not written to DALES input)
        for lu in lu_types:
            ## LU type (-)
            varname = 'type_'+lu
            setattr(self, varname, zeros) 

        # if debug:
        #     # Init all values at a large negative number
        #     for field in self.fields:
        #         data = getattr(self, field)
        #         data[:] = -1e9

        if debug:
            # Init all values at NaN
            for field in self.fields:
                data = getattr(self, field)
                if data.dtype == dtype_int:
                    continue
                data[:] = np.nan


    def _setpar(self, fields, parname, lu, zeros):
        '''
        Function that initializes parameter fields for each land use type

        Parameters
        ----------
        fields : list
            List of parameter fields.
        parname : str
            Parameter name.
        lu : str
            Land use name.
        zeros : np.array
            Array with zeros.

        Returns
        -------
        fields : list
            List of parameter fields.

        '''
        varname = '_'.join([parname,lu])
        setattr(self, varname, zeros) 
        fields.append(varname)
        return fields
    
    
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
        return


    def save_netcdf(self, nc_file):
        """
        Save to NetCDF for visualisation et al.
        """
        nc = nc4.Dataset(nc_file, 'w')

        nc.createDimension('x', self.itot)
        nc.createDimension('y', self.jtot)
        nc.createDimension('z', self.ktot)
        nc.createDimension('nlu', self.nlu)
        nc.createDimension('str3', size=3)
        nc.createDimension('str32', size=32)
        nc.createDimension('str1', size=1)


        var_x = nc.createVariable('x', float, 'x')
        var_y = nc.createVariable('y', float, 'y')

        var_x[:] = self.x[:]
        var_y[:] = self.y[:]

        # # Fields needed for offline LSM:
        # bonus = ['type_lv', 'type_hv', 'type_bs', 'type_aq', 'type_ap', 'lon', 'lat']
        bonus = ['lon', 'lat']
        
        for field in self.fields + bonus:
            data = getattr(self, field)
            dims = ['y', 'x'] if data.ndim == 2 else ['z', 'y', 'x']
            var  = nc.createVariable(field, float, dims)
            var[:] = data[:]

        luname = nc4.stringtochar(np.array(self.luname, 'S32'))
        var_lun = nc.createVariable('luname', 
                                    datatype='S1', 
                                    dimensions=('nlu','str32'))
        var_lun[:,:] = luname

        lushort = nc4.stringtochar(np.array(self.lushort,'S3'))
        var_lus = nc.createVariable('lushort', 
                                    datatype='S1', 
                                    dimensions=('nlu','str3'))
        var_lus[:,:] = lushort
        
        lveg    = nc4.stringtochar(np.array([str(b)[0] for b in self.lveg],'S1'))
        var_lveg = nc.createVariable('lveg',
                                     datatype='S1',
                                     dimensions=('nlu','str1'))
        var_lveg[:,:] = lveg
        
        laqu    = nc4.stringtochar(np.array([str(b)[0] for b in self.laqu],'S1'))
        var_laqu = nc.createVariable('laqu',
                                     datatype='S1',
                                     dimensions=('nlu','str1'))
        var_laqu[:,:] = laqu

        ilu     = np.array(self.ilu, dtype=int)
        var_ilu = nc.createVariable('ilu',int,('nlu',))
        var_ilu[:] = ilu
        
        nc.close()
 
        return


if __name__ == '__main__':
    """ Just for testing... """

    from landuse_types import lu_types_basic, lu_types_depac

    itot = 128
    jtot = 128
    kmax_soil = 4
    lu_types = lu_types_depac
    parnames_lsm = ['cover','c_veg','z0m','z0h','lai','ar','br',
                'lambda_s','lambda_us','rs_min','gD','tskin']
    parnames_dep = ['R_inc_b','R_inc_h','SAI_a','SAI_b',
                    'fmin','alpha','Tmin','Topt','Tmax','gs_max',
                    'vpd_min','vpd_max','R_soil','gamma_stom','gamma_soil_c_fac',
                    'gamma_soil_default','R_soilwet','R_soilfrozen']
    parnames = parnames_lsm + parnames_dep
        
    lsm_input = LSM_input_DALES(itot=itot, 
                                jtot=jtot, 
                                ktot=kmax_soil, 
                                lu_types=lu_types,
                                parnames = parnames,
                                debug=False)
    lsm_input.x[:] = np.arange(itot)
    lsm_input.y[:] = np.arange(itot)

    # ... set values ...

    # # Save DALES input:
    # lsm_input.save_binaries(nprocx=4, nprocy=2, exp_id=8, path='tmp/')

    # Save NetCDF output (for e.g. visualisation):
    lsm_input.save_netcdf('tmp/test1.nc')
