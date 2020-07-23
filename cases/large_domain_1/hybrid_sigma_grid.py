import numpy as np
from numba import jit


class Sigma_grid:
    """
    Various tools to calculate e.g. properties of the vertical IFS/ERA/HARMONIE grid
    NOTE: for optimization, most functions operate on 2D or 3D fields, using Numpy's
    vectorized instructions. Shape of the input arrays is assumed to {z,y,x} or {y,x}!
    """
    def __init__(self, grid_description):

        # Constants (see IFS part IV, chapter 12)
        print('BvS; reminder, still using IFS constants....')
        self.grav  = 9.80665
        self.Rd    = 287.0597
        self.Rv    = 461.5250
        self.eps   = self.Rv/self.Rd-1.
        self.cpd   = 1004.7090
        self.Lv    = 2.5008e6

        # Read the table with the vertical grid properties/parameters
        f = np.loadtxt(grid_description)

        # Half and full level number
        self.nh  = f[:,0]
        self.nf  = 0.5 * (self.nh[1:] + self.nh[:-1])

        # Reverse all arrays (::-1) from top-to-bottom to bottom-to-top
        self.a   = f[ :,1][::-1]       # a-coefficient pressure calculation
        self.b   = f[ :,2][::-1]       # b-coefficient pressure calculation


    def calc_half_level_pressure(self, ps):
        """
        Calculate half level pressure
        See IFS part III, eq. 2.11
        Equation: p = a + b * ps
        Top value is set to a small non-zero number to prevent div-by-0's
        Keyword arguments: 
            ps -- surface pressure (Pa)
        """

        ph = self.a[:,None,None] + self.b[:,None,None] * ps[:,:]
        ph[-1,:,:] = 10   # Optimized for HARMONIE grid...

        return ph


    def calc_full_level_pressure(self, ph):
        """
        Calculate full level pressure as a linear interpolation of the half level pressure
        Equation: p = a + b * ps
        See IFS part III, eq. 2.11
        Keyword arguments: 
            ps -- surface pressure (Pa)
        """

        return 0.5 * (ph[1:,:,:] + ph[:-1,:,:])


    def calc_half_level_Zg(self, ph, Tv):
        """
        Calculate half level geopotential height
        Equation: sums dZg = -Rd / g * Tv * ln(p+ / p-)
        See IFS part III, eq. 2.20-2.21
        Keyword arguments: 
            ph -- half level pressure (Pa)
            Tv -- full level virtual temperature (K) 
        """

        Zg = np.zeros_like(ph, dtype=np.float)
    
        pfrac = ph[1:,:,:] / ph[:-1,:,:]
        dZg   = -self.Rd * Tv * np.log(pfrac) / self.grav
        Zg[1:,:,:] = np.cumsum(dZg, axis=0) 
    
        return Zg


    def calc_full_level_Zg(self, Zgh):
        """
        Calculate full level geopotential height
        Equation: sums dZg = -Rd / g * Tv * ln(p+ / p-)
        See IFS part III, eq. 2.20-2.21
        Keyword arguments: 
            ph -- half level pressure (Pa)
            Tv -- full level virtual temperature (K) 
        """

        return 0.5 * (Zgh[1:,:,:] + Zgh[:-1,:,:])


    def calc_virtual_temp(self, T, qv, ql=0, qi=0, qr=0, qs=0):
        """
        Calculate the virtual temperature
        Equation: Tv = T * ([Rv/Rd-1]*qv - ql - qi - qr - qs)
        See IFS part IV, eq. 12.6
        Keyword arguments:
            T -- absolute temperature (K)
            q* -- specific humidities (kg kg-1):
                qv = vapor
                ql = liquid (optional)
                qi = ice    (optional)
                qr = rain   (optional)
                qs = snow   (optional)
        """

        return T * (1+self.eps*qv - ql - qi - qr - qs)


    def calc_exner(self, p):
        return (p/1e5)**(self.Rd/self.cpd)
