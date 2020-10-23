#
# Fast interpolation of HARMONIE data to LES grid
# Uses the Numba library with JIT compilation to speed the calculations up.
# Bart van Stratum (KNMI), Dec. 2018
#

import numpy as np
from numba import jit

@jit(nopython=True, nogil=True)
def interpolate_kernel_3d(field_LES, field_LS, i0, j0, k0, ifac, jfac, kfac):
    """
    Tri-linear interpolation of HARMONIE field onto LES grid
    Kept out of class to allow acceleration with Numba
    """

    itot = field_LES.shape[0]
    jtot = field_LES.shape[1]
    ktot = field_LES.shape[2]

    for i in range(itot):
        for j in range(jtot):
            for k in range(ktot):

                il = i0[i]
                jl = j0[j]

                field_LES[i,j,k] = jfac[j]     * (  ifac[i]  * ( kfac[il,   jl,   k]  * field_LS[ k0[il,   jl,   k],   jl,   il   ] +    \
                                                              (1-kfac[il,   jl,   k]) * field_LS[ k0[il,   jl,   k]+1, jl,   il   ] )  + \
                                                 (1-ifac[i]) * ( kfac[il+1, jl,   k]  * field_LS[ k0[il+1, jl,   k],   jl,   il+1 ] +    \
                                                              (1-kfac[il+1, jl,   k]) * field_LS[ k0[il+1, jl,   k]+1, jl,   il+1 ] )) + \
                                   (1-jfac[j]) * (  ifac[i]  * ( kfac[il,   jl+1, k]  * field_LS[ k0[il,   jl+1, k],   jl+1, il   ] +    \
                                                              (1-kfac[il,   jl+1, k]) * field_LS[ k0[il,   jl+1, k]+1, jl+1, il   ] )  + \
                                                 (1-ifac[i]) * ( kfac[il+1, jl+1, k]  * field_LS[ k0[il+1, jl+1, k],   jl+1, il+1 ] +    \
                                                              (1-kfac[il+1, jl+1, k]) * field_LS[ k0[il+1, jl+1, k]+1, jl+1, il+1 ] ))


@jit(nopython=True, nogil=True)
def interpolate_kernel_2d(field_LES, field_LS, i0, j0, ifac, jfac):
    """
    Bi-linear interpolation of HARMONIE field onto LES grid
    Kept out of class to allow acceleration with Numba
    """

    itot = field_LES.shape[0]
    jtot = field_LES.shape[1]

    for i in range(itot):
        for j in range(jtot):

            il = i0[i]
            jl = j0[j]

            field_LES[i,j] = jfac[j]     * (ifac[i] * field_LS[jl  , il] + (1-ifac[i]) * field_LS[jl  , il+1]) + \
                             (1-jfac[j]) * (ifac[i] * field_LS[jl+1, il] + (1-ifac[i]) * field_LS[jl+1, il+1])


@jit(nopython=True, nogil=True)
def calc_horz_interpolation_factors(i0, fi, x_LS, x):
    """
    Calculate the interpolation settings (horizontal):
    i0 : first index in x_LS left of each LES grid point
    fi : interpolation factor (phi_int = fi * phi_LS[i0] + (1-fi) * phi_LS[i0+1]
    """

    for i in range(x.size):
        i0[i] = np.where(x_LS - x[i] <= 0)[0][-1]
        fi[i] = 1.-((x[i] - x_LS[i0[i]]) / (x_LS[i0[i]+1] - x_LS[i0[i]]))


@jit(nopython=True, nogil=True)
def calc_vert_interpolation_factors(k0, fk, z_LS, z):
    """
    Calculate the interpolation settings (vertical):
    In HARMONIE (and others) the vertical grid differs in space,
    so all interpolation factors are 3D fields :-(

    k0 : first index in x_LS below each LES grid point
    fk : interpolation factor (phi = fi * phi_LS[i0] + (1-fi) * phi_LS[i0+1]
    """

    for i in range(k0.shape[0]):
        for j in range(k0.shape[1]):
            for k in range(z.size):
                if z[k] >= z_LS[0,j,i]:
                    k0[i,j,k] = np.where(z_LS[:,j,i] - z[k] <= 0)[0][-1]
                else:
                    k0[i,j,k] = 0

                fk[i,j,k] = 1.-((z[k] - z_LS[k0[i,j,k],j,i]) / (z_LS[k0[i,j,k]+1,j,i] - z_LS[k0[i,j,k],j,i]))


class Grid_interpolator:
    """
    Interpolate functions from a non-staggered grid (e.g. Harmonie; x_LS, y_LS, z_LS)
    to a higher resolution staggered grid (LES)
    """
    def __init__(self, x_LS, y_LS, z_LS, x, y, z, xh, yh, zh, x0, y0):

        # Calculate interpolation indices and factors
        # Full level (grid center):
        self.i0   = np.zeros_like(x, dtype=np.int)
        self.ifac = np.zeros_like(x, dtype=np.float)

        self.j0   = np.zeros_like(y, dtype=np.int)
        self.jfac = np.zeros_like(y, dtype=np.float)

        calc_horz_interpolation_factors(self.i0, self.ifac, x_LS, x+x0)
        calc_horz_interpolation_factors(self.j0, self.jfac, y_LS, y+y0)

        if z is not None:
            self.k0   = np.zeros((x_LS.size, y_LS.size, z.size), dtype=np.int)
            self.kfac = np.zeros((x_LS.size, y_LS.size, z.size), dtype=np.float)

            calc_vert_interpolation_factors(self.k0, self.kfac, z_LS, z)

        # Half level (grid edge):
        self.ih0   = np.zeros_like(x, dtype=np.int)
        self.ihfac = np.zeros_like(x, dtype=np.float)

        self.jh0   = np.zeros_like(y, dtype=np.int)
        self.jhfac = np.zeros_like(y, dtype=np.float)

        calc_horz_interpolation_factors(self.ih0, self.ihfac, x_LS, xh+x0)
        calc_horz_interpolation_factors(self.jh0, self.jhfac, y_LS, yh+y0)

        #if z is not None:
        #    self.kh0   = np.zeros((x_LS.size, y_LS.size, z.size), dtype=np.int)
        #    self.khfac = np.zeros((x_LS.size, y_LS.size, z.size), dtype=np.float)

        #    calc_vert_interpolation_factors(self.kh0, self.khfac, z_LS, zh)


    def interpolate_3d(self, field_LS, locx, locy, locz):
        """
        Interpolate `field_LS` onto the LES grid, at the specified location (x={x,xh}, y={y,yh}, z={z,zh})
        """

        # Switch between full and half levels
        i0 = self.i0 if locx == 'x' else self.ih0
        j0 = self.j0 if locy == 'y' else self.jh0
        k0 = self.k0 if locz == 'z' else self.kh0

        ifac = self.ifac if locx == 'x' else self.ihfac
        jfac = self.jfac if locy == 'y' else self.jhfac
        kfac = self.kfac if locz == 'z' else self.khfac

        field_LES = np.zeros((i0.size, j0.size, k0.shape[2]), dtype=np.float)

        interpolate_kernel_3d(field_LES, field_LS, i0, j0, k0, ifac, jfac, kfac)

        return field_LES


    def interpolate_2d(self, field_LS, locx, locy):
        """
        Interpolate `field_LS` onto the LES grid, at the specified location (x={x,xh}, y={y,yh})
        """

        # Switch between full and half levels
        i0 = self.i0 if locx == 'x' else self.ih0
        j0 = self.j0 if locy == 'y' else self.jh0

        ifac = self.ifac if locx == 'x' else self.ihfac
        jfac = self.jfac if locy == 'y' else self.jhfac

        field_LES = np.zeros((i0.size, j0.size), dtype=np.float)

        interpolate_kernel_2d(field_LES, field_LS, i0, j0, ifac, jfac)

        return field_LES
