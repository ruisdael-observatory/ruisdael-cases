import numpy as np
from numba import jit, prange
from progress.bar import Bar
import sys

@jit(nopython=True, nogil=True, fastmath=True)
def _nearest(array, value, size):
    """
    Get index of nearest `value` in `array`
    """
    min_dist = 1e12
    i_min = -1
    for i in range(size):
        dist = np.abs(array[i]-value)
        if dist < min_dist:
            min_dist = dist
            i_min = i
    return i_min


@jit(nopython=True, nogil=True, fastmath=True)
def _interp_block(
        field_out, frac_out, x_out, y_out, field_in, x_in, y_in, count, valid_codes, max_code,
        blocksize_x, blocksize_y, bi, bj, nn):
    """
    Interpolate sub-block of `field_in` onto `field_out`, using the most occuring
    value in `field_in` on a stencil of +/-nn grid points.
    """

    x_in_size = x_in.size
    y_in_size = y_in.size
    n_valid_codes = valid_codes.size
    stencil_size = (2*nn+1)**2

    # NOTE to myself: don't run this in parallel...
    for sj in range(blocksize_y):
        for si in range(blocksize_x):

            # Index in output data:
            io = bi*blocksize_x + si
            jo = bj*blocksize_y + sj

            # Find nearest grid point in global data:
            ig = _nearest(x_in, x_out[jo,io], x_in_size)
            jg = _nearest(y_in, y_out[jo,io], y_in_size)

            # Reset counter:
            for i in range(max_code+1):
                count[i] = 0

            # Loop over stencil in global (field_in) field:
            for j in range(jg-nn, jg+nn+1):
                for i in range(ig-nn, ig+nn+1):
                    code = field_in[j,i]

                    for k in range(n_valid_codes):
                        if code == valid_codes[k]:
                            count[code] += 1
                            break

            # Find most dominant type/code:
            max_count = 0
            max_val = -1
            for i in range(max_code+1):
                if count[i] > max_count:
                    max_count = count[i]
                    max_val = i

            field_out[jo,io] = max_val
            frac_out [jo,io] = max_count / stencil_size


def interp_dominant(x_out, y_out, field_in, valid_codes, max_code, nn, nblockx, nblocky, dx):
    """
    Interpolate field (field_in), using the most occuring value in a stencil of +/-nn grid points.
    Interpolation is done in `nblockx * nblocky` blocks, to prevent memory -> BOEM.
    """

    jtot = x_out.shape[0]
    itot = x_out.shape[1]

    blocksize_x = itot//nblockx
    blocksize_y = jtot//nblocky

    field_out = np.zeros((jtot, itot), dtype=int)
    frac_out  = np.zeros((jtot, itot), dtype=float)
    count     = np.zeros(max_code+1, dtype=int)

    suffix = '%(percent).0f%% | elapsed=%(elapsed).2fs | eta=%(eta).2fs | %(index)d/%(max)d'
    bar = Bar('Interpolating', max=nblocky*nblockx, suffix=suffix)
    for bj in range (nblocky):
        for bi in range(nblockx):
            ss = np.s_[bj*blocksize_y:(bj+1)*blocksize_y, bi*blocksize_x:(bi+1)*blocksize_x]

            # Get bounds of current sub-block:
            x0 = x_out[ss].min()-1000
            x1 = x_out[ss].max()+1000
            y0 = y_out[ss].min()-1000
            y1 = y_out[ss].max()+1000

            # Slice the input field:
            field_loc = field_in.sel(x=slice(x0,x1), y=slice(y0,y1))

            # Read data to memory, and remove NANs:
            data = field_loc.values.astype(int)
            data[np.isnan(data)] = -1
            x = field_loc.x.values
            y = field_loc.y.values

            # Interpolate using fast Numba kernel:
            _interp_block(
                    field_out, frac_out, x_out, y_out,
                    data, x, y, count,
                    valid_codes, max_code,
                    blocksize_x, blocksize_y, bi, bj, nn)

            bar.next()
    bar.finish()

    return field_out, frac_out


@jit(nopython=True, nogil=True, fastmath=True)
def interp_soil(
        soil_index, z_soil, bf_code,
        soil_id_lu, z_mid_prof, n_layers, lookup_index,
        itot, jtot, ktot):
    """
    Interpolate BOFEK soil profiles onto LES grid
    """

    for k in range(ktot):
        for j in range(jtot):
            for i in range(itot):
                if bf_code[j,i] > 0:
                    i_bf = soil_id_lu[bf_code[j,i]]

                    # Find nearest layer in soil profile
                    min_dist = 1e12
                    k_min = 0
                    for kk in range(n_layers[i_bf]):
                        dist = np.abs(z_mid_prof[i_bf,kk]-z_soil[k])
                        if dist < min_dist:
                            min_dist = dist
                            k_min = kk

                    soil_index[k,j,i] = lookup_index[i_bf,k_min]


@jit(nopython=True, nogil=True, fastmath=True)
def _find_left_index(arr, value, n):
    for i in range(n):
        if i == n-1:
            return n-2
        if arr[i] <= value and arr[i+1] > value:
            return i
    return 0


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _calc_era5_interpolation_factors(i0, j0, fx0, fy0, lon, lat, e5_lon, e5_lat, itot, jtot):
    n_e5_lon = e5_lon.size
    n_e5_lat = e5_lat.size

    for j in prange(jtot):
        for i in range(itot):
            i0[j,i] = _find_left_index(e5_lon, lon[j,i], n_e5_lon)
            j0[j,i] = _find_left_index(e5_lat, lat[j,i], n_e5_lat)

            lon_l = e5_lon[i0[j,i]  ]
            lon_r = e5_lon[i0[j,i]+1]

            lat_l = e5_lat[j0[j,i]  ]
            lat_r = e5_lat[j0[j,i]+1]

            fx0[j,i] = 1.-(lon[j,i]-lon_l) / (lon_r-lon_l)
            fy0[j,i] = 1.-(lat[j,i]-lat_l) / (lat_r-lat_l)


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _interpolate_era5(fld_out, fld_in, i0, j0, fx0, fy0, itot, jtot):
    for j in prange(jtot):
        for i in range(itot):
            # Short-cuts:
            ii = i0[j,i]
            jj = j0[j,i]

            fx = fx0[j,i]
            fy = fy0[j,i]

            # Bilinear interpolation:
            fld_out[j,i] =  fy  * (fx * fld_in[jj,  ii] + (1.-fx) * fld_in[jj,  ii+1]) + \
                        (1.-fy) * (fx * fld_in[jj+1,ii] + (1.-fx) * fld_in[jj+1,ii+1])


class Interpolate_era5:
    def __init__(self, lon_model, lat_model, lon_era, lat_era, itot, jtot):

        self._itot = itot
        self._jtot = jtot

        self._i0 = np.zeros((jtot, itot), dtype=int)
        self._j0 = np.zeros((jtot, itot), dtype=int)

        self._fx0 = np.zeros((jtot, itot), dtype=float)
        self._fy0 = np.zeros((jtot, itot), dtype=float)

        # Calculate interpolation factors
        _calc_era5_interpolation_factors(
                self._i0, self._j0, self._fx0, self._fy0,
                lon_model, lat_model, lon_era, lat_era, itot, jtot)

    def interpolate(self, fld_out, fld_in):

        # Call Numba kernel for fast interpolation
        _interpolate_era5(
                fld_out, fld_in, self._i0, self._j0,
                self._fx0, self._fy0, self._itot, self._jtot)
