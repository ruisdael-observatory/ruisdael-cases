import matplotlib.pyplot as pl
import numpy as np

pl.close('all')

# Numerical grid
xsize = 64*200
ysize = 64*200
dx    = 200
dy    = 200
x     = np.arange(0.5*dx, xsize, dx)
y     = np.arange(0.5*dy, ysize, dy)

# Boundary settings
nudge_offset = 2500
nudge_width  = 600
nudge_radius = nudge_offset

perturb_offset = 3500
perturb_width  = 600
perturb_radius = perturb_offset



def corner_factor(x, y, x_center, y_center, radius, width):
    D = np.sqrt((x-x_center)**2 + (y-y_center)**2) - radius
    return np.exp(-0.5*(D/width)**2)

def calc_factor(x, y, xsize, ysize, offset, width, radius):

    # 2D factor:
    f = np.zeros((y.size, x.size), dtype=np.float)

    # Center of nudging area
    xcW = offset
    xcE = xsize - offset
    
    ycS = offset
    ycN = ysize - offset
    
    # Total size of the corners
    dc =  offset + radius

    # Calculate factor
    for i in range(x.size):
        for j in range(y.size):
    
            if y[j] < dc and x[i] < dc:
                # SW-corner
                f[j,i] += corner_factor(x[i], y[j], dc, dc, radius, width)
    
            elif y[j] < dc and x[i] > xsize-dc:
                # SE-corner
                f[j,i] += corner_factor(x[i], y[j], xsize-dc, dc, radius, width)
    
            elif y[j] > ysize-dc and x[i] < dc:
                # NW-corner
                f[j,i] += corner_factor(x[i], y[j], dc, ysize-dc, radius, width)
    
            elif y[j] > ysize-dc and x[i] > xsize-dc:
                # NE-corner
                f[j,i] += corner_factor(x[i], y[j], xsize-dc, ysize-dc, radius, width)
    
            else:
                f[j,i] += np.exp(-0.5*((x[i]-xcW)/width)**2)
                f[j,i] += np.exp(-0.5*((x[i]-xcE)/width)**2)
                f[j,i] += np.exp(-0.5*((y[j]-ycS)/width)**2)
                f[j,i] += np.exp(-0.5*((y[j]-ycN)/width)**2)

    return f


f_nudge = calc_factor(x, y, xsize, ysize, nudge_offset, nudge_width, nudge_radius)
f_pert  = calc_factor(x, y, xsize, ysize, perturb_offset, perturb_width, perturb_radius)


pl.figure()
ax=pl.subplot(131, aspect='equal')
pl.pcolormesh(x, y, f_nudge)
pl.colorbar()

ax=pl.subplot(132, aspect='equal')
pl.pcolormesh(x, y, f_pert)
pl.colorbar()

ax=pl.subplot(133)
pl.plot(x, f_nudge[int(y.size/2),:])
pl.plot(x, f_pert [int(y.size/2),:])

