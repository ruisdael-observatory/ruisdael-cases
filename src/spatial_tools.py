import numpy as np

r_earth = 6367.47 * 1000    # Radius earth (m)

def dx(lonW, lonE, lat):
    """ Distance between longitudes in spherical coordinates """
    return r_earth * np.cos(np.deg2rad(lat)) * np.deg2rad(lonE - lonW)

def dy(latS, latN):
    """ Distance between latitudes in spherical coordinates """
    return r_earth * np.deg2rad(latN-latS)

def dlon(dx, lat):
    """ East-west distance in degrees for given dx """
    return np.rad2deg(dx / (r_earth * np.cos(np.deg2rad(lat))))

def dlat(dy):
    """ North-south distance in degrees for given dy """
    return np.rad2deg(dy / r_earth)


def find_nearest_latlon(lats, lons, goal_lat, goal_lon, silent=False):
    """
    Find nearest latitude/longitude in 2D lat/lon grid
    """

    dist_x = np.absolute(dx(lons, goal_lon, lats))
    dist_y = np.absolute(dy(lats, goal_lat))
    dist   = (dist_x**2. + dist_y**2)**0.5

    j,i = np.unravel_index(dist.argmin(), dist.shape)
    
    if not silent:
        print('Requested lat/lon = {0:.2f}/{1:.2f}, using {2:.2f}/{3:.2f}, distance = {4:.2f} km (i,j={5:},{6:})'\
            .format(goal_lat, goal_lon, lats[j,i], lons[j,i], dist.min()/1000., j, i))

    return j,i
