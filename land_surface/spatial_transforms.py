import numpy as np
import pyproj

#
# Proj.4 Rijksdriehoekscoordinaten
#
proj4_str_rd = '+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +towgs84=565.417,50.3319,465.552,-0.398957,0.343988,-1.8774,4.0725 +units=m +no_defs'
proj4_rd = pyproj.Proj(proj4_str_rd, preserve_units=True)

#
# Proj.4 HARMONIE grid
#
proj4_str_hm = '+proj=lcc +lat_1=52.500000 +lat_2=52.500000 +lat_0=52.500000 +lon_0=.000000 +k_0=1.0 +x_0=649536.512574 +y_0=1032883.739533 +a=6371220.000000 +b=6371220.000000'
proj4_hm = pyproj.Proj(proj4_str_hm, preserve_units=True)
