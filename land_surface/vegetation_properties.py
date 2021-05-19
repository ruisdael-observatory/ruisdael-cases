#
# This file is part of LASSIE.
#
# Copyright (c) 2019-2020 Bart van Stratum
#
# LASSIE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LASSIE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LASSIE.  If not, see <http://www.gnu.org/licenses/>.
#

from collections import namedtuple
import numpy as np

class _IFS_vegetation:
    def __init__(self):

        raw_data = np.array([
            # vt    z0m      z0h        rsmin cveg  gD         a_r     b_r     L_us  L_st  f_rs   rs  lai
            # 0     1        2          3     4     5          6       7       8     9     10     11  12
            [ 0,    0.250,   0.25e-2,   100,  0.90, 0.00/100., 5.558,  2.614,  10.0, 10.0, 0.05,  1,  3  ],    # 0  Crops, mixed farming
            [ 0,    0.200,   0.2e-2,    100,  0.85, 0.00/100., 10.739, 2.608,  10.0, 10.0, 0.05,  1,  2  ],    # 1  Short grass
            [ 1,    2.000,   2.0,       250,  0.90, 0.03/100., 6.706,  2.175,  40.0, 15.0, 0.03,  2,  5  ],    # 2  Evergreen needleleaf
            [ 1,    2.000,   2.0,       250,  0.90, 0.03/100., 7.066,  1.953,  40.0, 15.0, 0.03,  2,  5  ],    # 3  Deciduous needleleaf
            [ 1,    2.000,   2.0,       175,  0.90, 0.03/100., 5.990,  1.955,  40.0, 15.0, 0.03,  2,  5  ],    # 4  Deciduous broadleaf
            [ 1,    2.000,   2.0,       240,  0.99, 0.03/100., 7.344,  1.303,  40.0, 15.0, 0.035, 2,  6  ],    # 5  Evergreen broadleaf
            [ 0,    0.470,   0.47e-2,   100,  0.70, 0.00/100., 8.235,  1.627,  10.0, 15.0, 0.05,  1,  2  ],    # 6  Tall grass
            [ -1,   0.013,   0.013e-2,  250,  0,    0.00/100., 4.372,  0.978,  15.0, 15.0, 0.00,  1,  0.5],    # 7  Desert
            [ 0,    0.034,   0.034e-2,  80,   0.50, 0.00/100., 8.992,  8.992,  10.0, 10.0, 0.05,  1,  1  ],    # 8  Tundra
            [ 0,    0.500,   0.5e-2,    180,  0.90, 0.00/100., 5.558,  2.614,  10.0, 10.0, 0.05,  1,  3  ],    # 9  Irrigated crops
            [ 0,    0.170,   0.17e-2,   150,  0.10, 0.00/100., 4.372,  0.978,  10.0, 10.0, 0.05,  1,  0.5],    # 10 Semidesert
            [ -1,   1.3e-10, 1.3e-2,    -1,   -1,   -1,        -1,     -1,     58.0, 58.0, 0.00,  0,  -1 ],    # 11 Ice caps and glaciers
            [ 0,    0.830,   0.83e-2,   240,  0.60, 0.00/100., 7.344,  1.303,  10.0, 10.0, 0.05,  1,  0.6],    # 12 Bogs and marshes
            [ -1,   -1   ,   -1,        -1,   -1,   -1,        -1,     -1,     -1,   -1,   0.00,  0,  -1 ],    # 13 Inland water
            [ -1,   -1   ,   -1,        -1,   -1,   -1,        -1,     -1,     -1,   -1,   0.00,  0,  -1 ],    # 14 Ocean
            [ 0,    0.100,   0.1e-2,    225,  0.50, 0.00/100., 6.326,  1.567,  10.0, 10.0, 0.05,  1,  3  ],    # 15 Evergreen shrubs
            [ 0,    0.250,   0.25e-2,   225,  0.50, 0.00/100., 6.326,  1.567,  10.0, 10.0, 0.05,  1,  1.5],    # 16 Deciduous shrubs
            [ 1,    2.000,   2.0,       250,  0.90, 0.03/100., 4.453,  1.631,  40.0, 15.0, 0.03,  2,  5  ],    # 17 Mixed forest- Wood
            [ 1,    1.100,   1.1,       175,  0.90, 0.03/100., 4.453,  1.631,  40.0, 15.0, 0.03,  2,  2.5],    # 18 Interrupted forest
            [ 0,    -1,      -1,        150,  0.60, 0.00/100., -1,     -1,     -1,   -1,   0.00,  0,  4  ],    # 19 Water -land mixtures
            [ 0,    1.0,     1.0,       100,  0.50, 0.00/100., 10.739, 2.608,  30.0, 30.0, 0.00,  1,  2  ],    # 20 "Urban"
            [ 0,    0.1,     0.1e-2,    1e9,  0.00, 0.00/100., 5.558,  2.614,  30.0, 30.0, 0.00,  1,  0  ],    # 21 "Roads"
            [ 0,    0.1,     0.1e-2,    1e9,  0.00, 0.00/100., 5.558,  2.614,   0.0,  0.0, 0.00,  1,  0  ]])   # 22 Water

        self.z0m = raw_data[:,1]
        self.z0h = raw_data[:,2]
        self.rs_min = raw_data[:,3]
        self.c_veg = raw_data[:,4]
        self.gD = raw_data[:,5]
        self.a_r = raw_data[:,6]
        self.b_r = raw_data[:,7]
        self.lambda_us = raw_data[:,8]
        self.lambda_s = raw_data[:,9]
        self.f_rs = raw_data[:,10]
        self.rs = raw_data[:,11]
        self.lai = raw_data[:,12]

        self.name = [
                 'crops_mixed_farming', 'short_grass', 'evergreen_needleleaf', 'deciduous_needleleaf',
                 'deciduous_broadleaf', 'evergreen_broadleaf', 'tall_grass', 'desert', 'tundra',
                 'irrigated_crops', 'semidesert', 'ice_caps_glaciers', 'bogs_marshes', 'inland_water',
                 'ocean', 'evergreen_shrubs', 'deciduous_shrubs', 'mixed_forest_wood', 'interrupted_forest',
                 'water_land_mixtures', 'urban', 'road']

ifs_vegetation = _IFS_vegetation()

#
# Lookup table: land-use name -> index in Top10NL dataset
#
top10_ids = {
    'bos_loofbos': 1,
    'bos_naaldbos': 2,
    'bos_gemengd_bos': 3,
    'bos_griend': 4,
    'populieren': 5,
    'boomkwekerij': 6,
    'fruitkwekerij': 7,
    'boomgaard': 8,
    'heide': 9,
    'grasland': 10,
    'akkerland': 11,
    'zand': 12,
    'duin': 13,
    'water_waterloop': 14,
    'water_meer_plas': 15,
    'water_zee': 16,
    'water_droogvallend': 17,
    'water_droogvallend_LAT': 18,
    'aanlegsteiger': 19,
    'weg_verhard': 20,
    'weg_onverhard': 21,
    'weg_half_verhard': 22,
    'weg_onbekend': 23,
    'spoorbaanlichaam': 24,
    'basaltblokken_steenglooiing': 25,
    'dodenakker': 26,
    'dodenakker_met_bos': 27,
    'braakliggend': 28,
    'bebouwd_gebied': 29,
    'overig': 30}

# Reverse of top10_ids table:
top10_names = {v: k for k, v in top10_ids.items()}

#
# Lookup table: Top10NL index -> IFS vegetation index
#
top10_to_ifs = {
    # High vegetation:
     1 : 4,     # loofbos -> decidious broadleaf
     2 : 2,     # naaldbos -> evergreen needleleaf
     3 : 17,    # gemengd bos -> mixed forest-wood
     4 : 16,    # griend -> decidious shrubs
     5 : 4,     # populieren -> decidious broadleaf
     6 : 16,    # boomkwekerij -> decidious shrubs
     7 : 16,    # fruitkwekerij -> decidious shrubs
     8 : 16,    # boomgaard -> decidious shrubs

    # Low vegetation:
     9 : 8,      # heide -> tundra
    10 : 1,      # grasland -> short grass
    11 : 9,      # akkerland -> irrigated crops
    12 : 10,     # zand -> semi-desert
    13 : 10,     # duin -> semi-desert

    # 14 -> 19 = water..
    14: 22,
    15: 22,
    16: 22,
    17: 22,
    18: 22,
    19: 22,

    # Road et al.:
    20 : 21,     # weg_verhard -> road
    21 : 10,     # weg_onverhard -> semi-desert
    22 : 10,     # weg_half_verhard -> semi-desert
    23 : 21,     # weg_onbekend -> road
    24 : 21,     # spoorbaanlichaam -> road

    # Misc:
    25 : 10,     # basaltblokken_steenglooiing -> semi-desert
    26 : 10,     # dodenakker -> semi-desert
    27 : 17,     # dodenakker met bos -> mixed forest-wood
    28 : 1,      # braakliggend -> short grass

    # Urban
    29 : 20,    # bebouwd gebied -> urban
    30 : 20}    # overig (=mostly urban) -> urban


if __name__ == '__main__':

    # Check/print lookup tables
    for i in range(1,31):
        ii = top10_to_ifs[i]
        ifs_name = ifs_vegetation.name[ii] if ii>0 else 'None'
        print('{0:30s} ({1:2d}) = {2}'.format(top10_names[i], i, ifs_name))
