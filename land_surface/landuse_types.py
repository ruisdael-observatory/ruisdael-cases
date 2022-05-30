import numpy as np

### DEFAULT
# define lu type characteristics
lv = {'lu_long':'low vegetation', 
      'lu_short':'lv', 
      'lu_ids': np.array([9,10,11,13,26]),  # without urban
      'lveg':True}
hv = {'lu_long':'high vegetation', 
      'lu_short':'hv', 
      'lu_ids': np.array([1,2,3,4,5,6,7,8,27]),
       'lveg':True}
aq = {'lu_long':'water surface', 
      'lu_short':'aq', 
      'lu_ids': np.array([14,15,16,17,18]),
      'lveg':False}
ap = {'lu_long':'asphalt', 
      'lu_short':'ap', 
      'lu_ids': np.array([20,22,23,24,25,29,30]),
      'lveg':False}
bs = {'lu_long':'bare soil', 
      'lu_short':'bs', 
      'lu_ids': np.array([12,28]),
      'lveg':False}

# basic set of land use types
lu_types_basic = {'lv':lv,
                  'hv':hv, 
                  'bs':bs, 
                  'aq':aq,
                  'ap':ap } 


### with buildings and asphalt
# define lu type characteristics
lv = {'lu_long':'low vegetation', 
      'lu_short':'lv', 
      'lu_ids': np.array([9,10,11,13,26]),  # without urban
      'lveg':True}
hv = {'lu_long':'high vegetation', 
      'lu_short':'hv', 
      'lu_ids': np.array([1,2,3,4,5,6,7,8,27]),
       'lveg':True}
aq = {'lu_long':'water surface', 
      'lu_short':'aq', 
      'lu_ids': np.array([14,15,16,17,18]),
      'lveg':False}
ap = {'lu_long':'asphalt', 
      'lu_short':'ap', 
      'lu_ids': np.array([20,22,23,24,25]),
      'lveg':False}
bu = {'lu_long':'buildings', 
      'lu_short':'bu', 
      'lu_ids': np.array([29,30]),
      'lveg':False}
bs = {'lu_long':'bare soil', 
      'lu_short':'bs', 
      'lu_ids': np.array([12,28]),
      'lveg':False}

# set of land use types with 'buildings'
lu_types_build = {'lv':lv,
                  'hv':hv, 
                  'bs':bs, 
                  'aq':aq,
                  'ap':ap,
                  'bu':bu} 

### with crops
# define lu type characteristics
lv = {'lu_long':'low vegetation', 
      'lu_short':'lv', 
      'lu_ids': np.array([9,10,11,13,26]),  # without urban, arable land and grassland
      'lveg':True}
hv = {'lu_long':'high vegetation', 
      'lu_short':'hv', 
      'lu_ids': np.array([1,2,3,4,5,6,7,8,27]),
       'lveg':True}
aq = {'lu_long':'water surface', 
      'lu_short':'aq', 
      'lu_ids': np.array([14,15,16,17,18]),
      'lveg':False}
ap = {'lu_long':'asphalt', 
      'lu_short':'ap', 
      'lu_ids': np.array([20,22,23,24,25]),
      'lveg':False}
bu = {'lu_long':'buildings', 
      'lu_short':'bu', 
      'lu_ids': np.array([29,30]),
      'lveg':False}
bs = {'lu_long':'bare soil', 
      'lu_short':'bs', 
      'lu_ids': np.array([12,28]),
      'lveg':False}
# cr = {'lu_long':'crops', 
#       'lu_short':'cr', 
#       'lu_ids': np.arange(1101,1112,1),
#       'lveg':True}
ba = {'lu_long':'barley', 
      'lu_short':'ba', 
      'lu_ids': np.array([1101]),
      'lveg':True}
fl = {'lu_long':'flower', 
      'lu_short':'fl', 
      'lu_ids': np.array([1102]),
      'lveg':True}
fo = {'lu_long':'fodder', 
      'lu_short':'fo', 
      'lu_ids': np.array([1103]),
      'lveg':True}
ma = {'lu_long':'maize', 
      'lu_short':'ma', 
      'lu_ids': np.array([1104]),
      'lveg':True}
oa = {'lu_long':'oat', 
      'lu_short':'oa', 
      'lu_ids': np.array([1105]),
      'lveg':True}
oc = {'lu_long':'other cereal', 
      'lu_short':'oc', 
      'lu_ids': np.array([1106]),
      'lveg':True}
po = {'lu_long':'potato', 
      'lu_short':'po', 
      'lu_ids': np.array([1107]),
      'lveg':True}
ra = {'lu_long':'rapeseed', 
      'lu_short':'ra', 
      'lu_ids': np.array([1108]),
      'lveg':True}
ry = {'lu_long':'rye', 
      'lu_short':'ry', 
      'lu_ids': np.array([1109]),
      'lveg':True}
su = {'lu_long':'sugar beet', 
      'lu_short':'su', 
      'lu_ids': np.array([1110]),
      'lveg':True}
wh = {'lu_long':'wheat', 
      'lu_short':'wh', 
      'lu_ids': np.array([1111]),
      'lveg':True}

# set of land use types with 'buildings'
lu_types_crop = {'lv':lv,
                 'hv':hv, 
                 'bs':bs, 
                 'aq':aq,
                 'ap':ap,
                 'bu':bu,
                 'ba':ba,
                 'fl':fl,
                 'fo':fo,
                 'ma':ma,
                 'oa':oa,
                 'oc':oc,
                 'po':po,
                 'ra':ra,
                 'ry':ry,
                 'su':su,
                 'wh':wh,
                 } 