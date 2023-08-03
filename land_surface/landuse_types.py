import numpy as np

### DEFAULT
# define lu type characteristics
lv = {'lu_long':'low vegetation', 
      'lu_short':'lv', 
      'lu_ids': np.array([9,10,11,13,26]),  # without urban
      'lveg':True, 'laqu':False}
hv = {'lu_long':'high vegetation', 
      'lu_short':'hv', 
      'lu_ids': np.array([1,2,3,4,5,6,7,8,27]),
       'lveg':True, 'laqu':False}
aq = {'lu_long':'water surface', 
      'lu_short':'aq', 
      'lu_ids': np.array([14,15,16,17,18]),
      'lveg':False, 'laqu':True}
ap = {'lu_long':'asphalt', 
      'lu_short':'ap', 
      'lu_ids': np.array([20,22,23,24,25,29,30]),
      'lveg':False, 'laqu':False}
bs = {'lu_long':'bare soil', 
      'lu_short':'bs', 
      'lu_ids': np.array([12,28]),
      'lveg':False, 'laqu':False}

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
      'lveg':True, 'laqu':False}
hv = {'lu_long':'high vegetation', 
      'lu_short':'hv', 
      'lu_ids': np.array([1,2,3,4,5,6,7,8,27]),
       'lveg':True, 'laqu':False}
aq = {'lu_long':'water surface', 
      'lu_short':'aq', 
      'lu_ids': np.array([14,15,16,17,18]),
      'lveg':False, 'laqu':True}
ap = {'lu_long':'asphalt', 
      'lu_short':'ap', 
      'lu_ids': np.array([20,22,23,24,25]),
      'lveg':False, 'laqu':False}
bu = {'lu_long':'buildings', 
      'lu_short':'bu', 
      'lu_ids': np.array([29,30]),
      'lveg':False, 'laqu':False}
bs = {'lu_long':'bare soil', 
      'lu_short':'bs', 
      'lu_ids': np.array([12,28]),
      'lveg':False, 'laqu':False}

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
      'lveg':True, 'laqu':False}
hv = {'lu_long':'high vegetation', 
      'lu_short':'hv', 
      'lu_ids': np.array([1,2,3,4,5,6,7,8,27]),
       'lveg':True, 'laqu':False}
aq = {'lu_long':'water surface', 
      'lu_short':'aq', 
      'lu_ids': np.array([14,15,16,17,18]),
      'lveg':False, 'laqu':True}
ap = {'lu_long':'asphalt', 
      'lu_short':'ap', 
      'lu_ids': np.array([20,22,23,24,25]),
      'lveg':False, 'laqu':False}
bu = {'lu_long':'buildings', 
      'lu_short':'bu', 
      'lu_ids': np.array([29,30]),
      'lveg':False, 'laqu':False}
bs = {'lu_long':'bare soil', 
      'lu_short':'bs', 
      'lu_ids': np.array([12,28]),
      'lveg':False, 'laqu':False}
# cr = {'lu_long':'crops', 
#       'lu_short':'cr', 
#       'lu_ids': np.arange(1101,1112,1),
#       'lveg':True}
ba = {'lu_long':'barley', 
      'lu_short':'ba', 
      'lu_ids': np.array([1101]),
      'lveg':True, 'laqu':False}
fl = {'lu_long':'flower', 
      'lu_short':'fl', 
      'lu_ids': np.array([1102]),
      'lveg':True, 'laqu':False}
fo = {'lu_long':'fodder', 
      'lu_short':'fo', 
      'lu_ids': np.array([1103]),
      'lveg':True, 'laqu':False}
ma = {'lu_long':'maize', 
      'lu_short':'ma', 
      'lu_ids': np.array([1104]),
      'lveg':True, 'laqu':False}
oa = {'lu_long':'oat', 
      'lu_short':'oa', 
      'lu_ids': np.array([1105]),
      'lveg':True, 'laqu':False}
oc = {'lu_long':'other cereal', 
      'lu_short':'oc', 
      'lu_ids': np.array([1106]),
      'lveg':True, 'laqu':False}
po = {'lu_long':'potato', 
      'lu_short':'po', 
      'lu_ids': np.array([1107]),
      'lveg':True, 'laqu':False}
ra = {'lu_long':'rapeseed', 
      'lu_short':'ra', 
      'lu_ids': np.array([1108]),
      'lveg':True, 'laqu':False}
ry = {'lu_long':'rye', 
      'lu_short':'ry', 
      'lu_ids': np.array([1109]),
      'lveg':True, 'laqu':False}
su = {'lu_long':'sugar beet', 
      'lu_short':'su', 
      'lu_ids': np.array([1110]),
      'lveg':True, 'laqu':False}
wh = {'lu_long':'wheat', 
      'lu_short':'wh', 
      'lu_ids': np.array([1111]),
      'lveg':True, 'laqu':False}

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


### DEPAC LU types
# define lu type characteristics
ara = {'lu_long':'Arable land', 
      'lu_short':'ara', 
      'lu_ids': np.array([11]),  
      'lveg':True, 'laqu':False}
crp = {'lu_long':'Permanent crops', 
      'lu_short':'crp', 
      'lu_ids': np.array([6,7,8]),
       'lveg':True, 'laqu':False}
# fcd = {'lu_long':'Coniferous deciduous forest', 
#       'lu_short':'fcd', 
#       'lu_ids': np.array(None),
#       'lveg':True, 'laqu':False}
fce = {'lu_long':'Coniferous evergreen forest', 
      'lu_short':'fce', 
      'lu_ids': np.array([2,3]),
      'lveg':True, 'laqu':False}
fbd = {'lu_long':'Broadleaf deciduous forest', 
      'lu_short':'fbd', 
      'lu_ids': np.array([1,4,5,27]),
      'lveg':True, 'laqu':False}
# fbe = {'lu_long':'Broadleaf evergreen forest', 
#       'lu_short':'fbe', 
#       'lu_ids': np.array(None),
#       'lveg':True, 'laqu':False}
aqu = {'lu_long':'Aquatic', 
      'lu_short':'aqu', 
      'lu_ids': np.array([14,15,16,17,18,19]),
      'lveg':False, 'laqu':True}
brn = {'lu_long':'Barren land', 
      'lu_short':'brn', 
      'lu_ids': np.array([12,21,28]),
      'lveg':False, 'laqu':False}
sem = {'lu_long':'Semi-natural vegetation', 
      'lu_short':'sem', 
      'lu_ids': np.array([9,13,26]),
      'lveg':True, 'laqu':False}
grs = {'lu_long':'Grassland', 
      'lu_short':'grs', 
      'lu_ids': np.array([10]),  
      'lveg':True, 'laqu':False}
urb = {'lu_long':'Urban', 
      'lu_short':'urb', 
      'lu_ids': np.array([20,22,23,24,25,29,30]),
      'lveg':False, 'laqu':False}


# set of land use types with 'buildings'
lu_types_depac = {'ara':ara,
                  'crp':crp, 
                  # 'fcd':fcd, 
                  'fce':fce,
                  'fbd':fbd,
                  # 'fbe':fbe,
                  'aqu':aqu,
                  'brn':brn,
                  'sem':sem,
                  'grs':grs,
                  'urb':urb}