import glob
import os

all_files = glob.glob('*')
exclude   = ['namoptions','run','dales4','lon_LES.npy','lat_LES.npy']

for f in all_files:
    delete = True
    if '.py' in f:
        delete = False
    for excl in exclude:
        if excl in f:
            delete = False

    if delete:
        try:
            os.remove(f)
        except:
            print('Can not remove {}'.format(f))
