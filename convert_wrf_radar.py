import read_zlib_files as rz
import glob
import numpy as np

fn = glob.glob('/g/data/w40/esh563/lind04/dbz/2006*alllevels_zlib.ascii')
for i in range(len(fn)):
    print('Converting file ' + str(i))
    data = rz.read_wrf(fn[i], 0, 241, 241)
    data = np.reshape(data, (40, 241, 241), order='C')
    np.save(fn[i][:-6]+'.npy', data)
