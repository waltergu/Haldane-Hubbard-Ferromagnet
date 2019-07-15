from os import listdir
from os.path import isfile
import numpy as np

for fname in listdir('.'):
    if isfile(fname) and fname[-4:]=='.dat':
        print('Compression of %s'%fname)
        data=np.loadtxt(fname)
        np.savez_compressed(fname[:-4],data=data)