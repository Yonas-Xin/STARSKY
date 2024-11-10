import os

import numpy as np

def _loaddata(file,split=False):
    path=os.path.dirname(os.path.abspath(__file__))
    file=os.path.join(path,file)
    with np.load(file) as data:
        x = data['x_dataset']
        t = data['t_dataset']
    if split:
        x_train=x[:int(len(x)*0.8)]
        t_train=t[:int(len(t)*0.8)]
        x_test=x[int(len(x)*0.8):]
        t_test=t[int(len(t)*0.8):]
        return x_train,t_train,x_test,t_test
    return x,t

# x_train,t_train = _loaddata('dataset.npz')