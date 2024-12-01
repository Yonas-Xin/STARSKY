import os

import numpy as np
from skystar.sky_dataset.create_dataset import init_dir
def load(file,split=False,split_rate=0.8):
    '''
    split=false retrun: x,t np.ndarray
    split=true retrun: x_train,t_train,x_test,t_test np.ndarray
    '''
    path = init_dir()
    file=os.path.join(path,file)
    with np.load(file) as data:
        x = data['x_dataset']
        t = data['t_dataset']
    if split:
        x_train=x[:int(len(x)*split_rate)]
        t_train=t[:int(len(t)*split_rate)]
        x_test=x[int(len(x)*split_rate):]
        t_test=t[int(len(t)*split_rate):]
        return x_train,t_train,x_test,t_test
    return x,t

if __name__ == '__main__':
    x_train,t_train = load('dataset.npz')