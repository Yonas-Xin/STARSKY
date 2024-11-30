import numpy as np
gpu_enable=True
try:#试着能否导入cupy
    import cupy as cp
    cupy=cp
    # print('GPU enabled')
except ImportError:
    gpu_enable=False
from skystar.core import Variable

def get_array_module(x):#获取数据的模块，既numpy 或者 cupy
    if isinstance(x,Variable):
        x=x.data
    if not gpu_enable:
        return np
    xp=cp.get_array_module(x)
    return xp

def as_numpy(x):#把数据转化为numpy.ndarray
    if isinstance(x,Variable):
        x=x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x,np.ndarray):
        return x
    return cp.asnumpy(x)

def as_cupy(x):#把数据转化为cupy.ndarray
    if isinstance(x,Variable):
        x=x.data

    if not gpu_enable:
        raise Exception('CuPy cannot be loaded.Install CuPy')
    return cp.asarray(x)