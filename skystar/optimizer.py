import numpy as np

class Optimizer:
    '''
    target:目标model
    hooks:储存预处理功能函数
    '''
    def __init__(self,use_iter=False):
        self.use_iter=use_iter
        self.iter=0
        self.target=None
        self.hooks=[]
    '''target为model'''
    def setup(self,target):
        self.target=target
        return self

    def to_iter(self):
        self.use_iter=True

    def update(self):
        params=[param for param in self.target.params() if param.grad is not None]

        #可选功能，预处理，比如权重衰减
        for f in self.hooks:
            f(params)

        if not self.use_iter:
            for param in params:
                self.update_one(param)
        else:
            self.iter+=1
            for param in params:
                self.update_one(param,self.iter)

    def update_one(self,param,iter=0):
        raise NotImplementedError

    def add_hook(self,f):#使用该功能添加预处理方法
        self.hooks.append(f)


class SGD(Optimizer):
    '''这种更新方法学习率保持不变'''
    def __init__(self,lr=0.01):
        super().__init__()
        self.lr=lr

    def update_one(self,param):
        param.data-=self.lr*param.grad.data

class MomentumSGD(Optimizer):
    '''这种更新方法所有参数学习率一起变化，逐渐减小，直到稳定'''
    def __init__(self,lr=0.01,momentum=0.9):
        super().__init__()
        self.lr=lr
        self.momentum=momentum
        self.vs={}

    def update_one(self,param):
        v_key=id(param)
        if v_key not in self.vs:
            self.vs[v_key]=np.zeros_like(param.data)

        v=self.momentum*self.vs[v_key]-self.lr*param.grad.data
        param.data+=v


class AdaGrad(Optimizer):
    '''这种更新方法每个参数的学习率的变化速度不一样'''
    def __init__(self,lr=0.01):
        super().__init__()
        self.lr=lr
        self.hs={}

    def update_one(self,param):
        h_key=id(param)
        if h_key not in self.hs:
            self.hs[h_key]=np.zeros_like(param.data)

        self.hs[h_key]+=param.grad.data*param.grad.data
        param.data-=self.lr*param.grad.data/(np.sqrt(self.hs[h_key]+1e-6))


class Adam(Optimizer):
    """这种更新方法集合了momentum法与Adagrad的优点。Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        super(Adam, self).__init__(use_iter=True)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.ms = {}
        self.vs = {}

    def update_one(self, param, iter=0):
        _key=id(param)
        if _key not in self.ms:
            self.ms[_key]=np.zeros_like(param.data)
            self.vs[_key]=np.zeros_like(param.data)

        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** iter) / (1.0 - self.beta1 ** iter)
        self.ms[_key] += (1 - self.beta1) * (param.grad.data - self.ms[_key])
        self.vs[_key] += (1 - self.beta2) * (param.grad.data ** 2 - self.vs[_key])
        param.data -= lr_t * self.ms[_key] / (np.sqrt(self.vs[_key]) + 6)
