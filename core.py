import weakref
import numpy as np
import contextlib
import skystar
try:
    import cupy
    array_types=(np.ndarray,cupy.ndarray)
except ImportError:
    array_types=(np.ndarray)

class Variable:
    '''--------
    data：输入数据，仅支持ndarray结构
    name:数据名称
    creator:创造输出的函数，该函数具有input和output，其中input==data
    grad：函数节点处的梯度
    generation:数据的辈分，用于反向传播时决定传播的顺序
    '''
    def __init__(self,data,name=None,):
        if data is not None:#如果数据不是ndarray则报错
            if not isinstance(data, array_types):
                raise TypeError('{} is not supposed'.format(type(data)))
        self.data=data
        self.name=name
        self.grad=None
        self.creator=None
        self.generation=0

    '''为每个数据连接函数并辈分'''
    def set_creator(self,func):
        self.creator=func
        self.generation=func.generation+1
    def cleangrad(self):#用以删除传递中间过程产生数据的梯度，释放内存
        self.grad=None
    def unchain(self):#删除数据与函数的链接
        self.creator=None
    def unchain_backward(self):
        if self.creator is not None:
            funcs=[self.creator]
            while funcs:
                f=funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()
    def image_show(self,mode='feature',label=None):
        if self.ndim!=4:
            if self.ndim==3:
                C,H,W=self.data.shape
                self.data=self.data.reshape(1,C,H,W)
            if self.ndim==2:
                H,W=self.data.shape
                self.data=self.data.reshape(1,1,H,W)
        skystar.utils.images_show(self.data, mode=mode,label=label)
    def to_cpu(self):
        if self.data is not None:
            self.data=skystar.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data=skystar.cuda.as_cupy(self.data)

    def backward(self,retain_grad=False,create_graph=False):
        '''
        循环结构完成反向传播
        retain_grad:是否保留反向过程中产生的grad
        create_grad:是否启用高阶求导
        '''
        if Config.enable_backprop==False:#反向传播的开关，默认为Ture
            return None
        if self.grad is None:#初始化为Variable，便于计算导数的导数
            xp=skystar.cuda.get_array_module(self.data)
            self.grad=Variable(xp.ones_like(self.data))
        funcs=[self.creator]
        while funcs:
            f=funcs.pop()
            dys = [output().grad for output in f.outputs]#list，元素为Variable
            with config_using('enable_backprop',create_graph):
                '''如果create_graph为False,在下面的计算中enable_backprop设置为False，由于运算符重载，那么下面的dxs = f.backward(*dys)将会
                调用Function.__call__方法，Function会根据enable_backprop禁止grad（Variable）链接函数f，由此梯度的反向传播无法进行。'''
                dxs = f.backward(*dys)  # 多个值返回元组，一个值返回单个元素，元素为Variable
                if not isinstance(dxs, tuple):
                    dxs = (dxs,)  # 单个值转化为元组，元素为Variable

                '''对每个input设置导数,如果输入两个数据是同一个数据，那么该数据的导数要叠加。这里
                同时会产生一个问题，如果对一个相同的变量反复反向传播，那么它的梯度会叠加，因此需要在
                每次传播后使用self.cleangrad()初始化梯度'''
                for x, dx in zip(f.inputs, dxs):  # x:Variable,dx:Variable
                    if x.grad is None:
                        x.grad = dx
                    else:
                        x.grad = x.grad + dx
                    if x.creator is not None:
                        if x.creator not in funcs:  # 这里避免相同函数的backward被调用两次
                            funcs.append(x.creator)
                            funcs.sort(key=lambda x: x.generation)  # key为函数，指定取待排序元素的哪一项进行排序

                    if not retain_grad:  # 这里删除了中间过程产生的实例变量的grad，释放内存，由用户输入的数据grad的保存了下来
                        for y in f.outputs:
                            y().grad = None
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list) or axes[0] is None):
                axes = axes[0]
        return transpose(self, axes)

    def sum(self,axis=None,keepdims=False):
        return sum(self,axis,keepdims)
    '''静态方法，可以在调用时不加括号，比如x=Variable(np.array),可以直接实现x.shape查看数据形状'''
    @property
    def shape(self):
        return self.data.shape
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def size(self):
        return self.data.size
    @property
    def dtype(self):
        return self.data.dtype
    @property
    def T(self):
        return transpose(self)

    __array_priority__=200#设置运算优先级，防止ndarray与Variable运算时，调用的是numpy的运算符
    def __len__(self):#实现Variable实例直接调用len函数
        return len(self.data)

    def __repr__(self):#改变Variable实例的打印方式
        if self.data is None:
            return 'Variable(None)'
        p=str(self.data).replace('\n','\n'+' '*9)
        return 'Variable('+p+')'

    '''重载运算符，实现Variable实例 加减乘除等方式,函数setup_variable也实现了该功能'''

class Parameter(Variable):#创建子集Parameter，用来存储模型的参数
    pass



class Function:
    '''----------
    inputs：#可以是多个数据，使用加减乘除时必须包含一个Variable实例
    self.inputs：#list，元素为Variable
    self.outputs：#list,元素为Variable实例,使用了弱引用，调用的话需要写为：元素（）
    generation：函数的辈分
    '''
    def __call__(self, *inputs):#添加*号，可以同时输入数据，将多个数据变为元组tuple
        inputs=[as_variable(x) for x in inputs]#将输入变为list,元素为Variable实例
        xs=[x.data for x in inputs]#list,元素为ndarray
        ys=self.forward(*xs)#使用*号解包，传向forward的inputs为元组或单个元素，元素为ndarray
        if not isinstance(ys,tuple):#如果不是元组进行处理
            ys=(ys,)
        outputs=[Variable(as_scalar(y)) for y in ys]#list，元素为Variable

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

        self.inputs=inputs#list,元素为Variable实例
        self.outputs=[weakref.ref(output) for output in outputs]#这里添加了弱引用，减少内存占用，如果要调用列表元素x的值则需要使用x()

        if len(outputs)>1:
            return outputs
        else:
            return outputs[0]#Variable

    def forward(self,x):
        raise NotImplementedError()

    def backward(self,dout):
        raise NotImplementedError()

# =============================================================================
'''四则运算和常用运算'''
# =============================================================================
class Exp(Function):
    def forward(self,x):
        xp=skystar.cuda.get_array_module(x)
        y=xp.exp(x)
        return y
    def backward(self,dout):
        x=self.inputs[0]
        xp=skystar.cuda.get_array_module(dout)
        dout=xp.exp(x)*dout
        return dout
class Square(Function):
    def forward(self,x):
        y=x**2
        return y
    def backward(self,dout):
        dout=2*self.inputs[0]*dout
        return dout
class Add(Function):#加法,实现了广播功能
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        x1=as_scalar(x1)
        return x0+x1
    def backward(self,dout):
        dout0,dout1=dout,dout
        if self.x0_shape != self.x1_shape:
            dout0=skystar.utils.sum_to(dout0,self.x0_shape)
            dout1=skystar.utils.sum_to(dout1,self.x1_shape)
        return dout0,dout1
class Mul(Function):#乘法
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        x1=as_scalar(x1)
        return x0*x1
    def backward(self,dout):
        x0,x1=self.inputs
        if self.x0_shape != self.x1_shape:
            dout0=skystar.utils.sum_to(dout*x1,self.x0_shape)
            dout1=skystar.utils.sum_to(dout*x0,self.x1_shape)
        return dout0,dout1
class Neg(Function):#负数
    def forward(self,x):
        return -x
    def backward(self,dout):
        return -1*dout
class Sub(Function):#减法
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        return x0-x1
    def backward(self,dout):
        dout0=dout
        dout1=-dout
        if self.x0_shape != self.x1_shape:
            dout0=skystar.utils.sum_to(dout0,self.x0_shape)
            dout1=skystar.utils.sum_to(dout1,self.x1_shape)
        return dout0,dout1
class Div(Function):#除法
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        return x0/x1
    def backward(self,dout):
        x0,x1=self.inputs
        dout1=dout/x1
        dout2=-dout*x0/x1**2
        if self.x0_shape != self.x1_shape:
            dout1=skystar.utils.sum_to(dout1,self.x0_shape)
            dout2=skystar.utils.sum_to(dout2,self.x1_shape)
        return dout1,dout2
class Pow(Function):#幂函数
    def __init__(self,c):
        self.c=c
    def forward(self,x):
        return x**self.c
    def backward(self,dout):
        x=self.inputs
        dout=dout*self.c*x**(self.c-1)
        return dout

def exp(x):
    func=Exp()
    return func(x)
def sqaure(x):
    func=Square()
    return func(x)
def add(x0,x1):
    x1 = as_scalar(x1,skystar.cuda.get_array_module(x0.data))#转换为ndarray，实现Variable与int float的直接运算
    func=Add()
    return func(x0,x1)
def mul(x0,x1):
    x1 = as_scalar(x1,skystar.cuda.get_array_module(x0.data))
    func=Mul()
    return func(x0,x1)
def neg(x):
    func=Neg()
    return func(x)
def sub(x0,x1):
    x1=as_scalar(x1,skystar.cuda.get_array_module(x0.data))
    func=Sub()
    return func(x0,x1)
def rsub(x0,x1):
    x1=as_scalar(x1,skystar.cuda.get_array_module(x0.data))
    func=Sub()
    return func(x1,x0)
def div(x0,x1):
    x1=as_scalar(x1,skystar.cuda.get_array_module(x0.data))
    func=Div()
    return func(x0,x1)
def rdiv(x0,x1):
    x1=as_scalar(x1,skystar.cuda.get_array_module(x0.data))
    func=Div()
    return func(x1,x0)
def pow(x,c):
    func=Pow(c)
    return func(x)

# =============================================================================
'''辅助函数'''
# =============================================================================
class Config:#允许反向传播的开关
    enable_backprop=True
@contextlib.contextmanager
def config_using(name, value):
    oldvalue = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, oldvalue)
def no_grad():
    return config_using('enable_backprop', False)
def as_scalar(x,array_module=np):#把数据变为np.ndarray或者cp.ndarray
    if np.isscalar(x):
        return array_module.array(x)
    return x
def as_variable(x):#把数据变为Variable实例
    if isinstance(x,Variable):
        return x
    return Variable(x)

def setup_variable():#重载运算符
    Variable.__mul__=mul
    Variable.__rmul__=mul
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__neg__ = neg
    Variable.__truediv__=div
    Variable.__rtruediv__=rdiv
    Variable.__pow__=pow


# =============================================================================
'''一般函数'''
# =============================================================================
'''
x：函数前向传播的inputs，类型为tuple或单个元素，元素类型为Variable,经__call__方法在forward中变为ndarray
y：函数前向传播的outputs，这里输出类型为ndarray，但会在__call__方法中变为Variable
dout：Variabel实例
self.inputs：#list，元素为Variable
self.outputs：#list,元素为Variable实例,使用了弱引用，调用的话需要写为：元素（）
'''
class Sin(Function):
    def forward(self,x):
        xp=skystar.cuda.get_array_module(x)
        y=xp.sin(x)
        return y
    def backward(self,dout):
        x=self.inputs[0]
        dout=dout*cos(x)
        return dout
def sin(x):
    return Sin()(x)
class Cos(Function):
    def forward(self,x):
        xp=skystar.cuda.get_array_module(x)
        y=xp.cos(x)
        return y
    def backward(self,dout):
        x=self.inputs[0]
        dout=dout*-sin(x)
        return dout
def cos(x):
    return Cos()(x)
class Tanh(Function):
    def forward(self,x):
        xp=skystar.cuda.get_array_module(x)
        y=(xp.exp(x)-xp.exp(-x))/(xp.exp(x)+xp.exp(-x))
        return y
    def backward(self,dout):
        y=self.outputs[0]()
        dout=dout*(1-y*y)
        return dout
def tanh(x):
    return Tanh()(x)
class Reshape(Function):
    def __init__(self,shape):
        self.shape=shape
    def forward(self,x):
        self.x_shape=x.shape
        x=x.reshape(self.shape)
        return x
    def backward(self,dout):
        dout=dout.reshape(self.x_shape)
        return dout
def reshape(x,shape):
    if x.shape==shape:
        return as_variable(x)
    return Reshape(shape)(x)
class Dot(Function):#点积
    def forward(self,x0,x1):
        xp=skystar.cuda.get_array_module(x0)
        y=xp.dot(x0,x1)
        return y
    def backward(self,dout):
        x0,x1=self.inputs
        dout0=dot(dout,x1.T)
        dout1=dot(x0.T,dout)
        return dout0,dout1
def dot(x0,x1):
    return Dot()(x0,x1)

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes
    def forward(self, x):
        y = x.transpose(self.axes)
        return y
    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
        xp=skystar.cuda.get_array_module(gy)
        axes_len = len(self.axes)
        inv_axes = tuple(xp.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)
def transpose(x, axes=None):
    return Transpose(axes)(x)

class Sum(Function):
    def __init__(self,axis,keepdims):
        self.axis=axis
        self.keepdims=keepdims
    def forward(self,x):
        self.x_shape=x.shape
        y=x.sum(axis=self.axis,keepdims=self.keepdims)
        return y
    def backward(self,dout):
        dout=skystar.utils.reshape_sum_backward(dout,self.x_shape,self.axis,self.keepdims)
        dout=broadcast_to(dout,self.x_shape)
        return dout
def sum(x,axis=None,keepdims=False):
    return Sum(axis,keepdims)(x)
class BroadcastTo(Function):#广播
    def __init__(self,shape):
        self.shape=shape
    def forward(self,x):
        self.x_shape=x.shape
        xp=skystar.cuda.get_array_module(x)
        y=xp.broadcast_to(x,self.shape)
        return y
    def backward(self,dout):
        dout=sum_to(dout,self.x_shape)
        return dout
def broadcast_to(x,shape):
    if x.shape==shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)
class SumTo(Function):#广播
    def __init__(self,shape):
        self.shape=shape
    def forward(self,x):
        self.x_shape=x.shape
        y=skystar.utils.sum_to(x,self.shape)
        return y
    def backward(self,dout):
        dout=broadcast_to(dout,self.x_shape)
        return dout
def sum_to(x,shape):
    if x.shape==shape:
        return as_variable(x)
    return SumTo(shape)(x)


class Concat(Function):
    def __init__(self, axis=1):
        self.axis = axis  # 指定拼接的维度轴
    def forward(self, x0, x1):
        # 获取所有输入的形状
        self.shapes = [x0.shape,x1.shape]
        xp=skystar.cuda.get_array_module(x0)
        # 在指定轴上拼接输入
        y = xp.concatenate((x0,x1), axis=self.axis)
        return y
    def backward(self, dout):
        # 根据 forward 时记录的形状信息，将 dout 拆分成与 x0 和 x1 形状相同的部分
        axis = self.axis
        xp=skystar.cuda.get_array_module(dout)
        x0_shape, x1_shape = self.shapes
        # 使用 split 将 dout 分割回 x0 和 x1 的大小
        dx0, dx1 = xp.split(dout.data, [x0_shape[axis]], axis=axis)
        return dx0, dx1
def concat(x0, x1, axis=0):
    return Concat(axis)(x0,x1)

# =============================================================================
'''激活函数'''
# =============================================================================
class Sigmoid(Function):
    def forward(self,x):
        xp=skystar.cuda.get_array_module(x)
        y=1/(1+xp.exp(-x))
        return y
    def backward(self,dout):
        y=self.outputs[0]()
        dout = dout * y * (1 - y)
        return dout
def sigmoid(x):
    func=Sigmoid()
    return func(x)

class Relu(Function):
    def __init__(self):
        self.mask=None
    def forward(self,x):
        self.mask=(x<=0)#标记小于等于0的位置
        y=x.copy()#制作一个副本，避免x被改变
        y[self.mask]=0#小于等于0的值输出0，其它数值原样输出
        return y
    def backward(self,dout):
        dout.data[self.mask]=0
        return dout
def ReLU(x):
    return Relu()(x)

class Softmax(Function):#用于分类的激活函数,用于输出和输入相等
    def forward(self,x):
        y=skystar.utils.softmax(x)
        return y
    def backward(self,dout):
        y=self.outputs[0]()
        return dout*y*(1-y)
def softmax(x):
    return Softmax()(x)
# =============================================================================
'''loss函数'''
# =============================================================================
class SoftmaxCrossEntropyLoss(Function):#该函数结合了softmax和loss交叉熵误差函数，可以用作分类问题的最后一个激活函数
    def forward(self,x0,x1):
        y=skystar.utils.softmax(x0)
        loss=skystar.utils.cross_entropy_error(y,x1)
        return loss
    def backward(self,dout):#输出层反向传播输入导数1
        xp=skystar.cuda.get_array_module(dout)
        x0,t=self.inputs
        y=skystar.utils.softmax(x0.data)#ndarray
        batch_size=y.shape[0]
        if t.ndim!=4:
            if y.size == t.size:
                dout = (y - t) / batch_size  # 除以batch_size，传递单个数据的误差
            else:  # 当t为非one—hot形式时，传递梯度
                dout = Variable(y.copy())
                dout.data[xp.arange(batch_size), t.data] -= 1
                dout = dout / batch_size
        else:
            batch, num_class, H, W = y.shape
            if y.size != t.size:
                t=Variable(skystar.utils.onehot(t.data,num_classes=num_class))
            dout=(y - t) / batch
        return dout
def softmaxwithloss(x,t):
    return SoftmaxCrossEntropyLoss()(x, t)

class MeanSquaredError(Function):
    def forward(self,x0,x1):
        diff=x0-x1
        y=(diff**2).sum()/len(diff)
        return y
    def backward(self,dout):
        x0,x1=self.inputs
        diff=x0-x1
        gx0=dout*diff*(2./len(diff))
        gx1=-gx0
        return gx0,gx1
def mean_squared_error(x0,x1):
    return MeanSquaredError()(x0,x1)

# =============================================================================
'''层函数'''
# =============================================================================
class Affine(Function):
    '''
    affine只进行二维矩阵的点积
    x：输入数据，如果输入数据是多维，则先变为2维
    W：参数权重，为二维的Varaible
    b，参数偏置，为一维的Variable
    affine层可以单纯由dot与add实现
    '''
    def forward(self,x,W,b):
        xp=skystar.cuda.get_array_module(x)
        self.original_x_shape=x.shape
        x=x.reshape(x.shape[0],-1)
        y=xp.dot(x,W)+b
        return y
    def backward(self,dout):
        x,W,b=self.inputs
        x=x.reshape(x.shape[0],-1)
        dW=dot(x.T,dout)
        db=sum(dout,axis=0)
        dout=dot(dout,W.T)
        dout=dout.reshape(*self.original_x_shape)
        return dout,dW,db
def affine(x,W,b):
    return Affine()(x,W,b)

class BatchNormalization(Function):
    def __init__(self,gamma=1.0,beta=0,momentum=0.9,training=True):
        self.gamma=gamma#ndarray或float
        self.beta=beta#ndarray或float
        self.momentum=momentum#float
        self.xv=None#ndarray
        self.var=None#ndarray
        self.mean=None#ndarray

        self.test_mean=None#ndarray
        self.test_var=None#ndarray
        self.training=training
    def forward(self,x):
        xp=skystar.cuda.get_array_module(x)
        if self.test_mean is None:
            sum = xp.sum(x, axis=0)
            self.test_mean=xp.zeros_like(sum)
            self.test_var=xp.zeros_like(sum)
        if self.training:#训练时用当前批量输入的均值方差
            sum = xp.sum(x, axis=0)
            N = x.shape[0]
            mean = sum / N
            var = xp.sum((x - mean) ** 2, axis=0) / N
            self.mean = mean
            self.var = var
            '''更新全局均值方差，用于测试用'''
            self.test_mean=self.momentum * self.test_mean + (1 - self.momentum) * self.mean
            self.test_var=self.momentum * self.test_var + (1 - self.momentum) * self.var
        else:#测试用全局均值方差
            mean=self.test_mean
            var=self.test_var
        self.xv = (x - mean) / xp.sqrt(var + 1e-7)
        y = self.gamma * self.xv + self.beta
        return y
    def backward(self,dout):
        xp=skystar.cuda.get_array_module(dout)
        x=self.inputs[0]#Variable
        N=dout.shape[0]#ndarray
        dgamma=sum(dout*self.xv,axis=0)#Variable
        dbeta=sum(dout,axis=0)#Variable
        dxv=dout*self.gamma#Variable
        dvar=sum(dxv*(x-self.mean)*xp.power(self.var + 1e-7, -1.5)*(-0.5),axis=0)#Variable
        dmean=-1.0*sum(dxv,axis=0)/xp.sqrt(self.var+1e-7)+dvar*sum(-2.0*(x-self.mean),axis=0)/N#Variable
        dx=dxv/xp.sqrt(self.var+1e-7)+dvar*2*(x-self.mean)/N+dmean/N#Variable
        self.gamma-=0.01*dgamma.data
        self.beta-=0.01*dbeta.data
        return dx

class Dropout(Function):
    '''训练时随机删除神经元节点，有效避免过拟合'''
    def __init__(self,ratio=0.5,training=True):
        self.ratio=ratio
        self.mask=None
        self.training=training
    def forward(self,x):
        xp=skystar.cuda.get_array_module(x)
        if self.training:
            rand = xp.random.rand(*x.shape)  # 生成与x形状相同的随机数
            self.mask = (rand < self.ratio)
            y = x.copy()
            y[self.mask] = 0#被标记的神经元
            y/=(1.0 - self.ratio)#缩放处理，保持输出一致性
        else:
            y=x#测试时不对数据进行缩放，因为所有神经元都要输出
        return y
    def backward(self, dout):
         dout.data[self.mask]=0
         return dout
def dropout(x,ratio=0.5,training=True):
    return Dropout(ratio=ratio,training=training)(x)

# =============================================================================
'''卷积网络层函数'''
# =============================================================================
class Conv(Function):
    '''
    卷积层
    W：Variable,四维shape(FN,C,FH,FW)
    B：Variable,一维shape(FN,)
    stride:步长
    pad:填充
    to_affine:如果下一层是affine层，将输出调整为矩阵，而不是四维张量
    '''
    def __init__(self,stride=1,pad=0):
        self.stride=stride#int
        self.pad=pad#int

        # 中间数据（backward时使用）
        self.col = None#array,shape(N*oh*ow, c*fh*fw)
        self.col_W = None#array,shape(c*fh*fw,FN)
    def forward(self,x,Weight,b):
        xp=skystar.cuda.get_array_module(x)
        N,C,H,W=x.shape
        FN,C,FH,FW=Weight.shape
        out_h=int((H+2*self.pad-FH)//self.stride+1)
        out_w=int((W+2*self.pad-FW)//self.stride+1)
        self.col=skystar.utils.im2col(x,FH,FW,stride=self.stride,pad=self.pad)#shape(N*oh*ow, c*fh*fw)
        self.col_W=Weight.reshape(FN,-1).T#shape(c*fh*fw,FN)

        out = xp.dot(self.col, self.col_W)
        if b is not None:
            out +=b  # shape(N*oh*ow,FN))
        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)#shape(N,FN,oh,ow)
        return out

    def backward(self,dout):
        x,W,b=self.inputs[0],self.inputs[1],self.inputs[2]
        FN, C, FH, FW = W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)#(N*oh*ow,FN)

        if b is not None:
            b.grad = sum(dout, axis=0)  # (FN,)
        W.grad = dot(self.col.T, dout).transpose(1,0).reshape(FN,C,FH,FW)

        dcol = dot(dout, self.col_W.T)#Variable(N*oh*ow,c*fh*fw)
        dx = skystar.utils.col2im(dcol.data, x.shape, FH, FW, self.stride, self.pad)#shape(N,C,H,W)
        dx=Variable(dx)
        return dx
def convolution(x,W,b,stride=1,pad=0):
    return Conv(stride,pad)(x,W,b)

class ConvTranspose(Function):
    def __init__(self,stride=1,pad=0):
        self.stride=stride#int
        self.pad=pad#int

        # 中间数据（backward时使用）
        self.col = None#array,shape(N*oh*ow, c*fh*fw)
        self.col_W = None#array,shape(c*fh*fw,FN)
    def forward(self,x,Weight,b):
        '''2*2-->填充k=3,s=2,p=0--> 7*7-->标准卷积k=3,s=1,p=0-->5*5
        o' = s(i' - 1) + k-2p ，i'为原输入尺寸，o'为输出尺寸 （s=32,p=16,k=64）适合图像尺寸224*224卷积为7*7的网络反卷积'''
        xp=skystar.cuda.get_array_module(x)#3*3
        FN, C, FH, FW = Weight.shape
        x=skystar.utils.transconv_pad(x,stride=self.stride,kernel_size=FH,pad=self.pad)#对x进行反卷积的填充，7*7
        reWeight = xp.flip(Weight, axis=(2, 3))  # 对矩阵进行翻转
        N, C, H, W = x.shape
        self.shape=x.shape

        #标准卷积
        out_h=int((H+2*0-FH)//1+1)#5
        out_w=int((W+2*0-FW)//1+1)#5
        self.col=skystar.utils.im2col(x,FH,FW,stride=1,pad=0)#shape(N*oh*ow, c*fh*fw)
        self.col_W=reWeight.reshape(FN,-1).T#shape(c*fh*fw,FN)
        out=xp.dot(self.col, self.col_W)
        if b is not None:
            out += b  # shape(N*oh*ow,FN)
        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)#shape(N,FN,oh,ow)5*5

        return out
    def backward(self,dout):
        x,W,b=self.inputs[0],self.inputs[1],self.inputs[2]#这里的W是Variable实例
        xp = skystar.cuda.get_array_module(x)
        FN, C, FH, FW = W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)#(N*oh*ow,FN)

        if b is not None:
            b.grad = sum(dout, axis=0)  # (FN,)
        W.grad = dot(self.col.T, dout).transpose(1, 0).reshape(FN, C, FH, FW)#这里得到的梯度是翻转后W的梯度，因此梯度需要翻转
        W.grad.data=xp.flip(W.grad.data,axis=(2,3))#将W的梯度进行翻转

        dcol = dot(dout, self.col_W.T)#Variable(N*oh*ow,c*fh*fw)
        dx = skystar.utils.col2im(dcol.data, self.shape, FH, FW, stride=1,pad=0)#shape(N,C,H,W)
        dx=Variable(skystar.utils.back_transcov_pad(dx,stride=self.stride,kernel_size=FH,pad=self.pad))
        return dx
def transposed_convolution(x,weight,b,stride=1,pad=0):
    return ConvTranspose(stride=stride, pad=pad)(x, weight, b)

class MaxPool(Function):
    '''池化层，取池化窗口的最大值'''
    def __init__(self, pool_size, stride=1, pad=0):
        self.pool_size = pool_size#int
        self.stride = stride#int
        self.pad = pad#int
        self.arg_max = None#array

    def forward(self, x):
        xp=skystar.cuda.get_array_module(x)
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_size) / self.stride)
        out_w = int(1 + (W - self.pool_size) / self.stride)

        '''展开为二维数组'''
        col = skystar.utils.im2col(x, self.pool_size, self.pool_size, self.stride, self.pad)
        col = col.reshape(-1, self.pool_size * self.pool_size)

        '''计算最大值'''
        out = xp.max(col, axis=1)
        self.arg_max = xp.argmax(col, axis=1)  # 记录最大值的位置
        '''转换为四维数组'''
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        xp=skystar.cuda.get_array_module(dout)
        x = self.inputs[0]
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_size * self.pool_size
        dmax = xp.zeros((dout.size, pool_size))
        dmax[xp.arange(self.arg_max.size), self.arg_max.flatten()] = dout.data.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = skystar.utils.col2im(dcol, x.shape, self.pool_size, self.pool_size, self.stride, self.pad)
        dx = Variable(dx)
        return dx
def maxpool(x,pool_size,stride=1,pad=0):
    return MaxPool(pool_size,stride,pad)(x)

class AveragePool(Function):
    '''平均池化层，取池化窗口的平均值'''
    def __init__(self, pool_size, stride=1, pad=0):
        self.pool_size = pool_size  # int
        self.stride = stride  # int
        self.pad = pad  # int
    def forward(self, x):
        xp = skystar.cuda.get_array_module(x)
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_size) / self.stride)
        out_w = int(1 + (W - self.pool_size) / self.stride)
        # 展开为二维数组
        col = skystar.utils.im2col(x, self.pool_size, self.pool_size, self.stride, self.pad)
        col = col.reshape(-1, self.pool_size * self.pool_size)
        # 计算平均值
        out = xp.mean(col, axis=1)
        # 转换为四维数组
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out
    def backward(self, dout):
        x = self.inputs[0]
        dout = dout.transpose(0, 2, 3, 1)
        # 计算反向传播
        pool_size = self.pool_size * self.pool_size
        dmean = dout / pool_size
        dmean = Variable(dmean.data.repeat(pool_size, axis=-1))

        dcol = dmean.reshape(dmean.shape[0] * dmean.shape[1] * dmean.shape[2], -1)
        dx = skystar.utils.col2im(dcol.data, x.shape, self.pool_size, self.pool_size, self.stride, self.pad)
        dx = Variable(dx)
        return dx
def avgpool(x,pool_size,stride=1,pad=0):
    return AveragePool(pool_size,stride,pad)(x)