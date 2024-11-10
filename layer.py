import os.path

import skystar.utils
from skystar.core import *
from skystar import cuda
import numpy as np

# =============================================================================
'''Layer类'''
# =============================================================================

class Layer:
    '''training：某些层的使用分训练和测试两种类型，模型使用时默认training为True，
    如果训练完毕需要使用accurancy预测，请将training设置为False,一些不分测试，训
    练的模型，training的值不影响结果'''
    def __init__(self):
        self._params=set()#创建空集合，集合存储了实例的属性名，使用__dict__[name]可以取出属性值，集合的值无序且唯一，便于更新权重

    def __setattr__(self, name, value):#重写__setattr__，改变或添加实例属性的值时，会调用__setattr__
        if isinstance(value,(Parameter,Layer)):
            self._params.add(name)
        super().__setattr__(name,value)

    def __call__(self, *inputs,training=True):
        outputs=self.forward(*inputs,training=training)
        if not isinstance(outputs,tuple):
            outputs=(outputs,)
        self.inputs=[weakref.ref(x) for x in inputs]
        self.outputs=[weakref.ref(y) for y in outputs]
        if len(outputs)>1:
            return outputs
        else:
            return outputs[0]

    def forward(self,x,training=True):
        raise NotImplementedError

    def params(self):#生成器
        '''先从模型中迭代Layer属性，再从Layer中迭代它的Parameter属性，由此可迭代出模型里所有Layer的所有_params'''
        for name in self._params:
            obj=self.__dict__[name]
            if isinstance(obj,Layer):#如果对象是Layer，迭代返回_params
                yield from obj.params()
            # yield  暂停处理并返回值，再次调用params，暂停的处理再次运行，这样可以按顺序取出_params所有值
            #比如[1,2,3],使用yield取出一，下次循环时取出2，直到所有元素取出，而return只能返回1
            else:
                yield obj

    def cleangrads(self):
        for param in self.params():
            param.cleangrad()

    def _flatten_params(self,params_dict,parent_key=''):
        '''该函数使得params_dict变为name：Variabl的字典'''
        for name in self._params:
            obj=self.__dict__[name]
            key=parent_key+'/'+name if parent_key else name

            if isinstance(obj,Layer):
                obj._flatten_params(params_dict,key)
            else:
                params_dict[key]=obj

    def save_weights(self,path):
        '''获取当前脚本目录，并在目录下创建model_params用来储存参数'''
        self.to_cpu()
        dir=os.path.dirname(os.path.abspath(__file__))
        path=os.path.join(dir,'model_params',path)
        params_dict={}
        self._flatten_params(params_dict)
        array_dict={key:param.data for key,param in params_dict.items() if param is not None}

        try:#如果系统中断了正在保存的文件，则将文件删除，避免文件不完整
            np.savez_compressed(path,**array_dict)
            print(f'参数保存成功！path:{path}')
        except (Exception,KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
                print('保存中断，已删除文件')
            raise

    def load_weights(self,path):
        dir=os.path.dirname(os.path.abspath(__file__))
        path=os.path.join(dir,'model_params',path)
        if not os.path.exists(path):#如果不存在该文件，直接结束函数
            print('权重文件不存在，请训练网络！')
            return
        npz=np.load(path)
        params_dict={}
        self._flatten_params(params_dict)

        for name,param in params_dict.items():
            param.data=npz[name]
        print(f'网络参数加载成功！请注意参数类型为np.ndarray,训练或测试时根据需要转换参数格式path:{path}')


    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def weight_show(self,mode='weight',label=None):
        W = self.W.data
        if W is not None:
            if W.ndim ==4:
                skystar.utils.images_show(W,mode=mode,label=label)
            else:
                print(f'权重值维度不匹配：{W.ndim}！=4')
        else:
            print('权重尚未初始化：None')
class Affine(Layer):
    '''全连接层,只需要输入out_size,in_size可根据要传递的x大小自动计算得出'''
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self,xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O) * xp.sqrt(1 / I)
        W_data=W_data.astype(self.dtype)
        self.W.data = W_data

    def forward(self, x,training=True):
        xp=cuda.get_array_module(x)
        if self.W.data is None:
            self.in_size = x.reshape(x.shape[0],-1).shape[1]#如果x的维度是四维，那么变形之后取它的shape[1]
            self._init_W(xp)

        y = affine(x, self.W, self.b)
        return y

'''Batch_Norm层，放在激活函数层前或后，改善激活值的分布'''
class BatchNorm(Layer):
    '''self.test_mean,self.test_var:储存全局均值和方差用于模型预测阶段，如果training为True，每次运行forward，数据会更新'''
    def __init__(self,gamma=1.0,beta=0,momentum=0.9):
        super().__init__()
        self.batchnorm_func=BatchNormalization(gamma=gamma,beta=beta,momentum=momentum)

    def forward(self,x,training=True):
        self.batchnorm_func.train=training
        x=self.batchnorm_func(x)
        return x


class Convolution(Layer):
    '''卷积层：
    FN：核的数量，也是输出的通道数
    FH：核的高
    FW：核的宽
    in_channels：输入的通道数，也是核的通道数'''
    def __init__(self, FN, FH, FW, stride=1, pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.FN = FN
        self.FH = FH
        self.FW = FW
        self.stride = stride
        self.pad = pad
        self.in_channels = in_channels
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(FN, dtype=dtype), name='b')

    def _init_W(self,xp=np):
        I, FH, FW = self.in_channels, self.FH, self.FW
        W_data = xp.random.randn(self.FN, I, FH, FW) * xp.sqrt(1 / (I * FH * FW))
        W_data = W_data.astype(self.dtype)
        self.W.data = W_data

    def forward(self, x, training=True):
        xp=cuda.get_array_module(x)
        if self.W.data is None:
            self.in_channels = x.shape[1]
            self._init_W(xp)

        y = convolution(x, self.W, self.b, self.stride, self.pad)
        return y

class Transpose_Convlution(Layer):
    def __init__(self, FN, FH, FW, stride=1, pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.FN = FN
        self.FH = FH
        self.FW = FW
        self.stride = stride
        self.pad = pad
        self.in_channels = in_channels
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(FN, dtype=dtype), name='b')

    def _init_W(self,xp=np):
        '''初始化权重'''
        I, FN, K = self.in_channels, self.FN, self.FW
        W_data = skystar.utils.bilinear_kernel(in_channels=I,out_channels=FN,kernel_size=K,xp=xp)
        W_data = W_data.astype(self.dtype)
        self.W.data = W_data

    def forward(self, x, training=True):
        xp=cuda.get_array_module(x)
        if self.W.data is None:
            self.in_channels = x.shape[1]
            self._init_W(xp)

        y = transposed_convolution(x, self.W, self.b, self.stride, self.pad)
        return y

#残差块
class ResidualBlock(Layer):
    def __init__(self,num_channels, stride=1, nobias=False, dtype=np.float32, use_conv1x1=False):
        super().__init__()
        self.conv1=Convolution(FN=num_channels,FH=3,FW=3,stride=stride,pad=1,nobias=nobias, dtype=dtype)
        self.conv2=Convolution(FN=num_channels,FH=3,FW=3,stride=1,pad=1,nobias=nobias, dtype=dtype)
        if use_conv1x1:
            self.conv3=Convolution(FN=num_channels,FH=1,FW=1,stride=stride,pad=0,nobias=nobias, dtype=dtype)
        else:
            self.conv3=None
        self.bn1=BatchNorm()
        self.bn2=BatchNorm()
    def forward(self,x,training=True):#（在使用残差块建立网络时），需要注意残差块的前向传播中已经使用了批量归一化与激活函数
        y=self.bn1(self.conv1(x,training=training),training=training)
        y=ReLU(y)
        y=self.bn2(self.conv2(y,training=training),training=training)
        if self.conv3:
            x=self.conv3(x,training=training)
        y=ReLU(y+x)
        return y


class Pooling(Layer):
    '''池化层：
    pool_size：池化窗口大小
    stride：步长
    pad：填充'''
    def __init__(self,pool_size,stride=1,pad=0):
        super().__init__()
        self.pool_size=pool_size
        self.stride=stride
        self.pad=pad

    def forward(self,x,training=True):
        y=pooling(x,self.pool_size,self.stride,self.pad)
        return y

class RNN(Layer):
    '''self.h:既是自己的状态，也是自己的输出，自己的状态状态同时影响了输出'''
    def __init__(self,hidden_size,in_size=None):
        super().__init__()
        self.x2h=Affine(hidden_size,in_size=in_size)
        self.h2h=Affine(hidden_size,in_size=in_size,nobias=True)#不要偏置b
        self.h=None
    def reset_state(self):
        self.h=None
    def forward(self,x,training=True):
        if self.h is None:
            h_new=tanh(self.x2h(x))
        else:
            h_new=tanh(self.x2h(x))+tanh(self.h2h(self.h))

        self.h=h_new
        return self.h

class LSTM(Layer):
    '''比一般RNN更好的时间序列预测层'''
    def __init__(self,hidden_size,in_size=None):
        super().__init__()
        H,I=hidden_size,in_size
        self.x2f=Affine(H,in_size=I)
        self.x2i=Affine(H,in_size=I)
        self.x2o=Affine(H,in_size=I)
        self.x2u=Affine(H,in_size=I)
        self.h2f=Affine(H,in_size=I,nobias=True)
        self.h2i=Affine(H,in_size=I,nobias=True)
        self.h2o=Affine(H,in_size=I,nobias=True)
        self.h2u=Affine(H,in_size=I,nobias=True)

        self.reset_state()
    def reset_state(self):
        self.h=None
        self.c=None

    def forward(self,x,training=True):
        if self.h is None:
            f=sigmoid(self.x2f(x))
            i=sigmoid(self.x2i(x))
            o=sigmoid(self.x2o(x))
            u=tanh(self.x2u(x))
        else:
            f=sigmoid(self.x2f(x)+self.h2f(self.h))
            i=sigmoid(self.x2i(x)+self.h2i(self.h))
            o=sigmoid(self.x2o(x)+self.h2o(self.h))
            u=tanh(self.x2u(x)+self.h2u(self.h))
        if self.c is None:
            c_new=(i*u)
        else:
            c_new=(f*self.c)+(i*u)
        h_new=o*tanh(c_new)
        self.h,self.c=h_new,c_new
        return h_new