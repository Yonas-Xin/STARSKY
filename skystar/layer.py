import os.path
from collections import OrderedDict
from skystar.core import *
from skystar import cuda

# =============================================================================
'''Layer类'''


# =============================================================================
class Layer:
    '''training：某些层的使用分训练和测试两种类型，模型使用时默认training为True，
    如果训练完毕需要使用accurancy预测，请将training设置为False,一些不分测试，训
    练的模型，training的值不影响结果'''

    def __init__(self):
        self._LayerIndex=1
        self._params = set()  # 创建空集合，集合存储了实例的属性名，使用__dict__[name]可以取出属性值，集合的值无序且唯一，便于更新权重
        self._layers=[]

    def __setattr__(self, name, value):  # 重写__setattr__，改变或添加实例属性的值时，会调用__setattr__
        if isinstance(value, (Parameter, Layer, OrderedDict)):
            self._params.add(name)
        if isinstance(value, Layer):
            self._layers.append(name)
        super().__setattr__(name, value)
    def addlayer(self,layer,index=None):
        if index is None:
            name = f"L{self._LayerIndex}_" + layer.__class__.__name__
            self._LayerIndex += 1
            self.__setattr__(name,layer)
        else:
            index-=1
            name = f"Insert_" + layer.__class__.__name__
            self.__setattr__(name,layer)
            name=self._layers.pop()
            self._layers.insert(index,name)

    def deletelayer(self,layernum=None):
        if layernum is None:
            layername=self._layers.pop()
            self._params.remove(layername)
            self.__delattr__(layername)
        else:
            for _ in range(layernum):
                layername=self._layers.pop()
                self._params.remove(layername)
                self.__delattr__(layername)
    def __repr__(self):
        name=self.__class__.__name__
        if self._layers:#layer里面有block的情况
            name+='\n'
            for layername in self._layers:
                layer = self.__dict__[layername]
                name += "    Block:"
                layerinfo = layer.__repr__() + '\n'
                name +=layerinfo
        else:
            if not self._params:
                name += '--NoParams'
            else:
                name += "--Params:"
                for paramname in self._params:#有参数的情况
                    param=self.__dict__[paramname]
                    if param.data is not None:
                        name += f' {paramname}<shape={param.shape} dtype={param.dtype}>'
                    else:
                        name += f' {paramname}<None>'
        return name
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        if len(outputs) > 1:
            return outputs
        else:
            return outputs[0]

    def forward(self, x):
        raise NotImplementedError

    def params(self):  # 生成器
        '''先从模型中迭代Layer属性，再从Layer中迭代它的Parameter属性，由此可迭代出模型里所有Layer的所有_params'''
        for name in self._params:
            obj = getattr(self, name)  # 使用 getattr 获取属性，避免频繁查找
            if isinstance(obj, (Layer, OrderedDict)):  # 合并 Layer 和 OrderedDict 的处理逻辑
                for layer in obj.values() if isinstance(obj, OrderedDict) else [obj]:
                    yield from layer.params()#嵌套
            else:
                yield obj

    def cleangrads(self):
        for param in self.params():
            param.cleangrad()

    def _flatten_params(self, params_dict, parent_key=''):
        '''该函数使得params_dict变为name：Variabl的字典'''
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, filename):
        '''获取当前脚本目录，并在目录下创建model_params用来储存参数'''
        self.to_cpu()
        dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dir, 'model_params', filename)
        params_dict = {}
        self._flatten_params(params_dict)
        if '_blocks' in params_dict:
            val=params_dict.pop('_blocks')
            for key,layer in val.items():
                dict={}
                layer._flatten_params(dict)
                params_dict=params_dict|dict
            array_dict = {key: param.data for key, param in params_dict.items() if param is not None}
        else:
            array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:  # 如果系统中断了正在保存的文件，则将文件删除，避免文件不完整
            np.savez_compressed(filename, **array_dict)
            print(f'Params saved！path:{filename}')
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(filename):
                os.remove(filename)
                print('保存中断，已删除文件')
            raise

    def load_weights(self, filename):
        dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dir, 'model_params', filename)
        if not os.path.exists(filename):  # 如果不存在该文件，直接结束函数
            print('权重文件不存在，请训练网络！')
            return
        npz = np.load(filename,allow_pickle=True)
        params_dict = {}
        self._flatten_params(params_dict)
        if '_blocks' in params_dict:
            val=params_dict.pop('_blocks')
            for key,layer in val.items():
                dict={}
                layer._flatten_params(dict)
                params_dict=params_dict|dict
        for name, param in params_dict.items():
            param.data = npz[name]
        print(f'网络参数加载成功！请注意参数类型为np.ndarray,训练或测试时根据需要转换参数格式path:{filename}')

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def weight_show(self, mode='weight', label=None):
        W = self.W.data
        if W is not None:
            if W.ndim == 4:
                skystar.utils.images_show(W, mode=mode, label=label)
            else:
                print(f'权重值维度不匹配：{W.ndim}！=4')
        else:
            print('权重尚未初始化：None')


# =============================================================================
# 激活函数块，用于Sequential模型组合
# =============================================================================
class ReluBlock(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'ReluBlock'
    def forward(self, x):
        return ReLU(x)
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'SigmoidBlock'
    def forward(self, x):
        return sigmoid(x)

# =============================================================================
# 全连接层
# =============================================================================
class Affine(Layer):
    '''全连接层,只需要输入out_size,in_size可根据要传递的x大小自动计算得出'''

    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.name = 'Affine'
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

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O) * xp.sqrt(1 / I).astype(self.dtype)
        W_data = W_data.astype(self.dtype)
        self.W.data = W_data

    def forward(self, x):
        xp = cuda.get_array_module(x)
        if self.W.data is None:
            self.in_size = x.reshape(x.shape[0], -1).shape[1]  # 如果x的维度是四维，那么变形之后取它的shape[1]
            self._init_W(xp)

        if x.ndim>2:
            x = x.reshape(x.shape[0], -1)
        if self.b is not None:
            y = dot(x, self.W) + self.b
        else:
            y = dot(x, self.W)
        return y

class Gemm(Layer):
    '''矩阵乘,只需要输入out_size,in_size可根据要传递的x大小自动计算得出'''

    def __init__(self, out_size, alpha=1.0, beta=1.0,transA=False, transB=False, nobias=False,
                 dtype=np.float32, in_size=None):
        super().__init__()
        self.name = 'Gemm'
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB

        self.W = Parameter(None, name='B')
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='C')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O) * xp.sqrt(1 / I)
        W_data = W_data.astype(self.dtype)
        self.W.data = W_data

    def forward(self, x):
        xp = cuda.get_array_module(x)
        if self.W.data is None:
            self.in_size = x.reshape(x.shape[0], -1).shape[1]  # 如果x的维度是四维，那么变形之后取它的shape[1]
            self._init_W(xp)
        if x.ndim>2:
            x = x.reshape(x.shape[0], -1)#如果是四维数据，转变为二维
        y = gemm(x, self.W, self.b, self.alpha, self.beta, self.transA, self.transB)
        return y

# =============================================================================
# 批量归一化
# =============================================================================
class BatchNorm(Layer):
    '''self.test_mean,self.test_var:储存全局均值和方差用于模型预测阶段，如果training为True，每次运行forward，数据会更新'''

    def __init__(self, gamma=1.0, beta=0, momentum=0.9):
        super().__init__()
        self.name = 'BatchNorm'
        self.scale=Parameter(np.array(gamma), name="scale")
        self.B=Parameter(np.array(beta), name="B")
        self.input_mean=Parameter(None, name="input_mean")
        self.input_var=Parameter(None, name="input_var")
        self.batchnorm_func = BatchNormalization(momentum=momentum)

    def forward(self, x):
        xp=skystar.cuda.get_array_module(x)
        if self.input_mean.data is None:#参数初始化
            self.input_mean.data = xp.zeros(x.shape[1]).astype('float32')
            self.input_var.data = xp.zeros(x.shape[1]).astype('float32')
            self.scale.data=xp.array([self.scale.data]*x.shape[1]).astype('float32')
            self.B.data=xp.array([self.B.data]*x.shape[1]).astype('float32')
        x = self.batchnorm_func(x,self.scale,self.B,self.input_mean,self.input_var)

        self.input_mean.data=self.batchnorm_func.test_mean#training模式下input_mean会改变
        self.input_var.data=self.batchnorm_func.test_var
        return x


# =============================================================================
# 卷积块
# =============================================================================
class Convolution(Layer):
    '''卷积层：
    FN：核的数量，也是输出的通道数
    FH：核的高
    FW：核的宽
    in_channels：输入的通道数，也是核的通道数'''

    def __init__(self, out_channels, FH, FW, stride=1, pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.name = 'Convolution'
        self.out_channels = out_channels
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
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, FH, FW = self.in_channels, self.FH, self.FW
        W_data = xp.random.randn(self.out_channels, I, FH, FW) * xp.sqrt(1 / (I * FH * FW))
        W_data = W_data.astype(self.dtype)
        self.W.data = W_data

    def forward(self, x):
        xp = cuda.get_array_module(x)
        if self.W.data is None:
            self.in_channels = x.shape[1]
            self._init_W(xp)

        y = convolution(x, self.W, self.b, self.stride, self.pad)
        return y


# =============================================================================
# 反卷积块
# =============================================================================
class Transpose_Convolution(Layer):
    def __init__(self, out_channels, FH, FW, stride=1, pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.name = 'Transpose_Convolution'
        self.out_channels = out_channels
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
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        '''初始化权重'''
        I, out_channels, K = self.in_channels, self.out_channels, self.FW
        W_data = skystar.utils.bilinear_kernel(in_channels=I, out_channels=out_channels, kernel_size=K, xp=xp)
        W_data = W_data.astype(self.dtype)
        self.W.data = W_data

    def forward(self, x):
        xp = cuda.get_array_module(x)
        if self.W.data is None:
            self.in_channels = x.shape[1]
            self._init_W(xp)

        y = transposed_convolution(x, self.W, self.b, self.stride, self.pad)
        return y


# =============================================================================
# 残差块
# =============================================================================
class ResidualBlock(Layer):
    def __init__(self, num_channels, stride=1, nobias=False, dtype=np.float32, use_conv1x1=False):
        super().__init__()
        self.name = 'ResidualBlock'
        self.conv1 = Convolution(FN=num_channels, FH=3, FW=3, stride=stride, pad=1, nobias=nobias, dtype=dtype)
        self.conv2 = Convolution(FN=num_channels, FH=3, FW=3, stride=1, pad=1, nobias=nobias, dtype=dtype)
        if use_conv1x1:
            self.conv3 = Convolution(FN=num_channels, FH=1, FW=1, stride=stride, pad=0, nobias=nobias, dtype=dtype)
        else:
            self.conv3 = None
        self.bn1 = BatchNorm()
        self.bn2 = BatchNorm()

    def forward(self, x):  # （在使用残差块建立网络时），需要注意残差块的前向传播中已经使用了批量归一化与激活函数
        y = self.bn1(self.conv1(x))
        y = ReLU(y)
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = ReLU(y + x)
        return y


# =============================================================================
# 稠密块
# =============================================================================
class DenseBlock(Layer):
    def __init__(self, num_channels, num_convs):
        super().__init__()
        self.name = 'DenseBlock'
        for _ in range(num_convs):
            self.addlayer(BatchNorm())
            self.addlayer(Convolution(FN=num_channels, FH=3, FW=3, stride=1, pad=1, nobias=False))
    def forward(self, x):
        for i in range(len(self._layers)//2):
            y=self.__dict__[self._layers[i]](x)
            y = ReLU(y)
            y=self.__dict__[self._layers[i+1]](y)
            x = concat(x, y, axis=1)
        return y


# =============================================================================
# 过渡层，用在稠密层之后
# =============================================================================
class TransitionBlock(Layer):
    def __init__(self, num_channels):
        super().__init__()
        self.name = 'Transition'
        self.BN = BatchNorm()
        self.Conv1x1 = Convolution(num_channels, 1, 1)
        self.pool1 = Pooling(pool_size=2, stride=2, pad=0, mode='avg')

    def forward(self, x):
        y = self.BN(x)
        y = ReLU(y)
        y = self.Conv1x1(y)
        y = self.pool1(y)
        return y


# =============================================================================
# 池化块
# =============================================================================
class Pooling(Layer):
    '''
    池化层：
    pool_size：池化窗口大小
    stride：步长
    pad：填充
    mode：池化模式，"max" 表示最大池化，"avg" 表示平均池化
    '''
    def __init__(self, pool_size, stride=1, pad=0, mode="max"):
        super().__init__()
        self.name = mode + "pool"
        self.pool_size = pool_size
        self.stride = stride
        self.pad = pad
        self.mode = mode  # 新增参数，选择 "max" 或 "avg"

    def forward(self, x):
        if self.mode == "max":
            y = maxpool(x, self.pool_size, self.stride, self.pad)
        elif self.mode == "avg":
            y = avgpool(x, self.pool_size, self.stride, self.pad)
        else:
            raise ValueError("mode 参数必须是 'max' 或 'avg'")
        return y


# =============================================================================
# 裁剪复制块，用于U-net
# =============================================================================
class Slice(Layer):
    def __init__(self, starts,ends,axis=None, steps=None):
        super().__init__()
        self.name = 'Slice'
        if axis is None:
            axis = list(range(len(starts)))  # 默认为所有轴
        if steps is None:
            steps = [1] * len(starts)  # 默认为步长为 1
        if isinstance(starts, list):
            starts = np.array(starts).astype(np.int32)
            ends = np.array(ends).astype(np.int32)
            steps = np.array(steps).astype(np.int32)
            axis = np.array(axis).astype(np.int32)
        self.starts = Parameter(starts, name='starts')
        self.ends = Parameter(ends, name='ends')
        self.axis = Parameter(axis, name='axis')
        self.steps = Parameter(steps, name='steps')
    def forward(self, x):
        return my_slice(x, self.starts, self.ends, self.axis, self.steps)
class CopyAndCrop(Layer):
    def __init__(self, cropsize):
        super().__init__()
        self.name = 'CopyAndCrop'
        self.cropsize = cropsize

    def forward(self, x):
        N, C, H, W = x.shape
        crop_h, crop_w = self.cropsize
        cropmid_h = int(crop_h / 2)
        cropmid_w = int(crop_w / 2)
        mid_h, mid_w= H // 2,W // 2

        min_h = mid_h - cropmid_h
        min_w = mid_w - cropmid_w
        max_h = mid_h + cropmid_h
        max_w = mid_w + cropmid_w
        if crop_h % 2 > 0:
            max_h += 1
        if crop_w % 2 > 0:
            max_w += 1
        return Slice([0,0,min_h,min_w],[N,C,max_h,max_w])(x)

# =============================================================================
# 循环神经块
# =============================================================================
class RNN(Layer):
    '''self.h:既是自己的状态，也是自己的输出，自己的状态状态同时影响了输出'''

    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.name = 'RNN'
        self.x2h = Affine(hidden_size, in_size=in_size)
        self.h2h = Affine(hidden_size, in_size=in_size, nobias=True)  # 不要偏置b
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = tanh(self.x2h(x))
        else:
            h_new = tanh(self.x2h(x)) + tanh(self.h2h(self.h))

        self.h = h_new
        return self.h


# =============================================================================
# 长短时记忆块
# =============================================================================
class LSTM(Layer):
    '''比一般RNN更好的时间序列预测层'''

    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.name = 'LSTM'
        H, I = hidden_size, in_size
        self.x2f = Affine(H, in_size=I)
        self.x2i = Affine(H, in_size=I)
        self.x2o = Affine(H, in_size=I)
        self.x2u = Affine(H, in_size=I)
        self.h2f = Affine(H, in_size=I, nobias=True)
        self.h2i = Affine(H, in_size=I, nobias=True)
        self.h2o = Affine(H, in_size=I, nobias=True)
        self.h2u = Affine(H, in_size=I, nobias=True)

        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = sigmoid(self.x2f(x))
            i = sigmoid(self.x2i(x))
            o = sigmoid(self.x2o(x))
            u = tanh(self.x2u(x))
        else:
            f = sigmoid(self.x2f(x) + self.h2f(self.h))
            i = sigmoid(self.x2i(x) + self.h2i(self.h))
            o = sigmoid(self.x2o(x) + self.h2o(self.h))
            u = tanh(self.x2u(x) + self.h2u(self.h))
        if self.c is None:
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)
        h_new = o * tanh(c_new)
        self.h, self.c = h_new, c_new
        return h_new
