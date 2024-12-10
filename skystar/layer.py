import os.path
import weakref
import skystar.utils
import numpy as np
import time
from skystar.graph import create_graph, save_graph
from skystar.core import Parameter, sigmoid, ReLU, dot, gemm, BatchNormalization, convolution, transposed_convolution, \
    dropout, concat, maxpool, my_slice, avgpool, broadcast_to, gather, tanh, layernorm, my_mean, transpose, \
    Variable, softmax, matmul, as_variable
from skystar import cuda
from skystar.tansformer import padding_mask, sequence_mask


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
        if isinstance(value, (Parameter, Layer)):
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
    def setname(self,name):
        self.name=name
    def set_trainingmode(self, trainingmode):
        skystar.core.Set_TrainingMode(trainingmode)
    def __repr__(self,blank=''):
        name=self.name
        if self._layers:#layer里面有block的情况
            blank+=' '
            name+=('\n'+blank)
            for layername in self._layers:
                layer = self.__dict__[layername]
                name += "Block:"
                layerinfo = layer.__repr__(blank)+'\n'+blank
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
            if isinstance(obj, Layer):  # 合并 Layer 的处理逻辑
                    yield from obj.params()#嵌套,迭代出layer的所有参数
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
        if filename is None:
            raise NameError('Must suitable filename')
        self.to_cpu()
        date=time.strftime("%Y.%m.%d-%H%M%S")
        if '.npz' not in filename:
            filename = filename+date+'.npz'
        else:
            filename = filename.replace('.npz', date+'.npz')
        dir = os.getcwd()
        dir = os.path.join(dir, 'model_params')
        if not os.path.exists(dir):
            os.makedirs(dir)
        filename = os.path.join(dir, filename)
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
            print(f'Weight params saved！path:{filename}')
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(filename):
                os.remove(filename)
                print('保存中断，已删除文件')
            raise

    def load_weights(self, filename):
        if not os.path.exists(filename):
            dir = os.getcwd()
            filename = os.path.join(dir, 'model_params',filename)
        if not os.path.exists(filename):  # 如果不存在该文件，直接结束函数
            print('The network parameters are not exist！path:{}'.format(filename))
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
        print(f'The network parameters are loaded successfully！The params type:np.ndarray path:{filename}')

    def save_to_onnx(self, *inputs, name=None, ifsimplify=True, version=17):#17版支持layernorm
        '''
        :param input: 需要使用一个模型的输入
        :param name: 不支持绝对路径
        :return:
        '''
        model_name = self.__class__.__name__
        date=time.strftime("%Y.%m.%d-%H%M%S")
        if name is None:
            name = model_name + date + ".onnx"
        else:
            if '.onnx' not in name:
                name = name + date+ '.onnx'
            else:
                name = name.replace('.onnx', date+'.onnx')
        self.to_cpu()#把模型数据和输入全部变为np.array格式
        inputs =[skystar.cuda.as_numpy(input) for input in inputs]

        dir = os.getcwd()
        dir = os.path.join(dir, 'model_params')
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(dir, name)
        y = self(*inputs)
        graph = create_graph(y)
        save_graph(graph, model_name, file_name=path, ifsimplify=ifsimplify, version=version)
        return

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    # def weight_show(self, mode='weight', label=None):
    #     W = self.W.data
    #     if W is not None:
    #         if W.ndim == 4:
    #             skystar.utils.images_show(W, mode=mode, label=label)
    #         else:
    #             print(f'权重值维度不匹配：{W.ndim}！=4')
    #     else:
    #         print('权重尚未初始化：None')

# =============================================================================
'''使用Sequential类自由组合block，打造model。自由生成的Sequential可以使用Model的所有函数'''
# =============================================================================
class Sequential(Layer):
    def __init__(self,*layers):
        super().__init__()
        self.name='Sequential'
        if not isinstance(layers,tuple):
            layers=(layers,)
        for layer in layers:
            self.addlayer(layer)

    def forward(self,x):
        for layername in self._layers:
            layer = self.__dict__[layername]
            x = layer(x)
        return x

    def CreateModel(self,model):
        for layername in model._layers:
            layer = model.__dict__[layername]
            self.__setattr__(layername,layer)

# =============================================================================
# 激活函数块，用于Sequential模型组合
# =============================================================================
class ReluBlock(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'ReluBlock'
    def forward(self, x):
        return ReLU(x)
class SigmoidBlock(Layer):
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
        self.scale=Parameter(np.array(gamma,dtype=np.float32), name="scale")
        self.B=Parameter(np.array(beta,dtype=np.float32), name="B")
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
class DropoutBlock(Layer):
    def __init__(self, rate=0.5):
        super().__init__()
        self.name = 'DropoutBlock'
        self.rate = rate
    def forward(self, x):
        return dropout(x, self.rate)

# =============================================================================
# 残差块
# =============================================================================
class ResidualBlock(Layer):
    def __init__(self, num_channels, stride=1, nobias=False, dtype=np.float32, use_conv1x1=False):
        super().__init__()
        self.name = 'ResidualBlock'
        self.conv1 = Convolution(out_channels=num_channels, FH=3, FW=3, stride=stride, pad=1, nobias=nobias, dtype=dtype)
        self.conv2 = Convolution(out_channels=num_channels, FH=3, FW=3, stride=1, pad=1, nobias=nobias, dtype=dtype)
        if use_conv1x1:
            self.conv3 = Convolution(out_channels=num_channels, FH=1, FW=1, stride=stride, pad=0, nobias=nobias, dtype=dtype)
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
            self.addlayer(Convolution(out_channels=num_channels, FH=3, FW=3, stride=1, pad=1, nobias=False))
    def forward(self, x):
        for i in range(len(self._layers)//2):
            y=self.__dict__[self._layers[i]](x)
            y = ReLU(y)
            y=self.__dict__[self._layers[i+1]](y)
            x = concat(x, y, axis=1)
        return x


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
        return my_slice(x,[0,0,min_h,min_w],[N,C,max_h,max_w])
# =============================================================================
# 嵌入层
# =============================================================================
class Embedding(Layer):
    def __init__(self, num_embeddings, embedding_dim):
        '''
        输入为（N，seq_len），输出为（N,seq_len,embedding_dim）
        嵌入层的实现有以下两个方法：
        1、把输入变为onehot形式（N,seq_len,num_embeddings）,与参数矩阵W（num_embeddings,embedding_dim）点乘拼接
            得到输出（N,seq_len,embedding_dim）
        2、参数矩阵W为（num_embeddings,embedding_dim），根据（N，seq_len）索引直接映射输出（N,seq_len,embedding_dim）
        本层使用第二种方法
        '''
        super().__init__()
        self.name = 'Embedding'
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.We=Parameter(np.random.randn(num_embeddings, embedding_dim) * np.sqrt(1 / num_embeddings), name='W_embedding')
        self.We.data=self.We.data.astype(np.float32)
    def forward(self, x):
        batch_size, seq_len = x.shape
        x=skystar.core.reshape(x,(batch_size, seq_len, 1))#指定使用reshape建立连接
        We=broadcast_to(self.We,(batch_size,self.num_embeddings,self.embedding_dim))
        # x=broadcast_to(x,(batch_size,seq_len,self.embedding_dim))
        y=gather(We,x,axis=1)#目前gather函数专用于embedding层
        return y

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

# =============================================================================
# 层归一化，用于多头注意力的transformer模型
# =============================================================================
class LayerNorm(Layer):
    def __init__(self, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.name='LayerNorm'
        self.gamma=Parameter(None, name='gamma')
        self.beta=Parameter(None, name='beta')
        self.variance_epsilon = eps
    def forward(self, x):
        xp = skystar.cuda.get_array_module(x)
        if self.gamma.data is None:
            self.gamma.data=xp.ones(x.shape[-1],dtype=xp.float32)
            self.beta.data=xp.zeros(x.shape[-1],dtype=xp.float32)
        return layernorm(x,self.gamma,self.beta,self.variance_epsilon)
class TestLayerNorm(Layer):
    def __init__(self, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(TestLayerNorm, self).__init__()
        self.name='LayerNorm'
        self.W=Parameter(None, name='W')
        self.b=Parameter(None, name='b')
        self.variance_epsilon = eps

    def forward(self, x):
        xp = skystar.cuda.get_array_module(x)
        if self.W.data is None:
            self.W.data=xp.ones(x.shape[-1],dtype=xp.float32)
            self.b.data=xp.zeros(x.shape[-1],dtype=xp.float32)
        u = my_mean(x, -1, keepdims=True)
        s = my_mean(pow((x - u),2),-1, keepdims=True)
        x = (x - u) / pow(s + self.variance_epsilon,0.5)
        return self.W*x+self.b
class PositionalEncoding(Layer):
    def __init__(self, d_model, dropout_rate=0.1, max_len=100):
        '''根据三角函数生成的位置编码
        input：[batch_size, seq_len, d_model]
        max_len: 最长序列编码，要大于输入数据序列长度seq_len
        d_model：生成编码的输出维度（batch， seq_len, d_model）
        '''
        super(PositionalEncoding, self).__init__()
        self.name = 'PositionalEncoding'
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = DropoutBlock(dropout_rate)
        self.pos_table = None
    def init_table(self,xp):
        pos_table = xp.array([
            [pos / xp.power(10000, 2 * i / self.d_model) for i in range(self.d_model)]
            if pos != 0 else xp.zeros(self.d_model) for pos in range(self.max_len)],dtype=xp.float32)
        pos_table[1:, 0::2] = xp.sin(pos_table[1:, 0::2],dtype=xp.float32)  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = xp.cos(pos_table[1:, 1::2],dtype=xp.float32)  # 字嵌入维度为奇数时
        self.pos_table = pos_table  # enc_inputs: [seq_len, d_model]
    def forward(self, enc_inputs):  # enc_inputs: [batch_size, seq_len, d_model]
        xp = cuda.get_array_module(enc_inputs)
        if self.pos_table is None:
            self.init_table(xp)
        enc_inputs += self.pos_table[:enc_inputs.shape[1], :]
        return self.dropout(enc_inputs)
# =============================================================================
# 自注意力与多头注意力
# =============================================================================
class Self_Attention(Layer):
    '''自注意力机制'''
    def __init__(self,hidden_size):
        '''假设输入向量x（num，num_ebedding）'''
        super().__init__()
        self.name = 'Self_Attention'
        self.wq=Gemm(hidden_size,nobias=True)#查询向量矩阵，（num_ebedding，num_out）
        self.wq.setname('W_query')
        self.wk=Gemm(hidden_size,nobias=True)
        self.wk.setname('W_key')
        self.wv=Gemm(hidden_size,nobias=True)
        self.wv.setname('W_value')

    def forward(self, x):
        '''输入一个词向量，词向量是一个一位矩阵，多个词输入即为二维矩阵'''
        xp=cuda.get_array_module(x)
        query=self.wq(x)#查询向量,(num,hidden_size)
        key=self.wk(x)#键向量,(num,hidden_size)
        value=self.wv(x)
        scalar=Variable(xp.array([8],dtype=xp.float32),name='scalar')#取得标量，设为8
        score=dot(query,transpose(key,axes=(-1,-2)))/scalar#打分,(num,num)
        score_softmax=softmax(score,axis=-1)#归一化(num,num)，沿着行方向
        Z=dot(score_softmax,value)
        return Z#(num,hidden_size)


# noinspection PyTestUnpassedFixture
class MultiHeadAttention(Layer):
    '''多头注意力机制，包含多个自注意力,与一个额外矩阵'''
    def __init__(self,head_num,dkv,embedding_dim=512,mask='padding_mask'):
        super().__init__()
        self.name = 'MultiHeadAttention'
        self.head_num=head_num
        self.hidden_size=dkv
        self.mask=mask
        self.wq=Gemm(dkv*head_num,nobias=True)
        self.wk=Gemm(dkv*head_num,nobias=True)
        self.wv=Gemm(dkv*head_num,nobias=True)
        self.fc=Gemm(embedding_dim,nobias=True)
        self.layer_norm=LayerNorm(eps=1e-5)

        self.wq.setname('W_query')
        self.wk.setname('W_key')
        self.wv.setname('W_value')
    def forward(self, input_Q,input_K,input_V, mask):#input[batch,sqr_len,embedding_dim]
        '''input_Q,input_K,input_V是已经embedding的词向量，用来计算Q，K，V'''
        xp = skystar.cuda.get_array_module(input_Q)
        residual = input_Q

        batch, seqlen, embedding_dim = input_Q.shape
        input_Q = input_Q.reshape(batch * seqlen, embedding_dim)
        input_K = input_K.reshape(batch * seqlen, embedding_dim)
        input_V = input_V.reshape(batch * seqlen, embedding_dim)

        query = self.wq(input_Q)  # 查询向量,(batch*seqlen,hidden_size*head_num)
        key = self.wk(input_K)  # 键向量,(batch*seqlen,hidden_size*head_num)
        value = self.wv(input_V)  # 值向量，(batch*seqlen,hidden_size*head_num)

        # (batch,head_num,sqrlen,hidden_size)
        query = query.reshape(batch, seqlen, self.head_num, self.hidden_size).transpose(0, 2, 1,3)
        key = key.reshape(batch, seqlen, self.head_num, self.hidden_size).transpose(0, 2, 1,3)
        value = value.reshape(batch, seqlen, self.head_num, self.hidden_size).transpose(0, 2, 1,3)
        scalar = Variable(xp.array([xp.sqrt(self.hidden_size / self.head_num)], dtype=xp.float32),name='scalar')
        score = matmul(query, transpose(key, axes=(0, 1, 3, 2))) / scalar # 打分,(batch,head_num,sqrlen,sqrlen)

        mask = mask.reshape(batch, 1, seqlen, seqlen)  # (batch,head_num,sqrlen,sqrlen)
        score -= mask  # 掩膜pad信息, 这里采用的是减去极大值(batch,head_num,sqrlen,sqrlen)
        self.attention = softmax(score, axis=-1)  # 归一化(num,num)，沿着行方向(batch,head_num,sqrlen,sqrlen)
        Z = matmul(self.attention, value)  # (batch,head_num,sqrlen,hidden_size)
        Z = Z.transpose(0, 2, 1, 3).reshape(batch * seqlen, -1)
        Z = self.fc(Z)  # (batch,sqrlen,embedding_dim)
        Z = Z.reshape(batch, seqlen, -1)
        return self.layer_norm(Z + residual)  # 残差结构

class Feed_forward(Layer):
    def __init__(self,hidden_size,embedding_dim=512):
        super().__init__()
        self.name = 'Feed_forward'
        self.gemm1=Gemm(hidden_size)
        self.gemm2=Gemm(embedding_dim)
        self.LayerNorm=LayerNorm()
    def forward(self,x):#(batch,sqrlen,embedding_dim)
        batch, sqrlen, embedding_dim=x.shape
        x1=x.reshape(batch*sqrlen,embedding_dim)
        y=ReLU(self.gemm1(x1))#(batch*sqrlen,hidden_size)
        y=self.gemm2(y)#(batch*sqrlen,embedding_dim)
        y=y.reshape(batch,sqrlen,embedding_dim)
        y=self.LayerNorm(y+x)
        return y
class Encoder(Layer):
    def __init__(self,word_num=10,embedding_dim=512, dff=2048, dkv=64, n_heads=8):
        '''
        :param word_num:目标单词的总数
        :param embedding_dim:词嵌入的维度
        :param dff:前向传播层的隐藏层维度
        :param dkv:自注意力层k，v的维度
        :param n_heads:多头注意力的头数
        '''
        super().__init__()
        self.name='Encoder'
        self.embedding=Embedding(word_num,embedding_dim)
        self.positional_encoding=PositionalEncoding(embedding_dim)
        self.MutiHead=MultiHeadAttention(n_heads,dkv=dkv,embedding_dim=embedding_dim)
        self.fc=Sequential(Feed_forward(dff),
                           Feed_forward(dff),
                           Feed_forward(dff),
                           Feed_forward(dff),
                           Feed_forward(dff),)
    def forward(self,x):
        mask=padding_mask(x,x)*1e9#(batch,sqr_len,sqr_len)
        x=self.embedding(x)#[batch,sqr_len,embedding_dim]
        x=self.positional_encoding(x)#[batch,sqr_len,embedding_dim]
        x=self.MutiHead(x,x,x,mask)
        x=self.fc(x)
        return x

class Decoder(Layer):
    def __init__(self,word_num=10,embedding_dim=512,dkv=64,dff=2048,n_heads=8):
        '''
        :param word_num:目标单词的总数
        :param embedding_dim:词嵌入的维度
        :param dff:前向传播层的隐藏层维度
        :param dkv:自注意力层k，v的维度
        :param n_heads:多头注意力的头数
        '''
        super().__init__()
        self.name='Decoder'
        self.embedding=Embedding(word_num,embedding_dim)
        self.positional_encoding=PositionalEncoding(embedding_dim)
        self.MutiHead=MultiHeadAttention(n_heads,dkv,embedding_dim=embedding_dim)
        self.CorssMutiHead=MultiHeadAttention(n_heads,dkv,embedding_dim=embedding_dim)
        self.fc=Sequential(Feed_forward(dff),
                           Feed_forward(dff),
                           Feed_forward(dff),
                           Feed_forward(dff),)

    def forward(self, dec_input, enc_input, enc_output):
        '''
        x[batch,sqrlen_tgt]
        enc_input[batch,sqrlen_src]
        enc_output[batch,sqrlen_src,embedding_dim]
        '''
        #获取mask信息
        mask1=sequence_mask(dec_input)
        mask2=padding_mask(dec_input,dec_input)
        mh_mask=(mask1+mask2)*1e9

        cmh_mask=padding_mask(dec_input,enc_input)*1e9
        #词向量化
        dec_output=self.embedding(dec_input)#[batch,sqr_len,embedding_dim]
        dec_output=self.positional_encoding(dec_output)#[batch,sqr_len,embedding_dim]

        #输入输出
        dec_output=self.MutiHead(dec_output,dec_output,dec_output,mh_mask)
        dec_output=self.CorssMutiHead(dec_output,enc_output,enc_output,cmh_mask)
        dec_output=self.fc(dec_output)
        return dec_output