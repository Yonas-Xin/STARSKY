import os.path

from skystar.layer import *
from skystar.utils import plot_dot_graph
from skystar.graph import create_graph, save_graph
from skystar.optimizer import Adam
from tqdm import tqdm  # 添加进度条
import time

# =============================================================================
'''model类（继承自Layer）'''
# =============================================================================
class Model(Layer):
    def plot(self, *inputs):
        y = self.forward(*inputs)
        plot_dot_graph(y)

    def predict(self, *inputs):
        y = self.forward(*inputs)
        return y

    def __repr__(self):
        info='----------'+self.__class__.__name__+'----------\n'
        for layername in self._layers:
            layer=self.__dict__[layername]
            info+=layer.__repr__()+'\n'
        return info

    def accuracy(self, x_test, t):
        with no_grad():
            y = self.predict(x_test)
            if t.ndim == 2:
                t = np.argmax(t, axis=1)
            y.data = np.argmax(y.data, axis=1)
            sum = t.size
            return np.sum(y.data == t) / sum

    def save_to_onnx(self, input, name=None, ifsimplify=True):
        '''
        :param input: 需要使用一个模型的输入
        :param name: 不支持绝对路径
        :return:
        '''
        model_name = self.__class__.__name__
        if name is None:
            name = model_name + ".onnx"
        self.to_cpu()#把模型数据和输入全部变为np.array
        input = skystar.cuda.as_numpy(input)
        dir = os.getcwd()
        dir = os.path.join(dir, 'model_params')
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(dir, name)
        y = self.forward(input)
        graph = create_graph(y)
        save_graph(graph, model_name, file_name=path, ifsimplify=ifsimplify)
        return

    def train(self, train, test=None, lr=0.001, epoch=300, plot=False, plot_rate=0.1,
              loss_func=skystar.core.softmaxwithloss,
              accuracy=skystar.utils.accuracy, optimizer=Adam, use_gpu=True, save_model=True, autodecrese=False):
        '''
        :param train: 训练数据迭代器，需要dataloader
        :param lr: 学习率，默认为0.001
        :param epoch: 学习轮次
        :param test: 测试数据迭代器，默认为None
        :param loss_func: 指定损失函数，默认为softmaxwithloss
        :param accuracy: 指定计算准确率的函数，默认为工具中的accuracy
        :param plot: 是否在过程中画图，画图功能只有在test中才可用
        :param plot_rate: 画图比率
        :param optimizer: 优化器，默认为Adam
        :param use_gpu: 是否使用gpu，默认为Ture
        :return: 无返回，后续考虑返回准确率的和损失值的列表
        '''
        print('Training begin：')
        if use_gpu:
            print('GPU is available, and all parameters are converted to cp.ndarray')
            if skystar.cuda.gpu_enable:
                self.to_gpu()
                train.to_gpu()
                if test is not None:
                    test.to_gpu()
        self.list = [[], [], []]
        optimizer = optimizer(lr).setup(self)
        for i in range(epoch):
            if not Get_TrainingMode():
                Set_TrainingMode(True)
            if i % int(epoch // 2) == 0 and autodecrese:
                '''后期训练把学习率降低'''
                lr = lr / 10
                optimizer.lr = lr
            print(f'Epoch {i + 1}:')
            sum_acc = 0.0
            sum_loss = 0.0
            time.sleep(0.1)  # 停顿0.1秒，避免输出条与输出字符串同步
            # 使用 tqdm 包裹训练数据集
            for x, t in tqdm(train, desc='Training', total=len(train) / train.batch_size):
                y = self(x)
                loss = loss_func(y, t)
                self.cleangrads()
                loss.backward()
                optimizer.update()
                if accuracy is not None:
                    sum_acc += accuracy(y, t) * len(t)
                sum_loss += loss.data

            print(f'Train_Loss {sum_loss / len(train)}')
            self.list[0].append(sum_loss / len(train))
            if accuracy is not None:
                print(f'Train_Acc {sum_acc / len(train)}')
                self.list[1].append(sum_acc / len(train))

            time.sleep(0.1)  # 停顿0.1秒，避免输出条与输出字符串同步
            # 如果输入了测试集，则进行网络测试
            if test is not None:
                if Get_TrainingMode():
                    Set_TrainingMode(False)
                sum_acc = 0.0
                num = 1 * test.batch_size
                plot_num = int(plot_rate * len(test))
                with skystar.core.no_grad():
                    for x, t in tqdm(test, desc='Testing', total=len(test) // test.batch_size):
                        y = self(x)
                        if accuracy is not None:
                            sum_acc += accuracy(y, t) * len(t)
                        if num % plot_num == 0 and plot:
                            if i == 0: Variable(x).image_show(mode='feature', label=f' Epoch{i + 1} Real' + str(num))
                            if t.ndim >= 3:
                                if i == 0: Variable(t).image_show(mode='label', label=f' Epoch{i + 1} Real' + str(num))
                                y_predict = Variable(y.data.argmax(axis=1))
                                y_predict.image_show(mode='label', label=f' Epoch{i + 1} Prediction' + str(num))

                        num += 1 * test.batch_size
                print(f'Test_Acc: {sum_acc / len(test)}')
                self.list[2].append(sum_acc / len(test))
        if save_model:
            name = self.__class__.__name__
            print('Saving the model params......')
            self.save_weights(filename=name)

    def test(self, test, use_gpu=True, plot=True, plot_rate=0.1, accuracy=skystar.utils.accuracy):
        Set_TrainingMode(False)
        print('Test begin：')
        if use_gpu:
            print('GPU is available, and all parameters are converted to cp.ndarray')
            if skystar.cuda.gpu_enable:
                self.to_gpu()
                test.to_gpu()
        sum_acc = 0.0
        num, i = 1 * test.batch_size, 0
        plot_num = int(plot_rate * len(test))
        with skystar.core.no_grad():
            for x, t in tqdm(test, desc='Testing', total=len(test) // test.batch_size):
                y = self(x)
                if accuracy is not None:
                    sum_acc += accuracy(y, t) * len(t)
                if num % plot_num == 0 and plot:
                    Variable(x).image_show(mode='feature', label=f' Real' + str(num))
                    if t.ndim >= 3:
                        Variable(t).image_show(mode='label', label=f' Real' + str(num))
                        y_predict = Variable(y.data.argmax(axis=1))
                        y_predict.image_show(mode='label', label=f' Prediction' + str(num))
                num += 1 * test.batch_size
        print(f'Test_Acc: {sum_acc / len(test)}')

    def TrainForSeq(self, train, test=None, lr=0.001, epoch=300, BPTT=30,loss_func=skystar.core.mean_squared_error,
                    optimizer=Adam, use_gpu=True, save_model=True, autodecrese=False, leave=True, sleep=False):
        '''专门用于序列模型的训练函数'''
        Set_TrainingMode(True)
        print('Training begin：')
        if use_gpu:
            print('GPU is available, and all parameters are converted to cp.ndarray')
            if skystar.cuda.gpu_enable:
                self.to_gpu()
                train.to_gpu()
                if test is not None:
                    test.to_gpu()
        self.list = [[], [], []]
        optimizer = optimizer(lr).setup(self)
        for i in range(epoch):#训练开始
            self.reset_state()
            if i % int(epoch // 2) == 0 and autodecrese:
                '''后期训练把学习率降低'''
                lr = lr / 10
                optimizer.lr = lr
            print(f'Epoch {i + 1}:')
            loss,count = 0.0, 0
            if sleep:
                time.sleep(0.1)  # 停顿0.1秒，避免输出条与输出字符串同步
            for x, t in tqdm(train, desc='Training', total=len(train), leave=leave):
                y=self(x)
                loss+=loss_func(y, t)
                count+=1
                if count%BPTT == 0 or count == len(train):
                    self.cleangrads()
                    loss.backward()
                    loss.unchain_backward()
                    optimizer.update()
            avg_loss = loss / count
            self.list[0].append(avg_loss)
            print(f'Train_Loss {avg_loss}')
        if save_model:
            name = self.__class__.__name__
            print('Saving the model params......')
            self.save_weights(filename=name)


# =============================================================================
'''使用Sequential类自由组合block，打造model。自由生成的Sequential可以使用Model的所有函数'''
# =============================================================================
class Sequential(Model):
    def __init__(self):
        super().__init__()

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
'''一些经典的model'''
# =============================================================================
class MLP(Model):
    '''fc_output_sizes：tuple，输入层的神经元个数，全连接层神经网络'''

    def __init__(self, fc_output_sizes, activation=sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = Gemm(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


class Batch_norm_MLP(Model):
    '''添加了batchnorm的全连接层，分测试和训练阶段'''

    def __init__(self, fc_output_sizes, activation=sigmoid, gamma=1.0, beta=0, momentum=0.9):
        super().__init__()
        self.activation = activation
        self.layers = []
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum

        for i, outsize in enumerate(fc_output_sizes):
            layer1 = Affine(outsize)
            setattr(self, 'Affine' + str(i), layer1)
            self.layers.append(layer1)
            layer2 = BatchNorm(self.gamma, self.beta, self.momentum)
            setattr(self, 'Batch_norm' + str(i), layer2)
            self.layers.append(layer2)

    def forward(self, x):
        for i in range(0, len(self.layers) - 1):
            layer = self.layers[i]
            x = layer(x)
            if i % 2 != 0:
                x = self.activation(x)

        layer = self.layers[-1]
        x = layer(x)
        return x


class Batchnorm_dropout_MLP(Model):
    '''添加了batchnorm的dropout的全连接层，分测试和训练阶段,dropout层在sigmoid后面'''

    def __init__(self, fc_output_sizes, activation=sigmoid, use_dropout=False, dropout_ratio=0.5, gamma=1.0, beta=0,
                 momentum=0.9):
        super().__init__()
        self.activation = activation
        self.layers = []
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum

        for i, outsize in enumerate(fc_output_sizes):
            layer1 = Affine(outsize)
            setattr(self, 'Affine' + str(i), layer1)
            self.layers.append(layer1)
            layer2 = BatchNorm(self.gamma, self.beta, self.momentum)
            setattr(self, 'Batch_norm' + str(i), layer2)
            self.layers.append(layer2)

    def forward(self, x):
        for i in range(0, len(self.layers) - 1):
            layer = self.layers[i]
            x = layer(x)
            if i % 2 != 0:
                x = self.activation(x)
                if self.use_dropout:
                    x = dropout(x)

        layer = self.layers[-1]
        x = layer(x)
        return x


class Simple_CNN(Model):
    def __init__(self, activation=ReLU):
        super().__init__()
        self.activation = activation
        self.convolution1 = Convolution(16, 3, 3)
        self.pooling1 = Pooling(2, stride=2)
        self.convolution2 = Convolution(32, 3, 3)
        self.pooling2 = Pooling(2, stride=2)
        self.affine = Gemm(10)

    def forward(self, x):
        y1 = self.convolution1(x)
        y1 = self.activation(y1)
        y1 = self.pooling1(y1)

        y2 = self.convolution2(y1)
        y2 = self.activation(y2)
        y2 = self.pooling2(y2)

        out = self.affine(y2)
        return out


class Simple_RNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.RNN = RNN(hidden_size=hidden_size)  # 输出状态
        self.affine = Affine(out_size)  # 使用affine层输出状态

    def reset_state(self):
        self.RNN.reset_state()

    def forward(self, x):
        x = self.RNN(x)
        x = self.affine(x)

        return x


class Better_RNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.RNN = LSTM(hidden_size=hidden_size)
        self.affine = Affine(out_size)

    def reset_state(self):
        self.RNN.reset_state()

    def forward(self, x):
        x = self.RNN(x)
        x = self.affine(x)

        return x


class VGG(Model):
    def __init__(self, output=1000, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.conv1_1 = Convolution(16, 3, 3, pad=1)
        self.conv1_2 = Convolution(16, 3, 3, pad=1)
        self.pool1 = Pooling(2, stride=2)
        self.conv2_1 = Convolution(128, 3, 3, pad=1)
        self.conv2_2 = Convolution(128, 3, 3, pad=1)
        self.pool2 = Pooling(2, stride=2)
        self.conv3_1 = Convolution(256, 3, 3, pad=1)
        self.conv3_2 = Convolution(256, 3, 3, pad=1)
        self.conv3_3 = Convolution(256, 3, 3, pad=1)
        self.pool3 = Pooling(2, stride=2)
        self.conv4_1 = Convolution(512, 3, 3, pad=1)
        self.conv4_2 = Convolution(512, 3, 3, pad=1)
        self.conv4_3 = Convolution(512, 3, 3, pad=1)
        self.pool4 = Pooling(2, stride=2)
        self.conv5_1 = Convolution(512, 3, 3, pad=1)
        self.conv5_2 = Convolution(512, 3, 3, pad=1)
        self.conv5_3 = Convolution(512, 3, 3, pad=1)
        self.pool5 = Pooling(2, stride=2)
        self.affine1 = Affine(4096)
        self.affine2 = Affine(4096)
        self.affine3 = Affine(output)

    def forward(self, x):
        '''激活函数用ReLU'''
        x = ReLU(self.conv1_1(x))
        x = ReLU(self.conv1_2(x))
        x = self.pool1(x)
        x = ReLU(self.conv2_1(x))
        x = ReLU(self.conv2_2(x))
        x = self.pool2(x)
        x = ReLU(self.conv3_1(x))
        x = ReLU(self.conv3_2(x))
        x = ReLU(self.conv3_3(x))
        x = self.pool3(x)
        x = ReLU(self.conv4_1(x))
        x = ReLU(self.conv4_2(x))
        x = ReLU(self.conv4_3(x))
        x = self.pool4(x)
        x = ReLU(self.conv5_1(x))
        x = ReLU(self.conv5_2(x))
        x = ReLU(self.conv5_3(x))
        x = self.pool5(x)
        x = ReLU(self.affine1(x))
        x = dropout(x, ratio=self.ratio)
        x = ReLU(self.affine2(x))
        x = dropout(x, ratio=self.ratio)
        x = self.affine3(x)

        return x


class STL_10_CNN(Model):
    def __init__(self, ratio=0.5, output_size=10):
        super().__init__()
        self.ratio = ratio
        self.conv1_1 = Convolution(16, 3, 3, pad=1)
        self.conv1_2 = Convolution(16, 3, 3, pad=1)
        self.pool1 = Pooling(2, stride=2)
        self.conv2_1 = Convolution(128, 3, 3, pad=1)
        self.conv2_2 = Convolution(128, 3, 3, pad=1)
        self.pool2 = Pooling(2, stride=2)
        self.conv3_1 = Convolution(256, 3, 3, pad=1)
        self.conv3_2 = Convolution(256, 3, 3, pad=1)
        self.pool3 = Pooling(2, stride=2)
        # self.conv4_1=Convolution(512,3,3,pad=1)
        # self.conv4_2=Convolution(512,3,3,pad=1)
        # self.pool4=Pooling(2,stride=2)
        # self.conv5_1=Convolution(512,3,3,pad=1)
        # self.conv5_2=Convolution(512,3,3,pad=1)
        # self.pool5=Pooling(2,stride=2)
        self.affine1 = Affine(100)
        self.affine2 = Affine(100)
        self.affine3 = Affine(output_size)

    def forward(self, x):
        '''激活函数用ReLU'''
        x = ReLU(self.conv1_1(x))
        x = ReLU(self.conv1_2(x))
        x = self.pool1(x)
        x = ReLU(self.conv2_1(x))
        x = ReLU(self.conv2_2(x))
        x = self.pool2(x)
        x = ReLU(self.conv3_1(x))
        x = ReLU(self.conv3_2(x))
        x = self.pool3(x)
        # x=ReLU(self.conv4_1(x))
        # x=ReLU(self.conv4_2(x))
        # x=self.pool4(x)
        # x=ReLU(self.conv5_1(x))
        # x=ReLU(self.conv5_2(x))
        # x=self.pool5(x)
        x = ReLU(self.affine1(x))
        x = dropout(x, ratio=self.ratio)
        x = ReLU(self.affine2(x))
        x = dropout(x, ratio=self.ratio)
        x = self.affine3(x)

        return x


class Simple_FCN(Model):
    def __init__(self, activation=ReLU):
        super().__init__()
        self.activation = activation
        self.conv1_1 = Convolution(16, 3, 3, pad=1)
        self.conv1_2 = Convolution(16, 3, 3, pad=1)
        self.pool1 = Pooling(2, stride=2)
        self.conv2_1 = Convolution(32, 3, 3, pad=1)
        self.conv2_2 = Convolution(32, 3, 3, pad=1)
        self.pool2 = Pooling(2, stride=2)
        self.conv3_1 = Convolution(64, 3, 3, pad=1)
        self.conv3_2 = Convolution(64, 3, 3, pad=1)
        self.pool3 = Pooling(2, stride=2)
        self.conv_31 = Convolution(2, 1, 1)

        self.conv4_1 = Convolution(128, 3, 3, pad=1)
        self.pool4 = Pooling(2, stride=2)
        self.conv_21 = Convolution(2, 1, 1)

        self.conv5_1 = Convolution(256, 3, 3, pad=1)
        self.pool5 = Pooling(2, stride=2)
        self.conv_11 = Convolution(4096, 1, 1)
        self.conv_12 = Convolution(4096, 1, 1)
        self.conv_13 = Convolution(2, 1, 1)
        self.transposed_conv11 = Transpose_Convolution(2, 4, 4, stride=2, pad=1, nobias=True)
        self.transposed_conv12 = Transpose_Convolution(2, 4, 4, stride=2, pad=1, nobias=True)
        self.transposed_conv13 = Transpose_Convolution(2, 32, 24, stride=8, pad=8, nobias=True)

    def forward(self, x):
        y1 = self.conv1_1(x)
        y1 = self.conv1_2(y1)
        y1 = self.activation(y1)
        y1 = self.pool1(y1)

        y1 = self.conv2_1(y1)
        y1 = self.conv2_2(y1)
        y1 = self.activation(y1)
        y1 = self.pool2(y1)

        y1 = self.conv3_1(y1)
        y1 = self.conv3_2(y1)
        y1 = self.activation(y1)
        y1 = self.pool3(y1)
        y3 = self.conv_31(y1)

        y1 = self.conv4_1(y1)
        y1 = self.activation(y1)
        y1 = self.pool4(y1)
        y2 = self.conv_21(y1)

        y1 = self.conv5_1(y1)
        y1 = self.activation(y1)
        y1 = self.pool5(y1)

        y1 = self.conv_11(y1)
        y1 = self.conv_12(y1)
        y1 = self.conv_13(y1)

        y1 = self.transposed_conv11(y1)
        y1 = y1 + y2
        y1 = self.transposed_conv12(y1)
        y1 = y1 + y3
        y1 = self.transposed_conv13(y1)
        return y1


class Simple_ResNet(Model):
    def __init__(self, activation=ReLU):
        super().__init__()
        self.activation = activation
        self.conv1 = Convolution(16, 3, 3, pad=1)
        self.pool1 = Pooling(2, stride=2)
        self.residual1_1 = ResidualBlock(16, stride=1, use_conv1x1=False)
        self.residual1_2 = ResidualBlock(16, stride=1, use_conv1x1=False)
        self.residual2_1 = ResidualBlock(32, stride=2, use_conv1x1=True)
        self.residual2_2 = ResidualBlock(32, stride=1, use_conv1x1=False)
        self.residual3_1 = ResidualBlock(64, stride=2, use_conv1x1=True)
        self.residual3_2 = ResidualBlock(64, stride=1, use_conv1x1=False)
        self.affine = Affine(10)

    def forward(self, x):
        x = ReLU(self.conv1(x))
        x = self.pool1(x)
        x = self.residual1_1(x)
        x = self.residual1_2(x)
        x = self.residual2_1(x)
        x = self.residual2_2(x)
        x = self.residual3_1(x)
        x = self.residual3_2(x)
        x = self.affine(x)
        return x


class Simple_densenet(Model):
    def __init__(self, activation=ReLU, to_gpu=True):
        super().__init__()
        self.activation = activation
        self.conv1 = Convolution(16, 3, 3, pad=1)
        self.BN1 = BatchNorm()
        self.pool1 = Pooling(2, stride=2)
        self.dense1 = DenseBlock(32, 4)
        self.transition1 = TransitionBlock(32)
        self.dense2 = DenseBlock(64, 4)
        self.transition2 = TransitionBlock(64)
        self.dense3 = DenseBlock(64, 4)
        self.transition3 = TransitionBlock(64)
        self.affine = Affine(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.BN1(x)
        x = ReLU(x)
        x = self.pool1(x)
        x = self.dense1(x)
        x = self.transition1(x)
        x = self.dense2(x)
        x = self.transition2(x)
        x = self.dense3(x)
        x = self.transition3(x)
        x = self.affine(x)
        return x


class U_net(Model):
    def __init__(self):
        super().__init__()
        self.conv1_1 = Convolution(16, 3, 3)
        self.conv1_2 = Convolution(16, 3, 3)
        self.pool1 = Pooling(2, stride=2)
        self.conv2_1 = Convolution(128, 3, 3)
        self.conv2_2 = Convolution(128, 3, 3)
        self.pool2 = Pooling(2, stride=2)
        self.conv3_1 = Convolution(256, 3, 3)
        self.conv3_2 = Convolution(256, 3, 3)
        self.pool3 = Pooling(2, stride=2)
        self.conv4_1 = Convolution(512, 3, 3)
        self.conv4_2 = Convolution(512, 3, 3)
        self.pool4 = Pooling(2, stride=2)

        self.conv5_1 = Convolution(512, 3, 3)
        self.conv5_2 = Convolution(512, 3, 3)
        self.transconv1 = Transpose_Convolution(512, 2, 2, stride=2, nobias=True)
        self.conv6_1 = Convolution(512, 3, 3)
        self.conv6_2 = Convolution(512, 3, 3)
        self.transconv2 = Transpose_Convolution(256, 2, 2, stride=2, nobias=True)
        self.conv7_1 = Convolution(256, 3, 3)
        self.conv7_2 = Convolution(256, 3, 3)
        self.transconv3 = Transpose_Convolution(128, 2, 2, stride=2, nobias=True)
        self.conv8_1 = Convolution(128, 3, 3)
        self.conv8_2 = Convolution(128, 3, 3)
        self.transconv4 = Transpose_Convolution(16, 2, 2, stride=2, nobias=True)
        self.conv9_1 = Convolution(16, 3, 3)
        self.conv9_2 = Convolution(16, 3, 3)

    def forward(self, x):
        x1 = ReLU(self.conv1_1(x))  # 572
        x1 = ReLU(self.conv1_2(x1))  # 570
        x1_crop = CopyAndCrop((392, 392))(x1)

        x2 = self.pool1(x1)  # 568
        x2 = ReLU(self.conv2_1(x2))  # 284
        x2 = ReLU(self.conv2_2(x2))  # 282
        x2_crop = CopyAndCrop((200, 200))(x2)
        # x2_crop = util.copyandcrop(x2, (200, 200))

        x3 = self.pool2(x2)  # 280
        x3 = ReLU(self.conv3_1(x3))  # 140
        x3 = ReLU(self.conv3_2(x3))  # 138
        x3_crop = CopyAndCrop((104, 104))(x3)
        # x3_crop = util.copyandcrop(x3, (104, 104))

        x4 = self.pool3(x3)  # 136
        x4 = ReLU(self.conv4_1(x4))  # 68
        x4 = ReLU(self.conv4_2(x4))  # 66
        x4_crop = CopyAndCrop((56, 56))(x4)
        # x4_crop = util.copyandcrop(x4, (56, 56))

        x = self.pool4(x4)  # 64
        x = ReLU(self.conv5_1(x))  # 32
        x = ReLU(self.conv5_2(x))  # 30

        x = self.transconv1(x) + x4_crop  # 512 28
        x = ReLU(self.conv6_1(x))  # 512 56
        x = ReLU(self.conv6_2(x))

        x = self.transconv2(x) + x3_crop  # 512 52
        x = ReLU(self.conv7_1(x))  # 256 104
        x = ReLU(self.conv7_2(x))

        x = self.transconv3(x) + x2_crop  # 256 100
        x = ReLU(self.conv8_1(x))  # 128 200
        x = ReLU(self.conv8_2(x))

        x = self.transconv4(x) + x1_crop  # 128 196
        x = ReLU(self.conv9_1(x))  # 16 392
        x = ReLU(self.conv9_2(x))  # 16 390
        return x