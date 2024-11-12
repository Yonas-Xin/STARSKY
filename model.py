import skystar.utils
from skystar.layer import *
from skystar.utils import plot_dot_graph
from skystar.graph import create_graph,save_graph
from skystar.optimizer import Adam
from tqdm import tqdm#添加进度条
import time
# =============================================================================
'''model类（继承自Layer）'''
# =============================================================================
class Model(Layer):
    def plot(self,*inputs):
        y=self.forward(*inputs)
        plot_dot_graph(y)

    def predict(self,*inputs,training=True):
        y=self.forward(*inputs,training=training)
        return y

    def accuracy(self,x_test,t,training=True):
        with no_grad():
            y = self.predict(x_test, training=training)
            if t.ndim == 2:
                t = np.argmax(t, axis=1)
            y.data = np.argmax(y.data, axis=1)
            sum = t.size
            return np.sum(y.data == t) / sum
    def save_to_onnx(self,*inputs,name=None,ifsimplify=True):
        '''
        :param inputs: 需要使用一个模型的输入
        :param name:
        :return:
        '''
        model_name=self.__class__.__name__
        if name is None:
            name=model_name+".onnx"
        self.to_cpu()
        dir=os.path.dirname(os.path.abspath(__file__))
        path=os.path.join(dir,'model_params',name)
        y=self.forward(*inputs)
        graph=create_graph(y)
        save_graph(graph,model_name,file_name=path,ifsimplify=ifsimplify)
        return
    def train(self, train, lr=0.001, epoch=100, test=None, plot=False, plot_rate=0.1, loss_func=skystar.core.softmaxwithloss,
              accuracy=skystar.utils.accuracy, optimizer=Adam, use_gpu=True, save_model=True,autodecrese=False):
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
        name = self.__class__.__name__
        print('Training begin：')
        if use_gpu:
            print('GPU加速已开启，所有参数已转换为cp.ndarray')
            if skystar.cuda.gpu_enable:
                self.to_gpu()
                train.to_gpu()
                if test is not None:
                    test.to_gpu()
        self.list = [[], [], []]
        optimizer = optimizer(lr).setup(self)
        for i in range(epoch):
            if i % int(epoch // 2) == 0 and autodecrese:
                '''后期训练把学习率降低'''
                lr = lr / 10
                optimizer.lr = lr
            tqdm.write(f'Epoch {i + 1}:')
            sum_acc = 0.0
            sum_loss = 0.0
            time.sleep(0.1)  # 停顿0.1秒，避免输出条与输出字符串同步
            # 使用 tqdm 包裹训练数据集
            for x, t in tqdm(train, desc='Training', total=len(train) / train.batch_size):
                y = self(x, training=True)
                loss = loss_func(y, t)
                self.cleangrads()
                loss.backward()
                optimizer.update()
                if accuracy is not None:
                    sum_acc += accuracy(y, t) * len(t)
                sum_loss += loss.data
            # 使用 tqdm.write 替代 print
            tqdm.write(f'Train_Loss {sum_loss / len(train)}')
            self.list[0].append(sum_loss / len(train))
            if accuracy is not None:
                tqdm.write(f'Train_Acc {sum_acc / len(train)}')
                self.list[1].append(sum_acc / len(train))

            time.sleep(0.1)  # 停顿0.1秒，避免输出条与输出字符串同步
            # 如果输入了测试集，则进行网络测试
            if test is not None:
                sum_acc = 0.0
                num = 1 * test.batch_size
                plot_num = int(plot_rate * len(test))
                with skystar.core.no_grad():
                    for x, t in tqdm(test, desc='Testing', total=len(test) // test.batch_size):
                        y = self(x, training=False)
                        if accuracy is not None:
                            sum_acc += accuracy(y, t) * len(t)
                        if num % plot_num == 0 and plot:
                            if i == 0: Variable(x).image_show(mode='feature', label=f' Epoch{i + 1} Real' + str(num))
                            if t.ndim >= 3:
                                if i == 0: Variable(t).image_show(mode='label', label=f' Epoch{i + 1} Real' + str(num))
                                y_predict = Variable(y.data.argmax(axis=1))
                                y_predict.image_show(mode='label', label=f' Epoch{i + 1} Prediction' + str(num))

                        num += 1 * test.batch_size
                tqdm.write(f'Test_Acc: {sum_acc / len(test)}')
                self.list[2].append(sum_acc / len(test))
        if save_model:
            name = self.__class__.__name__
            print('保存模型参数中......')
            self.save_weights(path=name)

    def test(self, test,use_gpu=True,plot=True,plot_rate=0.1,accuracy=skystar.utils.accuracy):
        print('Test begin：')
        if use_gpu:
            print('GPU加速已开启，所有参数已转换为cp.ndarray')
            if skystar.cuda.gpu_enable:
                self.to_gpu()
                test.to_gpu()
        sum_acc = 0.0
        num,i = 1 * test.batch_size,0
        plot_num = int(plot_rate * len(test))
        with skystar.core.no_grad():
            for x, t in tqdm(test, desc='Testing', total=len(test) // test.batch_size):
                y = self(x, training=False)
                if accuracy is not None:
                    sum_acc += accuracy(y, t) * len(t)
                if num % plot_num == 0 and plot:
                    Variable(x).image_show(mode='feature', label=f' Real' + str(num))
                    if t.ndim >= 3:
                        Variable(t).image_show(mode='label', label=f' Real' + str(num))
                        y_predict = Variable(y.data.argmax(axis=1))
                        y_predict.image_show(mode='label', label=f' Prediction' + str(num))
                num += 1 * test.batch_size
        tqdm.write(f'Test_Acc: {sum_acc / len(test)}')

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
            layer = Affine(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x,training=True):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)

class Batch_norm_MLP(Model):
    '''添加了batchnorm的全连接层，分测试和训练阶段'''
    def __init__(self, fc_output_sizes, activation=sigmoid,gamma=1.0,beta=0,momentum=0.9):
        super().__init__()
        self.activation = activation
        self.layers = []
        self.gamma=gamma
        self.beta=beta
        self.momentum=momentum

        for i, outsize in enumerate(fc_output_sizes):
            layer1 = Affine(outsize)
            setattr(self, 'Affine' + str(i), layer1)
            self.layers.append(layer1)
            layer2 = BatchNorm(self.gamma, self.beta, self.momentum)
            setattr(self, 'Batch_norm' + str(i), layer2)
            self.layers.append(layer2)

    def forward(self, x,training=True):
        for i in range(0, len(self.layers) - 1):
            layer = self.layers[i]
            x = layer(x,training=training)
            if i % 2 != 0:
                x = self.activation(x)

        layer = self.layers[-1]
        x = layer(x,training=training)
        return x


class Batchnorm_dropout_MLP(Model):
    '''添加了batchnorm的dropout的全连接层，分测试和训练阶段,dropout层在sigmoid后面'''
    def __init__(self, fc_output_sizes, activation=sigmoid,use_dropout=False,dropout_ratio=0.5,gamma=1.0,beta=0,momentum=0.9):
        super().__init__()
        self.activation = activation
        self.layers = []
        self.use_dropout=use_dropout
        self.dropout_ratio=dropout_ratio
        self.gamma=gamma
        self.beta=beta
        self.momentum=momentum

        for i, outsize in enumerate(fc_output_sizes):
            layer1 = Affine(outsize)
            setattr(self, 'Affine' + str(i), layer1)
            self.layers.append(layer1)
            layer2 = BatchNorm(self.gamma, self.beta, self.momentum)
            setattr(self, 'Batch_norm' + str(i), layer2)
            self.layers.append(layer2)

    def forward(self, x,training=True):
        for i in range(0, len(self.layers) - 1):
            layer = self.layers[i]
            x = layer(x,training=training)
            if i % 2 != 0:
                x = self.activation(x)
                if self.use_dropout:
                    x = dropout(x,training=training)

        layer = self.layers[-1]
        x = layer(x,training=training)
        return x

class Simple_CNN(Model):
    def __init__(self,activation=ReLU):
        super().__init__()
        self.activation=activation
        self.convolution1=Convolution(16,3,3)
        self.pooling1=Pooling(2,stride=2)
        self.convolution2=Convolution(32,3,3)
        self.pooling2=Pooling(2,stride=2)
        self.affine=Affine(10)

    def forward(self,x,training=True):
        y1=self.convolution1(x)
        y1=self.activation(y1)
        y1=self.pooling1(y1)

        y2=self.convolution2(y1)
        y2=self.activation(y2)
        y2=self.pooling2(y2)

        out=self.affine(y2)
        return out

class Simple_RNN(Model):
    def __init__(self,hidden_size,out_size):
        super().__init__()
        self.RNN=RNN(hidden_size=hidden_size)#输出状态
        self.affine=Affine(out_size)#使用affine层输出状态

    def reset_state(self):
        self.RNN.reset_state()

    def forward(self,x,training=True):
        x=self.RNN(x)
        x=self.affine(x)

        return x

class Better_RNN(Model):
    def __init__(self,hidden_size,out_size):
        super().__init__()
        self.RNN=LSTM(hidden_size=hidden_size)
        self.affine=Affine(out_size)

    def reset_state(self):
        self.RNN.reset_state()

    def forward(self,x,training=True):
        x=self.RNN(x)
        x=self.affine(x)

        return x

class VGG(Model):
    '''使用该网络时请注意显存gpu的容量'''
    def __init__(self,output=1000,ratio=0.5):
        super().__init__()
        self.ratio=ratio
        self.conv1_1=Convolution(16,3,3,pad=1)
        self.conv1_2=Convolution(16,3,3,pad=1)
        self.pool1=Pooling(2,stride=2)
        self.conv2_1=Convolution(128,3,3,pad=1)
        self.conv2_2=Convolution(128,3,3,pad=1)
        self.pool2=Pooling(2,stride=2)
        self.conv3_1=Convolution(256,3,3,pad=1)
        self.conv3_2=Convolution(256,3,3,pad=1)
        self.conv3_3=Convolution(256,3,3,pad=1)
        self.pool3=Pooling(2,stride=2)
        self.conv4_1=Convolution(512,3,3,pad=1)
        self.conv4_2=Convolution(512,3,3,pad=1)
        self.conv4_3=Convolution(512,3,3,pad=1)
        self.pool4=Pooling(2,stride=2)
        self.conv5_1=Convolution(512,3,3,pad=1)
        self.conv5_2=Convolution(512,3,3,pad=1)
        self.conv5_3=Convolution(512,3,3,pad=1)
        self.pool5=Pooling(2,stride=2)
        self.affine1=Affine(4096)
        self.affine2=Affine(4096)
        self.affine3=Affine(output)

    def forward(self,x,training=True):
        '''激活函数用ReLU'''
        x=ReLU(self.conv1_1(x))
        x=ReLU(self.conv1_2(x))
        x=self.pool1(x)
        x=ReLU(self.conv2_1(x))
        x=ReLU(self.conv2_2(x))
        x=self.pool2(x)
        x=ReLU(self.conv3_1(x))
        x=ReLU(self.conv3_2(x))
        x=ReLU(self.conv3_3(x))
        x=self.pool3(x)
        x=ReLU(self.conv4_1(x))
        x=ReLU(self.conv4_2(x))
        x=ReLU(self.conv4_3(x))
        x=self.pool4(x)
        x=ReLU(self.conv5_1(x))
        x=ReLU(self.conv5_2(x))
        x=ReLU(self.conv5_3(x))
        x=self.pool5(x)
        x=ReLU(self.affine1(x))
        x=dropout(x,training=training,ratio=self.ratio)
        x=ReLU(self.affine2(x))
        x=dropout(x,training=training,ratio=self.ratio)
        x=self.affine3(x)

        return x

class STL_10_CNN(Model):
    def __init__(self,ratio=0.5,output_size=10):
        super().__init__()
        self.ratio=ratio
        self.conv1_1=Convolution(16,3,3,pad=1)
        self.conv1_2=Convolution(16,3,3,pad=1)
        self.pool1=Pooling(2,stride=2)
        self.conv2_1=Convolution(128,3,3,pad=1)
        self.conv2_2=Convolution(128,3,3,pad=1)
        self.pool2=Pooling(2,stride=2)
        self.conv3_1=Convolution(256,3,3,pad=1)
        self.conv3_2=Convolution(256,3,3,pad=1)
        self.pool3=Pooling(2,stride=2)
        # self.conv4_1=Convolution(512,3,3,pad=1)
        # self.conv4_2=Convolution(512,3,3,pad=1)
        # self.pool4=Pooling(2,stride=2)
        # self.conv5_1=Convolution(512,3,3,pad=1)
        # self.conv5_2=Convolution(512,3,3,pad=1)
        # self.pool5=Pooling(2,stride=2)
        self.affine1=Affine(100)
        self.affine2=Affine(100)
        self.affine3=Affine(output_size)

    def forward(self,x,training=True):
        '''激活函数用ReLU'''
        x=ReLU(self.conv1_1(x))
        x=ReLU(self.conv1_2(x))
        x=self.pool1(x)
        x=ReLU(self.conv2_1(x))
        x=ReLU(self.conv2_2(x))
        x=self.pool2(x)
        x=ReLU(self.conv3_1(x))
        x=ReLU(self.conv3_2(x))
        x=self.pool3(x)
        # x=ReLU(self.conv4_1(x))
        # x=ReLU(self.conv4_2(x))
        # x=self.pool4(x)
        # x=ReLU(self.conv5_1(x))
        # x=ReLU(self.conv5_2(x))
        # x=self.pool5(x)
        x=ReLU(self.affine1(x))
        x=dropout(x,training=training,ratio=self.ratio)
        x=ReLU(self.affine2(x))
        x=dropout(x,training=training,ratio=self.ratio)
        x=self.affine3(x)

        return x



class Simple_FCN(Model):
    def __init__(self,activation=ReLU):
        super().__init__()
        self.activation=activation
        self.conv1_1=Convolution(16,3,3,pad=1)
        self.conv1_2=Convolution(16,3,3,pad=1)
        self.pool1=Pooling(2,stride=2)
        self.conv2_1=Convolution(32,3,3,pad=1)
        self.conv2_2=Convolution(32,3,3,pad=1)
        self.pool2=Pooling(2,stride=2)
        self.conv3_1=Convolution(64,3,3,pad=1)
        self.conv3_2=Convolution(64,3,3,pad=1)
        self.pool3=Pooling(2,stride=2)
        self.conv_31=Convolution(2,1,1)

        self.conv4_1=Convolution(128,3,3,pad=1)
        self.pool4=Pooling(2,stride=2)
        self.conv_21=Convolution(2,1,1)

        self.conv5_1=Convolution(256,3,3,pad=1)
        self.pool5=Pooling(2,stride=2)
        self.conv_11=Convolution(4096,1,1)
        self.conv_12=Convolution(4096,1,1)
        self.conv_13=Convolution(2,1,1)
        self.transposed_conv11=Transpose_Convlution(2, 4, 4, stride=2, pad=1, nobias=True)
        self.transposed_conv12=Transpose_Convlution(2, 4, 4, stride=2, pad=1, nobias=True)
        self.transposed_conv13=Transpose_Convlution(2, 32, 24, stride=8, pad=8, nobias=True)

    def forward(self,x,training=True):
        y1=self.conv1_1(x)
        y1=self.conv1_2(y1)
        y1=self.activation(y1)
        y1=self.pool1(y1)

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
        y1=y1+y2
        y1 = self.transposed_conv12(y1)
        y1=y1+y3
        y1 = self.transposed_conv13(y1)
        return y1

class Simple_ResNet(Model):
    def __init__(self,activation=ReLU):
        super().__init__()
        self.activation=activation
        self.conv1=Convolution(16,3,3,pad=1)
        self.pool1=Pooling(2,stride=2)
        self.residual1_1=ResidualBlock(16,stride=1,use_conv1x1=False)
        self.residual1_2=ResidualBlock(16,stride=1,use_conv1x1=False)
        self.residual2_1=ResidualBlock(32,stride=2,use_conv1x1=True)
        self.residual2_2=ResidualBlock(32,stride=1,use_conv1x1=False)
        self.residual3_1=ResidualBlock(64,stride=2,use_conv1x1=True)
        self.residual3_2=ResidualBlock(64,stride=1,use_conv1x1=False)
    def forward(self,x,training=True):
        x=ReLU(self.conv1(x))
        x=self.pool1(x)
        x=self.residual1_1(x)
        x=self.residual1_2(x)
        x=self.residual2_1(x)
        x=self.residual2_2(x)
        x=self.residual3_1(x)
        x=self.residual3_2(x)
        return x

class Simple_densenet(Model):
    def __init__(self,activation=ReLU):
        super().__init__()
        self.activation=activation
        self.conv1=Convolution(16,3,3,pad=1)
        self.BN1=BatchNorm()
        self.pool1=Pooling(2,stride=2)
        self.dense1=DenseBlock(32,4)
        self.transition1=TransitionBlock(32)
        self.dense2=DenseBlock(64,4)
        self.transition2=TransitionBlock(64)
        self.dense3=DenseBlock(64,4)
        self.transition3=TransitionBlock(64)
    def forward(self,x,training=True):
        x=self.conv1(x)
        x=self.BN1(x)
        x=ReLU(x)
        x=self.pool1(x)
        x=self.dense1(x)
        x=self.transition1(x)
        x=self.dense2(x)
        x=self.transition2(x)
        x=self.dense3(x)
        x=self.transition3(x)
        return x


x=np.random.random((3,3,224,224))
model=Simple_densenet()
y=model(x)
y.backward()
model.save_to_onnx(x,ifsimplify=False)

