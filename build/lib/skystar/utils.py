import os
import subprocess
import urllib.request
from skystar import no_grad
from skystar.cuda import get_array_module
from skystar.cuda import as_numpy
from skystar.sky_dataset import data_to_npz
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

from skystar.voc2012 import VOC_COLORMAP  # 颜色条

def make_dir(dirname):
    dir = os.getcwd()
    dir = os.path.join(dir, dirname)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
def pair():
    pass
def _dot_var(v, verbose=False):  # v是Variable实例,该函数实现单个元素的dot转换
    def shape_txt(x):
        label = 'shape['
        for i in range(len(x.shape)):
            _str = str(x.shape[i])
            if i != len(x.shape) - 1:
                label += _str + ' '
            else:
                label += _str
        label += ']'
        return label

    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    if v.name is None:
        name = ' '
    else:
        name = v.name

    if v.data is None:
        name += 'None'
    if verbose and v.data is not None:
        if v.name is not None:
            name += '\n'
        else:
            name = 'X\n'
        name += shape_txt(v) + '\ndtype:' + str(v.dtype)

    return dot_var.format(id(v), name)


def _dot_func(f):  # f是函数，该函数实现输入函数的dot转化
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # y是weakref
    return txt


def get_dot_var(output, verbose=False):
    txt = ''
    funcs = [output.creator]
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for input in func.inputs:
            txt += _dot_var(input, verbose)
            if input.creator is not None:
                if input.creator not in funcs:
                    funcs.append(input.creator)

    return 'digraph g{\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='C:\\Users\\85002\\Desktop\\graph.png'):  # 计算图默认生成在桌面
    dot_graph = get_dot_var(output, verbose)

    # 将dot保存为文件，位置在C:\Uers\85002\.dezer_Cgraph
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero_Cgraph')  # 获取文件夹路径/.dezero_Cgraph
    if not os.path.exists(tmp_dir):  # 若文件夹不在，则创建文件夹
        os.makedirs(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')  # 获取创建文件的路径及名称

    with open(graph_path, 'w') as f:  # 创建文件
        f.write(dot_graph)

    # 外部调用dot命令
    dot_path = r'C:\Program Files\Graphviz\bin\dot.exe'  # dot命令的位置
    extension = os.path.splitext(to_file)[1][1:]  # 获取拓展名
    cmd = f'"{dot_path}" "{graph_path}" -T {extension} -o "{to_file}"'
    ''' cmd=C:\\Program Files\\Graphviz\\bin\\dot.exe C:\\Users\\85002\\.dezero_Cgraph\\tmp_graph.dot -T png -o C:\\Users\\85002\\Desktop\\graph.png'''
    subprocess.run(cmd, shell=True)  # 终端执行cmd命令

    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass


def sum_to(x, shape):
    """沿轴对元素求和，以输出具有给定形状的数组。
    Parameters：
    x (ndarray): 输入数组。
    shape: 目标形状。
    return：
    ndarray: 具有目标形状的输出数组。
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.data.squeeze(lead_axis)
        y = skystar.Variable(y)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """适当地重塑梯度以用于dezero.functions.sum的反向传播。
    Parameters：
    gy (dezero.Variable): 反向传播中从输出获得的梯度变量。
    x_shape (tuple): 在sum函数的前向传播中使用的形状。
    axis (None或int或int的元组): 在sum函数的前向传播中使用的轴。
    keepdims (bool): 在sum函数的前向传播中使用的keepdims参数。
    return：
    dezero.Variable: 适当重塑的梯度变量。
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not hasattr(axis, 'len'):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)
    return gy


def cross_entropy_error(y, t):  # 交叉熵误差，当t为非0-1形式时
    xp = get_array_module(y)
    if y.ndim <= 2:
        if y.ndim == 1:  # 1把一维数据变为一行的二维数据
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)

        if t.size == y.size:  # 当t为0-1形式时提取最大值的索引,变为非0-1形式
            t = t.argmax(axis=1)
        batch_size = y.shape[0]  # 如果是一维数组的话，这里就会返回不正常值
        pt = -xp.sum(xp.log(y[xp.arange(batch_size), t]) + 1e-5) / batch_size
        return pt
    else:
        '''这里加入了语义分割模型下，像素级的交叉熵误差'''
        if y.ndim == 3:
            y = y.reshape(1, y.shape[0], y.shape[1], y.shape[2])
        batch_size, num_class, H, W = y.shape
        if t.ndim != 4:
            t = t.reshape(1, 1, H, W)
        '''将y变为二维数组，与t变为非0-1形式二维数组'''
        t = onehot(t, num_class)
        pt = -xp.sum(t * xp.log(y + 1e-7)) / H / W / batch_size
        return pt


def softmax(x):  # Softmax函数,输出层激活函数,用于多维数组
    xp = get_array_module(x)
    if x.ndim == 2:
        x = x.T  # 转置操作，每一列代表一个样本，如果是行的话，广播会出错
        c = xp.max(x, axis=0)  # 求出最大值，避免数据溢出
        x_exp = xp.exp(x - c)  # 利用了广播，每一列的样本减去相同的值
        sum = xp.sum(x_exp, axis=0)
        y = x_exp / sum
        return y.T

    if x.ndim > 2:
        '''这里加入了语义分割模型下，像素级的softmax分类'''
        if x.ndim == 3:
            axis = 0
        else:
            axis = 1
        c = xp.max(x, axis=axis, keepdims=True)
        x_exp = xp.exp(x - c)
        sum = xp.sum(x_exp, axis=axis, keepdims=True)
        y = x_exp / sum
        return y

    x = x - xp.max(x)  # 溢出对策
    return xp.exp(x) / xp.sum(xp.exp(x))


# def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
#     """
#     将四维图像变为二维矩阵
#     Parameters
#     ----------
#     input_data : 由(数据量, 通道, 高, 宽)的4维数组构成的输入数据
#     filter_h : 滤波器的高
#     filter_w : 滤波器的宽
#     stride : 步幅
#     pad : 填充
#
#     Returns
#     -------
#     col : 2维数组
#     """
#     xp =get_array_module(input_data)
#     N, C, H, W = input_data.shape
#     out_h = (H + 2*pad - filter_h)//stride + 1#输出窗口高
#     out_w = (W + 2*pad - filter_w)//stride + 1#输出窗口宽
#
#     img = xp.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
#     col = xp.zeros((N, C, filter_h, filter_w, out_h, out_w))
#     '''经过下面的循环，将col变成（N,C,filter_h,filter_w,out_h,out_w）'''
#     for y in range(filter_h):
#         y_max = y + stride*out_h
#         for x in range(filter_w):
#             x_max = x + stride*out_w
#             col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
#     '''数据的展开是按照0，1，2，3....的顺序依次展开的，注意核作用的区域大小为（filter_h,filter_w）
#     所以为了正确展开，对数据的轴重新排列（N,out_h,out_w,C,filter_h,filter_w）'''
#     col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
#     return col#(N*oh*ow,c*fh*fw)
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):  # 一种更加高效的im2col方法
    xp = get_array_module(input_data)
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    # 添加填充
    img = xp.pad(input_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    # 生成一个索引数组
    i0 = xp.repeat(xp.arange(filter_h), filter_w)
    i1 = xp.tile(xp.arange(filter_w), filter_h)
    j = xp.repeat(xp.arange(out_h), out_w)
    k = xp.tile(xp.arange(out_w), out_h)
    i = i0[:, None] + j[None, :] * stride
    j = i1[:, None] + k[None, :] * stride
    col = img[:, :, i, j].transpose(0, 3, 1, 2).reshape(N * out_h * out_w, -1)
    return col


# def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
#     """
#     将二维矩阵变为四维图像
#     Parameters
#     ----------
#     col :
#     input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
#     filter_h :窗口高度
#     filter_w:窗口宽度
#     stride:步长
#     pad:填充
#
#     Returns
#     -------
#     """
#     xp =get_array_module(col)
#     N, C, H, W = input_shape
#     out_h = (H + 2*pad - filter_h)//stride + 1
#     out_w = (W + 2*pad - filter_w)//stride + 1
#     col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
#
#     img = xp.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
#     for y in range(filter_h):
#         y_max = y + stride*out_h
#         for x in range(filter_w):
#             x_max = x + stride*out_w
#             img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
#
#     return img[:, :, pad:H + pad, pad:W + pad]#(N,C,H,W)
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    xp = get_array_module(col)
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    # 重塑并转置col
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    # 初始化图像
    img = xp.zeros((N, C, H + 2 * pad, W + 2 * pad))
    # 计算每个窗口的填充区域并进行累加
    for y in range(filter_h):
        for x in range(filter_w):
            img[:, :, y:y + out_h * stride:stride, x:x + out_w * stride:stride] += col[:, :, y, x, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]  # 返回裁剪后的图像


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError


def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')


# cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')#'C:\\Users\\85002\\.dezero'
cache_dir = 'D:\\Programing\\pythonProject\\Dezero\\train_image'


def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


def transconv_pad(x, stride=1, kernel_size=3, pad=0):
    '''对x实现边缘填充和中间填充，扩展图像的尺寸'''
    xp = get_array_module(x)
    middle_pad = stride - 1
    edge_pad = kernel_size - 1 - pad  # 这里说明pad影响了边缘填充
    N, C, h, w = x.shape

    # 计算中间填充后图像尺寸
    new_h = h + (h - 1) * middle_pad
    new_w = w + (w - 1) * middle_pad

    mid_demox = xp.zeros((N, C, new_h, new_w))

    for i in range(h):
        for j in range(w):
            mid_demox[:, :, i * (1 + middle_pad), j * (1 + middle_pad)] = x[:, :, i, j]

    # 使用edge_pad在图像边缘添加填充
    x = xp.pad(mid_demox, [(0, 0), (0, 0), (edge_pad, edge_pad), (edge_pad, edge_pad)], mode='constant')
    return x


def back_transcov_pad(x, stride=1, kernel_size=3, pad=0):
    '''从反卷积结果中还原原始图像，用于上采样层的反向传播'''
    xp = get_array_module(x)
    edge_pad = kernel_size - 1 - pad
    middle_pad = stride - 1
    N, C, h, w = x.shape

    # 去除边缘填充
    x = x[:, :, edge_pad:h - edge_pad, edge_pad:w - edge_pad]

    h, w = x.shape[2], x.shape[3]
    # 计算去除中间填充后的尺寸
    new_h = int((h + middle_pad) // (1 + middle_pad))
    new_w = int((w + middle_pad) // (1 + middle_pad))
    out = xp.zeros((N, C, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            out[:, :, i, j] = x[:, :, i * stride, j * stride]

    return out


def bilinear_kernel(in_channels, out_channels, kernel_size, xp=np):
    '''创建一个双线性内插值的核，用于初始化上采样层的权重'''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (xp.arange(kernel_size).reshape(-1, 1),
          xp.arange(kernel_size).reshape(1, -1))
    filt = (1 - xp.abs(og[0] - center) / factor) * \
           (1 - xp.abs(og[1] - center) / factor)
    weight = xp.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[:] = filt
    return xp.array(weight)


def onehot(x, num_classes):
    '''此函数用于把标签t变为onehot形式'''
    xp = get_array_module(x)
    if x.ndim == 2:
        '''这里可以扩展一维标签变为onehot形式'''
        pass
    if x.ndim != 4 and x.ndim != 2:
        try:
            C, H, W = x.shape
            x = x.reshape(1, C, H, W)
        except ValueError:
            H, W = x.shape
            x = x.reshape(1, 1, H, W)
    batch, C, H, W = x.shape
    x = x.reshape(-1)
    rex = xp.zeros((x.size, num_classes))
    rex[xp.arange(len(x)), x] = 1
    rex = rex.reshape(batch, H, W, num_classes)
    rex = rex.transpose(0, 3, 1, 2)
    return rex


def accuracy(y, t):
    '''
    y:Variable
    t:ndarray
    '''
    with no_grad():
        xp = get_array_module(y)
        if t.ndim > 3:
            xp = get_array_module(y)
            y = y.data.argmax(axis=1)
            num_classes, H, W = y.shape
            y = y.reshape(num_classes, 1, H, W)
            acc = xp.sum(y == t) / y.size
            return acc
        else:
            if t.ndim == 2:
                t = xp.argmax(t, axis=1)
            y = xp.argmax(y.data, axis=1)
            sum = t.size
            return xp.sum(y == t) / sum


def mean_IoU(y, t):
    '''
    y:Variable 预测值 四维
    t:ndarray 真实标签 四维
    '''
    xp = get_array_module(y)
    if y.data.size != t.size:
        y = y.data.argmax(axis=1)  # 预测标签
    num_classes, H, W = y.shape
    y = y.reshape(num_classes, 1, H, W)
    unique = xp.unique(t)
    iou_list = []
    for i in unique:
        true_positive = xp.sum((t == i) & (y == i))  # 正确分类的总像素
        false_positive = xp.sum((t != i) & (y == i))  # 错误分类的总像素
        false_negative = xp.sum((t == i) & (y != i))  # 被漏掉的总像素
        denominator = true_positive + false_positive + false_negative
        if denominator == 0:  # 避免除以零
            iou = 0.0
        else:
            iou = true_positive / denominator
        iou_list.append(iou)
    mean_iou = xp.mean(xp.array(iou_list))  # 计算平均IoU
    mean_iou = float(mean_iou)
    return mean_iou
def dataset_to_npz(data_txt_name, outfile_name):
    '''输入txt文件，返回数据集，数据集的字典为x_dataset,t_dataset'''
    return data_to_npz(data_txt_name, outfile_name)

def data_norm(x, _range=(0, 1),dtype=np.float32):
    '''
    对单幅图像进行归一化
    :param x: 图像的矩阵，shape：num,C,H,W
    :param _range: 归一化范围
    :return: 返回归一化结果
    '''
    range_min = _range[0]
    range_max = _range[1]
    _max = np.max(x)
    _min = np.min(x)
    k = (range_max - range_min) / (_max - _min + 1e-5)
    x = k * (x - _min)
    x=x.astype(dtype)
    return x


# def datatofeature(data,mode='feature',pad=1):
#     '''图像预处理'''
#     data = skystar.cuda.as_numpy(data)
#     if mode == 'feature' or mode == 'weight':
#         data = data_norm(data, _range=(0, 255))  # 把数据映射到0~255
#     data = data.astype(np.float32)
#     data = data.transpose(0, 2, 3, 1)
#
#     '''额外添加图片，满足图像x*x排列'''
#     num, H, W, C = data.shape
#     x = int_square_root(num)  # x*x
#     if x * x > num:
#         add_num = x * x - num
#         new_data = np.zeros((add_num, H, W, C), dtype=np.float32)
#         if mode == 'feature' or 'weight':
#             new_data += 255  # 生成白图
#         elif mode == 'label':
#             new_data += np.max(data)
#         data = np.concatenate((data, new_data), axis=0)
#     full_image = splicing(data, x, pad=pad, padnum=255)  # 图像填充和拼接
#     return full_image
# def int_square_root(x):
#     '''确定x的整数平方根，若无整数平方根，x+1直到能取整数平方根为之'''
#     sqrt = np.sqrt(x)
#     y = asif_int(sqrt)
#     while not y:
#         x += 1
#         sqrt = np.sqrt(x)
#         y = asif_int(sqrt)
#     return int(sqrt)
# def asif_int(x):
#     '''判断数是否为整数'''
#     x = ((x - int(x)) == 0)
#     return x
# def splicing(data, P_num, pad=1, padnum=255):
#     '''该函数用于将四维的多个图片变为二维矩阵，即将图像二维排列,P_num:排列数'''
#     num, H, W, C = data.shape  # 原图大小
#     data = data.reshape(P_num, P_num, H, W, C)
#     # 按行和列拼接数据，形成完整的图像
#     rows = []
#     for i in range(P_num):
#         row = np.concatenate(data[i, :, :, :, :],
#                              axis=1)  # 拼接行data[i, :, :, :, :]取出 i行(x, H_pad, W_pad, C)数据,拼接后的数据降维（H，W，C），可以理解为axis=1在W方向上拼接
#         rows.append(row)
#     full_image = np.concatenate(rows, axis=0)
#     # 给不同图像之间添加分界线用以区别
#     H_full, W_full, C = full_image.shape
#     re_img = (np.zeros((H_full + P_num - 1, W_full + P_num - 1, C)) + padnum)
#     for i in range(P_num):
#         x = i * (H + pad)
#         for j in range(P_num):
#             y = j * (W + pad)
#             re_img[x:x + H, y:y + W, :] = full_image[i * H:i * H + H, j * W:j * W + W, :]
#     return re_img
# def X_mode_show(feature, label,alpha=1):
#     H, W, C = feature.shape
#     if C == 1:  # 通道为1显示灰度图
#         plt.imshow(feature, cmap='gray')
#     else:  # 真彩色
#         re_img = np.zeros((H, W, 3))
#         re_img[:, :, 0] = feature[:, :, 0]
#         re_img[:, :, 1] = feature[:, :, 1]
#         if C == 2:
#             re_img[:, :, 2] += 255
#         else:
#             re_img[:, :, 2] = feature[:, :, 2]
#         re_img = np.uint8(re_img)
#         plt.imshow(re_img,alpha=alpha)
#     plt.axis('off')
#     plt.title('Feature' + label)
#     plt.show()
#
# def T_mode_show(feature, label):
#     feature = np.squeeze(feature)
#     unique_labels = np.unique(feature)
#     cmap = voc_colormap2cmap()  # 使用自定义的颜色条映射
#     # 创建标签与颜色之间的映射
#     norm = mcolors.BoundaryNorm(boundaries=np.arange(len(unique_labels) + 1) - 0.5, ncolors=len(unique_labels))
#     plt.imshow(feature, cmap=cmap, norm=norm,alpha=1)
#     plt.axis('off')
#     plt.title('Label' + label)
#     # 显示图像
#     plt.show()
# def Weight_mode_show(feature, label):
#     H, W, C = feature.shape
#     if C == 1:  # 通道为1显示灰度图
#         plt.imshow(feature, cmap='gray')
#     else:
#         re_img = np.zeros((H, W, 3))
#         re_img[:, :, 0] = feature[:, :, 0]
#         re_img[:, :, 1] = feature[:, :, 1]
#         if C == 2:
#             re_img[:, :, 2] += 255
#         else:
#             re_img[:, :, 2] = feature[:, :, 2]
#         re_img = re_img[:, :, 0] * 0.2989 + re_img[:, :, 1] * 0.5870 + re_img[:, :, 2] * 0.1140
#         re_img = np.uint8(re_img)
#         plt.imshow(re_img, cmap='gray')
#     plt.axis('off')
#     plt.title('Feature' + label)
#     plt.show()
#     return
# def images_show(data, pad=1, mode='feature', label=None):
#     '''
#     :param data: ndarray shape: num,C,H,W
#     :param pad: 中间填充数量，用于区分图像，默认为1
#     :param mode: 模式，默认为’X‘  可选X T
#     :param label: 图名
#     :return: 显示平铺的图形
#     '''
#     '''图像预处理'''
#     full_image=datatofeature(data,mode=mode,pad=pad)
#     '''图像显示'''
#     if label is None:
#         label = ''
#     else:
#         label = ':' + label
#     if mode == 'feature':
#         X_mode_show(full_image, label)
#     elif mode == 'label':
#         T_mode_show(full_image, label)
#     elif mode == 'weight':
#         Weight_mode_show(full_image, label)
#     else:
#         print(f'not support mode:{label}')

# def init_graph():
#     figure,axes= plt.subplots(nrows=1, ncols=3)
#     for ax in axes.flat:  # 对于多行，遍历所有子图并关闭轴
#         ax.axis('off')
#     return figure,axes
# def subplots_show(axes, x, t, predict, t_alpha=1):
#     feature = datatofeature(x, mode='feature')
#     t_feature=datatofeature(t,mode='label')
#     predict_feature=datatofeature(predict,mode='label')
#     fh,fw,fc=feature.shape
#     re_img = np.zeros((fh, fw, 3))
#     if fc>=3:
#         re_img[:,:,0:3]=feature[:,:,0:3]
#         re_img = np.uint8(re_img)
#         axes[0].imshow(re_img)
#
#     else:
#         print('真彩图的通道数少于3，无法正常显示')
#         raise
#     t_feature = np.squeeze(t_feature)
#     predict_feature = np.squeeze(predict_feature)
#
#     unique_labels = np.unique(t_feature)
#     cmap = voc_colormap2cmap()  # 使用自定义的颜色条映射
#     # 创建标签与颜色之间的映射
#     norm = mcolors.BoundaryNorm(boundaries=np.arange(len(unique_labels) + 1) - 0.5, ncolors=len(unique_labels))
#     axes[1].imshow(t_feature, cmap=cmap, norm=norm)
#     axes[2].imshow(predict_feature, cmap=cmap, norm=norm,alpha=t_alpha)

# def save_figure(figure, label=None):
#     dir = os.getcwd()
#     dir = os.path.join(dir, 'figures')
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     filename = os.path.join(dir, label)
#     figure.savefig(filename)
#     print(f'Figture creted!path:{filename}')

def write_text(list,filename):
    dir = os.getcwd()
    dir = os.path.join(dir, 'results')
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename=os.path.join(dir, filename)
    with open(filename,'w',encoding='utf-8') as f:
        for item in list:
            f.write(str(item)+'\n')
    print(f'list.txt created path:{filename}')
def read_text(filename):
    dir = os.getcwd()
    dir = os.path.join(dir, 'results')
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename=os.path.join(dir, filename)
    with open(filename,'r',encoding='utf-8') as f:
        text_list = [float(line.strip()) for line in f if line.strip()]
    return text_list
# def voc_colormap2cmap():
#     """将自定义的颜色条转换为 matplotlib 可识别的 ListedColormap"""
#     return mcolors.ListedColormap(np.array(VOC_COLORMAP) / 255.0)
def To_PIL(data_array):
    if data_array.shape[0]==3:
        data_array=data_array.transpose((1,2,0))
    img=Image.fromarray(data_array)
    return img
def label_to_rgb(t):
    '''根据颜色条将label映射到rgb图像'''
    H, W = t.shape
    t=t.reshape(-1)
    rgb=[VOC_COLORMAP[i] for i in t ]
    rgb=np.array(rgb,dtype=np.uint8)
    rgb=rgb.reshape(H,W,3)
    return rgb
def image_pad(im):
    '''给图像填充1像素的黑边'''
    data=np.array(im)
    data=np.pad(data,((1,1),(1,1),(0,0)), 'constant',constant_values=0)
    return To_PIL(data)
def cal_premutation(num):
    '''输入图像的数量，计算最佳排列，如12->3*4,13->4*4'''
    sqrt=np.sqrt(num)
    int_sqrt=int(sqrt)
    if num==int_sqrt*int_sqrt:
        return (int_sqrt,int_sqrt)
    elif (int_sqrt*(int_sqrt+1)>=num) and (num>int_sqrt*int_sqrt):
        return (int_sqrt,int_sqrt+1)
    else:
        return (int_sqrt+1,int_sqrt+1)

def splicing(data):
    '''把四维的array数据转化为多个图像并拼接'''
    # 把四维data转化为多个三维datas
    data = as_numpy(data)
    num, C, H, W = data.shape
    if C==3:
        datas=[data_norm(data[i],(0,255),np.uint8) for i in range(num)]#对数据进行归一化
    elif C==1:
        datas=[label_to_rgb(data[i][0]) for i in range(num)]
    else:
        raise
    im_list=[To_PIL(data) for data in datas]
    im_list=[image_pad(im) for im in im_list]

    # 单幅图像尺寸
    width, height = im_list[0].size

    # 创建空图
    h_num,w_num=cal_premutation(len(im_list))
    result=Image.new(im_list[0].mode, (width*w_num, height*h_num))

    # 图像拼接
    for i in range(h_num):
        for j in range(w_num):
            index=i*w_num+j
            result.paste(im_list[index],(width*j,height*i))
            if (index+1)==len(im_list):
                break
    return result
def Create_ImgForSeg(datas):
    '''接受一个元组，为元组里的数据创建对比图，期望元组（x,t,predict），数据的shape（N，C，H，W）'''
    datas=[as_numpy(data) for data in datas]
    w_num=len(datas)
    h_num=len(datas[0])
    imgs=[]
    for i in range(w_num):
        data=datas[i]
        N, C, H, W = data.shape
        if C==3:
            imgdatas=[data_norm(data[i],(0,255),np.uint8) for i in range(N)]
        elif C==1:
            imgdatas=[label_to_rgb(data[i][0]) for i in range(N)]
        else:
            raise ValueError('数据的通道数不符合要求，要求通道为3或1')
        imglist = [To_PIL(data) for data in imgdatas]
        imglist = [image_pad(im) for im in imglist]
        imgs.append(imglist)
    width, height = imgs[0][0].size
    result=Image.new(imgs[0][0].mode,(width*w_num,height*h_num))
    for i in range(w_num):
        for j in range(h_num):
            imglist=imgs[i]
            result.paste(imglist[j],(width*i,height*j))
    return result


def save_img(im,name='demo.jpg'):
    dir = make_dir('figures')
    if '.jpg' not in name:
        name+='.jpg'
    # 保存图片
    im.save(os.path.join(dir,name))