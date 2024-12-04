'''Info:该脚本用于制作Voc2012的数据集，但是删除了matplotlib的接口，后续考虑使用PIL接口'''
# from matplotlib import image
import os
import numpy as np

'''颜色条'''
VOC_COLORMAP = [[245, 245, 245], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
'''类别'''
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
def pad(feature,label,height,width):
    '''如果图像不支持切割成预定的大小，则对图像进行扩充'''
    H,W,C=feature.shape
    if H<=height:
        pad=height-H
        pad+=10#上下加10个像素
        feature=np.pad(feature,[(pad,pad),(0,0),(0,0)],mode='constant',constant_values=0)
        try:
            label = np.pad(label, [(pad, pad), (0, 0), (0, 0)], mode='constant', constant_values=0)
        except ValueError:
            # plt.imshow(label)
            # plt.show()
            pass
            label = np.pad(label, [(pad, pad), (0, 0)], mode='constant', constant_values=0)
    if W<=width:
        pad=width-W
        pad+=10
        feature=np.pad(feature,[(0,0),(pad,pad),(0,0)],mode='constant',constant_values=0)
        label=np.pad(label,[(0,0),(pad,pad),(0,0)],mode='constant',constant_values=0)
    return feature,label
def read_voc_images(voc_dir, is_train=True):
    """
    读取所有VOC图像并标注
    :param voc_dir: 储存voc官方数据的根目录
    :param is_train: bool，Ture读取训练数据，False读取测试数据
    :return: list，元素为ndarry
    """
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        pass
        # features.append(image.imread(os.path.join(
        #     voc_dir, 'JPEGImages', f'{fname}.jpg')))
        # labels.append((image.imread(os.path.join(
        #     voc_dir, 'SegmentationClass', f'{fname}.png'))*255).astype(np.uint8))
    return features, labels
#@save
def voc_colormap2label():
    '''
    构建从RGB到VOC类别索引的映射
    :return: 返回一维矩阵，如RGB=1，2，3 对应索引为1 则colormap2label[(1*256+2)*256+3]=1
    '''
    colormap2label = np.zeros(256**3)
    for i, colormap in enumerate(VOC_COLORMAP):
        index=(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]
        colormap2label[index] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    '''
    将VOC标签中的RGB值映射到它们的类别索引
    :param colormap: 输入图像，三通道标签图shape（H，W，C）
    :param colormap2label: RGB到索引的映射矩阵，一维ndarray
    :return: 单通道标签图shape（H，W）
    '''
    colormap = colormap.astype(np.float32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    idx = idx.astype(np.int32)
    # idx:目标矩阵，元素值为colormap2label的索引 #colormap2label：映射矩阵，元素值为标签
    return colormap2label[idx]
def rand_crop(orijin_feature, orijin_label, height, width,num):
    '''
    :param orijin_feature: 输入图像，ndarray，shape(height, width, c)
    :param orijin_label: 输入标签，ndarray，shape(height, width, c)
    :param height: 裁剪高度
    :param width: 裁剪宽度
    :param num: 裁剪数量
    :return: ndarray shape（num，height，width，c）
    '''
    orijin_feature,orijin_label=pad(orijin_feature,orijin_label,height,width)#这里如果图像大小不符合，则对图像进行扩充
    features,labels=[],[]
    for i in range(num):
        feature, label = voc_rand_crop(orijin_feature, orijin_label, height, width)
        features.append(feature)
        labels.append(label)
    features=np.array(features)
    labels=np.array(labels)
    return features, labels

def dataset_rand_crop(feature_dataset, label_dataset1, height, width,mutify=10):
    '''
    :param feature_dataset: 输入图像，list，元素为ndarray
    :param label_dataset1: 输入标签，list，元素为ndarray
    :param height: 裁剪高度
    :param width: 裁剪宽度
    :param mutify: 扩充倍数
    :return: ndarray，四维矩阵
    '''
    x_data,t_data=[],[]
    for i in range(len(feature_dataset)):
        features,labels=rand_crop(feature_dataset[i], label_dataset1[i], height, width,num=mutify)
        x_data.append(features)
        t_data.append(labels)
    x_data=np.array(x_data)
    num1,num2,H,W,C=x_data.shape
    x_data=x_data.reshape(num1*num2,H,W,C)
    t_data=np.array(t_data)
    num1,num2,H,W,C=t_data.shape
    t_data=t_data.reshape(num1*num2,H,W,C)
    return x_data,t_data
def voc_rand_crop(feature, label, height, width):
    '''裁剪图像和标签图像'''
    def random_crop(feature, crop_size):
        """对输入的图像进行随机裁剪，返回裁剪后的图像及裁剪区域"""
        crop_width, crop_height = crop_size
        height, width = feature.shape[:2]
        top = np.random.randint(0, height - crop_height + 1)
        left = np.random.randint(0, width - crop_width + 1)
        cropped_feature = feature[top:top + crop_height, left:left + crop_width]
        return cropped_feature, (left, top, crop_width, crop_height)
    def fixed_crop(image, left, top, crop_width, crop_height):
        """从输入的图像中根据给定的矩形区域裁剪出图像"""
        return image[top:top + crop_height, left:left + crop_width]
    """随机裁剪特征和标签图像"""
    feature, rect = random_crop(feature, (width, height))
    label = fixed_crop(label, *rect)
    return feature, label
def create_data(path, is_train=True,height=224,width=224,mutify=10):
    '''
    :param path: 存放数据的根目录
    :param is_train: 训练数据or测试数据
    :param height: 裁剪高度
    :param width: 裁剪宽度
    :param mutify: 扩充倍数
    :return: 数据集，四维
    '''
    feature,label=read_voc_images(path, is_train)#从根目录获取图像数据，list
    feature,label=dataset_rand_crop(feature,label,height,width,mutify=mutify)#随机裁剪，扩充图像数量，ndarray
    t=[]
    for i in range(len(label)):
        t.append(voc_label_indices(label[i],voc_colormap2label()))
    t=np.array(t).astype(np.uint8)#三通道标签数据转为单通道标签数据
    return feature,t
#请指定VOC2012数据的绝对位置，如下所示
vocdir='D:\\Programing\\pythonProject\\Dezero\\voc2012_dataset\\VOC2012'

# feature1,label1=create_data(vocdir,is_train=True,mutify=10)
# feature2,label2=create_data(vocdir,is_train=False,mutify=1)
# np.savez_compressed('voc2012_train.npz',feature=feature1,label=label1)
# np.savez_compressed('voc2012_test.npz',feature=feature2,label=label2)