import numpy as np
try:
    from skystar.dataset_mnist.mnist import load_mnist
    # from skystar.STL10 import read_all_images, read_labels
    # from skystar.tiny_image import load_data
    from skystar.sky_dataset.load import _loaddata
except ImportError:
    print('please check dataset path')
import os
class Dataset:
    def __init__(self,training=True,transform=None,target_transform=None):
        self.training=training
        self.transform=transform
        self.target_transform=target_transform
        if self.transform is None:
            self.transform=lambda x:x#设置临时函数，将数据按照原样输出
        if self.target_transform is None:
            self.target_transform=lambda x:x
        self.data=None
        self.label=None
        self.prepare()

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):  # 处理单个标量索引
            if self.label is None:
                return self.transform(self.data[index]), None
            else:
                return self.transform(self.data[index]), self.target_transform(self.label[index])
        elif isinstance(index, slice):  # 处理切片索引
            if self.label is None:
                return self.transform(self.data[index]), None
            else:
                return (self.transform(self.data[index]),
                        self.target_transform(self.label[index]))
        else:
            raise TypeError(f'Invalid index type: {type(index)}')

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass

class BigData(Dataset):
    def __getitem__(self, index):
        x=np.load(f'data/{index}.npy')
        t=np.load(f'label/{index}.npy')

        return x,t
    def __len__(self):
        return 1000000

class MathImg_data(Dataset):
    def __init__(self,normalize=True,flatten=True,one_hot_label=False,training=True,transform=None,target_transform=None):
        self.normalize=normalize
        self.flatten=flatten
        self.one_hot_label=one_hot_label
        super().__init__(training,transform,target_transform)
    def prepare(self):
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=self.normalize, flatten=self.flatten, one_hot_label=self.one_hot_label)
        if self.training:
            self.data=x_train
            self.label=t_train
        else:
            self.data=x_test
            self.label=t_test

class Stl_10(Dataset):
    def __init__(self,training=True,transform=None,target_transform=None):
        super().__init__(training,transform,target_transform)

    def prepare(self):
        path=os.path.abspath(__file__)
        path=os.path.dirname(path)
        if self.training:
            X_path=os.path.join(path,'STL10\\stl10_binary','train_X.bin')
            Y_path=os.path.join(path,'STL10\\stl10_binary','train_y.bin')
            x_train = read_all_images(X_path)
            self.data = x_train.transpose(0, 3, 1, 2)
            self.label = read_labels(Y_path)
            self.label=self.label-1#该数据的标签从1开始
        else:
            X_path=os.path.join(path,'STL10\\stl10_binary','test_X.bin')
            Y_path=os.path.join(path,'STL10\\stl10_binary','test_y.bin')
            x_test=read_all_images(X_path)
            self.data=x_test.transpose(0,3,1,2)
            self.label=read_labels(Y_path)
            self.label=self.label-1#该数据的标签从1开始

# class Tiny_image(Dataset):
#     def __init__(self,training=True,transform=None,target_transform=None):
#         super().__init__(training,transform,target_transform)
#
#     def prepare(self):
#         x_train, t_train, x_test, t_test=load_data()
#         if self.training:
#             self.data,self.label=x_train,t_train
#         else:
#             self.data,self.label=x_test,t_test

class selfdata_starsky(Dataset):
    def __init__(self,filename,training=True,transform=None,target_transform=None,split=True):
        '''默认将数据集分割为训练集与验证集'''
        self.filename=filename
        self.split=split
        super().__init__(training,transform,target_transform)

    def prepare(self):
        if self.split:
            x_train, t_train, x_test, t_test = _loaddata(self.filename,split=self.split)
            if self.training:
                self.data, self.label = x_train, t_train
            else:
                self.data, self.label = x_test, t_test
        else:
            x, t = _loaddata(self.filename,split=self.split)
            return x,t