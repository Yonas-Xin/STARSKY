from PIL import Image
import numpy as np
import os

def init_dir():#初始化一个储存数据的文件夹
    dir_path=os.getcwd()
    dir_path=os.path.join(dir_path,'skystar_dataset')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        os.mkdir(os.path.join(dir_path,'data_txt'))
        print('Dir skystar_dataset created successfully! Please place the .txt file in the data_txt directory')
    return dir_path

def get_names(path):
    '''从txt文本中获取图像的路径'''
    dir_path=init_dir()
    if not os.path.isdir(path):#路径不是绝对路径
        path=os.path.join(dir_path,'data_txt',path)
        if not os.path.exists(path):
            print('The path {} does not exist!Please place the .txt file in the data_txt directory'.format(path))
            raise FileNotFoundError

    with open(path, 'r', encoding='utf-8') as f:
        dataset = f.readlines()
    train_names = []
    label_names = []
    for name in dataset:
        name = name.split()
        mid = len(name) // 2
        name1, name2 = '', ''
        for i in range(mid):
            if i != mid - 1:
                name1 += (name[i] + ' ')
                name2 += (name[i + mid] + ' ')
            else:
                name1 += name[i]
                name2 += name[i + mid]
        train_names.append(name1)
        label_names.append(name2)
    return train_names, label_names
def image_generator(path):
    '''生成器，按需从路径中加载图片并生成图像数组和标签'''
    names1,names2=get_names(path)
    for i in range(len(names1)):
        name1 = names1[i]
        name2 = names2[i]
        img1 = Image.open(name1)
        img_array1 = np.array(img1)
        img_array1 = img_array1.transpose(2, 0, 1)# 将通道维度放在最前面

        img2 = Image.open(name2)
        img_array2 = np.array(img2)
        yield img_array1, img_array2

def create_dataset(path,shuffle=True):
    '''该函数返回训练数据集与标签数据集，shuffle指定是否打乱数据集的排序，默认打乱'''
    x_dataset,t_dataset=[],[]
    for data1,data2 in image_generator(path):
        x_dataset.append(data1)
        t_dataset.append(data2)
    if shuffle:
        index = np.random.permutation(len(x_dataset))
        x_dataset =np.array([x_dataset[i] for i in index])
        t_dataset=np.array([t_dataset[i] for i in index])
    batch,H,W=t_dataset.shape
    t_dataset=t_dataset.reshape(batch,1,H,W)
    return x_dataset,t_dataset

def data_to_npz(data_txt_path,outfile_name):
    '''使用该函数直接将数据集保存为.npz格式'''
    print('Data Craating......')
    x_dataset,t_dataset=create_dataset(data_txt_path)
    if not os.path.isdir(outfile_name):
        dir_path=init_dir()
        outfile_name=os.path.join(dir_path,outfile_name)
    np.savez_compressed(outfile_name,x_dataset=x_dataset,t_dataset=t_dataset)
    print('Data Craating......Done!')
    return

if __name__ == '__main__':
    data_to_npz('dataset.txt','dataset.npz')