import math
import numpy as np
from skystar import cuda

class Dataloader:
    '''从Dataloader取出的数据维度ndim一致'''
    def __init__(self,dataset,batch_size,shuffle=True,gpu=False,dtype=np.float32):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.data_size=len(dataset)
        self.max_iter=math.ceil(self.data_size/batch_size)#向上取整
        self.gpu=gpu

        self.reset()

    def reset(self):
        self.iteration=0
        if self.shuffle:
            self.index=np.random.permutation(self.data_size)
        else:
            self.index=np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration>=self.max_iter:
            self.reset()
            raise StopIteration

        '''按照batch大小取出数据，如果最后一部分数据少于batch，那么直接取出剩余数据'''
        i,batch_size=self.iteration,self.batch_size
        batch_index=self.index[i*batch_size:(i+1)*batch_size]
        batch=[self.dataset[i] for i in batch_index]
        xp=cuda.cupy if self.gpu else np
        x=xp.array([example[0] for example in batch],dtype=xp.float32)
        t=xp.array([example[1] for example in batch],dtype=xp.int32)

        self.iteration+=1
        return x,t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu=False

    def to_gpu(self):
        self.gpu=True

    def __len__(self):
        return self.max_iter



class SeqDataloader(Dataloader):
    '''创建一个用于时间序列模型的数据加载器，如果数据量为1500，batch=3，那么每一小批量数据将分别从[1,501,1001]开始，一直到[500,100,1500]'''
    def __init__(self,dataset,batch_size,gpu=False):
        super().__init__(dataset=dataset,batch_size=batch_size,shuffle=False,gpu=gpu)#时间序列模型训练数据不打乱

    def __next__(self):
        if self.iteration>=self.max_iter:
            self.reset()
            raise StopIteration
        '''偏移：假如一个时序数据（1，2，3，4，5，6），batch为2，那么数据分为两组，取三次数据（1，4）（2，5）（3，6）
        如果数据本身就是一个序列，如（1，4）是两个具有高维的序列数据，偏移与否不影响最后的输出'''
        jump=self.data_size//self.batch_size#获取偏移量,用于时序偏移
        batch_index=[(i*jump+self.iteration)%self.data_size for i in range(self.batch_size)]
        batch=[self.dataset[i] for i in batch_index]
        xp = cuda.cupy if self.gpu else np
        if len(batch[0])==2:
            x=xp.array([example[0] for example in batch],dtype=xp.float32)
            t=xp.array([example[1] for example in batch],dtype=xp.int32)

            #如果数据集是一维，将他们变为二维，每一列为一个批量
            if x.ndim==1:
                x=x.reshape(-1,1)
            if t.ndim==1:
                t=t.reshape(-1,1)
            self.iteration+=1
            return x,t
        elif len(batch[0])==3:
            x = xp.array([example[0] for example in batch],dtype=xp.float32)
            y = xp.array([example[1] for example in batch],dtype=xp.int32)
            t = xp.array([example[2] for example in batch],dtype=xp.int32)

            # 如果数据集是一维，将他们变为二维，每一列为一个批量
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if t.ndim == 1:
                t = t.reshape(-1, 1)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            self.iteration += 1
            return x,y,t
        else:
            raise StopIteration

    def __len__(self):
        '''序列数据的长度等于序列长度除以批次，向上取整'''
        return self.max_iter