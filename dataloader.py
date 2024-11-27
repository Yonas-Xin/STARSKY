import math
import numpy as np
from skystar import cuda

class Dataloader:
    def __init__(self,dataset,batch_size,shuffle=True,gpu=False):
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

        i,batch_size=self.iteration,self.batch_size
        batch_index=self.index[i*batch_size:(i+1)*batch_size]
        batch=[self.dataset[i] for i in batch_index]
        xp=cuda.cupy if self.gpu else np
        x=xp.array([example[0] for example in batch])
        t=xp.array([example[1] for example in batch])

        self.iteration+=1
        return x,t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu=False

    def to_gpu(self):
        self.gpu=True

    def __len__(self):
        return self.data_size


class SeqDataloader(Dataloader):
    '''创建一个用于时间序列模型的数据加载器，如果数据量为1500，batch=3，那么每一小批量数据将分别从[1,501,1001]开始，一直到[500,100,1500]'''
    def __init__(self,dataset,batch_size,gpu=False):
        super().__init__(dataset=dataset,batch_size=batch_size,shuffle=False,gpu=gpu)#时间序列模型训练数据不打乱

    def __next__(self):
        if self.iteration>=self.max_iter:
            self.reset()
            raise StopIteration

        jump=self.data_size//self.batch_size#获取偏移量
        batch_index=[(i*jump+self.iteration)%self.data_size for i in range(self.batch_size)]
        batch=[self.dataset[i] for i in batch_index]

        xp=cuda.cupy if self.gpu else np
        x=xp.array([example[0] for example in batch])
        t=xp.array([example[1] for example in batch])

        #如果数据集是一维，将他们变为二维，每一列为一个批量
        if x.ndim==1:
            x=x.reshape(-1,1)
        if t.ndim==1:
            t=t.reshape(-1,1)
        self.iteration+=1
        return x,t

    def __len__(self):
        '''序列数据的长度等于序列长度除以批次，向上取整'''
        return math.ceil(self.data_size/self.batch_size)