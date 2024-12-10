import numpy as np
import skystar.cuda as cuda
'''两个重要概念，句子长度和句字长度。src_len表示中文句子固定最大长度，tgt_len 表示英文句子固定最大长度'''

'''参数设置
d_model = 512  # 字 Embedding 的维度
d_ff = 2048  # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度 
n_layers = 6  # 有多少个encoder和decoder
n_heads = 8  # Multi-Head Attention设置为8
'''
def sequence_mask(seq):  # seq: [batch_size, tgt_len]
    '''掩膜未来输入信息，用于Decoder'''
    xp = cuda.get_array_module(seq)
    attn_shape = [seq.shape[0], seq.shape[1], seq.shape[1]]
    subsequence_mask = xp.triu(xp.ones(attn_shape,dtype=xp.float32), k=1)  # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]  # [batch_size, tgt_len, tgt_len]
    return subsequence_mask


def padding_mask(seq_q, seq_k):  # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    '''掩膜没有实际意义的占位符'''
    xp = cuda.get_array_module(seq_q)
    batch_size, len_q = seq_q.shape
    pad_attn_mask = (seq_k == 0)# 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    pad_attn_mask = xp.expand_dims(pad_attn_mask, axis=1)#[batch_size, 1, len_k]
    pad_attn_mask = xp.repeat(pad_attn_mask, len_q, axis=1)#复制扩充，扩展成多维度
    pad_attn_mask = xp.array(pad_attn_mask,dtype=xp.float32)
    return pad_attn_mask
