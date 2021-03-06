"""
@project : Translate-en-cn
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : test.py
@ide     : PyCharm
@time    : 2022-06-23

"""


import os
import math
import copy
import time
import json
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nltk import word_tokenize
import nltk
# nltk.download('punkt')
from collections import Counter
from torch.autograd import Variable

# 初始化参数设置
UNK = 0           # The id of unknown word in the vocabulary
PAD = 1           # The id of padding word in the vocabulary
BATCH_SIZE = 64   # Batch size, data number in a data
EPOCHS = 20       # Epochs
LAYERS = 6        # encoder and decoder blocks number in the transformer
H_NUM = 8         # multihead attention hidden个数
D_MODEL = 256     # embedding dimensions
D_FF = 1024       # feed forward dimensions
DROPOUT = 0.1     # dropout rate
MAX_LENGTH = 60   # The max length of a sentence

TRAIN_FILE = 'corpus/train.json'    # Train data
DEV_FILE = "corpus/dev.json"        # Develop / Evaluate data
SAVE_FILE = 'save/model.pt'         # model saving path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def seq_padding(X, padding=0):
    """
    TODO: Padding the sentence with padding id. Make sentence into same length
    X = [["I", "love", "you"],
         ["Me", "too"],
         ["This", "is", "a", "little", "cat"]]

    X = np.array([['I', 'love', 'you', '0', '0'],
                  ['Me', 'too', '0', '0', '0'],
                  ['This', 'is', 'a', 'little', 'cat']])
    """
    # Calculate the length of these sentences in a Batch
    L = [len(x) for x in X]

    # Get the max length of these sentences
    ML = max(L)

    # If the length of a sentence is less than the max length, fill the last ML-len(x) tokens with padding id
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])



class PrepareData:
    def __init__(self, train_path, dev_path):

        # Load the data and split into word tokens
        self.train_en, self.train_cn = self.load_data(train_path)
        self.dev_en, self.dev_cn = self.load_data(dev_path)

        # Build the vocabulary
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)

        # Change the word token into id
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)

        # Pad the data, and split the data into batches
        self.train_data = self.splitBatch(self.train_en, self.train_cn, BATCH_SIZE)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_cn, BATCH_SIZE)


    def load_data(self, path):
        """
        read the bilinguistic corpus in English and Chinese
        use nltk word_tokenize to change corpus into word tokens

        formation: en = [['BOS', 'i', 'love', 'you', 'EOS'],
                         ['BOS', 'me', 'too', 'EOS'], ...]
                   cn = [['BOS', '我', '爱', '你', 'EOS'],
                         ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        en = []
        cn = []
        

        json_list = json.load(open(path))
        for list_content in json_list:
            en.append(['BOS'] + word_tokenize(list_content["english"]) + ['EOS'])
            cn.append(['BOS'] + [ch for ch in list_content["chinese"]] + ['EOS'])


        return en, cn



    def build_dict(self, sentences, max_words = 50000):
        """
        Load the tokenized data after load_data
        Build the word token dictionary {word_token: token_id}
        """
        # Count the token number in all the corpus
        word_count = Counter()

        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1


        # Only reserve the top max_words frequency words
        # Add UNK and PAD, for 0 and 1
        ls = word_count.most_common(max_words)
        
        # Static the total occurence of each word
        total_words = len(ls) + 2       # Because 0 and 1 have been taken placed
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD
        
        # Build another reverse dictionary from id to tokens
        index_dict = {v: k for k, v in word_dict.items()}

        return word_dict, total_words, index_dict

    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):
        """
        This method can change the English tokens and Chinese tokens to id
        If the sort attribute is True, the token id will be sorted by the length
            of English sentence length
        This is for the padding procedure to reduce the total length of a batch

        """
        # Calculate the English Sentence Number
        length = len(en)
        
        # TODO: Change the word token into id for English and Chinese
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        # Build a function for return the order of sentence length
        def len_argsort(seq):
            """
            seq: [sentence0, sentence1, ..., sentenceN]
            return: The sentence index by the order of sentence length / token number
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # Sort the sentence by the sentence length
        if sort:
            # Base on the english sentence length
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]
            
        return out_en_ids, out_cn_ids


    def splitBatch(self, en, cn, batch_size, shuffle=True):
        """
        TODO: Split the data into different data batch
        If shuffle is true, the order of the data batch is shuffled
        """
        idx_list = np.arange(0, len(en), batch_size)

        # If shuffle is true, the order of the train data is shuffled
        if shuffle:
            np.random.shuffle(idx_list)

        # batch_indexs is a multilayer list, stores the index of each label
        batch_indexs = []
        for idx in idx_list:
            # The order of the largest batch maybe beyond the data range
            # if idx is more than the total dataset range, take the minimum one
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))

        # Index the sentence order by the batch_indexs
        batches = []
        for batch_index in batch_indexs:

            # index the batch data for English and Chinese
            batch_en = [en[index] for index in batch_index]  
            batch_cn = [cn[index] for index in batch_index]

            # Perform padding operation for each pad.
            # shape -> (bath_size, max_sentence_size)
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)

            # Instance a Batch Object and append it to the list
            batches.append(Batch(batch_en, batch_cn))

        return batches


class Batch:
    """
    TODO:Object for holding a batch of data with mask during training.
    """

    def __init__(self, src, trg=None, pad=0):
        """
        src: source data like token id after padding in a batch  src.shape -> (N, L_src)
        trg: target data like token id after padding in a batch  trg.shape -> (N, L_trg)
        """

        # Change numpy array into torch.Tensor using long int
        src = torch.from_numpy(src).to(DEVICE).long()
        trg = torch.from_numpy(trg).to(DEVICE).long()
        self.src = src

        # Get the masked boolean matrix
        # Add one -2 dimension unsqueeze(-2), shape in (N, 1, L)
        self.src_mask = (src != pad).unsqueeze(-2)

        # If there is the target data, add mask to the target data in decoder
        if trg is not None:
            # Because we use the seq to seq model, we have to predict the token one by one
            # decoder using the left without the last one as the input
            self.trg = trg[:, :-1]

            # the real token is the last one of the target data when training the decoder
            self.trg_y = trg[:, 1:]

            # using the input part to create the attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)

            # statistic the real token numbers in the output target value
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask mechanism
    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        tgt.shape -> (N, L)
        """

        # tgt = torch.randint(5, (4, 10))

        # If element is not equal to padding value, the mask value is true
        # Then add a new dimension at -2, to spreed the -1 dimension (N, L) -> (N, 1, L)
        tgt_mask = (tgt != pad).unsqueeze(-2)

        # subsequent_mask function to generate a mask square with the last dimension L
        mask_square = subsequent_mask(tgt.size(-1))
        mask_square = Variable(mask_square.type_as(tgt_mask.data))

        # Spread the new -2 dimension from 1 to L, with the value as both tgt_mask and mask_square
        # tgt_mask.shape    -> (N, 1, L)
        # mask_square.shape -> (1, L, L)
        # attn_mask.shape   -> (N, L, L)
        attn_mask = tgt_mask & mask_square

        return attn_mask


def subsequent_mask(L_length):
    """
    TODO: Generate the Self-Mask for the Masked-Self-Attention matrix with a length
    L_length: The lengths of a sentence
    size = 4
    First:                Second:
    [[0, 1, 1, 1],        [[ True, False, False, False],
     [0, 0, 1, 1],         [ True,  True, False, False],
     [0, 0, 0, 1],         [ True,  True,  True, False],
     [0, 0, 0, 0]]         [ True,  True,  True,  True]]
    """
    # Set the mask square size
    attn_shape = (1, L_length, L_length)

    # First: Generate a triangle matrix with top right 1 (without eye) and left bottom 0
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # Second: Generate a triangle matrix with top right False (without eye) and left bottom True
    subsequent_mask = torch.from_numpy(subsequent_mask) == 0

    # subsequent_mask.shape -> (1, L_length, L_length)
    return subsequent_mask



class Embeddings(nn.Module):
    """
    TODO: Automatic Embedding. Change the dimension of the vocabulary data.
    From the 1-hot vocabulary data to d_model dimensions
    """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()

        # Embedding using the embedding method of PyTorch
        self.lut = nn.Embedding(vocab, d_model)

        # Embedding Dimensions
        self.d_model = d_model

    def forward(self, x):
        # Return the embedding matrix corresponding with x
        # The Embedding need to multiply the sqrt of the dimension
        return self.lut(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    """
    TODO: positional encoding
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化一个size为 max_len(设定的最大长度)×embedding维度 的全零矩阵
        # 来存放所有小于这个长度位置对应的porisional embedding

        # Initilize an zero matrix with the size as (max_len, d_model)
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        """
        Formation like:
        tensor([[0.],
                [1.],
                [2.],
                [3.],
                [4.],
                ...])
        """
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)

        # Realize the calucation of the positional embedding by torch.exp() and math.log()
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))
        
        # TODO: 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 加1个维度，使得pe维度变为：1×max_len×embedding维度
        # (方便后续与一个batch的句子所有词的embedding批量相加)
        pe = pe.unsqueeze(0) 
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    """
    TODO: Realize the attention machinism
    query = (N, L, D), key = (N, L, D), value = (N, L, D)
    N: Batch size
    L: Token Number / Sentence Length
    D: Token Dimensions

    mask: mask matrix
    dropout: dropout layer
    """
    
    # d_k is the last dimension with means the token dimension
    d_k = query.size(-1)

    
    # Score = (Q * K^T) / sqrt(d_k)
    # Transpose the matrix with the -1 & -2 dimension, 
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores.shape -> (N, L, L)

    # using einsum
    # scores = torch.einsum("nld,nhd->nlh", query, key) / math.sqrt(d_k)
    
    # mask here is a layer
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)     # mask.shape -> (N, L, L)
        
    # Using Softmax(Score)
    p_attn = F.softmax(scores, dim = -1)
    
    # dropout here is a function or None
    if dropout is not None:
        Dropout = nn.Dropout(dropout)
        p_attn = Dropout(p_attn)
    
    # return the attention score = (softmax(Q*K^T)/sqrt(d))*V and the attention matrix

    return torch.matmul(p_attn, value), p_attn



class MultiHeadedAttention_v2(nn.Module):
    """
    TODO: Simplex method 
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention_v2, self).__init__()

        # assert that the original dimension can be fully divided by head numbers
        assert d_model & h == 0
        self.model_dimension = d_model
        self.dropout = dropout
        self.H = h                     # Number of head
        self.D = int(d_model / h)
        
        # The weights of Q, K and V
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # Fully connected layer attached to the final attention score
        self.FC = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask):
    
        dropout = self.dropout
        N = Q.shape[0]   # Size of one batch
        H = self.H       # Number of heads
        D = self.D       # Separated dimension
        L = Q.shape[1]   # Number of tokens in a sentence

        # Get the value of Q, K, V
        # Change the dimension of L and H to perform multiheaded attention
        Q = self.Wq(Q).view(N, L, H, D).transpose(1, 2)
        K = self.Wk(K).view(N, L, H, D).transpose(1, 2)
        V = self.Wv(V).view(N, L, H, D).transpose(1, 2)

        # Call the attention function
        x, atten = attention(Q, K, V, mask, dropout)
        # print(atten.shape)
        self.atten = atten

        # Concatenate the separated heads into the original shape
        x = x.transpose(1, 2).contiguous().view(N, L, H*D)
        
        x = self.FC(x)

        return x



class MultiHeadedAttention(nn.Module):
    """
    Define the multi- head attention layer
    """
    
    def __init__(self, h, d_model, dropout=0.1):
        """
        h: number of head
        d_model: input length
        """
        
        super(MultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        self.d_k = d_model // h   # Length of a head
        self.h = h                # head numbers

        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        # The 0 dimension is the batch size
        N = query.size(0)
        H = self.h
        D = self.d_k
        
        # l: linear module
        # x: query, key, value: the data logits as x
        # l(x): Q=Wq*x, K=Wk*x, V=Wv+x
        query, key, value = [l(x).view(N, -1, H, D).transpose(1, 2) 
                             for l, x in zip(self.linears, (query, key, value))]
        
        # Use attention module to get the attention matrix and the
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # Concatenate the separated heads into the original shape
        x = x.transpose(1, 2).contiguous().view(N, -1, self.h * self.d_k)
        
        # Add one fully connected layer. Use the last one of the linears module list
        return self.linears[-1](x)





class LayerNorm(nn.Module):
    """
    TODO: LayerNorm
    """
    def __init__(self, size, eps=1e-6):
        """
        size: input dimension
        """
        super(LayerNorm, self).__init__()

        # initialization with α = 1, β = 0
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

        # smooth item eps, set is 1e-6
        self.eps = eps

    def forward(self, x):
        # Calculate the mean and the standard error using for layer norm
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        # Weight * (x - E(x)) / sqrt(var(x) + epsilon) + Bias
        output = self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2
        return output




class AddAndNormLayer(nn.Module):
    """
    Like Residual Network
    Add the original data and the function layer data together
    function layer data can be MHA or FF
    There should be a Layer norm after each Layer
    """
    def __init__(self, size, dropout, functionlayer):
        super(AddAndNormLayer, self).__init__()
        
        # Using LayerNorm defined before
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.functionlayer = functionlayer

        # functionlayer = self_attn
        # functionlayer = feed_forward


    def forward(self, org_x, *args):
        
        # Return the addition of original data and the function layer data
        org_x = self.norm(org_x)
        residual = self.functionlayer(*args)
        residual = self.dropout(residual)
        output = org_x + residual
        
        return output


def clones(module, N):
    """
    TODO: Clone a layer for N times
    Retrun a module list, contain N same module without sharing the parameters
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])   # copy library





class PositionwiseFeedForward(nn.Module):
    """
     TODO: Realize the feed forward layer
     Two linear fully conected layer
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)   # Layer One
        self.w_2 = nn.Linear(d_ff, d_model)   # Later Two
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.w_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class Encoder(nn.Module):
    """
    TODO: Clone the layer structure for N times, N =6
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # Clone single layers for N times
        self.layers = clones(layer, N)
        # Layer Norm, size is the layer's size
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Clone the layer structure for N times, N =6
        Connect the layers in a Module List,
        Attach a Layer Norm to the last layer
        x: words tokens after embedding
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



class EncoderLayer(nn.Module):
    """
    TODO: Build the encoder layer
    X
    MHA = Multihead_Attention(X)
    AN1 = Add_Norm_Layer(X + MHA)
    FF = Feedforward_Layer(AN1)
    AN2 = Add_Norm_Layer(FF + AN1)
    return AN2
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        self_atten: Self-Attention Module
        feed_forward: Feed_forward Module
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.add_norm_1 = AddAndNormLayer(size, dropout, self.self_attn)
        self.add_norm_2 = AddAndNormLayer(size, dropout, self.feed_forward)

        # d_model
        self.size = size

    def forward(self, x, mask):
        """

        """
        
        # Data stream inside a 
        # x = self.self_attn(x, x, x, mask)
        x = self.add_norm_1(x, x, x, x, mask)
        # x = self.feed_forward(x)
        x = self.add_norm_2(x, x)

        return x



class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()

        # TODO: Build the Decoder
        # Duplicate N times of decoder layer
        self.layers = clones(layer, N)

        # Layer Norm
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Duplicate N times of decoder layer, here we use 6 times
        The Decoder layer will receive the outcome from the
        For each output, perform attention mask and subsequent mask
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    * Decoder Layer needs two attention layers
    * One for self-attention Layer
    * One for masked-attention Layer

    X
    MMHA = Masked_Multiheead_Attention(X)
    AN1  = Add_Norm_Layer(X + MMHA)
    MHA  = Multihead_Attention(Q->Encoder, K,V->AN1)
    AN2  = Add_Norm_Layer(AN1 + MHA)
    FF   = Feed_Forward(AN2)
    AN3  = Add_Norm_Layer(AN2 + FF)
    Y    = Linear(AN3)
    Y    = Softmax(Y)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()

        self.size = size

        # Self-Attention for the second
        self.self_attn = self_attn

        # Masked-Attention for the first
        self.src_attn = src_attn

        # Feed forward layer
        self.feed_forward = feed_forward

        # 3 add and norm layers
        # Add_Norm_Layer(X + MMHA)
        self.add_norm_1 = AddAndNormLayer(size, dropout, self.self_attn)
        self.add_norm_2 = AddAndNormLayer(size, dropout, self.src_attn)
        self.add_norm_3 = AddAndNormLayer(size, dropout, self.feed_forward)

    def forward(self, x, memory, src_mask, tgt_mask):

        # m used for storing the hidden outcome for Q and K from encoder
        m = memory

        # TODO: Follow the encoder layer finish the decoder layer
        # The first attention is self attention
        # The second attention is masked self attention

        x = self.add_norm_1(x, x, x, x, tgt_mask)
        x = self.add_norm_2(x, x, m, m, src_mask)
        x = self.add_norm_3(x, x)

        return x





# class Encoder(nn.Module):

#     def __init__(self, OneLayer, N):
#         super(Encoder, self).__init__()
#         self.Layers = clones(OneLayer, N)
#         self.LayerNorm = LayerNorm(OneLayer)

#     def forward(self, x, mask):
#         for Layer in self.Layers:
#             x = Layer(x, mask)
#         x = self.LayerNorm(x)
#         return x


class Transformer(nn.Module):
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        encoder: Encoder, N * EncoderLayer
        decoder: Decoder, N * DecoderLayer
        src_embed: The Embeddings of the source data
        tgt_embed: The Embeddings of the target data
        generator: Generator after the decoder
        """
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator 

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.encoder(src, src_mask)
        return src

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.decoder(tgt, memory, src_mask, tgt_mask)
        return tgt

    def forward(self, src, tgt, src_mask, tgt_mask):
        # The result of encoder should be taken as the input of the decoder memory
        # m = encoder(Q=x, K=x, V=x)
        # result = decoder(Q=x, K=m, V=m)
        memory = self.encode(src, src_mask)
        tgt = self.decode(memory, src_mask, tgt, tgt_mask)
        return tgt


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()

        # The data after the decode. Change the dimension from the d_model to the vocab
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # perform softmax and log for the outcome of the decoder
        x = self.proj(x)
        x = F.log_softmax(x, dim=-1)
        return x


def make_model(
    src_vocab,     # source vocabulary English
    tgt_vocab,     # target vocabulary Chinese
    N=6,           # Encoder layer number
    d_model=512,   # Token dimensions
    d_ff=2048,     # Feed Forward dimensions
    h = 8,         # Head numbers
    dropout=0.1    # dropout rate
    ):

    # N, d_model, d_ff, h, dropout = 6, 512, 2048, 8, 0.1

    c = copy.deepcopy
    
    # instantiate the Attention module
    attn = MultiHeadedAttention_v2(h, d_model).to(DEVICE)
    
    # instantiate the Feed Forward module
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    
    # instantiate the PositionalEncoding module
    position = PositionalEncoding(d_model, dropout).to(DEVICE)

    # encoder and decoder of the
    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE)

    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE)

    src_embed = nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position))

    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),

    generator = Generator(d_model, tgt_vocab)

    # instantiate the Transformer module
    model = Transformer(encoder, decoder, src_embed, tgt_embed, generator).to(DEVICE)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # Take the xavier method for initialization nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)



class LabelSmoothing(nn.Module):
    """
    TODO: Perform Label smoothing
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
 

 
class SimpleLossCompute:
    """
    Simple loss calculation and backward broadcast for training the new parameter
    """
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        # __call__() method to realize the class as a function
        x = self.generator(x)

        # loss = (x - y) / norm, x and y in (N, L*D)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm

        # Backward broadcast
        loss.backward()

        if self.opt is not None:

            # move the optimizer and zero gradient
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data.item() * norm.float()



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))




def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for i, batch in enumerate(data):

        # batch = next(iter(data))
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def train(dataset, model, criterion, optimizer):
    """
    Train the model and save it
    """
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_dev_loss = 1e5
    
    for epoch in range(EPOCHS):
        # 模型训练
        model.train()
        run_epoch(dataset.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        model.eval()

        # 在dev集上进行loss评估
        print('>>>>> Evaluate')
        dev_loss = run_epoch(dataset.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        print('<<<<< Evaluate loss: %f' % dev_loss)
        
        # TODO: 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), SAVE_FILE)
            best_dev_loss = dev_loss
        
        
        print()


# 数据预处理
dataset = PrepareData(TRAIN_FILE, DEV_FILE)
src_vocab = len(dataset.en_word_dict)
tgt_vocab = len(dataset.cn_word_dict)
print("src_vocab %d" % src_vocab)
print("tgt_vocab %d" % tgt_vocab)

# 初始化模型
model = make_model(
                    src_vocab, 
                    tgt_vocab, 
                    LAYERS, 
                    D_MODEL,
                    D_FF,
                    H_NUM,
                    DROPOUT
                )

# 训练
print(">>>>>>> start train")
train_start = time.time()
criterion = LabelSmoothing(tgt_vocab, padding_idx = 0, smoothing= 0.0)
optimizer = NoamOpt(D_MODEL, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))

train(data, model, criterion, optimizer)
print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    """
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len-1):
        # decode得到隐层表示
        out = model.decode(memory, 
                           src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def evaluate(data, model):
    """
    在data上用训练好的模型进行预测，打印模型翻译结果
    """
    # 梯度清零
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for i in range(len(data.dev_en)):
            # TODO: 打印待翻译的英文句子
            en_sent = " ".join([data.en_index_dict[w] for w in  data.dev_en[i]])
            print("\n" + en_sent)
            
            # TODO: 打印对应的中文句子答案
            cn_sent = " ".join([data.cn_index_dict[w] for w in  data.dev_cn[i]])
            print("".join(cn_sent))
            
            # 将当前以单词id表示的英文句子数据转为tensor，并放如DEVICE中
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(DEVICE)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])
            # 初始化一个用于存放模型翻译结果句子单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):
                # 获取当前下标的输出字符
                sym = data.cn_index_dict[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前句子的翻译结果列表
                if sym != 'EOS':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            # 打印模型翻译输出的中文句子结果
            print("translation: %s" % " ".join(translation))

# 预测
# 加载模型
model.load_state_dict(torch.load(SAVE_FILE))
# 开始预测
print(">>>>>>> start evaluate")
evaluate_start  = time.time()
evaluate(data, model)         
print(f"<<<<<<< finished evaluate, cost {time.time()-evaluate_start:.4f} seconds")

"""
He is my youngest brother . EOS
BOS 他 是 我 最 年 轻 的 兄 弟 。 EOS
translation: 他 是 我 最 年 轻 的 兄 弟 。

BOS Most boys like computer games . EOS
BOS 大 多 数 男 生 喜 欢 电 脑 游 戏 。 EOS
translation: 大 部 分 男 生 喜 欢 电 脑 游 戏 。

BOS Can I have a bite ? EOS
BOS 我 可 以 吃 一 口 嗎 ？ EOS
translation: 我 有 咬 吗 ？

BOS This is not very UNK . EOS
BOS 這 不 是 很 流 行 。 EOS
translation: 這 不 是 很 匆 忙 的 。

BOS There is a fork missing . EOS
BOS 少 一 把 叉 子 。 EOS
translation: 少 一 把 叉 子 。

BOS I love my yellow sweater . EOS
BOS 我 很 喜 欢 我 的 黄 色 套 衫 。 EOS
translation: 我 愛 頭 髮 師 為 我 的 毛 筆 。

BOS We UNK tea from India . EOS
BOS 我 們 從 印 度 進 口 茶 葉 。 EOS
translation: 我 們 從 印 度 進 口 茶 。

BOS He seldom goes to church . EOS
BOS 他 很 少 去 教 堂 。 EOS
translation: 他 很 少 去 教 堂 。

BOS It seemed to be cheap . EOS
BOS 似 乎 很 便 宜 。 EOS
translation: 似 乎 很 便 宜 。

BOS Tom can keep a secret . EOS
BOS 汤 姆 会 保 密 。 EOS
translation: 汤 姆 可 能 是 一 个 秘 密 。

BOS Tennis is my favorite sport . EOS
BOS 网 球 是 我 最 喜 欢 的 运 动 。 EOS
translation: 网 球 是 我 最 喜 欢 的 运 动 。

BOS When is your bed time ? EOS
BOS 你 什 么 时 候 睡 觉 ？ EOS
translation: 你 什 麼 時 候 睡 覺 ?

BOS Give me something to do . EOS
BOS 给 我 点 事 做 。 EOS
translation: 給 我 些 什 麼 事 做 。

BOS Nobody gave us a chance . EOS
BOS 没 人 给 我 们 机 会 。 EOS
translation: 没 人 给 我 们 机 会 。

BOS I am six feet tall . EOS
BOS 我 六 英 尺 高 。 EOS
translation: 我 六 點 高 。

BOS Where can you get tickets ? EOS
BOS 在 哪 里 可 以 买 到 车 票 ？ EOS
translation: 你 在 哪 里 可 以 搭 到 火 車 ？

BOS These pants fit me well . EOS
BOS 我 穿 這 條 褲 子 很 合 身 。 EOS
translation: 這 些 裤 子 我 穿 起 來 很 合 身 。

BOS He showed me her picture . EOS
BOS 他 給 我 看 了 她 的 照 片 。 EOS
translation: 他 給 我 看 了 她 的 照 片 。

BOS Are you afraid of UNK ? EOS
BOS 你 怕 虫 子 吗 ？ EOS
translation: 你 怕 虫 子 吗 ？

BOS How about taking a rest ? EOS
BOS 休 息 一 下 怎 麼 樣 ? EOS
translation: 休 息 一 下 怎 麼 樣 ?

BOS I need a UNK dictionary . EOS
BOS 我 需 要 一 本 日 英 字 典 。 EOS
translation: 我 需 要 一 本 的 字 典 。

BOS He 's a bit UNK . EOS
BOS 他 有 点 活 泼 。 EOS
translation: 他 有 点 活 泼 。

BOS He 's bound to forget . EOS
BOS 他 准 会 忘 。 EOS
translation: 他 會 忘 。

BOS The well has run dry . EOS
BOS 這 口 井 乾 涸 了 。 EOS
translation: 這 口 井 乾 涸 了 。
"""