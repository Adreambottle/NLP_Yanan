# Documents of transformer and Bert

## 1. Instruction
* 因为对 huggingface 开发的 Transformer 包进行了修改，所以将修改之后的 module 重命名为 adj_tf

* 调用的预训练模型是 Bert，已进行封装，位于 `"/data/bob/Daniel_STF/trans2gp/adj_tf/models/bert/modeling_bert.py"`

* 主要调用的 class 是 `BertForTokenClassification`

* 因为添加了 Gaussian Process 的部分，对 BertForTokenClassification model 最后的输出进行了修改

* 如果使用其他预训练模型，如 albert, roberta, tinyBert, 需要按照第4部分对预训练模型进行修改

***

## 2. Bert Structure
### 2.1 Model Inheritance Relationships 模型继承关系
```
PreTrainedModel   (download 已经 pre-train 好的 model)
    ↓
BertPreTrainedModel   （采用 Bert 版本的 pre-train model）
    ↓
BertModelForTokenClassification   (具体的 NER 任务)
    ↓
NERModel  (from simpletransformer)
```
***

### 2.2. BertModelForTokenClassification 的结构
<center>
![v2-09cd3fe6a7b7de05119cf3b8a2667ad6_1440w](media/16149339338517/v2-09cd3fe6a7b7de05119cf3b8a2667ad6_1440w.jpeg)

</center>

#### 2.2.1 BertModelForTokenClassification 的结构
* 第一层是 Embedding 层
    * 包括 word_embeddings,  position_embeddings, token_type_embeddings
    * LayerNorm
    * dropout
* 第二层是 BertEncoder 层
    * 这一层包括了12层相同的结构，用的是 Bert-base-model
        * 如果是 Bert-big-model 则使用了24层
    * 每一层是一个 BertLayer 单元
        * 第一层是 **attention** 层
        * 第二层是 intermediate 层
        * 第三层是 output 层
* 第三层是 dropout 层
* 第四层是 classifier 线性分类层

```
BertForTokenClassification(
    (bert): BertModel(
        (embeddings):(
            (word_embeddings)
            (position_embeddings)
            (token_type_embeddings)
            (LayerNorm) 
            (dropout) )      
        (encoder): BertEncoder()
            (layer): ModuleList(
                (0): BertLayer()
                (1): BertLayer()
                    ...
                (11): BertLayer()
            )
        (dropout): Dropout(p=0.1, inplace=False)
        (classifier): Linear(in_features=768, out_features=15, bias=True)
    )
)
```

***



#### 2.2.2 BertAttention 层的结构

* BertAttention 层
    * 分别计算 Q, K, V 三种线性变化
    * Dropout层
* Feedforward Neural Network
    * Linear线性层
    * LayerNorm层
    * Dropout层

```
(attention): BertAttention(
    (self): BertSelfAttention(
      (query): Linear(in_features=768, out_features=768, bias=True)
      (key): Linear(in_features=768, out_features=768, bias=True)
      (value): Linear(in_features=768, out_features=768, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (output): BertSelfOutput(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
)
```

***
## 3. Transformer BertSelfAttention 的代码文档

### 3.1 Attention 的公式
$$ Attention(K, Q, V) = softmax(\frac{QK^T}{\sqrt{d_k}}) V$$
<center>![-w383](media/16149339338517/16211666066146.jpg)
</center>
***

### 3.2 K, Q, V 矩阵的定义
$$Q = W_Q*X,\quad K = W_K*X,\quad V = W_K*V$$

```python
self.query = nn.Linear(config.hidden_size, self.all_head_size)
self.key   = nn.Linear(config.hidden_size, self.all_head_size)
self.value = nn.Linear(config.hidden_size, self.all_head_size)
```

* `self.query`, `self.key`, `self.value`  储存的是 $W_K$, $W_Q$, $W_V$ 的值

* `key_layer`, `query_layer`, `value_layer` 储存的是 K, Q, V 的值
* `self.query`, `self.key`, `self.value` 三层均是线性层，input 的维度是 768， output 的维度也是 768 

* 经过`self.query`, `self.key`, `self.value` 三层输入和输出的数据维度是 `(1, 128, 768)` 即 `(batch_size, token_length, embedding_dimension)`

* `config.hidden_size = 768`, `self.all_head_size = 768`

* K, Q, V 并没有进行 MultiHead 的矩阵拆分，MultiHead 的拆分是用`self.transpose_for_scores()` 函数进行的  

* 将拆分之后的 K, Q, V 的值保存在 `key_layer`, `query_layer`, `value_layer`

* 多头处理拆分之后的 K, Q, V 的维度分别是 `(1, 12, 128, 64)` 即 `(batch_size, head_number, token_length, head_dimension)`

<center> 
![v2-70a3f25fa1b618ecd9811567daca4ec9_1440w](media/16149339338517/v2-70a3f25fa1b618ecd9811567daca4ec9_1440w.jpeg)
</center>

```
key_layer    (1, 12, 128, 64)
query_layer  (1, 12, 128, 64)
value_layer  (1, 12, 128, 64)
```

***

### 3.3 矩阵维度的意义

```
(1, 128, 768)   (batch_size, token_length, embedding_dimension)
    ↓
self.transpose_for_scores()
    ↓
(1, 12, 128, 64)  (batch_size, head_number, token_length, head_dimension)
```
<br>

| 数字 | 名称 | 代表的含义 |
| --- | --- | --- |
| 1  | batch_size | 每个 batch 中有几条数据，总是第一个维度 |
| 128 | token_length | 每条数据被打散成了多少个 tokens |
| 768 | ebedding_dimension | 每个 token embedding 有多少个维度 |
| 12 | head_number | Multihead-Attention 中 head 的数量 |
| 64 | head_dimension | 每个 Head 的维度是 |

***

### 3.4 计算 Attention Score
* attention_scores 记录的是不同 Tokens 之间的相关度，类似 Correlation Matrix

* attention_scores 的维度是 `tokens_number * tokens_number`

* transpose 转置了 Tensor 最后两个维度的数据

* `torch.matmul` 将多头处理后的64个维度的 Embedding 信息进行矩阵相乘，得到一个方阵


```python
attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
```

* attention_scores 的维度是 (1, 12, 128, 128) 

***

### 3.5 标准化处理与 Softmax 处理
* 对 attention_scores 进行标准化处理

* 用 attention_scores 比上 attention_head_size 的平方根

```python
attention_scores = attention_scores / math.sqrt(self.attention_head_size)
```

* 对经过标准化处理后的数据的最后一个维度进行 Softmax 操作，

```python
attention_probs = nn.Softmax(dim=-1)(attention_scores)
attention_probs = self.dropout(attention_probs)
```

* attention_probs 的维度是 (1, 12, 128, 128) (batch_size, head_number, token_length, token_length)

***

### 3.6 context_layer 的计算

* context_layer 的维度

```
attention_probs   *   value_layer     =    context_layer
(1, 12, 128, 128)   (1, 12, 128, 64)      (1, 12, 128, 64)
```
* `torch.matmul(attention_probs, value_layer)`
    只用最后两个维度相乘，不改变前两个维度

* context_layer 的计算部分

```python
context_layer = torch.matmul(attention_probs, value_layer)
context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
context_layer = context_layer.view(*new_context_layer_shape)
```

* 拼合后 context_layer 的维度，需要将维度和 attention 之前的维度对应上

* context_layer_shape  (1, 128, 768)

***

### 3.7 output of BertSelfAttention

* outputs 包括 context_layer 和 attention_probs
* context_layer_shape  (1, 128, 768)
* attention_probs_shape  (1, 12, 128, 128)

```python
outputs = (context_layer, attention_probs)
```
***


## 4. 进行 Gaussian Process 在 Bert 最后一层进行修改

### 4.1 BertSelfOutput 层

* 在 Attention 部分结束后是 FFN 层

* FFN 即一层 Linear 层，并进行 LayerNorm 处理和 dropout 处理 

```python
self.dense = nn.Linear(config.hidden_size, config.hidden_size)
self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
self.dropout = nn.Dropout(config.hidden_dropout_prob)
```

***
### 4.2 BertForTokenClassification 中进行的修改

#### 4.2.1 BertForTokenClassification原本的输出
```python
self.bert = BertModel(config, add_pooling_layer=False)
self.dropout = nn.Dropout(config.hidden_dropout_prob)
self.classifier = nn.Linear(config.hidden_size, config.num_labels)

def forward( ... )
        ...
    outputs = self.bert(
                input_ids, attention_mask, token_type_ids, 
                position_ids, head_mask, inputs_embeds, 
                output_attentions, output_hidden_states, return_dict)
```

* 先调用之前定义的 Bert 类，并返回
```python
self.bert()
    return BaseModelOutputWithPoolingAndCrossAttentions( ... )

    # BaseModelOutputWithPoolingAndCrossAttentions()[0] = last_hidden_state
```
* `last_hidden_state` 是最后一层的logits(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
* Sequence of hidden-states at the output of the last layer of the model.

* classifier 线性层 input 是 embedding_dimension，output 是 label_numbers，完成每个tokens在的不同label上的映射

```python 
sequence_output = outputs[0]     # 即将 last_hidden_state 提取出来
sequence_output = self.dropout(sequence_output)  # 进行 drop_out
logits = self.classifier(sequence_output)     
```

***
#### 4.2.2 BertForTokenClassification 添加的部分
* 添加 `self.is_pred` 变量，如果是 predict 的时候，会多返回一部分
* 在 `predict()` 添加最后一层的返回值

```python
def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.bert = BertModel(config, add_pooling_layer=False)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    self.is_pred = False        
              ...
outputs = self.bert()
sequence_output = outputs[0]
sequence_output = self.dropout(sequence_output)
# sequence_output 即最后一层之前的输出

logits = self.classifier(sequence_output)   
```

```python     
if not self.is_pred:
    return TokenClassifierOutput()
else:
    return (TokenClassifierOutput(),
            sequence_output)
```
* 如果是 predict 的情况，会将最后一层之前的输出也返回

