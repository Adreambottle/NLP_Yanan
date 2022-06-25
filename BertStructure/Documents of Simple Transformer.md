# Documents of SimpleTransformer

## 1. Instruction
### 1.1 Background
* 因为对 SimpleTransformer 包进行了修改，所以将修改之后的 module 重命名为 adj_stf

* 调用的文件是 `ner_model`位于 `"/data/bob/Daniel_STF/trans2gp/adj_stf/ner/ner_model.py"`

* 主要调用的 class 是 `class NERModel(model)`

* 因为添加了 Gaussian Process 的部分，对 `NERModel` 进行了修改

***

### 1.2 调用说明
* `NERModel` 对 `BertModelForTokenClassification`进行了更深的封装，但是核心仍是 `BertModelForTokenClassification`

* `NERModel` 会调用 `DNN2GP` 中的函数进行 Gaussian Process 的运算，将结果储存在本地

***

### 1.3 NERModel 的模型继承关系
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

### 1.4 NERModel 的模型结构
* 数据读写，模型加载相关函数
* `train()`
* `predict()`
* 拼接 tokens 成 words

***

## 2. 数据读写，模型加载

### 2.1 可以使用的预训练模型
* "albert",
* "auto",
* "bert",
* "bertweet",
* "camembert",
* "distilbert",
* "electra",
* "layoutlm",
* "longformer",
* "mobilebert",
* "mpnet",
* "roberta",
* "squeezebert",
* "xlmroberta",
* "xlnet",

### 2.2 读取数据，调用相关函数，进行数据封装
```python
from adj_stf.ner.ner_utils import (InputExample,
                                   convert_examples_to_features,
                                   get_examples_from_df,
                                   get_labels,
                                   read_examples_from_file,
                                   )
```

***

### 2.3 读取数据，将数据变成 `InputExample` 的格式
#### 2.3.1 Contents
* `InputExample` 有三个子变量 `guid`, `words`, `labels`

*  `guid` 代表的是 data 也就是 sentence 的 index

*  `words` 是每条 data 里面的每个 feature 需要打上标签

*  `labels` 是个 feature 对应的 label

* 如果是 `predict()` 则 label 均为 -1

#### 2.3.2 Example

| tokens_index | sentence_id |   words   |              labels    |
|--------|----------|--------------------|------------------------|
| 0      |       0  |              2.2M  |              Resistance|
| 1      |       0  |              1/2W  |  RatedPowerDissipationP|
| 2      |       0  |                5%  |               Tolerance|
| 3      |       0  |THROUGH HOLE MOUNT  |         MountingFeature|
| 4      |       1  |               32M  |              Resistance|
| 5      |       1  |              0.83  |               Tolerance|
| 6      |       1  |              1206  |                SizeCode|
| 7      |       1  |              -55℃  | OperatingTemperatureMin|
***

### 2.4 将每条数据打散成 Tokens

* 因为数据中包含的特殊单词太多，矩阵会变得很大。需要将单词变成 Tokens

* 调用 `convert_example_to_feature` 函数
  
    * 将 word 即 features, 按照 Bert 的 vocabulary 打散成 word piece，也就是 Tokens
    
    * vocabulary 的位置在 pre-trained model 里可以找到，在这里用的是 `"/data/bob/ner/outputs/04-23 bert-base-cased/vocab.txt"`

    * `convert_example_to_feature`函数将也会每一个 example 封装成 `tensor` 的形式，作为Transformer的输入。

***

### 2.5 model之前的输入，即 train_dataset

* 按照 pre-trained 已保存的词表将 words/description 拆分成 tokens；

* 再将 `tokens` 在词表中的位置映射成一个`input_ids`向量；

* 将每个的 word 中的**第一个**小的 `token` 作为有用的 `label_ids`，其他的都标注为`-100`，在计算loss的时候会忽略标注为 `-100` 的token；

* `load_and_cache_examples()` 函数将数据进一步封装

* 最终作为train_dataset 或 eval_dataset 放入 model 的 DataLoader 中的TensorDataset 包括：
    * input_ids
    * input_mask
    * segment_ids
    * label_ids

***

```python
def load_and_cache_examples()
    # features 里面有 input_ids, input_mask, segment_ids, label_ids
         ...
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        ...
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset
```

***
### 2.6 Example

```
example_words = ['THROUGH HOLE MOUNT',
                 '4037n',
                 '125℃',
                 '3.5kV',
                 '02012F152Z7BP0D']

example_labels = ['MountingFeature',
                  'Capacitance',
                  'OperatingTemperatureMax',
                  'RatedDCVoltageURdc',
                  'MfrPartNumber']

tokens = ['[CLS]',
          'T', '##H', '##RO', '##U', '##G', '##H', 'H', '##OL', '##E', 'M', '##O', '##UN', '##T',
          '40', '##37', '##n',
          '[UNK]',
          '3', '.', '5', '##k', '##V',
          '02', '##01', '##2', '##F', '##15', '##2', '##Z', '##7', '##B', '##P', '##0', '##D',
          '[SEP]']

input_ids = [101,
             157, 3048, 21564, 2591, 2349, 3048, 145, 13901, 2036, 150, 2346, 27370, 1942,
             1969, 26303, 1179,
             100,
             124, 119, 126, 1377, 2559,
             5507, 24400, 1477, 2271, 16337, 1477, 5301, 1559, 2064, 2101, 1568, 2137,
             102, 0, 0, 0, ... , 0]
```

***

## 3. train

## 4. predict
### 4.1 Predict data 的封装

* 在预测 predict() 的过程中 `model.is_pred = True` 被激活

* 则 model 会多返回最后一层之前，数据经过 Bert 层返回的 logits

* 将最后一层之前的输出的 logits 和最后一层的参数进行 Gaussian Process

* predict 和 train 部分的 数据封装是一样的

* 用 `InputExample` 对预测数据进行封装。因为 `labels` 均为 -1

```python
# 先将 predict data 打包成 InputExample 
# 有三个子变量 `guid`, `words`, `labels`
predict_examples = [
    InputExample(i, sentence.split(), [self.args.labels_list[0] for word in sentence.split()])
    for i, sentence in enumerate(to_predict)
]
```
***

### 4.2 在 eval_dataloader 中提取最后一层之前的logits

* 新建 data_loader 将 predict数据打包放入

* 对每个 batch 进行循环，将最后一层之前的输出拼在一起

```python
eval_dataloader = DataLoader(eval_dataset,
                             sampler=eval_sampler,
                             batch_size=args.eval_batch_size)
            ···
for batch in tqdm(eval_dataloader): 
    outputs, before_last_layer_logits = model(**inputs)
    before_last_layer_total = torch.cat((before_last_layer_total, before_last_layer_logits), 0)
```

***

### 4.3 进行预测
* 对最后返回的 outputs 中的 logits 作为 preds

* 选出 `preds` 中每个token对应最大value的label作为该token的label

* 将 `preds` 中的内容本地保存，在数据分析的时候回用到

```python
preds = np.argmax(preds, axis=2)

saving_path = './savings'
np.save(saving_path + '/preds.npy', preds)
np.save(saving_path + '/out_label_ids.npy', out_label_ids)
np.save(saving_path + '/out_input_ids.npy', out_input_ids)
```

***

### 4.4 准备 Gaussian Process 的数据

* 最后一层之前的 logits 数据是 `Tensor.float` 格式，仍在计算图中

* 使用 `Tensor.clone().detach()` 使其脱离计算图

* 新建 GP 过程中的DataLoader，将最后一层之前的 logits 数据重新打包成 batch

* 将 `BertModelForTokenClassification` 中最后一层提取出来，

```python
# 提取最后一层之前的数据
before_last_layer_total = before_last_layer_total.clone().detach()

# 定义新的loader
gp_loader = DataLoader(before_last_layer_total, batch_size=gp_batch_size, shuffle=True)

# 保存最后一层
model_last_layer_path = "./savings/last_layer.pt"
model_last_layer = model.classifier
torch.save(model_last_layer, model_last_layer_path)
```

***

### 4.5 进行 Gaussian Process

* 需要关注的每个 token 关于每个 label 的 variance 会本地保存

* 在 DNN2GP 的代码中有更详细的说明

```python
post_prec = compute_laplace(model=model_last_layer,
                        train_loader=gp_loader,
                        prior_prec=1,
                        device=self.device)

compute_dnn2gp_quantities_seperate(model_last_layer,
                                  gp_loader,
                                  self.device,
                                  limit=1000,
                                  post_prec=post_prec,
                                 saving_path="./results")

```

***

## 5. 数据的拼接

### 5.1 数据拼接的代码
#### 5.1.1 设置 label 和 label_index 的对应字典
```python
label_map = {i: label for i, label in enumerate(self.args.labels_list)}
```


* 如果 `out_label_ids[i, j] != pad_token_label_id` 的情况下，将 preds_list 中将 token 对应的 label_index 转换成 

```python
out_label_list = [[] for _ in range(out_label_ids.shape[0])]
preds_list = [[] for _ in range(out_label_ids.shape[0])]

for i in range(out_label_ids.shape[0]):
    for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != pad_token_label_id:
            out_label_list[i].append(label_map[out_label_ids[i][j]])
            preds_list[i].append(label_map[preds[i][j]])
```

***

#### 5.1.2 将 Tokens 转换成其在每个 label 上的 logits
``` python
preds = [
    [{word: preds_list[i][j]} for j, word in enumerate(sentence.split()[: len(preds_list[i])])]
    for i, sentence in enumerate(to_predict)
]

word_tokens = []
for n, sentence in enumerate(to_predict):
    w_log = self._convert_tokens_to_word_logits(
        out_input_ids[n], out_label_ids[n], out_attention_mask[n], token_logits[n],
    )
    word_tokens.append(w_log)
```

***

#### 5.1.3 word_tokens 对应的嵌套结构

| 数据 | 嵌套的结构 | 可取的范围 | 代表的含义 |
| --- | --- | --- | --- |
| 最外面一层是几条 predict data |       word_tokens        |    [0~4] | 5 条 predict data |
| 第二层是该 predict data 包含几个单词 |      word_tokens[0]    |     [0~7]  |   第 0 条 predict data 有 8 个单词 |
| 第三层表示的是该单词能拆成几个tokens  |     word_tokens[0][0]   |   [0~3] | "resistors" 拆成三个 tokens ["resist##", "##or", "##s"]|
| 第四层表示每个 tokens 预测到每个 label 上的概率  | word_tokens[0][0][0])  |[0~120]| "resist##" tokens 在 120 个 label 上的概率 |

***


### 5.2 数据拼接的例子

#### 5.2.1 Original predict data

```
['Resistors 
  155℃ 
  P11P4F0GGSA20104MA 
  6669O 
  5% 
  -400,400ppm/℃ 
  1W 
  2512']
```

***

#### 5.2.2 Predict data before the model
```
   words              pred_label                 label_id
{'Resistors':          'others'},                    38
{'155℃':               'others'},                    38 
{'P11P4F0GGSA20104MA': 'others'},                    38
{'6669O':              'Resistance'},                59    
{'5%':                 'Tolerance'},                 36   
{'-400,400ppm/℃':      'others'},                    38
{'1W':                 'RatedPowerDissipationP'},    70
{'2512':               'SizeCode'}                   34
```

***

#### 5.2.3 Token ids for each token
```
   words                      token_id
'Resistors'             11336, 22398,  3864,   
'155℃'                    100,   
'P11P4F0GGSA20104MA'      153, 14541,  2101,  1527,  2271, 1568,  2349, 13472,  1592, 10973, 10424,  1527,  8271,  
'6669O'                  5046,  1545,  1580,  2346,   
'5%'                      126,   110,   
'-400,400ppm/℃'           118,  3434,   117,  3434,  8661,  1306, 120,   100,   
'1W'                      122,  2924, 
'2512'                  25101,  1477, 
```

***

#### 5.2.4 Predicted  label id for each token

```
   words               pred_label_id                        tokens_num    
'Resistors'           38 38 38                                   3
'155℃'                38                                         1
'P11P4F0GGSA20104MA'  38 38 38 38 38 38 38 38 38 38 38 38 38     13
'6669O'               59 38 38 38                                4
'5%'                  36 38                                      2
'-400,400ppm/℃'       38 38 38 77 38 38 38 38                    8
'1W'                  70 70                                      2
'2512'                34 38                                      2
```
