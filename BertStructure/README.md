## 1. Instruction
### 1.1 Background
* 因为对 SimpleTransformer 包进行了修改

* 调用的文件是 `ner_model`位于 `"./ner/ner_model.py"`

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