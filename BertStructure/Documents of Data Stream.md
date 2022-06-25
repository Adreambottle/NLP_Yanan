# Documents of Data Stream

## 1. 主函数需要定义的参数

* `MODEL_PATH = "./models/04-23_bert-base-cased/"`  => 模型存放的地址

* `PRED_DATA_PATH = '/data/bob/synthetic/2021-04-22_4m/ 1m_synthetic_cap.json'` => 预测数据存放的地址

* `PRED_DATA_LEN = 1` => 预测数据的长度，最好一次预测一条

* `VAR_NUM = 5` => 每条数据需要找到有最大不确定度标签的个数


```python
def main(MODEL_PATH, PRED_DATA_PATH, PRED_DATA_LEN, VAR_NUM):
    """
    :param MODEL_PATH: The path of the model
    :param PRED_DATA_PATH: The path of the data to predict
    :param PRED_DATA_LEN: The number of the data to predict
    :param VAR_NUM: The number of the labels of the highest variance 
    """
                ...
    preds, model_outputs, variances, token_id_dict = model.predict(input_data, gp_batch_size=4)
                ...
```

## 2. 重要的变量，即`predict()`的返回值

* `preds` => 对预测数据预测的结果

* `model_outputs` => 预测结果的 logits   修改变量名

* `variances` => 预测结果在多个标签上的 variance 

* `token_id_dict` => 储存了每条预测数据的 first token 的位置

* `top_k_label` => 储存了每个 word 有最大 variance 的 k 个 labels


***
## 3. 主函数调用的函数

```python
from adj_stf.ner.ner_model_direct import NERModel, NERArgs
```
* 主要调用函数储存在 `"adj_stf/ner/ner_model_direct.py"` 文件中的 `NERModel` 类中

* `ner_model_direct` 的说明详见 *_Documents of SimpleTransformer_*

* 在预测的时候用到的是 `predict()` 成员函数
```python
(preds, 
     model_outputs, 
     variances, 
     token_id_dict) = model.predict(input_data, post_prec=1, gp_batch_size=1)
```

    * `post_prec=1` => 是  $\delta$ 的值，默认为1
        
    * `gp_batch_size=1` => 是 `GP_Loader` 的 batch_size，默认为1

***

## 4. Predict 和 Gaussian 的过程
* 调用的函数来源于`"adj_stf/ner/DNN2GP_direct.py"`
    * `DNN2GP` 的说明详见 *_Documents of DNN2GP_*
    
    * `compute_laplace`用于计算后验分布的 covariance matrix
    
    * `compute_dnn2gp_quantities` 用于计算不同 tokens 在 GP 之后的 variance

* 在 `NERModel.predict()` 中进行 GP 的部分

```python
from adj_stf.ner.DNN2GP_direct import (compute_dnn2gp_quantities,
                                       compute_laplace,
                                       token_length_and_first_token_id)

Class NERModel():
    ...
    def predict():
        ...
        
        """GP的过程，可用于修改"""  
        token_id_dict = token_length_and_first_token_id(out_label_ids, out_input_ids)
        before_last_layer_total = before_last_layer_total.clone().detach()
        gp_loader = DataLoader(before_last_layer_total, batch_size=gp_batch_size, shuffle=True)
        model_last_layer = model.classifier
        
        post_prec = compute_laplace(model=model_last_layer,
                                    train_loader=gp_loader,
                                    prior_prec=prior_prec,
                                    device=self.device)
        
        variances = compute_dnn2gp_quantities(model=model_last_layer,
                                  data_loader=gp_loader,
                                  device=self.device,
                                  post_prec=post_prec)
        
        return preds, model_outputs, variances, token_id_dict
```