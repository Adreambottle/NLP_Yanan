# Documents of DNN2GP

## 1. Instruction
* DNN2GP 是进行 Gaussian Process 的部分

* 路径位于 `/data/bob/Daniel_STF/trans2gp/adj_stf/ner/DNN2GP.py`

* 包括五个函数 
    * `gradient(model)`
    
    * `weights(model)`
    
    * `compute_kernel(Jacobians, agg_type='diag')`
    
    * `compute_laplace(model, train_loader, prior_prec, device)`
    
    * `compute_dnn2gp_quantities(model, data_loader, device, post_prec=None, saving_path=None)`

***
## 2. DNN2GP 用到的辅助函数
### 2.1 提取所有的梯度和参数权重
* gradient 在计算 Jacobian 时用于求每个参数的一阶梯度

```python
def gradient(model):
    """
    提取所有的 gradient，flatten 成一个维度用于计算 Jacobian 矩阵
    """
    grad = torch.cat([p.grad.data.flatten() for p in model.parameters()])
    return grad.detach()
```

* weights 在计算 Jacobian 时用于将最后一层的线性层中参数提取出来

```python
def weights(model):
    """
    提取所有的参数权重，flatten 成一个维度用于计算 Jacobian 矩阵
    """
    wts = torch.cat([p.flatten() for p in model.parameters()])
    return wts
```
***
### 2.2 Gaussian Process 用到的核函数

#### 2.2.1 compute_kernel
* 作为相似度或距离的衡量 `κ(x, x‘)`

```python
def compute_kernel(Jacobians, agg_type='diag'):
    """
    用来计算相似度矩阵的
    默认选择 diag 的形式
    Compute kernel by various aggregation types based on Jacobians
    """
    if agg_type == 'diag':
        K = np.einsum('ikp,jkp->ij', Jacobians, Jacobians)  # one gp per class and then sum
    elif agg_type == 'sum':
        K = np.einsum('ikp,jlp->ij', Jacobians, Jacobians)  # sum kernel up
    elif agg_type == 'full':
        K = np.einsum('ikp,jlp->ijkl', Jacobians, Jacobians)  # full kernel NxNxKxK
    else:
        raise ValueError('agg_type not available')
    return K
```

#### 2.2.2 可选择的方式
* `diag  'ikp,jkp->ij'` 对 k, p 即第二和三两个维度进行内积运算，保留另外一个维度

* `sum   'ikp,jlp->ij'` 对 p 即第三个维度进行内积运算，保留第一个维度，加和第二个维度 

* `full  'ikp,jlp->ijkl'`  对 p 即第三个维度进行内积运算，并保留另外两个维度

* 默认选择 `diag` 的形式，不同 kernel 对预测的结果影响不大，但是 diag 有最小的复杂度


***
## 3. Gaussian Process 的过程
### 3.1 调用 `compute_laplace` 和 `compute_dnn2gp_quantities()` 

* `compute_laplace` 和 `compute_dnn2gp_quantities()`函数的调用位于 `adj_stf.ner.ner_model`

```python
model_last_layer = model.classifier

# 计算后验分布的 variance
post_prec = compute_laplace(model=model_last_layer,
                            train_loader=gp_loader,
                            prior_prec=1,
                            device=self.device)

# 将得到的后验带入，得到最后的参数分布情况
output2 = compute_dnn2gp_quantities(model=model_last_layer, 
                                    data_loader=gp_loader, 
                                    self.device, 
                                    limit=1000, 
                                    post_prec=post_prec)
（Jacobians, predictive_mean_GP, 
  predictive_var_f, predictive_noise, 
  predictive_mean) = output2
```

***

### 3.2 函数 `compute_laplace()` 和 `compute_dnn2gp_quantities()` 的定义

#### 3.2.1 `compute_laplace` 函数
* `compute_laplace` 用于计算 Laplace Process 的后验分布

* 返回的后验分布会传给 `compute_dnn2gp_quantities()` 

```python
def compute_laplace(model, train_loader, prior_prec, device):
    """
    用来计算后验概率的，返回的是后验分布 variance 的倒数 Sigma
    Compute diagonal posterior precision due to Laplace approximation
    :param model: pytorch neural network
    :param train_loader: data iterator of training set with features and labels
    :param prior_prec: prior precision scalar
    :param device: device to compute/backpropagate on (ideally GPU)
    """
                    ...
    return post_prec
```

#### 3.2.2 `compute_dnn2gp_quantities` 函数
* `compute_dnn2gp_quantities` 用于计算 GP_mean, GP_variance 等相关参数

* `compute_dnn2gp_quantities` 中计算的参数会以 nmp 的形式保存在本地

* 返回的数据会在之后的数据分析中读取 

```python
def compute_dnn2gp_quantities(model, data_loader, device, post_prec=None, saving_path=None):
    """
    用于计算 GP_variance 等相关参数，保存结果
    Compute reparameterized nn2gp quantities for softmax regression (multiclassification)
    :param model: pytorch function subclass with differentiable output
    :param data_loader: data iterator yielding tuples of features and labels
    :param device: device to do heavy compute on (saving and aggregation on CPU)
    :param post_prec: posterior precision (diagonal)
    """
```

***

### 3.3 函数 `compute_laplace()` 和 `compute_dnn2gp_quantities()` 的解释

#### 3.3.1 `compute_laplace` 函数，计算后验分布
```python
def compute_laplace(model, train_loader, prior_prec, device):
    theta_star = weights(model)
    post_prec = (torch.ones_like(theta_star) * prior_prec)

    batch_num = 0
    for data in tqdm(train_loader):
        batch_num += 1
        data = data.to(device).float()

        prediction = model.forward(data[0])
        p = torch.softmax(prediction, -1).detach()
        Lams = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)
        Jacs_in_label = list()
        for i in range(prediction.shape[0]):
            Jac = list()
            for j in range(prediction.shape[1]):
                rg = (i != (prediction.shape[0] - 1) or j != (prediction.shape[1] - 1))
                prediction[i, j].backward(retain_graph=rg)
                Jacs_in_token = gradient(model)
                Jac.append(Jacs_in_token)
                model.zero_grad()
            Jac = torch.stack(Jac).t()
            Jacs_in_label.append(Jac)
        Jacs_in_label = torch.stack(Jacs_in_label).detach()
        post_prec += torch.einsum('npj,nij,npi->p', Jacs_in_label, Lams, Jacs_in_label)
    return post_prec
```

<br>
##### 3.3.1.1 theta_star

* `theta_star = weights(model)`

* `theta_star` 是提取出的模型最后一层的参数，即已经得到的最小值的 $\theta*$

* $p(w|D) = N(w|\mu, \Sigma)$

<br>
##### 3.3.1.2 post_prec
* `post_prec = (torch.ones_like(theta_star) * prior_prec)`

* `prior_prec` 是传入参数的 scalor, 是 $\delta$ 

* `prior_prec` 是先验分布给出的权重参数，用于调节 variance 中估计项 `y_uct` 所占的权重

* $\widetilde y  = f(x) + \epsilon, \quad f(x) \sim GP(0, \delta^{-1}J_*(x)J_*(x^T))$


* 按照参数 $\theta_star$ 的shape，生成 element 全部是 $\delta$ 的矩阵

<br>
##### 3.3.1.3 用 Dataloader 对每个训练数据进行循环

```python
prediction = model.forward(data[0])
p = torch.softmax(prediction, -1).detach()
Lams = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)
```

* `prediction` 是 model `BertForTokenClassification` 之后的logits

* `p` 是 `prediction` 经过 softmax 之后处理的每个labels上的概率

* `Lams` 是 $\Lambda_w(x, y)$
    * $\Lambda_w(x, y) = \nabla^2_{ff}l(y, f)$
    
    * $\nabla^2_{ww}l(w) = J_w(x)^T\Lambda_w(x, y)J_w(x)$
    
    *  `torch.diag_embed(p)` 将 p 的值填充在指定的二维矩阵的diagonal位置
    
    *  `torch.einsum('ij,ik->ijk', p, p)` 对每个label上的概率求
    
    * 把这两部分做差: Covariance 减去 diagnal的影响
      

<br>
##### 3.3.1.4 更新 Jacobian 矩阵

```python
Jacs_in_label = list()

# i = 128 Tokens 的个数
for i in range(prediction.shape[0]):
    Jac = list()
    
    # j = 120 Lables 的个数
    for j in range(prediction.shape[1]):
        rg = (i != (prediction.shape[0] - 1) or j != (prediction.shape[1] - 1))
        prediction[i, j].backward(retain_graph=rg)
        Jacs_in_token = gradient(model)
        Jac.append(Jacs_in_token)
        model.zero_grad()
    Jac = torch.stack(Jac).t()
    Jacs_in_label.append(Jac)
Jacs_in_label = torch.stack(Jacs_in_label).detach()
```
* `Jacs_in_label` 对应的是是每个 token 对应到所有 label 的梯度向量

* `Jacs` 每个token分配到每个label上的参数的梯度

* 每次迭代用 `prediction` 中不同的元素`prediction[i, j]`进行BP，更新对应的 Jacs


<br>   
##### 3.3.1.5 计算后验分布
```python
post_prec += torch.einsum('npj,nij,npi->p', Jacs_in_label, Lams, Jacs_in_label)
```

* `post_prec` 是 $\Sigma^{-1}$

* $\widetilde y  = f(x) + \epsilon, \quad f(x) \sim GP(0, \delta^{-1}J_*(x)J_*(x^T))$


<br>

#### 3.3.2 `compute_dnn2gp_quantities` 函数，计算需要的参数

* Compute reparameterized nn2gp quantities for softmax regression (multiclassification)

* **loss function 采用的是 Logistic Loss**

* `compute_dnn2gp_quantities()` code

```python
def compute_dnn2gp_quantities(model, data_loader, device, post_prec=None, saving_path=None):

    batch_num = 0
    data_num = 0
    # 先对每个batch进行循环
    for batch in tqdm(data_loader):
        for n, data in enumerate(batch):
            Jacobians = list()
            predictive_mean_GP = list()
            predictive_var_f = list()
            predictive_noise = list()
            predictive_mean = list()
            
            theta_star = weights(model)

            data = data.to(device).float()
            prediction = model.forward(data)
            p = torch.softmax(prediction, -1).detach()
            Lams = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)
            y_uct = p - (p ** 2)

            # i = 128 Tokens 的个数
            for i in range(prediction.shape[0]):
                Jacs_in_label = list()
                kpreds = list()

                # j = 120 Lables 的个数
                for j in range(prediction.shape[1]):
                    rg = (i != (prediction.shape[0] - 1) or j != (prediction.shape[1] - 1))
                    prediction[i, j].backward(retain_graph=rg)
                    Jacs_in_token = gradient(model)

                    Jacs_in_label.append(Jacs_in_token)
                    with torch.no_grad():
                        kpreds.append(Jacs_in_token @ theta_star)
                    model.zero_grad()
                    # 这里计算完每一个矩阵的

                # 拼合每个 label 上的 Jacs 拼好
                Jacs_in_label = torch.stack(Jacs_in_label)
                jtheta_star = torch.stack(kpreds).flatten()

                # 在 tokens 的维度上append到新的列表中
                Jacobians.append(Jacs_in_label.to('cpu'))
                predictive_mean_GP.append(jtheta_star.to('cpu'))
                predictive_mean.append(p[i].to('cpu'))

                if post_prec is not None:
                    f_uct = torch.diag(Lams[i]
                                       @ torch.einsum('kp,p,mp->km', Jacs_in_label, 1/post_prec, Jacs_in_label)
                                       @ Lams[i])
                    predictive_var_f.append(f_uct.to('cpu'))
                    predictive_noise.append(y_uct[i].to('cpu'))

            # 将每条数据的都保存
            if post_prec is not None:

                predictive_mean = torch.stack(predictive_mean)
                predictive_mean_GP = torch.stack(predictive_mean_GP)
                predictive_var_f = torch.stack(predictive_var_f)
                predictive_noise = torch.stack(predictive_noise)

                np.save(saving_path + "/predictive_var_f" + "_data" + str(data_num) + ".npy", predictive_var_f.numpy())

            data_num += 1
```

##### 3.3.2.1 需要的公式

$$l(y, f_w(x)) = -log[p(y|h(f_w(x)))]$$
$$\widetilde y_* = J_*(x_*) - \lambda_*(x_*)^{-1}r_*(x_i, y_i)$$
$$\widetilde y_* = J_*(x_*) - \lambda_*(x_*)^{-1}(p_*(x_*)-y_*)$$
$$y_* = p_*(x_*) +  \lambda_*(x_*)\widetilde y_* - \lambda_*(x_*)J_*(x_*)w_*$$
$$(y_*|x_*,\widetilde D) \sim N(y_*|\sigma(f_{w_*}(x_*), \quad \lambda_*(x_*)+\lambda_*(x_*)^2·J_*(x_*)·\widetilde\Sigma ·J^T_*(x_*)\ )$$

<br>
##### 3.3.2.2 循环结构
```python
# 先对每个 gp_loader 中的 batch 进行循环
for batch in tqdm(data_loader):

    # 再对每个 batch 中的每条 data 进行循环
    # 将每条数据对应的参数都在本地保存下来
    for n, data in enumerate(batch):
    
        # 对每条 data 的所有 token 进行循环，共128个
        # 用于求 Jacobian
        for i in range(prediction.shape[0]):
        
            # 对每个 token 对应的所有 labels 进行循环
            for i in range(prediction.shape[0]): 
```

<br>
##### 3.3.2.3 需要的参数

* `Jacobians` => $J_*(x_*)$
* `predictive_mean` => $p_*(x_*)$ 
* `predictive_mean_GP` => $y_*|\sigma(f_{w_*}(x_*)$
* `predictive_var_f` => $J_*(x_*)·\widetilde\Sigma ·J^T_*(x_*)$
* `predictive_noise` => $\lambda_*(x)$

<br>
##### 3.3.2.4 参数对应的代码
```
predictive_mean_GP.append(jtheta_star)
predictive_mean.append(p[i])
predictive_var_f.append(f_uct)
predictive_noise.append(y_uct[i])
labels.append(label)
```

<br>
##### 3.3.2.5 计算Jacobian之前的步骤

* `theta_star = weights(model)`
    * `theta_star` 是提取出的模型最后一层的参数，即已经得到的最小值的 $\theta*$
    * $p(w|D) = N(w|\mu, \Sigma)$


* `post_prec` 的 variance 是通过 `compute_laplace` 计算出的 $\widetilde \Sigma$ 

    * $p(w|\widetilde D) = N(w|\mu,\widetilde \Sigma)$ 后验分布是符合这样的正态分布的。

*  `Lams = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)`

<br>
##### 3.3.2.6 计算 Jacobian 
```python
Jacobians = list()
# i = 128 Tokens 的个数
for i in range(prediction.shape[0]):
    Jacs_in_label = list()
    kpreds = list()

    # j = 120 Lables 的个数
    for j in range(prediction.shape[1]):
        rg = (i != (prediction.shape[0] - 1) or j != (prediction.shape[1] - 1))
        prediction[i, j].backward(retain_graph=rg)
        Jacs_in_token = gradient(model)

        Jacs_in_label.append(Jacs_in_token)
        with torch.no_grad():
            kpreds.append(Jacs_in_token @ theta_star)
        model.zero_grad()
        # 这里计算完每一个矩阵的

    # 拼合每个 label 上的 Jacs 拼好
    Jacs_in_label = torch.stack(Jacs_in_label)
    jtheta_star = torch.stack(kpreds).flatten()

    # 在 tokens 的维度上append到新的列表中
    Jacobians.append(Jacs_in_label.to('cpu'))
    predictive_mean_GP.append(jtheta_star.to('cpu'))
    predictive_mean.append(p[i].to('cpu'))
```

* `Jacobians` 储存的是每条 data 中所有 tokens 对应所有 labels 的 Jacobians

* `kpreds.append(Jacs_in_token @ theta_star)` 
    * 相当于是 $J_*(X_i)w_*$ 
    * 也是 `jtheta_star` flatten
    * 也是 `predictive_mean_GP` 的 item

* `Jacs_in_label` 储存的是每一个 tokens 对应的所有 labels 的 Jacobians

* `Jacs_in_tokens` 储存的是每一个 tokens 对应的所有 labels 的 Jacobians
    * `Jacs_in_token = gradient(model)`
    * `Jacs_in_token` 提取了model对应的所有的梯度

* 每次迭代用 `prediction` 中不同的元素`prediction[i, j]`进行BP，更新对应的 Jacs

<br>
##### 3.3.2.7 基于 Jacobian 的参数的部分

* `predictive_mean_GP` => $y_*|\sigma(f_{w_*}(x_*)$

* `predictive_mean` => 模型预测出的 softmax 之后的概率 => `p` => $p_*(x_*)$


* `predictive_var_f` => `f_uct` => GP的预测值$(y_*|x_*,\widetilde D)$的variance的第二部分

* `predictive_noise` => `y_uct` => $p(1-p)$ => $\lambda_*(x)$ => GP的预测值$(y_*|x_*,\widetilde D)$的variance的第一部分

    * `f_uct = torch.diag(Lams[i] @ torch.einsum('kp,p,mp->km', Jacs, 1/post_prec, Jacs) @ Lams[i])`
    
    * `torch.einsum('kp,p,mp->km', Jacs, 1/post_prec, Jacs)` 相当于是 $J_w(x)· \widetilde \Sigma ·J_w^T(x)$，
    
    * `y_uct = p - (p ** 2)`

<br>
##### 3.3.2.8 保存数据
* 数据保存在本地，会在数据分析的时候提取

```python
predictive_var_f = torch.stack(predictive_var_f)
np.save(saving_path + "/predictive_var_f" + "_data" + str(data_num) + ".npy", predictive_var_f.numpy())
```

***

## 4. 数据的维度

### 4.1 数据的维度
| 名称 | 数字 | 解释 |
| --- | --- | --- |
| batch_size | 8 | 每个 batch 里面有几条预测数据，来源于 gp_loader |
| label_number | 120 | 可供预测的 label 的个数，来源于 to_pred, 用户给出 |
| token_number  | 128 | 每条数据被打散成 128 个tokens |
| embedding_dimension  | 768 | 每个tokens对应的embedding的维度，BertTokenizer 是 768 |

***

### 4.2 作为Outputs的参数

#### 4.2.1 最后一层的线性层的模型参数的个数
* weight
* bias

```
weight_shape : torch.Size([120, 768])
bias_shape : torch.Size([120])
```

***

#### 4.2.2 数据流的维度
```
before_last_layer_total shape : torch.Size([8, 128, 768])
after_last_layer_total shape : torch.Size([8, 128, 120])
```

***

#### 4.2.3 GP输出的维度

```
Jacobians shape : torch.Size([768, 120, 9997])

post_prec shape : torch.Size([92160])   92160 = 768 * 120

predictive_mean_GP shape : torch.Size([768, 120])

predictive_var_f shape : torch.Size([768, 120])

predictive_noise shape : torch.Size([768, 120])
```
