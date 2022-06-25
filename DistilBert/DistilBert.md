# DistilBert 

## 1. Background

### 1.1. BERT model

<div align=center><img src="./plots/image.png" width="400"></div>

BERT (Bidirectional Encoder Representations from Transformers) 

#### 1.1.1. train task

* Only predict the masked words rather than reconstructing the entire input 
* Masked Language Model (MLM)  and next sentence prediction 

#### 1.1.2. Architecture

<div align=center><img src="./plots/image (1).png" width="400"></div>

* BERT’s model architecture is a multi-layer bidirectional Transformer encoder
* Layer number (Transformer blocks): `L` (Base 12, Large 24)
* Hidden size: `H` (Base 768, Large 1024)
* Number of self-attention heads: `A `(Base 12, Large 16)

#### 1.1.3. Input Representation

* Use WordPiece embeddings  (Wu, 2016)， 30,000 token
* Two methods of processing sentence pairs into one sentence: separated by the special token `[SEP]`; and by a new Embedding Mark which sentence each token belongs to
* Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways.
* The input embeddings are the sum of `token embeddings`, `segmentation embeddings`, and `position embeddings`.

#### 1.1.4. Mask Mechanism

* A downside is that they are creating a mismatch between pre-training and fine-tuning since the `MASK]` oken does not appear during fine-tuning. 
* To mitigate this, they do not always replace “masked” words with the actual `[MASK]` oken. The training data generator chooses 15% of the token positions at random for prediction. If the i-th token is chosen, we replace the i-th token with 

  * the `[MASK]` token 80% of the time 
  * a random token 10% of the time 
  * the unchanged i-th token 10% of the time. 

#### 1.1.5. NSP
* When choosing the sentences A and B for each pre-training example, 
	* 50% part of  B is the actual next sentence that follows A  => (labeled as `IsNext`), 
	* 50% part is a random sentence from the corpus  => (labeled as `NotNext`). 

#### 1.1.6 Output

* Then, Ti (output of hidden layer) will be used to predict the original token with cross entropy loss.

* The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. 

```
Input = [CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]
Label = IsNext
Input = [CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label = NotNext
```

#### 1.1.7 Fine-tune 

* The more classic tasks, including text classification, QA, and NER, can be done through a unified framework. By pretraining a unified large model, it can be used for various downstream tasks.

* Bert and its extensions achieve exciting performance(SOTA)
	* Text Classification
	* Question answering, Machine Reading Comprehension
	* Named Entity Recognition
	* Machine translation

#### 1.1.7 Parameter Numbers

* BERT-Base => 110M
* BERT-Large => 340M

***

### 1.2 Distillation

#### 1.2.1 Introduction

* Hinton et al.(2015) first proposed the concept of knowledge distillation in the article "Distilling the Knowledge in a Neural Network". 
* The core idea is to first train a complex network model, and then use the output of this complex network and the real label of the data to train a more Small network
* The knowledge distillation framework usually contains a complex model (called the Teacher model) and a small model (called the Student model).

<div align=center><img src="./plots/image (2).png" width="400"></div>


#### 1.2.2 Why use distillation
* Improve the accuracy of the model Imporve the Accuracy of the model
* Reduce model delay and compress network parameters
* Domain transfer between labels



#### 1.2.3 Different methods of Model compress

Model compression can be roughly divided into 5 types:

* **Network Pruning** 
	* weight pruning (Hard to implement, hard to speed up)
	* Neuron pruning
* **Knowledge Distillation** 
  * Provide a small netwrok without too much loss on accuracy comparing with original network
* **Parameter Quantization** 
  * Using less bits to represent a value 
  * Weight clustering
<div align=center><img src="./plots/image (3).png" width="400"></div>
  * Represent frequent clusters by less bits, represent rare clusters by more bits	

* * **Parameter matrix approximation**
	* The purpose of reducing matrix parameters is achieved by low-rank decomposition of the matrix or other methods* 
<div align=center><img src="./plots/image (4).png" width="400"></div>

* * **Weight sharing**
	* By sharing parameters, the network parameters can be reduced,
	* For example, Albert shares the transformer layer;

#### 1.2.4 Distillation in NLP 

* While large-scale pre-trained language models lead to significant improvement, they often have several hundred million parameters.
* The problem is the growing computational and memory requirements of these models may hamper wide adoption.
* Fortunately, it is possible to reach similar performances on many downstream-tasks using much smaller language models pre-trained with knowledge distillation, resulting in models that are lighter and faster at inference time, while also requiring a smaller computational training budget.

### 1.3 Distilled models 

* DistilBert
* MobileBert
* TinyBert
* Distilled BiLSTM

#### 1.3.1 Distilled BiLSTM

* Distilling Task-Specific Knowledge from BERT into Simple Neural Networks: https://arxiv.org/abs/1903.12136

<div align=center><img src="./plots/image (5).png" width="500"></div>


* The author proposed to distill knowledge from BERT into a single-layer BiLSTM. Across multiple datasets in paraphrasing, natural language inference, and sentiment classification, model achieve comparable results with ELMo, while using roughly 100 times fewer parameters and 15 times less inference time.

* For BERT, they use the large variant BERTLARGE (described below) as the teacher network, starting with the pretrained weights and following the original, task-specific fine-tuning procedure (Devlin et al., 2018)

##### 1.3.1.1 Data Augmentation for Distillation

The author proposed a novel, rule-based textual data augmentation approach for constructing the knowledge transfer set.

* **Masking.**  
	
	With probability $p_{mask} $ , they randomly replace a word with `[MASK]`  
e.g. `"I loved the comedy"`  => `"I [MASK] the comedy`
	
* **POS-guided word replacement.**  
  With probability $p_{cos}$, they replace a word with another of the same `POS` tag.  
  e.g. `"What do pigs eat?"` => `"How do pigs eat?"`

* **n-gram sampling.**  
  With probability $ P_{ng} $, they randomly sample an n-gram from the example, where n is randomly selected from {1, 2, . . . , 5}. Then mask them together.

* Fix  $P_{mask} = P_{pos} = 0.1$ and $P_{ng} = 0.25$ across all datasets.



#### 1.3.2 BERT-PKD 

* Patient Knowledge Distillation for BERT Model Compression: https://arxiv.org/abs/1908.09355

<div align=center><img src="./plots/image (6).png" width="500"></div>

* Loss:

<div align=center><img src="./plots/image (7).png" width="250"></div>
<div align=center><img src="./plots/image (8).png" width="250"></div>
<div align=center><img src="./plots/image (9).png" width="250"></div>

* The additional training loss introduced by the patient teacher is defined as the mean-square loss between the normalized hidden states:

<div align=center><img src="./plots/image (10).png" width="250"></div>

* Different from previous studies, BERT-PKD proposes Patient Knowledge Distillation, which extracts knowledge from the middle layer of the teacher model to avoid the phenomenon of overfitting when distilling the last layer.

* One hypothesis is that overfitting during knowledge distillation may lead to poor generalization. To mitigate this issue, instead of forcing the student to learn only from the logits of the last layer, authors propose a "patient" teacher-student mechanism to distill knowledge from the teacher's inter-mediate layers as well.

* For the distillation of the intermediate layer, the author uses the normalized MSE, called PT loss.

* The teacher model adopts a fine-tuned `BERT-base`, and the student model has a 6-layer and a 3-layer.

* In order to initialize a better student model, the author proposes two strategies, 
	* One is PKD-skip, which uses the `[2, 4, 6, 8, 10]` layers of `BERT-base`, 
	* The other is PKD-last , using layers `[7, 8, 9, 10, 11]` layers of `BERT-base` (slightly better (<0.01)). 

***

## 2. DistilBert Model

<div align=center><img src="./plots/image (11).png" width="300"></div>



* Developed by *Victor SANH, Lysandre DEBUT, Julien CHAUMOND, Thomas WOLF,* from HuggingFace, [**DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**](https://arxiv.org/pdf/1910.01108.pdf). 
* Due to the large size of BERT, it is difficult for it to put it into production. Suppose we want to use these models on mobile phones, so we require a less weight yet efficient model, that’s when DistilBERT comes into the picture. 
* Distil-BERT has 97% of BERT’s performance while being trained on half of the parameters of BERT. 


### 2.1 Model Structure
#### 2.1.1 MLM mechanism of DistilBert 
##### 2.1.1.1 Static mask

* When inputting, randomly cover or replace any word or word in a sentence, and then let the model predict which part is covered or replaced through contextual understanding, and then only calculate the tokens of the covered part when doing prediction.

* Randomly replace 15% of the tokens in a sentence with the following:
	* These tokens have an 80% chance of being replaced with [Mask];
	* There is a 10% chance of being replaced by any other;
	* There is a 10% chance of being left intact.

```python
# REAL_TOKENS: The real tokens 
_token_ids_real = token_ids[pred_mask]
# RANDOM_TOKENS: select n_tgt ids from vocab_size
_token_ids_rand = _token_ids_real.clone().random_(self.vocab_size)
# MASK_TOKENS: n_tgt number of mask id as [MASK]
_token_ids_mask = _token_ids_real.clone().fill_(
    self.params.special_tok_ids["mask_token"])

# pred_probs = torch.FloatTensor([0.8, 0.1, 0.1])
# Random select with the prob as pred_probs
probs = torch.multinomial(self.pred_probs, len(_token_ids_real), replacement=True)

# process the MASK Tokens with the prob of 80%，10%，10% 
# The final prob is: mask 15%*80%，real 15%*10%，random 15%*10%
_token_ids = (
    _token_ids_mask * (probs == 0).long()
    + _token_ids_real * (probs == 1).long()
    + _token_ids_rand * (probs == 2).long()
)
# using _token_ids take the place of original token ids
# (batch_size, seq_length)
token_ids = token_ids.masked_scatter(pred_mask, _token_ids)
```

##### 2.1.1.2 More attention to low-frequency words

Token_probs is used in the mask to make the selection of mask pay more attention to low-frequency words, so as to achieve smooth sampling of the mask (if sampling is evenly distributed, most of the masks obtained may be repeated high-frequency words).

```python
bs, max_seq_len = token_ids.size()
# copy token_ids
mlm_labels = token_ids.new(token_ids.size()).copy_(token_ids)

# The prob of each tokens 
x_prob = self.token_probs[token_ids.flatten()]

# mlm_mask_prop = 0.15, the prob of mask words is 15%
n_tgt = math.ceil(self.mlm_mask_prop * lengths.sum().item())

# Sample n_tgt words，with the prob of each token as x_prob, without replacement, return the ids of samples
tgt_ids = torch.multinomial(x_prob / x_prob.sum(), n_tgt, replacement=False)
```

#### 2.1.2 DistilBert Model Stcture

* On the basis of the 12-layer Transformer-encoder, one layer is removed from every two layers, and finally, the 12 layers are reduced to 6 layers.
* Removed the token type embedding and pooler.
* Use the soft target of the teacher model and the hidden layer parameters of the teacher model to train the student model.

<div align=center><img src="./plots/image (12).png" width="500"></div>


##### 2.1.2.1 Total Model of the DistilBertModel
```
(distilbert): DistilBertModel(
  
  (embeddings): Embeddings(
    (word_embeddings): Embedding(30522, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (LayerNorm): LayerNorm((768,)
    (dropout): Dropout(p=0.1, inplace=False)
  )

  (transformer): Transformer(
    (layer): ModuleList(
      (0): TransformerBlock(
        (attention): MultiHeadSelfAttention()
        (sa_layer_norm): LayerNorm((768,)
        (ffn): FFN( (dropout) (lin1) (lin2) )
        (output_layer_norm): LayerNorm((768,)
      )
      (1): TransformerBlock()
      (2): TransformerBlock()
      (3): TransformerBlock()
      (4): TransformerBlock()
      (5): TransformerBlock()     
    )
  )
)
```

##### 2.1.2.2 Detail Model Part of the Transformer part

```
(transformer): Transformer(
  (layer): ModuleList(
    (0): TransformerBlock(
      (attention): MultiHeadSelfAttention(
        (dropout): Dropout(p=0.1, inplace=False)
        (q_lin): Linear(in=768, out=768, bias=True)
        (k_lin): Linear(in=768, out=768, bias=True)
        (v_lin): Linear(in=768, out=768, bias=True)
        (out_lin): Linear(in=768, out=768, bias=True)
      )
      (sa_layer_norm): LayerNorm((768,)
      (ffn): FFN(
        (dropout): Dropout(p=0.1, inplace=False)
        (lin1): Linear(in=768, out=3072, bias=True)
        (lin2): Linear(in=3072, out=768, bias=True)
      )
      (output_layer_norm): LayerNorm((768,)
    )
        """  Repeat TransformerBlock 6 times   """
  )
)
```

### 2.2 Distiller

#### 2.2.1 Adjustment of the activate function
##### 2.2.1.1 Traditional softmax

* The neural network uses the softmax layer to convert logits to probabilities. The original softmax function:

$$ p_i = \frac{exp(z_i)}{\sum_{j}exp(z_j) } $$

<div align=center><img src="./plots/image (13).png" width="300"></div>

* But directly using the output value of the softmax layer as the soft target, this will bring another problem:

* When the entropy of the probability distribution output by softmax is relatively small, the value of the negative label is very close to 0, and the contribution to the loss function is very small, so small that it can be ignored. So the variable "temperature" comes in handy.

##### 2.2.1.2 Softmax-Temperature:

* The following formula is the softmax function after adding the temperature variable:

$$p_i = \frac{exp(z_i/T)}{\sum_{j}exp(z_j/T) } $$

<div align=center><img src="./plots/image (14).png" width="300"></div>

* Where qi is the probability of each category output, zi is the logits output of each category, and T is the temperature. When the temperature T = 1, this is the standard Softmax formula. 

* The higher the T, the smoother the output probability distribution of softmax, and the larger the entropy of the distribution, the information carried by the negative label will be relatively amplified, and the model training will pay more attention to the negative label.

#### 2.2.2 Loss 

The final loss function is a linear combination of Lce and masked language modeling loss Lmlm. In addition, the author found that adding cosine embedding loss (Lcos) is beneficial to make the direction of the hidden state vector of students and teachers consistent.

* Lce：The loss of the logits of the teacher model and the student model. Using T as the temperature adjustment in the softmax. Using Kullback-Leibler divergence as the loss function $ L_{ce} = \sum_{i}t_i * log(s_i) \quad $ 
	* t_i and s_i means the logits of the teacher model and student model
* Lmlm：the loss of the BERT-MLM task. Using cross-entropy loss function
* Lcos：the loss of the output vector of the teacher's model and the student model. Using cosine loss.

##### 2.2.2.1 Loss_ce

```python
# logits hidden_states(student&teacher) (batch_size, seq_lenth, vocab_size)
s_logits, s_hidden_states = student(input_ids=input_ids, attention_mask=attention_mask)
t_logits, t_hidden_states = teacher(input_ids=input_ids, attention_mask=attention_mask)

# choose masked logits (n_tgt, vocab_size)
s_logits_slct = torch.masked_select(s_logits, mask)
s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))
t_logits_slct = torch.masked_select(t_logits, mask)
t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))

# temperature = 2.0，
# ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
# define the loss fucntion as the KL diversity loss
loss_ce = self.ce_loss_fct(
    F.log_softmax(s_logits_slct / self.temperature, dim=-1),
    F.softmax(t_logits_slct / self.temperature, dim=-1)) * (self.temperature) ** 2
```
    
##### 2.2.2.2 Loss_mlm
```python
# lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
# (batch_size*seq_length, vocab_size), (batch_size*seq_length, )
loss_mlm = self.lm_loss_fct(s_logits.view(-1, s_logits.size(-1)), lm_labels.view(-1))
```

#### 2.2.2.3 Loss_cos
```python
# hidden_state of the last layer, (batch_size, seq_length, dim)
s_hidden_states = s_hidden_states[-1]  
t_hidden_states = t_hidden_states[-1]

# attention_mask: (batch_size, seq_length), mask: (batch_size, seq_length, dim)
mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)
dim = s_hidden_states.size(-1)

# (sum(lengths), dim) Change the shape of the data into 2 dimensions
s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)
s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)
t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)
t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)       

target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)

# cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")
loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
```
#### 2.2.2.3 Total_Loss

```python
# alpha_ce = 5.0
# alpha_mlm = 2.0 
# alpha_cos = 1.0 
# alpha_clm = 0.0
# alpha_mse = 0.0

loss = alpha_ce * loss_ce +
       alpha_mlm * loss_mlm +
       alpha_cos * loss_cos
```

### 2.3 Conclusion

The model can be understood perceptually. The first and third loss are guaranteed to be the same as the teacher model, and the second loss is the guarantee of self (Bert), which is very explanatory.

* Practice has proved that the universal language model can be successfully trained through distillation.
* Use the knowledge of the teacher model to initialize the student model.
* The use of the cosine loss function can have a better performance effect.
