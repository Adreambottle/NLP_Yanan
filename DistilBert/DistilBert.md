# DistilBert 

## 1. Background

### 1.1. BERT model

<img src="./plots/image.png" alt="image" style="zoom:25%;" />


BERT (Bidirectional Encoder Representations from Transformers) 

#### 1.1.1. train task

* Only predict the masked words rather than reconstructing the entire input 
* Masked Language Model (MLM)  and next sentence prediction 

#### 1.1.2. Architecture

<img src="./plots/image (1).png" alt="image (1)" style="zoom:30%;" />

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

<img src="./plots/image (2).png" alt="image (2)" style="zoom: 50%;" />

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
  	<img src="./plots/image (3).png" alt="image (3)" style="zoom:30%;" />
  * Represent frequent clusters by less bits, represent rare clusters by more bits	
* **Parameter matrix approximation**
	* The purpose of reducing matrix parameters is achieved by low-rank decomposition of the matrix or other methods* 
    <img src="./plots/image (4).png" alt="image (4)" style="zoom:30%;" />
* **Weight sharing**
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

<img src="./plots/image (5).png" alt="image (5)" style="zoom:40%;" />

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

* Fix  $p_{mask} = p_{pos} = 0.1$ and $P_{ng} = 0.25$ across all datasets.